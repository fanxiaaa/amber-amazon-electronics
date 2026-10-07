"""Build the archived 8192-user prefix protocol from the pinned public ZIP.

Download the archive separately; this command never downloads or extracts files.
"""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from amber.experiment import sha256, write

ARCHIVE_SHA256 = '83183787433dbf559744f00ef09af52e60239908946b33484576910143c8a5d3'
HEADER = ['label', 'user_id', 'item_id', 'cate_id', 'item_history', 'cate_history']


def rows(archive, name):
    matches = [n for n in archive.namelist() if n == name or n.endswith('/' + name)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one ' + name)
    with archive.open(matches[0]) as stream:
        reader = csv.reader(io.TextIOWrapper(stream, encoding='utf-8'))
        if next(reader) != HEADER:
            raise ValueError('Unexpected CSV fields')
        yield from enumerate(reader)


def rank_users(users, seed):
    return sorted(users, key=lambda u: (hashlib.sha256(f'{seed}:{u}'.encode()).digest(), u))


def encode_records(records, mapping):
    fields = ('row_id', 'stage', 'label', 'user_id', 'item', 'category', 'review_prefix', 'length')
    arrays = {k: np.empty(len(records), np.int64) for k in fields}
    arrays.update(history_item=np.zeros((len(records), 100), np.int32),
                  history_category=np.zeros((len(records), 100), np.int32))
    for j, (row, stage, label, user, item, category, hist, cats) in enumerate(records):
        prefix = len(hist)
        hist, cats = hist[-100:], cats[-100:]
        values = (row, stage, label, user, mapping['item'].get(item, 1),
                  mapping['category'].get(category, 1), prefix, len(hist))
        for key, value in zip(fields, values):
            arrays[key][j] = value
        for name, values in [('item', hist), ('category', cats)]:
            arrays['history_' + name][j, :len(values)] = [mapping[name].get(v, 1) for v in values]
    return arrays


def prepare(archive_path, output):
    if sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError('Use the pinned BARS archive; its SHA256 does not match')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(archive_path) as archive:
        prefixes = {}
        counts = {}
        for _, row in rows(archive, 'train.csv'):
            user = int(row[1])
            prefixes.setdefault(user, set()).add(len(row[4].split('^')))
            counts[user] = counts.get(user, 0) + 1
        # Reconstruct the prior cohort deterministically; no user-ID list is shipped.
        excluded = set(rank_users([u for u, p in prefixes.items() if len(p) >= 2], 20261010)[:8192])
        eligible = [u for u, p in prefixes.items() if len(p) >= 3 and u not in excluded]
        selected = set(rank_users(eligible, 20261015)[:8192])
        if len(selected) != 8192:
            raise ValueError('Insufficient eligible users')
        for u in selected:
            if prefixes[u] != set(range(1, max(prefixes[u]) + 1)) or counts[u] != 2 * len(prefixes[u]):
                raise ValueError('Expected contiguous paired user prefixes')
        maximum = {u: max(prefixes[u]) for u in selected}
        cut = {u: max(1, min(maximum[u] - 2, 4 * maximum[u] // 7)) for u in selected}
        records, pairs, vocabulary = [], {}, {'item': set(), 'category': set()}
        last_positive = {}
        for row_id, row in rows(archive, 'train.csv'):
            label, user, item, category = map(int, row[:4])
            if user not in selected:
                continue
            hist, cats = tuple(map(int, row[4].split('^'))), tuple(map(int, row[5].split('^')))
            n = len(hist)
            stage = 0 if n <= cut[user] else 1 if n < maximum[user] else 2
            key = (user, n)
            pair = pairs.setdefault(key, (hist, cats, set()))
            if len(cats) != n or label not in (0, 1) or pair[:2] != (hist, cats) or label in pair[2]:
                raise ValueError('Invalid paired training context')
            pair[2].add(label)
            if stage == 0:
                vocabulary['item'].update(hist + (item,))
                vocabulary['category'].update(cats + (category,))
            if stage == 2 and label == 1:
                last_positive[user] = (hist + (item,), cats + (category,))
            records.append((row_id, stage, label, user, item, category, hist, cats))
        if row_id + 1 != 2608764 or any(p[2] != {0, 1} for p in pairs.values()):
            raise ValueError('Training cardinality/pair mismatch')
        records.sort(key=lambda r: (len(r[6]), r[3], -r[2]))
        test_pairs = {}
        for row_id, row in rows(archive, 'test.csv'):
            label, user, item, category = map(int, row[:4])
            if user not in selected:
                continue
            hist, cats = tuple(map(int, row[4].split('^'))), tuple(map(int, row[5].split('^')))
            if (hist, cats) != last_positive[user] or label not in (0, 1):
                raise ValueError('Test history does not extend the validation positive')
            labels = test_pairs.setdefault(user, set())
            if label in labels:
                raise ValueError('Duplicate test candidate label')
            labels.add(label)
            records.append((2608764 + row_id, 3, label, user, item, category, hist, cats))
        if row_id + 1 != 384806 or set(test_pairs) != selected or any(v != {0, 1} for v in test_pairs.values()):
            raise ValueError('Test cardinality/pair mismatch')
    mapping = {k: {raw: i + 2 for i, raw in enumerate(sorted(v))} for k, v in vocabulary.items()}
    arrays = encode_records(records, mapping)
    stages = {str(s): int((arrays['stage'] == s).sum()) for s in range(4)}
    if stages != {'0': 54464, '1': 39330, '2': 16384, '3': 16384}:
        raise ValueError('Cohort stage counts differ from the archived protocol')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'inputs.npz', **arrays)
    write(output / 'vocabulary.json', mapping)
    write(output / 'manifest.json', dict(archive_sha256=ARCHIVE_SHA256,
          inputs_sha256=sha256(output / 'inputs.npz'), vocab_sizes={k: len(v) + 2 for k, v in mapping.items()},
          selected_users=len(selected), stages=stages, history_length=100,
          cohort_rule='Exclude first8192 hash20261010 users with>=2 contexts; select8192 hash20261015 users with>=3.',
          order='Train by prefix,user,positive-first; test retains published source order.',
          split='0 teacher; 1 downstream fit; 2 validation; 3 published paired test. Per-user ordinal prefixes, not global time.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output)
