"""External auxiliary-head KD; serving uses only the existing VM head."""
import torch
from torch import nn
from torch.nn import functional as F


class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.scale, None


class AuxiliaryKD(nn.Module):
    def __init__(self, vm, cfg):
        super().__init__()
        self.vm = vm
        self.beta = cfg['gradient_scale']
        # Branch before serving-only layers, after shared first 1024-wide ReLU.
        # No auxiliary dropout: constructing/using this branch leaves VM RNG intact.
        device = next(vm.parameters()).device
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        torch.manual_seed(cfg['auxiliary_seed'])
        self.auxiliary = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, device)
        self.hidden = None
        self.hook = vm.dnn.mlp[2].register_forward_hook(self.capture)

    def capture(self, module, args, output):
        self.hidden = output

    def forward(self, batch):
        return self.vm(batch)

    def training_logits(self, batch):
        serving = self.vm(batch)
        auxiliary = self.auxiliary(GradientScale.apply(self.hidden, self.beta)).squeeze(-1)
        return serving, auxiliary

    def regularization_loss(self):
        return self.vm.regularization_loss()


def kd_losses(serving, auxiliary, labels, teacher_logits, cfg):
    target = (torch.sigmoid(teacher_logits.detach()) * cfg['label_scale']).clamp(0, 1)
    task = F.binary_cross_entropy_with_logits(serving, labels)
    distillation = F.binary_cross_entropy_with_logits(auxiliary, target)
    return task, distillation, task + cfg['loss_weight'] * distillation
