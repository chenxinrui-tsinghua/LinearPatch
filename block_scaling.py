import torch.nn as nn
import torch
from rotate_utils.hadamard_utils import get_hadamard_matrix


def generate_symmetric_matrix(diag):
    orth = get_hadamard_matrix(diag.shape[-1], diag.device).to(torch.float32)
    diag = torch.diag(diag)
    symmetric_matrix = torch.matmul(orth, torch.matmul(diag, orth.T)).to(diag.device)
    return symmetric_matrix


class LinearPatch(nn.Module):
    def __init__(self, diag=None, rotated=True, weight=None):
        super(LinearPatch, self).__init__()
        if weight is not None:
            self.weight = nn.Parameter(weight)
        elif rotated and diag is not None:
            self.weight = nn.Parameter(generate_symmetric_matrix(diag))
        else:
            self.weight = nn.Parameter(torch.diag(diag))
    def forward(self, x):
        return torch.matmul(x.to(torch.float32), self.weight.to(torch.float32)).to(torch.float16)
    def get_weight(self):
        return self.weight.to(torch.float32)


def register_linear_patch(model, start_l, scale_param, dev, rotated=True, weight=None):
    def rotated_hook(module, input, output):
        output = list(output)
        output[0] = module.linear_patch(output[0])
        output = tuple(output)
        return output
    handles = []
    for i, layer in enumerate(model.model.layers):
        if i == start_l - 1:
            if scale_param is not None:
                linear_patch = LinearPatch(scale_param.to(torch.float32).to(dev), rotated, weight=None)
            else:
                linear_patch = LinearPatch(None, rotated, weight=weight.to(torch.float32).to(dev))
            setattr(layer, 'linear_patch', linear_patch)
            handle = layer.register_forward_hook(rotated_hook)
            handles.append(handle)
            break
    return handles