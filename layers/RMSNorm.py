import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, dim=-1, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(self.dim,-1)
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = self.weight * hidden_states
        hidden_states = hidden_states.transpose(self.dim,-1)
        return hidden_states
