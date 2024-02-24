import torch
import torch.nn as nn


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        # Compute the matrix multiplication between input and the tensor
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        # Calculate the L1 distance between samples
        diff = torch.abs(matrices.unsqueeze(0) - matrices.unsqueeze(1))
        diff = diff.sum(dim=3)
        diff = torch.exp(-diff)

        # Sum or mean the distances
        if self.mean:
            output = diff.mean(dim=1)
        else:
            output = diff.sum(dim=1)

        return torch.cat([x, output], 1)
