import torch
import torch.nn as nn
import torch.nn.functional as F


class ENLAttention(nn.Module):
    """
    Efficient Nonlocal Attention Module:
    First, the unified attention information is learned, and the nonlocal spatial attention is degenerated into a nonlocal channel attention
    Second, a 1-D convolution is used to interact for information interactions among channels.
    """

    def __init__(self, k):
        super(ENLAttention, self).__init__()
        self.conv_k = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
        # self.softmax = nn.Softmax(dim=2)
        self.conv_d = nn.Conv1d(1, 1, kernel_size=k, stride=1, padding=1)
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        # unified correlation matrix
        atten_map = self.conv_k(x)
        atten_map = atten_map.view(B, 1, H * W)
        atten_map = F.softmax(atten_map, dim=2)
        # atten_map = self.softmax(atten_map)
        chan_seq = x.view(B, H * W, C)
        atten_vec = torch.matmul(atten_map, chan_seq)
        atten_vec = self.conv_d(atten_vec)
        atten_vec = atten_vec.unsqueeze(1).permute(0, 3, 1, 2)
        x = x * atten_vec
        x = x * self.resweight

        identity = identity + x

        return identity
