import torch
import torch.nn.functional as F
import torch.nn as nn

# from ptflops import get_model_complexity_info


def pad_conv_stride(I, kernel_size, stride, shift=False):
    left_pad = stride
    right_pad = 0 if (I.shape[3] + left_pad - kernel_size) % stride == 0 else stride - ((I.shape[3] + left_pad - kernel_size) % stride)
    top_pad = stride
    bot_pad = 0 if (I.shape[2] + top_pad - kernel_size) % stride == 0 else stride - ((I.shape[2] + top_pad - kernel_size) % stride)
    right_pad += stride
    bot_pad += stride

    if not shift:
        I_padded = F.pad(I, (left_pad, right_pad, top_pad, bot_pad), mode='reflect')
        valids = F.pad(torch.ones_like(I), (left_pad, right_pad, top_pad, bot_pad), mode='constant')
    else:
        I_padded = torch.zeros(I.shape[0], stride**2, I.shape[1], top_pad + I.shape[2] + bot_pad, left_pad + I.shape[3] + right_pad).type_as(I)
        valids = torch.zeros_like(I_padded)
        for num, (row_shift, col_shift) in enumerate([(i, j) for i in range(stride) for j in range(stride)]):
            I_shift = F.pad(I, pad=(left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')

            valid = F.pad(torch.ones_like(I),
                          pad=(left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift),
                          mode='constant')

            I_padded[:, num, :, :, :] = I_shift
            valids[:, num, :, :, :] = valid

        I_padded = I_padded.reshape(-1, *I_padded.shape[2:])
        valids = valids.reshape(-1, *valids.shape[2:])

    return I_padded, valids


def conv_power_method(D, img_size, num_iters=100, stride=1, use_gpu=True):
    """
    Finds the maximal eigenvalue of D.T.dot(D) using the iterative power method
    :param D:
    :param num_needles:
    :param image_size:
    :param patch_size:
    :param num_iters:
    :return:
    """
    z = torch.zeros((D.shape[1], *img_size)).reshape(1, D.shape[1], *img_size)
    if use_gpu:
        z = z.cuda()
        D = D.cuda()

    z = F.conv2d(z, D, stride=stride)
    z = torch.randn_like(z)
    z = z / torch.norm(z.reshape(-1))
    L = None
    for _ in range(num_iters):
        Dz = F.conv_transpose2d(z, D, stride=stride)
        DTDz = F.conv2d(Dz, D, stride=stride)
        L = torch.norm(DTDz.reshape(-1))
        z = DTDz / L
    return L.item()


class SoftThreshold(nn.Module):

    def __init__(self, num_channels, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1, num_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        threshold = self.threshold
        out = torch.sign(x) * torch.clamp_min(x.abs() - threshold, 0)
        return out


class CSCNet(nn.Module):

    def __init__(self, in_ch, num_kernels, kernel_size, stride, num_iter) -> None:
        super().__init__()

        self.num_iter = num_iter

        self.stride = stride
        self.conv = torch.nn.Conv2d(in_ch, num_kernels, kernel_size, stride=stride, bias=False)
        self.convT = torch.nn.ConvTranspose2d(num_kernels, in_ch, kernel_size, stride=stride, bias=False)

        self.decode = torch.nn.ConvTranspose2d(num_kernels, in_ch, kernel_size, stride=stride, bias=False)
        self.active = SoftThreshold(num_kernels, 0.001)

        W = torch.clone(self.conv.weight.data)
        eigen = conv_power_method(W, img_size=[100, 100], stride=1, num_iters=100)
        W /= eigen**0.5

        self.convT.weight.data = torch.clone(W)
        self.conv.weight.data = torch.clone(W)
        self.decode.weight.data = torch.clone(W)

    def forward(self, I):
        mean = I.flatten(1).mean(dim=1).reshape(I.shape[0], 1, 1, 1)
        I = I - mean
        shape = I.shape
        I, valids = pad_conv_stride(I, self.conv.kernel_size[0], self.stride, True)
        code = self.conv(I)
        code = self.active(code)
        for i in range(self.num_iter):
            res = I - self.convT(code)
            code = code + self.conv(res)
            code = self.active(code)

        out = self.decode(code)
        out = torch.masked_select(out, valids.bool())
        out = out.reshape(shape[0], -1, *shape[1:])
        out = out.mean(dim=1, keepdim=False)

        return out + mean


class ECSCNet(nn.Module):

    def __init__(self, in_ch, num_kernels, kernel_size, stride, num_iter) -> None:
        super().__init__()
        self.num_iter = num_iter
        self.stride = stride

        self.init = torch.nn.Conv2d(in_ch, num_kernels, kernel_size, stride=stride, bias=False)
        self.conv = torch.nn.Conv2d(in_ch, num_kernels, kernel_size, stride=stride, bias=False)
        self.convT = torch.nn.ConvTranspose2d(num_kernels, in_ch, kernel_size, stride=stride, bias=False)

        self.decoder = torch.nn.ConvTranspose2d(num_kernels, in_ch, kernel_size, stride=stride, bias=False)

        self.elastic = torch.nn.ModuleList()
        self.active = SoftThreshold(num_kernels, 1e-3)

        for i in range(num_iter):
            projection = torch.nn.Conv2d(num_kernels, num_kernels, 1, 1, 0, bias=False)
            W = torch.clone(projection.weight.data)
            length = W.squeeze().pow(2).sum(dim=1).pow(0.5).view(-1, 1, 1, 1)
            W = W / length * 0.01
            projection.weight.data = torch.clone(W)
            self.elastic.append(projection)

        W = torch.clone(self.conv.weight.data)
        eigen = conv_power_method(W, img_size=[100, 100], stride=1, num_iters=100)
        W /= eigen**0.5

        self.init.weight.data = torch.clone(W)
        self.decoder.weight.data = torch.clone(W)
        self.convT.weight.data = torch.clone(W)
        self.conv.weight.data = torch.clone(W)

    def forward(self, I: torch.Tensor, return_code=False):
        shape = I.shape
        mean = 0.5
        I = I - mean
        I, valids = pad_conv_stride(I, self.conv.kernel_size[0], self.stride, shift=True)
        Dx = self.conv(I)
        z = self.init(I)
        z = self.active(z)
        for i in range(self.num_iter):
            DDTz = self.conv(self.convT(z))
            Qz = self.elastic[i](z)
            z = z - DDTz + Dx + Qz
            z = self.active(z)

        out = self.decoder(z)
        out = torch.masked_select(out, valids.bool())
        out = out.reshape(shape[0], -1, *shape[1:])
        out = out.mean(dim=1, keepdim=False) + mean
        if return_code:
            return out, z
        return out
