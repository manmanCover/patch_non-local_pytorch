import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True, nb_patches=None):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.nb_patches = nb_patches
        self.use_patch = False

        if nb_patches is not None:
            self.use_patch = True

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':           # ???
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        x = self._tensor2patch(x)
        output = self.operation_function(x)
        output = self._patch2tensor(output)
        return output

    def _tensor2patch(self, x):
        if not self.use_patch:
            return x

        tensor_size = x.size()

        if self.nb_patches is None:
            nb_patches = [1] * self.dimension
        elif isinstance(self.nb_patches, int):
            nb_patches = [self.nb_patches] * self.dimension
        else:
            nb_patches = self.nb_patches

        for dim in range(1, self.dimension + 1):    # dim \in {1, 2, 3}
            if not tensor_size[-dim] % nb_patches[-dim] == 0:
                raise ValueError('(WDX Warning) The patch size must be compatible with the feature map size.')
        self.nb_patches = nb_patches

        # x: (b, c, t, h, w)
        if self.dimension == 3:
            b, c, t, h, w = tensor_size
            x = x.view(b, c,
                       nb_patches[0], t/nb_patches[0],
                       nb_patches[1], h/nb_patches[1],
                       nb_patches[2], w/nb_patches[2])
            x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            x = x.view(-1, c, t/nb_patches[0], h/nb_patches[1], w/nb_patches[2])
        elif self.dimension == 2:
            b, c, h, w = tensor_size
            x = x.view(b, c,
                       nb_patches[0], h / nb_patches[0],
                       nb_patches[1], w / nb_patches[1])
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(-1, c, h/nb_patches[0], w/nb_patches[1])
        elif self.dimension == 1:
            b, c, w = tensor_size
            x = x.view(b, c,
                       nb_patches[0], w / nb_patches[0])
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(-1, c, w / nb_patches[0])
        return x

    def _patch2tensor(self, x):
        if not self.use_patch:
            return x
        tensor_size = x.size()
        nb_patches = self.nb_patches
        if self.dimension == 3:
            # x: (b*m*n, c, t, h, w)
            p, c, tt, hh, ww = tensor_size
            x = x.view(-1, nb_patches[0], nb_patches[1], nb_patches[2], c, tt, hh, ww)
            x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            x = x.view(-1, c, nb_patches[0] * tt, nb_patches[1] * hh, nb_patches[2] * ww)
        elif self.dimension == 2:
            p, c, hh, ww = tensor_size
            x = x.view(-1, nb_patches[0], nb_patches[1], c, hh, ww)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            x = x.view(-1, c, nb_patches[0] * hh, nb_patches[1] * ww)
        elif self.dimension == 1:
            p, c, ww = tensor_size
            x = x.view(-1, nb_patches[0], c, ww)
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(-1, c, nb_patches[0] * ww)
        return x

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True, nb_patches=None):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              nb_patches=nb_patches)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True, nb_patches=None):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              nb_patches=nb_patches)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True, nb_patches=None):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              nb_patches=nb_patches)


if __name__ == '__main__':
    from torch.autograd import Variable

    mode_list = ['embedded_gaussian', 'gaussian', 'dot_product', ]
    # mode_list = ['concatenation']

    for mode in mode_list:
        print(mode)
        img = Variable(torch.randn(2, 4, 5))
        net = NONLocalBlock1D(4, mode=mode, sub_sample=True)
        nn.init.uniform_(net.W[1].weight)
        nn.init.uniform_(net.W[1].bias)
        out1 = net(img)
        net.nb_patches = [1]
        out2 = net(img)
        print('\t', torch.sum(out2 - out1))

        img = Variable(torch.randn(2, 4, 10, 10))
        net = NONLocalBlock2D(4, mode=mode, sub_sample=False)
        nn.init.uniform_(net.W[1].weight)
        nn.init.uniform_(net.W[1].bias)
        out1 = net(img)
        net.nb_patches = [1, 1]
        out2 = net(img)
        print('\t', torch.sum(out2 - out1))

        img = Variable(torch.randn(2, 4, 5, 4, 5))
        net = NONLocalBlock3D(4, mode=mode)
        nn.init.uniform_(net.W[1].weight)
        nn.init.uniform_(net.W[1].bias)
        out1 = net(img)
        net.nb_patches = [1, 1, 1]
        out2 = net(img)
        print('\t', torch.sum(out2 - out1))

    import numpy as np
    x = torch.Tensor(np.arange(6).reshape(1, 1, 6))
    print(x)
    net = NONLocalBlock1D(4, mode=mode, sub_sample=True, nb_patches=[3])
    x = net._tensor2patch(x)
    print(x)
    x = net._patch2tensor(x)
    print(x)

    x = torch.Tensor(np.arange(16).reshape(1, 1, 4, 4))
    print(x)
    net = NONLocalBlock2D(4, mode=mode, sub_sample=True, nb_patches=[2, 2])
    x = net._tensor2patch(x)
    print(x)
    x = net._patch2tensor(x)
    print(x)
