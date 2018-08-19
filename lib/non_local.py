import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, mode='patch_embedded_gaussian',
                 sub_sample=True, bn_layer=True, nb_patches=None):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [2]
        assert mode in ['patch_embedded_gaussian', 'embedded_gaussian', 'gaussian']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.nb_patches = nb_patches

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            # nn.init.constant_(self.W[1].weight, 0)
            # nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            # nn.init.constant_(self.W.weight, 0)
            # nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None

        if mode in ['patch_embedded_gaussian', 'embedded_gaussian']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'patch_embedded_gaussian':
                self.operation_function = self._patch_embedded_gaussian
            elif mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))


    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """

        output = self.operation_function(x)
        return output

    def _patch_embedded_gaussian(self, x):
        batch_size = x.size(0)
        h = x.size(-2)
        w = x.size(-1)

        if self.nb_patches is None:
            nb_patches = [h, w]
        elif isinstance(self.nb_patches, int):
            nb_patches = [self.nb_patches, self.nb_patches]
        else:
            nb_patches = self.nb_patches
        if not h % nb_patches[0] == 0 or not w % nb_patches[1] == 0:
            raise ValueError('(WDX Warning) The patch size must be compatible with the feature map size.')

        # assuming nb_patch == 8,
        # g=>(b, c, h, w)->(b, 0.5c, h, w)->(b, 0.5b, 8, h/8, 8, w/8)
        g_x = self.g(x).view(batch_size, self.inter_channels,
                             nb_patches[0], h / nb_patches[0],
                             nb_patches[1], w / nb_patches[1])
        # g=>(b, 8, 8, 0.5c, h/8, w/8)->(b, 8*8, 0.5c*h/8*w/8)
        g_x = g_x.permute(0, 2, 4, 1, 3, 5).contiguous()
        g_x = g_x.view(batch_size, nb_patches[0] * nb_patches[1], -1)
        # print(g_x.size())
        # theta=>(b, c, h, w)[->(b, 0.5c, h, w)->(b, 0.5c, 8, h/8, 8, w/8)]->(b, 8*8, 0.5c*h/8*w/8)
        # phi  =>(b, c, h, w)[->(b, 0.5c, h, w)->(b, 0.5c, 8, h/8, 8, w/8)]->(b, 0.5c*h/8*w/8, 8*8)
        # f=>(b, 8*8, 0.5c*h/8*w/8)dot(b, 0.5c*h/8*w/8, 8*8) = (b, 8*8, 8*8)
        theta_x = self.theta(x).view(batch_size, self.inter_channels,
                                     nb_patches[0], h / nb_patches[0],
                                     nb_patches[1], w / nb_patches[1])
        theta_x = theta_x.permute(0, 2, 4, 1, 3, 5).contiguous()
        theta_x = theta_x.view(batch_size, nb_patches[0] * nb_patches[1], -1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels,
                                 nb_patches[0], h / nb_patches[0],
                                 nb_patches[1], w / nb_patches[1])
        phi_x = phi_x.permute(0, 2, 4, 1, 3, 5).contiguous()
        phi_x = phi_x.view(batch_size, nb_patches[0] * nb_patches[1], -1)
        phi_x = phi_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, 8*8, 8*8)dot(b, 8*8, 0.5c*h/8*w/8) = (b, 8*8, 0.5c*h/8*w/8)
        # ->(b, 8, 8, 0.5c, h/8, w/8)->(b, c, 8, h/8, 8, w/8)->(b, c, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, nb_patches[0], nb_patches[1],
                   self.inter_channels, h/nb_patches[0], w/nb_patches[1])
        y = y.permute(0, 3, 1, 4, 2, 5).contiguous()
        y = y.view(batch_size, self.inter_channels, h, w)

        # y = y.permute(0, 2, 1).contiguous()
        # y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

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


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='patch_embedded_gaussian',
                 sub_sample=True, bn_layer=True, nb_patches=None):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer,
                                              nb_patches=nb_patches)


if __name__ == '__main__':
    from torch.autograd import Variable

    mode_list = ['patch_embedded_gaussian']

    for mode in mode_list:
        img = Variable(torch.randn(2, 6, 10, 10))
        net = NONLocalBlock2D(6, mode=mode, sub_sample=False, bn_layer=False, nb_patches=10)
        out1 = net(img)
        print(out1.size())

        net.operation_function = net._embedded_gaussian
        out2 = net(img)
        print(out2.size())
        # print(out2)

        print(out1 - out2)

