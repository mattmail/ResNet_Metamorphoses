import torch.nn as nn
import torch
import kornia as K
from utils import deform_image

class res_block(nn.Module):

    def __init__(self, h):
        super().__init__()
        self.conv1 = nn.Conv2d(2, h, 3, bias=True, padding=1)
        self.conv2 = nn.Conv2d(h, h, 3, bias=True, padding=1)
        self.conv3 = nn.Conv2d(h, 1, 3, bias=True, padding=1)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, z, I):
        x = torch.cat([z,I], dim=1)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv3(self.leaky_relu(self.conv2(x)))
        return x

class meta_model(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0, h=100):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i])
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            f = self.res_list[i](self.residuals[i], image[i]) * 1/self.l
            self.residuals.append(self.residuals[i] + f)
            deformation = self.id_grid - self.field[i]/self.l
            image.append(deform_image(image[i], deformation) + self.residuals[i+1] * self.mu**2 / self.l)

        return image, self.field, self.residuals, self.grad

class meta_model_local(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0, h=100):
        super().__init__()
        self.l = l
        self.res_list = []
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(h))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source, source_seg):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i])
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            f = self.res_list[i](self.residuals[i], image[i]) * 1 / self.l
            self.residuals.append(self.residuals[i] + f)

            deformation = self.id_grid - self.field[i]/self.l
            source_seg = deform_image(source_seg, deformation)
            image.append(deform_image(image[i], deformation) + self.residuals[i+1] * mu**2 / self.l * source_seg)

        return image, self.field, self.residuals, self.grad