import torch.nn as nn
import torch
import kornia as K
from utils import deform_image
import reproducing_kernels as rk

class res_block(nn.Module):

    def __init__(self, h, n_in=2, n_out=1):
        super().__init__()
        self.n_in = n_in
        if n_in == 3:
            kernel_size = 3
            pad = 1
        else:
            kernel_size = 3
            pad = 1
        self.conv1 = nn.Conv2d(n_in, h, kernel_size, bias=True, padding=pad)
        self.conv2 = nn.Conv2d(h, n_out, kernel_size, bias=False, padding=pad)
        #self.conv3 = nn.Conv2d(h, n_out, kernel_size, bias=False, padding=pad)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, z, I):
        if self.n_in ==3:
            z = z.permute(0,3,1,2)
        x = torch.cat([z,I], dim=1)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)

        if self.n_in == 3:
            x = x.permute(0,2,3,1)
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


class meta_model_sharp(nn.Module):

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
        residuals_deformed = [self.residuals[0]]
        self.field = []
        self.grad = []
        self.phi = [torch.cat([self.id_grid for _ in range(source.shape[0])])]
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i])
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            deformation = self.id_grid - self.field[i] / self.l
            f = self.res_list[i](self.residuals[i], image[i]) * 1 / self.l
            self.residuals.append(self.residuals[i] + f)
            residuals_deformed.append(self.residuals[-1])
            residuals_deformed = [deform_image(z, deformation) for z in residuals_deformed]
            self.phi.append(deform_image(self.phi[-1].permute(0,3,1,2), deformation).permute(0,2,3,1))
            image.append(deform_image(image[0], self.phi[i+1]) + sum(residuals_deformed[1:]) * mu**2 / self.l)

        return image, self.field, self.residuals, self.grad

class meta_model_local_sharp(nn.Module):

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
        #self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
        #                                         (sigma_v, sigma_v),
        #                                         border_type='constant')

        self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v), border_type='constant')
        self.mu = mu

    def forward(self, source, source_seg):
        """image = []
        image.append(source)"""
        image = source.clone()
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        residuals_deformed = [self.residuals[0]]
        self.field = []
        self.grad = []
        self.phi = [torch.cat([self.id_grid for _ in range(source.shape[0])])]
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image)
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            deformation = self.id_grid - self.field[i] / self.l
            f = self.res_list[i](self.residuals[i], image) * 1 / self.l
            self.residuals.append(self.residuals[i] + f)
            residuals_deformed = [deform_image(z, deformation) for z in residuals_deformed]
            residuals_deformed.append(self.residuals[-1])
            self.phi.append(deform_image(self.phi[-1].permute(0,3,1,2), deformation).permute(0,2,3,1))

            mask = deform_image(source_seg, self.phi[i+1])
            image = deform_image(source, self.phi[i+1]) + sum(residuals_deformed[1:]) * mu**2 / self.l * mask
        self.seg = mask
        return image, self.field, self.residuals, self.grad

class double_resnet(nn.Module):

    def __init__(self, l, im_shape, device, mu, z0, h=100):
        super().__init__()
        self.l = l
        self.z_list = []
        self.v_list = []
        self.device = device
        for i in range(l):
            self.z_list.append(res_block(h))
            self.v_list.append(res_block(h, 3, 2))
        self.z_list = nn.ModuleList(self.z_list)
        self.v_list = nn.ModuleList(self.v_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.mu = mu
        self.kernel = K.filters.GaussianBlur2d((51, 51),
                                               (10, 10),
                                               border_type='constant')

    def forward(self, source, source_seg):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        residuals_deformed = [self.residuals[0]]
        deformation = self.id_grid.clone()
        mu = self.mu
        self.field = []
        self.grad = [K.filters.SpatialGradient(mode='sobel')(source)]
        for i in range(self.l):
            f = self.z_list[i](self.residuals[i], image[i]) * 1 / self.l
            self.residuals.append(self.residuals[i] + f)
            residuals_deformed = [deform_image(z, deformation) for z in residuals_deformed]
            residuals_deformed.append(self.residuals[-1])
            self.field.append(self.kernel(self.v_list[i](deformation, image[-1])))
            deformation = deformation - self.field[i] / self.l
            mask = deform_image(source_seg, deformation)
            image.append(deform_image(image[0], deformation) + sum(residuals_deformed[1:]) * mu**2 / self.l * mask)
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i+1])
            self.grad.append(grad_image)

        self.seg = mask
        return image, self.field, self.residuals, self.grad