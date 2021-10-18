import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
import kornia.filters as flt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import reproducing_kernels as rk
import seaborn as sns
import os
from time import time
import my_torchbox as tb
from scipy.stats.stats import pearsonr

class res_block(nn.Module):

    def __init__(self, h):
        super().__init__()
        self.conv1 = nn.Conv2d(2, h, 3, bias=True, padding=1)
        self.conv2 = nn.Conv2d(h, h, 3, bias=True, padding=1)
        self.conv3 = nn.Conv2d(h, 1, 3, bias=True, padding=1)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, z, I):
        x = torch.cat([z,I], dim=1)
        #x = z
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv3(self.leaky_relu(self.conv2(x)))
        return x

class meta_model(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0):
        super().__init__()
        self.l = l
        self.res_list = []
        self.im_shape = im_shape
        self.device = device
        for i in range(l):
            self.res_list.append(res_block(100))
        self.res_list = nn.ModuleList(self.res_list)

        self.z0 = nn.Parameter(z0)

        self.id_grid = K.utils.grid.create_meshgrid(im_shape[2], im_shape[3], False, device)
        self.kernel = K.filters.GaussianBlur2d((kernel_size,kernel_size),
                                                 (sigma_v, sigma_v),
                                                 border_type='constant')

        #self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v),
        #                                         border_type='constant')
        self.mu = mu

    def forward(self, source, target):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(self.z0)
        self.field = []
        self.grad = []
        self.div_diff = []
        self.corr = []
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i])
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            div = - tb.field_divergence((self.residuals[i] *self.field[i]).permute(0,2,3,1))
            self.field[i] = self.field[i].permute(0,2,3,1)
            f = self.res_list[i](self.residuals[i], image[i]) * 1/self.l

            self.div_diff.append((f-div).abs().sum().detach().cpu() / 208 / 208)
            self.residuals.append(self.residuals[i] + f)
            self.corr.append(pearsonr(f.flatten().detach().cpu().numpy(), div.flatten().detach().cpu().numpy())[0])
            deformation = self.id_grid - self.field[i]/self.l
            image.append(self.deform_image(image[i], deformation) + self.residuals[i+1] * self.mu**2 / self.l)


        return image, self.field, self.residuals, self.grad

    def deform_image(self, image, deformation, interpolation="bilinear"):
        _, _, H, W = image.shape
        mult = torch.tensor((2/(W-1), 2/(H-1))).unsqueeze(0).unsqueeze(0).to(device)
        deformation = deformation * mult - 1

        image = F.grid_sample(image, deformation, interpolation, padding_mode="border", align_corners=True)

        return image

def make_grid(size, step):
    grid = torch.ones(size)
    grid[:,:, ::step] = 0.
    grid[:,:,:, ::step] = 0.
    return grid

def integrate(v):
    l = len(v)
    id_grid = K.utils.grid.create_meshgrid(v[0].shape[1], v[0].shape[2], False, device)
    grid_def = id_grid - v[0]/l
    _, H, W, _ = grid_def.shape
    mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(device)

    for t in range(1, l):
        slave_grid = grid_def.detach()
        interp_vectField = F.grid_sample(v[t].permute(0,3,1,2), slave_grid * mult - 1, "nearest", padding_mode="border", align_corners=True).permute(0,2,3,1)

        grid_def -= interp_vectField/l

    return grid_def


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
print(os.getcwd())
#source = torch.from_numpy(cv2.resize(cv2.imread("reg_test_m0t.png", cv2.IMREAD_GRAYSCALE), (200, 200))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0) / 255.
#target = torch.from_numpy(cv2.resize(cv2.imread("reg_test_m0c.png", cv2.IMREAD_GRAYSCALE), (200, 200))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0) / 255.


sig = 1
smooth = flt.GaussianBlur2d((sig*6+1,sig*6+1), (sig, sig))
#source = smooth(source)
#target = smooth(target)

target = torch.from_numpy(np.transpose(np.load("../brats_2020_2D/healthy/BraTS20_Training_019/BraTS20_Training_019_t1ce.npy"))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
source = torch.from_numpy(np.transpose(np.load("../brats_2020_2D/cancerous/BraTS20_Training_010/BraTS20_Training_010_t1ce.npy"))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)

_,_,h,w = source.shape

n_iter = 500
l= 15
L2_weight = .5
v_weight = 3e-6/l
z_weight = 3e-6/l
mu = 0.015

z0 = torch.zeros(source.shape)

print("### Starting LDDMM")
print("L2_weight=", L2_weight)
print("z_weight=", z_weight)
print("v_weight=", v_weight)
print("n_iter=", n_iter)
print("mu=", mu)
t = time()
model = meta_model(l, source.shape, device, 51, 4., mu, z0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                              weight_decay=1e-8)

target_seg = torch.ones(source.shape)
rect_xmin = 28
rect_ymin = 85
rect_xlength = 40
rect_ylength = 29
target_seg[:,:,rect_ymin:rect_ymin+rect_ylength, rect_xmin:rect_xmin+rect_xlength] = 0
target_seg = target_seg == 1


#rect = patches.Rectangle((rect_xmin, rect_ymin), rect_xlength, rect_ylength, fill=False, edgecolor="red")

div_list = []
cor_list = []
for i in range(n_iter):
    """if i == 5:
        break"""
    source_deformed, fields, residuals, grad = model(source, target)
    div_list.append(model.div_diff)
    cor_list.append(model.corr)
    v_norm = torch.stack([(residuals[j]*grad[j].squeeze(1)*(-fields[j].permute(0,3,1,2))).sum() for j in range(len(fields))]).sum()
    residuals_norm = (torch.stack(residuals)**2).sum()
    L2_norm = ((source_deformed[l] - target)**2)[target_seg].sum()

    total_loss = L2_weight*L2_norm + v_weight*(v_norm) + mu*z_weight*residuals_norm
    print("Iteration %d: Total loss: %f   L2 norm: %f   V_reg: %f   Z_reg: %f" % (i, total_loss, L2_norm, v_norm, residuals_norm))
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    """if i > 500:
        if i % 30 ==0:
            if mu < 0.049:
                mu += 0.001
                model.mu = mu"""
    if i == 500:
        for g in optimizer.param_groups:
            g['lr'] = 5e-4
    if i == 1000:
        for g in optimizer.param_groups:
            g['lr'] = 1e-4
    if (i+1) % 500 == 0:
        print("current mu", mu)
        fig, ax = plt.subplots(2, 3, figsize=(7.5, 5), constrained_layout=True)
        ax[0, 0].imshow(source.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
        #rect = patches.Rectangle((rect_xmin, rect_ymin), rect_xlength, rect_ylength, fill=False, edgecolor="red")
        #ax[0, 0].add_patch(rect)
        ax[0, 0].set_title("Source")
        ax[0, 0].axis('off')
        ax[0, 1].imshow(target.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
        """rect = patches.Rectangle((rect_xmin, rect_ymin), rect_xlength, rect_ylength, fill=False, edgecolor="red")
        ax[0, 1].add_patch(rect)"""
        ax[0, 1].set_title("Target")
        ax[0, 1].axis('off')
        ax[1, 0].imshow((source_deformed[l]).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
        ax[1, 0].set_title("Deformed")
        ax[1, 0].axis('off')
        superposition = torch.stack([target.squeeze().detach().cpu(),
                                     source_deformed[l].squeeze().detach().cpu(),
                                     torch.zeros(source.squeeze().shape)], dim=2)
        ax[1, 1].imshow(superposition, vmin=0, vmax=1)
        ax[1, 1].set_title("Superposition")
        ax[1, 1].axis('off')
        """grid = make_grid(source.shape, 10).to(device)
        deform = integrate(fields)
        grid = model.deform_image(grid, deform, "bilinear")
        ax[1, 2].set_title('deformation grid')
        ax[1, 2].axis('off')
        ax[1, 2].imshow(grid.detach().cpu().squeeze(), cmap="gray")"""
        id_grid = K.utils.grid.create_meshgrid(h, w, False, device)
        residuals_deform = residuals[l]
        for r in range(1,l):
            res_tmp = residuals[r]
            for f in range(r,l):
                res_tmp = model.deform_image(res_tmp, id_grid - fields[f]/l)
            residuals_deform += res_tmp
        residuals_deform = residuals_deform*mu**2/l

        im = ax[1, 2].imshow(residuals_deform.squeeze().detach().cpu().numpy())
        cbar = ax[1, 2].figure.colorbar(im, ax=ax[1, 2])
        ax[1, 2].set_title("Residuals heatmap")
        ax[1, 2].axis('off')
        ax[0, 2].set_title('Shape deformation only')
        source_deformed_shape = source
        deformations = [source]
        for j in range(l):
            deformations.append(model.deform_image(deformations[j], id_grid - fields[j]/l))



        #source_deformed_shape = model.deform_image(source, deform, interpolation="bilinear")
        ax[0, 2].imshow(deformations[l].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
        ax[0, 2].axis('off')
        fig.suptitle('Metamorphoses, iteration: %d' % (i + 1))
        plt.savefig('Metamorphoses.png')
        plt.axis('off')
        plt.show()
        fig, ax = plt.subplots(l//5, 5,figsize=(10,10))
        for k in range(l):
            ax[k // 5, k % 5].imshow((source_deformed[k + 1]).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
            ax[k // 5, k % 5].set_title("t = %d" % (k + 1))
            ax[k // 5, k % 5].axis('off')
        fig.suptitle('Metamorphoses, iteration: %d' % (i + 1))
        plt.savefig('Metamorphoses_evolution.png')
        plt.show()


div_list = np.array(div_list)

plt.figure()
for i in range(div_list.shape[1]):
    plt.plot(div_list[:,i], label="div %d" %i)
    plt.legend()
plt.show()

cor_list = np.array(cor_list)
plt.figure()
for i in range(cor_list.shape[1]):
    plt.plot(cor_list[:,i], label="div %d" %i)
    plt.legend()
plt.show()



