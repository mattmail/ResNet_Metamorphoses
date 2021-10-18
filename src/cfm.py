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
from model import UNet
import nibabel as nib

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

    def forward(self, source):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(self.z0)
        self.field = []
        self.grad = []
        mu = self.mu
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i])
            self.grad.append(grad_image)
            self.field.append(self.kernel(-self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0,2,3,1)
            self.residuals.append(self.residuals[i] + self.res_list[i](self.residuals[i], image[i]) * 1/self.l)

            deformation = self.id_grid - self.field[i]/self.l
            image.append(self.deform_image(image[i], deformation) + self.residuals[i+1] * mu**2 / self.l)


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

def dice(pred, gt):
    eps = 1e-10
    tp = torch.sum(torch.mul(pred, gt))
    fp = torch.sum(torch.mul(pred, 1 - gt))
    fn = torch.sum(torch.mul(1 - pred, gt))
    dice_eps = (2. * tp + eps) / (2. * tp + fp + fn + eps)
    return dice_eps


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
MNI_img = nib.load("/usr/local/fsl/data/linearMNI/MNI152lin_T1_1mm_brain.nii.gz").get_fdata()
target_img = np.pad(MNI_img[:,:,93], ((18,18),(0,0)), "constant")
target_img = np.rot90(cv2.resize(target_img, (208,208)))
target_img = torch.from_numpy(target_img.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device) / 255.
seg = True
if seg:
    model_path = "/home/matthis/Nextcloud/ventricleSeg/results/baseline_1005_1502/pkl/Checkpoint_best.pkl"
    checkpoint = torch.load(model_path)
    seg_model = UNet(checkpoint['config']).to(device)
    seg_model.load_state_dict(checkpoint['model_state_dict'])
    mask = MNI_img != 0
    MNI_img = MNI_img / 255.
    MNI_img_scaled = MNI_img.copy()
    MNI_img_scaled = (MNI_img - MNI_img[mask].mean()) / (MNI_img[mask].std() + 1e-30)
    MNI_img_scaled = np.pad(MNI_img_scaled[:, :, 93], ((18, 18), (0, 0)), "constant")
    MNI_img_scaled = np.rot90(cv2.resize(MNI_img_scaled, (208, 208)))
    MNI_img_scaled = torch.from_numpy(MNI_img_scaled.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
    target_seg = seg_model(MNI_img_scaled)


test_list = []
with open("test_list.txt", "r") as f:
    test_paths = f.readlines()
    test_paths = [path[:-1] for path in test_paths]
    f.close()

for image in test_paths:
    test_seg = np.transpose(np.load("../data_miccai_2D_2021/" + image + "/" + image + "_seg.npy"))
    test_seg[test_seg == 2.] = 1.
    test_seg[test_seg == 4.] = 1.
    test_seg = torch.from_numpy(test_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    test_list.append(torch.cat([torch.from_numpy(
        np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
        torch.FloatTensor).unsqueeze(0).unsqueeze(0), test_seg]))


source = test_list[0][0].unsqueeze(0)
n_iter = 500
l= 15
L2_weight = .5
v_weight = 3e-5/l
z_weight = 3e-5/l
mu = 0.



print("### Starting LDDMM")
print("L2_weight=", L2_weight)
print("z_weight=", z_weight)
print("v_weight=", v_weight)
print("n_iter=", n_iter)
print("mu=", mu)

L2_norm_list = []
dice_list = []
time_list = []

for image in test_list:
    source = image[0].unsqueeze(0).to(device)
    source_seg = image[1].unsqueeze(0).to(device)
    _, _, h, w = source.shape
    source_seg = source_seg == 0
    print((1-source_seg*1).sum())
    t = time()
    z0 = torch.zeros(source.shape).float()
    model = meta_model(l, source.shape, device, 51, 10., mu, z0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-8)

    for i in range(n_iter):
        source_deformed, fields, residuals, grad = model(source)
        v_norm = torch.stack([(residuals[j] * grad[j].squeeze(1) * (-fields[j].permute(0, 3, 1, 2))).sum() for j in
                              range(len(fields))]).sum()
        residuals_norm = (torch.stack(residuals) ** 2).sum()
        L2_norm = ((source_deformed[l] - target_img)[source_seg] ** 2).sum()
        total_loss = L2_weight * L2_norm + v_weight * v_norm + mu * z_weight * residuals_norm
        #print("Iteration %d: Total loss: %f   L2 norm: %f   V_reg: %f   Z_reg: %f" % (i, total_loss, L2_norm, v_norm, residuals_norm))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        L2_norm = ((source_deformed[l] - target_img) ** 2).sum()
        if i == 500:
            for g in optimizer.param_groups:
                g['lr'] = 5e-4
        if i == 1000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
        if (i + 1) % 500 == 0:
            fig, ax = plt.subplots(2, 3, figsize=(7.5, 5), constrained_layout=True)
            ax[0, 0].imshow(source.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
            ax[0, 0].set_title("Source")
            ax[0, 0].axis('off')
            ax[0, 1].imshow(target_img.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
            ax[0, 1].set_title("Target")
            ax[0, 1].axis('off')
            arg = source_deformed[l].argmax()
            ax[1, 0].imshow(source_deformed[l].squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
            ax[1, 0].set_title("Deformed")
            ax[1, 0].axis('off')
            superposition = torch.stack([target_img.squeeze().detach().cpu(), source_deformed[l].squeeze().detach().cpu(),
                                         torch.zeros(source.squeeze().shape)], dim=2)
            ax[1, 1].imshow(superposition, vmin=0, vmax=1)
            ax[1, 1].set_title("Superposition")
            ax[1, 1].axis('off')
            id_grid = K.utils.grid.create_meshgrid(h, w, False, device)
            residuals_deform = residuals[l]
            for r in range(1, l):
                res_tmp = residuals[r]
                for f in range(r, l):
                    res_tmp = model.deform_image(res_tmp, id_grid - fields[f] / l)
                residuals_deform += res_tmp
            residuals_deform = residuals_deform * mu ** 2 / l

            im = ax[1, 2].imshow(residuals_deform.squeeze().detach().cpu().numpy())
            cbar = ax[1, 2].figure.colorbar(im, ax=ax[1, 2])
            ax[1, 2].set_title("Residuals heatmap")
            ax[1, 2].axis('off')
            ax[0, 2].set_title('Shape deformation only')
            # rect = patches.Rectangle((rect_xmin, rect_ymin), rect_xlength, rect_ylength, fill=False, edgecolor="red")
            # ax[0, 2].add_patch(rect)
            source_deformed_shape = source
            deformations = [source]
            for j in range(l):
                deformations.append(model.deform_image(deformations[j], id_grid - fields[j] / l))

            # source_deformed_shape = model.deform_image(source, deform, interpolation="bilinear")
            ax[0, 2].imshow(deformations[l].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
            ax[0, 2].axis('off')
            fig.suptitle('Metamorphoses, iteration: %d' % (i + 1))
            plt.savefig('Metamorphoses.png')
            plt.axis('off')
            plt.show()
    L2_norm_list.append(L2_norm.detach().cpu().item())
    time_list.append(time()-t)
    s_mask = (source_deformed[l] != 0)
    source_deformed[l] = (source_deformed[l] - source_deformed[l][s_mask].mean()) / (source_deformed[l][s_mask].std() + 1e-30)
    pred_seg = seg_model(source_deformed[l])
    pred_seg = 1. * (pred_seg > 0.5)
    dice_list.append(dice(pred_seg.float(), target_seg.squeeze(0)).detach().cpu())
    print("Validation L2:", L2_norm_list[-1])
    print("Validation dice:", dice_list[-1])
    print("Validation time:", time_list[-1])
L2_norm_list = np.array(L2_norm_list)
dice_list = np.array(dice_list)
time_list = np.array(time_list)
print("Validation L2 loss: %f" % (L2_norm_list.mean()), "std: %f" % (L2_norm_list.std()))
print("Validation dice: %f" % (dice_list.mean()), "std: %f" % (dice_list.std()))
print("Validation time:", time_list.mean(), "std: %f" % (time_list.std()))


