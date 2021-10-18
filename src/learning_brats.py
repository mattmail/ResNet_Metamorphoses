import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from kornia import filters as flt
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import subprocess as sp
import os
import nibabel as nib
from model import UNet
from time import time


def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    print(memory_free_values)
    return memory_free_values


class res_block(nn.Module):

    def __init__(self, h):
        super().__init__()
        self.conv1 = nn.Conv2d(2, h, 3, bias=True, padding=1)
        self.conv2 = nn.Conv2d(h, h, 3, bias=True, padding=1)
        self.conv3 = nn.Conv2d(h, 1, 3, bias=True, padding=1)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, z, I):
        x = torch.cat([z, I], dim=1)
        # x = z
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv3(self.leaky_relu(self.conv2(x)))
        return x


class meta_model(nn.Module):

    def __init__(self, l, im_shape, device, kernel_size, sigma_v, mu, z0, mu_val):
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
        self.kernel = K.filters.GaussianBlur2d((kernel_size, kernel_size),
                                               (sigma_v, sigma_v),
                                               border_type='constant')

        # self.kernel = rk.GaussianRKHS2d((sigma_v, sigma_v),
        #                               border_type='constant')
        self.mu = mu
        self.mu_val = mu_val
        res_shape = (4,1,208,208)
        self.max_val = torch.ones(res_shape).to(device)
        self.min_val = torch.zeros(res_shape).to(device)

    def forward(self, source, source_seg):
        image = []
        image.append(source)
        self.residuals = []
        self.residuals.append(torch.cat([self.z0 for _ in range(source.shape[0])]))
        self.field = []
        self.grad = []
        for i in range(self.l):
            grad_image = K.filters.SpatialGradient(mode='sobel')(image[i])
            self.grad.append(grad_image)
            self.field.append(-self.kernel(self.residuals[i] * grad_image.squeeze(1)))
            self.field[i] = self.field[i].permute(0, 2, 3, 1)
            self.residuals.append(self.residuals[i] + self.res_list[i](self.residuals[i], image[i]) * 1 / self.l)
            deformation = self.id_grid - self.field[i] / self.l
            source_seg = self.deform_image(source_seg, deformation)
            img = self.deform_image(image[i], deformation)
            #self.residuals[i + 1][source_seg != 0] = (torch.min(torch.stack([img + self.residuals[i + 1] * (self.mu_val) ** 2 / self.l, self.max_val]), 0)[0] - img)[source_seg != 0] /(self.mu_val) ** 2 * self.l
            #self.residuals[i + 1][source_seg != 0] = (torch.max(torch.stack([img + self.residuals[i + 1] * (self.mu_val) ** 2 / self.l, self.min_val]), 0)[0] - img)[source_seg != 0] / (self.mu_val) ** 2 * self.l
            image.append(img + self.residuals[i + 1] * (self.mu_val * source_seg) ** 2 / self.l)

        return image, self.field, self.residuals, self.grad

    def deform_image(self, image, deformation, interpolation="bilinear"):
        _, _, H, W = image.shape
        mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(device)
        deformation = deformation * mult - 1

        image = F.grid_sample(image, deformation, interpolation, padding_mode="border", align_corners=True)

        return image


def make_grid(size, step):
    grid = torch.ones(size)
    grid[:, :, ::step] = 0.
    grid[:, :, :, ::step] = 0.
    return grid

def dice(pred, gt):
    eps = 1e-10
    tp = torch.sum(torch.mul(pred, gt))
    fp = torch.sum(torch.mul(pred, 1 - gt))
    fn = torch.sum(torch.mul(1 - pred, gt))
    dice_eps = (2. * tp + eps) / (2. * tp + fp + fn + eps)
    return dice_eps


def eval(model, source, target, target_img, test_list, validation=True):
    with torch.no_grad():
        model.eval()
        L2_norm_mean = 0
        for i in range(test_list.shape[0]//4):
            source_img = test_list[i:i+4, 0].to(device)
            source_seg = test_list[i:i+4, 1].to(device)
            source_deformed, fields, residuals, _ = model(smooth(source_img), source_seg)
            L2_norm = ((source_deformed[l] - target[:4].unsqueeze(0)) ** 2).sum() / 4
            L2_norm_mean += L2_norm.detach().cpu()
            if (e + 1) % 10 == 0 and i == 0:
                for r in range(4):
                    fig, ax = plt.subplots(2, 3, figsize=(7.5, 5), constrained_layout=True)
                    ax[0, 0].imshow((test_list[r][0]).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                    ax[0, 0].set_title("Source")
                    ax[0, 0].axis('off')
                    ax[0, 1].imshow((target_img).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                    ax[0, 1].set_title("Target")
                    ax[0, 1].axis('off')
                    ax[1, 0].imshow((source_deformed[l][r]).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                    ax[1, 0].set_title("Deformed")
                    ax[1, 0].axis('off')
                    superposition = torch.stack([target_img.squeeze().detach().cpu(),
                                                 source_deformed[l][r].squeeze().detach().cpu(),
                                                 torch.zeros(source[r][0].squeeze().shape)], dim=2)
                    ax[1, 1].imshow(superposition, vmin=0, vmax=1)
                    ax[1, 1].set_title("Superposition")
                    ax[1, 1].axis('off')
                    ax[0, 2].set_title('Shape deformation only')
                    # source_deformed_shape = source[r]
                    id_grid = K.utils.grid.create_meshgrid(208, 208, False, device)
                    seg_r = source_seg[r].unsqueeze(0)
                    residuals_deform = torch.zeros(seg_r.shape).to(device)
                    for k in range(1, l):
                        seg_r = model.deform_image(seg_r, id_grid - fields[k-1][r] / l)
                        res_tmp = residuals[k][r].unsqueeze(0) * seg_r
                        for f in range(k, l):
                            res_tmp = model.deform_image(res_tmp, id_grid - fields[f][r] / l)
                        residuals_deform += res_tmp * mu_val ** 2 / l
                    seg_r = model.deform_image(seg_r, id_grid - fields[l-1][r] / l)
                    residuals_deform += residuals[l][r].unsqueeze(0) * seg_r * mu_val ** 2 / l


                    #im = ax[1, 2].imshow((target_img - source_deformed[l][r]).squeeze().detach().cpu().numpy(), cmap="gray")
                    im = ax[1, 2].imshow(residuals_deform.squeeze().detach().cpu().numpy())
                    cbar = ax[1, 2].figure.colorbar(im, ax=ax[1, 2])
                    ax[1, 2].set_title("Residuals heatmap")
                    ax[1, 2].axis('off')
                    deformations = [test_list[r][0].to(device).unsqueeze(0)]
                    for j in range(l):
                        deformations.append(model.deform_image(deformations[j], id_grid - fields[j][r] / l))

                    # source_deformed_shape = model.deform_image(source, deform, interpolation="bilinear")
                    ax[0, 2].imshow(deformations[l].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
                    ax[0, 2].axis('off')
                    fig.suptitle('Metamorphoses, iteration: %d' % (e + 1))
                    plt.savefig('Metamorphoses.png')
                    plt.axis('off')
                    plt.show()
                    fig, ax = plt.subplots(3, 5, figsize=(10, 10))
                    for k in range(l):
                        ax[k // 5, k % 5].imshow((source_deformed[k + 1][r]).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax[k // 5, k % 5].set_title("t = %d" % (k + 1))
                        ax[k // 5, k % 5].axis('off')
                    fig.suptitle('Metamorphoses, iteration: %d' % (e + 1))
                    plt.savefig('Metamorphoses_evolution.png')
                    plt.show()
                    torch.save(model, "../results/model.pt")
        print("Validation L2 loss: %f" %(L2_norm_mean / (test_list.shape[0] // 4)))
        L2_val.append(L2_norm_mean / (test_list.shape[0] // 4))
    return (L2_val)

def eval_dice(model, source, target, target_img, test_list, seg_model, target_seg, validation=True):
    with torch.no_grad():
        id_grid = K.utils.grid.create_meshgrid(208, 208, False, device)
        model.eval()
        L2_norm_list = []
        dice_list = []
        v_norm_list = []
        res_norm_list = []
        time_list = []
        for i in range(test_list.shape[0]//4):
            t = time()
            source_img = test_list[i:i+4, 0].to(device)
            source_seg = test_list[i:i+4, 1].to(device)
            source_deformed, fields, residuals, grad = model(source_img, source_seg)
            time_list.append(time()-t)
            v_norm = torch.stack([(residuals[j] * grad[j].squeeze(1) * (-fields[j].permute(0, 3, 1, 2))).sum() for j in
                                  range(len(fields))]).sum()
            residuals_norm = (torch.stack(residuals) ** 2).sum()
            v_norm_list.append((v_norm / source_img.shape[0] / l).cpu().item())
            res_norm_list.append((residuals_norm / source_img.shape[0] / l).cpu().item())
            source_deformed_tpn = source_deformed[l].clone()
            for r in range(4):
                deformations = [source_img[r].to(device).unsqueeze(0)]
                for j in range(l):
                    deformations.append(model.deform_image(deformations[j], id_grid - fields[j][r] / l))
                source_deformed_tpn[r] = deformations[l].squeeze(0).clone()
            L2_norm = ((source_deformed_tpn - target[:4].unsqueeze(0)) ** 2).sum() / 4
            L2_norm_list.append(L2_norm.detach().cpu())
            source_deformed_tpn = source_deformed_tpn.detach().cpu().numpy()
            for j in range(source_deformed_tpn.shape[0]):
                s_mask = (source_deformed[l][j] != 0).cpu()
                source_deformed_tpn[j] = (source_deformed_tpn[j] - source_deformed_tpn[j][s_mask].mean()) / (
                            source_deformed_tpn[j][s_mask].std() + 1e-30)
            source_deformed_tpn = torch.from_numpy(source_deformed_tpn).float().to(device)
            pred_seg = seg_model(source_deformed_tpn)
            pred_seg = 1. * (pred_seg > 0.5)
            for j in range(pred_seg.shape[0]):
                dice_list.append(dice(pred_seg[j].float(), target_seg.squeeze(0)).detach().cpu())
                if i == 0 and (e+1) % 5 == 0 :
                    fig, ax = plt.subplots(2,2)
                    ax[0, 0].imshow(target_img.squeeze().float().detach().cpu(), cmap="gray", vmin=0, vmax=1)
                    ax[0, 1].imshow(source_deformed[l][j].float().detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
                    ax[1, 0].imshow(target_seg.detach().float().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
                    ax[1, 1].imshow(pred_seg[j].detach().float().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
                    plt.title
                    plt.show()

            L2_norm = ((source_deformed[l] - target[:4].unsqueeze(0)) ** 2).sum() / 4
            L2_norm_list.append(L2_norm.detach().cpu())

            if (e + 1) % 5 == 0 and i == 0:
                for r in range(4):
                    fig, ax = plt.subplots(2, 3, figsize=(7.5, 5), constrained_layout=True)
                    ax[0, 0].imshow((test_list[r][0]).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                    ax[0, 0].set_title("Source")
                    ax[0, 0].axis('off')
                    ax[0, 1].imshow((target_img).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                    ax[0, 1].set_title("Target")
                    ax[0, 1].axis('off')
                    ax[1, 0].imshow((source_deformed[l][r]).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                    ax[1, 0].set_title("Deformed")
                    ax[1, 0].axis('off')
                    superposition = torch.stack([target_img.squeeze().detach().cpu(),
                                                 source_deformed[l][r].squeeze().detach().cpu(),
                                                 torch.zeros(source[r][0].squeeze().shape)], dim=2)
                    ax[1, 1].imshow(superposition, vmin=0, vmax=1)
                    ax[1, 1].set_title("Superposition")
                    ax[1, 1].axis('off')
                    ax[0, 2].set_title('Shape deformation only')
                    # source_deformed_shape = source[r]
                    seg_r = source_seg[r].unsqueeze(0)
                    residuals_deform = torch.zeros(seg_r.shape).to(device)
                    for k in range(1, l):
                        seg_r = model.deform_image(seg_r, id_grid - fields[k-1][r] / l)
                        res_tmp = residuals[k][r].unsqueeze(0) * seg_r
                        for f in range(k, l):
                            res_tmp = model.deform_image(res_tmp, id_grid - fields[f][r] / l)
                        residuals_deform += res_tmp * mu_val ** 2 / l
                    seg_r = model.deform_image(seg_r, id_grid - fields[l-1][r] / l)
                    residuals_deform += residuals[l][r].unsqueeze(0) * seg_r * mu_val ** 2 / l


                    #im = ax[1, 2].imshow((target_img - source_deformed[l][r]).squeeze().detach().cpu().numpy(), cmap="gray")
                    im = ax[1, 2].imshow(residuals_deform.squeeze().detach().cpu().numpy())
                    cbar = ax[1, 2].figure.colorbar(im, ax=ax[1, 2])
                    ax[1, 2].set_title("Residuals heatmap")
                    ax[1, 2].axis('off')
                    deformations = [test_list[r][0].to(device).unsqueeze(0)]
                    for j in range(l):
                        deformations.append(model.deform_image(deformations[j], id_grid - fields[j][r] / l))

                    # source_deformed_shape = model.deform_image(source, deform, interpolation="bilinear")
                    ax[0, 2].imshow(deformations[l].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
                    ax[0, 2].axis('off')
                    fig.suptitle('Metamorphoses, iteration: %d' % (e + 1))
                    plt.savefig('Metamorphoses.png')
                    plt.axis('off')
                    plt.show()
                    fig, ax = plt.subplots(3, 5, figsize=(10, 10))
                    for k in range(l):
                        ax[k // 5, k % 5].imshow((source_deformed[k + 1][r]).squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
                        ax[k // 5, k % 5].set_title("t = %d" % (k + 1))
                        ax[k // 5, k % 5].axis('off')
                    fig.suptitle('Metamorphoses, iteration: %d' % (e + 1))
                    plt.savefig('Metamorphoses_evolution.png')
                    plt.show()
                    torch.save(model, "../results/model.pt")
        L2_norm_list = np.array(L2_norm_list)
        dice_list = np.array(dice_list)
        v_norm_list = np.array(v_norm_list)
        res_norm_list = np.array(res_norm_list)
        time_list = np.array(time_list)
        print("Validation L2 loss: %f" % (L2_norm_list.mean()), "std: %f" % (L2_norm_list.std()))
        print("Validation dice: %f" % (dice_list.mean()), "std: %f" % (dice_list.std()))
        print("Validation z loss:", res_norm_list.mean(), "std: %f" % (res_norm_list.std()))
        print("Validation v loss:", v_norm_list.mean(), "std: %f" % (v_norm_list.std()))
        print("Validation time:", time_list.mean(), "std: %f" % (time_list.std()))
        L2_val.append(L2_norm_list.mean())
    return (L2_val)



def integrate(v):
    l = len(v)
    id_grid = K.utils.grid.create_meshgrid(v[0].shape[1], v[0].shape[2], False, device)
    grid_def = id_grid - v[0] / l
    _, H, W, _ = grid_def.shape
    mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(device)

    for t in range(1, l):
        slave_grid = grid_def.detach()
        interp_vectField = F.grid_sample(v[t].permute(0, 3, 1, 2), slave_grid * mult - 1, "nearest",
                                         padding_mode="border", align_corners=True).permute(0, 2, 3, 1)

        grid_def -= interp_vectField / l

    return grid_def


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
    neurite_path = "/home/matthis/datasets/neurite-oasis/2D_data/OASIS_OAS1_0056_MR1/slice_88/image.npy"
    neurite_hist = np.load(neurite_path)
    neurite_hist = neurite_hist[neurite_hist > 0]
#target_img = torch.from_numpy(np.transpose(np.load('../data_miccai_2D_2021/' + "BraTS2021_00162" + "/" + "BraTS2021_00162" + "_t1.npy"))).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
"""source_list = []
path_list = os.listdir('../data_miccai_2D_2021')
random.shuffle(path_list)
test_list = path_list[-40:]
with open("test_list.txt", "w") as f:
    for img in test_list:
        f.write(img+"\n")
    f.close()

for image in path_list:
    if image[:5] == "BraTS":
        source_seg = np.transpose(np.load("../data_miccai_2D_2021/" + image + "/" + image + "_seg.npy"))
        source_seg[source_seg == 2.] = 1.
        source_seg[source_seg == 4.] = 1.
        #source_seg = ndimage.morphology.binary_dilation(source_seg, iterations=2)
        source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        source_list.append(torch.cat([torch.from_numpy(np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))
"""
test_list = []
with open("test_list.txt", "r") as f:
    test_paths = f.readlines()
    test_paths = [path[:-1] for path in test_paths]
    f.close()
source_list = []
for image in os.listdir('../data_miccai_2D_2021/'):
    if image[:5] == "BraTS" and image not in test_paths:
        if image not in test_paths:
            source_seg = np.transpose(np.load("../data_miccai_2D_2021/" + image + "/" + image + "_seg.npy"))
            source_seg[source_seg == 2.] = 1.
            source_seg[source_seg == 4.] = 1.
            # source_seg = ndimage.morphology.binary_dilation(source_seg, iterations=2)
            source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            source_list.append(torch.cat([torch.from_numpy(
                np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))

print(len(source_list))
for image in test_paths:
    test_seg = np.transpose(np.load("../data_miccai_2D_2021/" + image + "/" + image + "_seg.npy"))
    test_seg[test_seg == 2.] = 1.
    test_seg[test_seg == 4.] = 1.
    test_seg = torch.from_numpy(test_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    test_list.append(torch.cat([torch.from_numpy(
        np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
        torch.FloatTensor).unsqueeze(0).unsqueeze(0), test_seg]))

"""target_img = torch.from_numpy(
    np.transpose(np.load("../brats_2020_2D/healthy/BraTS20_Training_019/BraTS20_Training_019_t1ce.npy"))).type(
    torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
source_list = []
for image in os.listdir('../brats_2020_2D/cancerous'):
    if image[:5] == "BraTS":
        source_seg = np.transpose(np.load("../brats_2020_2D/cancerous/" + image + "/" + image + "_seg.npy"))
        source_seg[source_seg == 2.] = 1.
        source_seg[source_seg == 4.] = 1.
        source_seg = ndimage.morphology.binary_dilation(source_seg, iterations=2)
        source_seg = torch.from_numpy(source_seg).unsqueeze(0).unsqueeze(0)
        source_list.append(torch.cat([torch.from_numpy(np.transpose(np.load('../brats_2020_2D/cancerous/' + image + "/" + image + "_t1ce.npy"))).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))"""

"""target_img = torch.from_numpy(cv2.resize(cv2.imread("reg_test_m0c.png", cv2.IMREAD_GRAYSCALE), (200, 200))).type(
    torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0) / 255.
test = torch.from_numpy(cv2.resize(cv2.imread("reg_test_m0t.png", cv2.IMREAD_GRAYSCALE), (200, 200))).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0) / 255.
test_seg = torch.zeros(test.shape)
rect_xmin = 15
rect_ymin = 80
rect_xlength = 67
rect_ylength = 39
seg=False
test_seg[:,:,rect_ymin:rect_ymin+rect_ylength, rect_xmin:rect_xmin+rect_xlength] = 1.
source_list = []
for image in os.listdir('../images'):
    if image[:6] == "source":
        #source_seg = cv2.imread("../images/" + image + "/" + "segmentation.png", cv2.IMREAD_GRAYSCALE)
        #source_seg = ndimage.morphology.binary_dilation(source_seg, iterations=2)
        #source_seg = torch.from_numpy(source_seg).unsqueeze(0).unsqueeze(0)
        source_list.append(torch.cat([torch.from_numpy(cv2.imread('../images/' + image, 0)).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)/255., test_seg]))
"""
sig = 1
smooth = flt.GaussianBlur2d((sig * 6 + 1, sig * 6 + 1), (sig, sig))

target_img = smooth(target_img)

#test_list = source_list[-40:]
#test_list.append(torch.cat([test, test_seg]))
#source_list = source_list[:-40]
source_list = torch.stack(source_list)
test_list = torch.stack(test_list)


l = 15
L2_weight = .5
v_weight = 3e-6 / l
z_weight = 3e-6 / l
mu_val = 0.015
mu = mu_val * torch.ones(target_img.shape).to(device)
n_images = len(source_list)
batch_size = 4
n_epoch = 30

z0 = torch.zeros(target_img.shape)
z0.requires_grad = True

print("### Starting Metamorphoses")
print("L2_weight=", L2_weight)
print("z_weight=", z_weight)
print("v_weight=", v_weight)
print("mu=", mu_val)
print("batch_size=", batch_size)
print("n_epoch=", n_epoch)
model = meta_model(l, target_img.shape, device, 51, 4., mu, z0, mu_val).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                             weight_decay=1e-8)
L2_loss = []
L2_val = []
dice_val = []
indexes = list(range(source_list.shape[0]))
for e in range(n_epoch):
    """if e == 5:
        break"""
    model.train()
    total_loss_avg = 0
    L2_norm_avg = 0
    residuals_norm_avg = 0
    v_norm_avg = 0
    for i in tqdm(range(n_images // batch_size)):
        """if i == 5:
            break"""
        source = source_list[indexes[i * batch_size:(i + 1) * batch_size]]
        source_img = source[:, 0]
        source_seg = source[:, 1]
        size_batch = source.shape[0]
        target = torch.cat([target_img for _ in range(size_batch)], dim=0).to(device)
        source_img = source_img.to(device)
        source_seg = source_seg.to(device)
        source_deformed, fields, residuals, grad = model(smooth(source_img), source_seg)
        v_norm = torch.stack([(residuals[j] * grad[j].squeeze(1) * (-fields[j].permute(0, 3, 1, 2))).sum() for j in
                              range(len(fields))]).sum()
        residuals_norm = (torch.stack(residuals) ** 2).sum()
        L2_norm = ((source_deformed[l] - target) ** 2).sum()
        total_loss = (L2_weight * L2_norm + v_weight * v_norm + mu_val * z_weight * residuals_norm) / size_batch
        total_loss_avg += total_loss / (n_images // batch_size)
        L2_norm_avg += L2_norm / size_batch / (n_images // batch_size)
        residuals_norm_avg += residuals_norm / size_batch / (n_images // batch_size)
        v_norm_avg += v_norm / size_batch / (n_images // batch_size)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    random.shuffle(indexes)
    L2_loss.append(L2_norm_avg.detach().cpu())

    print("Epoch %d, total loss: %f, L2 norm: %f, v_norm: %f, residuals: %f" % (
    e + 1, total_loss_avg, L2_norm_avg, v_norm_avg, residuals_norm_avg))

    if e == 10:
        for g in optimizer.param_groups:
            g['lr'] = 5e-4
    if e == 20:
        for g in optimizer.param_groups:
            g['lr'] = 1e-4

    ### Validation

    if seg:
        L2_val = eval_dice(model, source, target, target_img, test_list, seg_model, target_seg)
    else:
        L2_val = eval(model, source, target, target_img, test_list)
    #eval(model, source, target, target_img, source_list[-3:], False)

    plt.figure()
    x = np.linspace(1, e + 1, e + 1)
    plt.plot(x, L2_loss, color='blue', label="Training")
    plt.plot(x, L2_val, color='red', label="Validation")
    plt.title('L2 norm during training and validation ')
    plt.xlabel('epoch')
    plt.ylabel('L2 norm')
    plt.legend()
    plt.savefig('../results/loss.png')
    plt.clf()
    """if seg:
        plt.figure()
        x = np.linspace(1, e + 1, e + 1)
        plt.plot(x, dice_val, color='red', label="Validation")
        plt.title('dice score during validation ')
        plt.xlabel('epoch')
        plt.ylabel('Dice')
        plt.legend()
        plt.savefig('../results/dice.png')
        plt.clf()"""

"""source_deformed, _, _, _ = model(smooth(test).to(device), test_seg.to(device))
fig,ax = plt.subplots(figsize=(8,8))
plt.imshow((source_deformed[l] - target[0]).abs().detach().cpu().squeeze(), cmap="jet", vmin=0, vmax=1)
fig.subplots_adjust(
    top=0.981,
    bottom=0.019,
    left=0.019,
    right=0.981,
    hspace=0.2,
    wspace=0.2
)
plt.axis('off')
plt.show()

print("L2 on test ", ((source_deformed[l]- target[0])**2).sum().detach().cpu())"""


