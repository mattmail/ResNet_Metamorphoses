import matplotlib.pyplot as plt
import torch
import kornia.filters as flt
import numpy as np
import cv2
import os
from time import time
from models import meta_model, meta_model_local, meta_model_sharp, meta_model_local_sharp, double_resnet
from train import train_opt
import nibabel as nib
from scipy.ndimage import binary_dilation
from skimage.exposure import match_histograms


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    """source_img = torch.from_numpy(cv2.resize(cv2.imread("../images/reg_test_m0t.png", cv2.IMREAD_GRAYSCALE), (200, 200))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0) / 255.
    target = torch.from_numpy(cv2.resize(cv2.imread("../images/reg_test_m0c.png", cv2.IMREAD_GRAYSCALE), (200, 200))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0) / 255.
    source_seg = torch.zeros(source_img.shape, device=device, dtype=torch.float32)
    min_x, min_y = 90, 10
    x_length, y_length = 27, 80
    source_seg[:,:,min_x:min_x+x_length, min_y:min_y+y_length] = 1.
    plt.imshow(source_seg.detach().cpu().squeeze(), cmap='gray')
    plt.show()
    sig = 1"""
    """smooth = flt.GaussianBlur2d((sig*6+1,sig*6+1), (sig, sig))
    source_img = smooth(source)
    target = smooth(target)"""

    """source_path = "BraTS20_Training_010"
    target = torch.from_numpy(np.transpose(np.load("../brats_2020_2D/healthy/BraTS20_Training_019/BraTS20_Training_019_t1ce.npy"))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    source_img = torch.from_numpy(np.transpose(np.load("../brats_2020_2D/cancerous/" + source_path + "/" + source_path + "_t1ce.npy"))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    source_seg = np.transpose(np.load("../brats_2020_2D/cancerous/" + source_path + "/" + source_path + "_seg.npy"))
    source_seg[source_seg == 2.] = 1.
    source_seg[source_seg == 4.] = 1.
    source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)"""

    #target_path = "BraTS2021_00162"
    #target = torch.from_numpy(np.transpose(np.load('../data_miccai_2D_2021/' + target_path + "/" + target_path + "_t1.npy"))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    source_path = "BraTS2021_00750"
    #source_path = "BraTS2021_00045"
    #source_path = "BraTS2021_01074"

    MNI_img = nib.load("/home/matthis/datasets/skull_stripped.nii.gz").get_fdata()
    MNI_img = (MNI_img - MNI_img.min()) / (MNI_img.max() - MNI_img.min())
    MNI_img = np.pad(MNI_img, ((6,5),(0,0),(0,0)))
    MNI_img = np.rot90(MNI_img[:, 12:220, 93])
    source = np.transpose(np.load('../data_miccai_2D_2021/' + source_path + "/" + source_path + "_t1.npy"))
    source[source > 0] = match_histograms(source[source > 0], MNI_img[MNI_img > 0])
    target = torch.from_numpy(MNI_img.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
    source_img = torch.from_numpy(
        source).type(
        torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)

    source_seg = np.transpose(np.load('../data_miccai_2D_2021/' + source_path + "/" + source_path + "_seg.npy"))
    source_seg[source_seg == 2.] = 1.
    source_seg[source_seg == 4.] = 1.
    source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)

    """laplacian = torch.tensor([[0.,-1.,0.],[-1.,4.,-1.],[0., -1., 0.]]).float().to(device)[None, None]
    contours = torch.nn.functional.conv2d(source_seg, laplacian, padding=1)
    contours = (contours > 0)*1
    np_contours = contours.squeeze().detach().cpu().numpy()
    np_source_seg = source_seg.detach().cpu().squeeze().numpy()
    dilated_contours = binary_dilation(np_contours)
    inside_contours = 1*(np_source_seg + dilated_contours - np_contours == 2)
    outside_contours = dilated_contours - np_contours - inside_contours
    inside_contours = torch.from_numpy(inside_contours).to(device)[None, None]
    outside_contours = torch.from_numpy(outside_contours).to(device)[None, None]"""

    use_segmentation = True
    if use_segmentation:
        source = [source_img, source_seg]
    else:
        source = source_img

    two_resnets = False

    be_sharp = True
    n_iter = 2000
    l = 40
    L2_weight = .5
    v_weight = 3e-6/l
    z_weight = 3e-8/l
    mu = 0.01
    sigma = 10.
    kernel_size = 51
    debug = False

    z0 = torch.zeros(source_img.shape)

    print("### Starting Metamorphoses ###")
    print("L2_weight=", L2_weight)
    print("z_weight=", z_weight)
    print("v_weight=", v_weight)
    print("n_iter=", n_iter)
    print("mu=", mu)
    print("sigma=", sigma)
    t = time()
    if use_segmentation:
        if be_sharp:
            model = meta_model_local_sharp(l, source_img.shape, device, kernel_size, sigma, mu, z0).to(device)
        elif two_resnets:
            model = double_resnet(l, source_img.shape, device, mu, z0).to(device)
        else:
            model = meta_model_local(l, source_img.shape, device, kernel_size, sigma, mu, z0).to(device)
    else:
        if be_sharp:
            model = meta_model_sharp(l, source_img.shape, device, kernel_size, sigma, mu, z0).to(device)
        else:
            model = meta_model(l, source_img.shape, device, kernel_size, sigma, mu, z0).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + [z0], lr=5e-3, weight_decay=1e-8)

    train_opt(model, source, target, optimizer, device, n_iter=n_iter, local_reg=use_segmentation, double_resnets=two_resnets, debug=debug, plot_iter=500, L2_weight=L2_weight, v_weight=v_weight, z_weight=z_weight)





