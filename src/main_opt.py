import torch
import kornia.filters as flt
import numpy as np
import cv2
import os
from time import time
from models import meta_model, meta_model_local
from train import train_opt


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    #source = torch.from_numpy(cv2.resize(cv2.imread("../images/reg_test_m0t.png", cv2.IMREAD_GRAYSCALE), (200, 200))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0) / 255.
    #target = torch.from_numpy(cv2.resize(cv2.imread("../images/reg_test_m0c.png", cv2.IMREAD_GRAYSCALE), (200, 200))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0) / 255.
    #sig = 1
    #smooth = flt.GaussianBlur2d((sig*6+1,sig*6+1), (sig, sig))
    #source = smooth(source)
    #target = smooth(target)

    source_path = "BraTS20_Training_010"
    target = torch.from_numpy(np.transpose(np.load("../brats_2020_2D/healthy/BraTS20_Training_019/BraTS20_Training_019_t1ce.npy"))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    source_img = torch.from_numpy(np.transpose(np.load("../brats_2020_2D/cancerous/" + source_path + "/" + source_path + "_t1ce.npy"))).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    source_seg = np.transpose(np.load("../brats_2020_2D/cancerous/" + source_path + "/" + source_path + "_seg.npy"))
    source_seg[source_seg == 2.] = 1.
    source_seg[source_seg == 4.] = 1.
    source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)

    use_segmentation = False
    if use_segmentation:
        source = [source_img, source_seg]
    else:
        source = source_img



    n_iter = 1000
    l = 15
    L2_weight = .5
    v_weight = 3e-7/l
    z_weight = 3e-7/l
    mu = 0.015

    z0 = torch.zeros(source_img.shape)

    print("### Starting Metamorphoses ###")
    print("L2_weight=", L2_weight)
    print("z_weight=", z_weight)
    print("v_weight=", v_weight)
    print("n_iter=", n_iter)
    print("mu=", mu)
    t = time()
    if use_segmentation:
        model = meta_model_local(l, source_img.shape, device, 51, 4., mu, z0).to(device)
    else:
        model = meta_model(l, source_img.shape, device, 51, 4., mu, z0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                  weight_decay=1e-8)

    train_opt(model, source, target, optimizer, device, n_iter=n_iter, local_reg=use_segmentation, debug=False, plot_iter=500, L2_weight=L2_weight, v_weight=v_weight, z_weight=z_weight)





