import torch
import kornia.filters as flt
import numpy as np
import cv2
import os
from time import time
from models import meta_model, meta_model_local, meta_model_sharp, meta_model_local_sharp
from train import train_learning
from prepare_data import load_brats_2021, load_brats_2020


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    use_segmentation = True

    n_epoch = 50
    l = 15
    L2_weight = .5
    v_weight = 3e-6/l
    z_weight = 3e-8/l
    mu = 0.025
    batch_size = 4
    kernel_size = 51
    sigma = 6.
    debug = False

    train_list, test_list, target_img = load_brats_2021(device, use_segmentation, chose_randomly=True)

    be_sharp = True
    z0 = torch.zeros(target_img.shape)

    print("### Starting Metamorphoses ###")
    print("L2_weight=", L2_weight)
    print("z_weight=", z_weight)
    print("v_weight=", v_weight)
    print("n_epoch=", n_epoch)
    print("mu=", mu)
    t = time()
    if use_segmentation:
        if be_sharp:
            model = meta_model_local_sharp(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)
        else:
            model = meta_model_local(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)
    else:
        if be_sharp:
            model = meta_model_sharp(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)
        else:
            model = meta_model(l, target_img.shape, device, kernel_size, sigma, mu, z0).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + [target_img], lr=1e-3,
                                 weight_decay=1e-8)
    train_learning(model, train_list, test_list, target_img, optimizer, device, batch_size=batch_size, n_epoch=n_epoch, local_reg=use_segmentation, debug=debug, plot_epoch=1, L2_weight=L2_weight, v_weight=v_weight, z_weight=z_weight)





