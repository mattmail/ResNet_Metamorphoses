import numpy as np
import torch
import nibabel as nib
import os
import cv2
import random

def load_brats_2021(device, use_segmentation):
    test_list = []
    with open("test_list.txt", "r") as f:
        test_paths = f.readlines()
        test_paths = [path[:-1] for path in test_paths]
        f.close()
    source_list = []
    for image in os.listdir('../data_miccai_2D_2021/'):
        if image[:5] == "BraTS" and image not in test_paths:
            if image not in test_paths:
                if use_segmentation:
                    source_seg = np.transpose(np.load("../data_miccai_2D_2021/" + image + "/" + image + "_seg.npy"))
                    source_seg[source_seg == 2.] = 1.
                    source_seg[source_seg == 4.] = 1.
                    source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                    source_list.append(torch.cat([torch.from_numpy(
                        np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))
                else:
                    source_list.append(torch.from_numpy(
                        np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                        torch.FloatTensor).unsqueeze(0))

    print("Number of training images:", len(source_list))
    for image in test_paths:
        if use_segmentation:
            test_seg = np.transpose(np.load("../data_miccai_2D_2021/" + image + "/" + image + "_seg.npy"))
            test_seg[test_seg == 2.] = 1.
            test_seg[test_seg == 4.] = 1.
            test_seg = torch.from_numpy(test_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            test_list.append(torch.cat([torch.from_numpy(
                np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                torch.FloatTensor).unsqueeze(0).unsqueeze(0), test_seg]))
        else:
            test_list.append(torch.from_numpy(
                np.transpose(np.load('../data_miccai_2D_2021/' + image + "/" + image + "_t1.npy"))).type(
                torch.FloatTensor).unsqueeze(0))
    print("Number of test images:", len(test_list))

    MNI_img = nib.load("/usr/local/fsl/data/linearMNI/MNI152lin_T1_1mm_brain.nii.gz").get_fdata()
    target_img = np.pad(MNI_img[:, :, 93], ((18, 18), (0, 0)), "constant")
    target_img = np.rot90(cv2.resize(target_img, (208, 208)))
    target_img = torch.from_numpy(target_img.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device) / 255.
    return torch.stack(source_list), torch.stack(test_list), target_img

def load_brats_2020(device, use_segmentation, test_size=40):
    target_img = torch.from_numpy(
        np.transpose(np.load("../brats_2020_2D/healthy/BraTS20_Training_019/BraTS20_Training_019_t1ce.npy"))).type(
        torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)
    source_list = []
    for image in os.listdir('../brats_2020_2D/cancerous'):
        if image[:5] == "BraTS":
            if use_segmentation:
                source_seg = np.transpose(np.load("../brats_2020_2D/cancerous/" + image + "/" + image + "_seg.npy")).astype(float)
                source_seg[source_seg == 2.] = 1.
                source_seg[source_seg == 4.] = 1.
                source_seg = torch.from_numpy(source_seg).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                source_list.append(torch.cat([torch.from_numpy(
                    np.transpose(np.load('../brats_2020_2D/cancerous/' + image + "/" + image + "_t1ce.npy"))).type(
                    torch.FloatTensor).unsqueeze(0).unsqueeze(0), source_seg]))
            else:
                source_list.append(torch.from_numpy(
                    np.transpose(np.load('../brats_2020_2D/cancerous/' + image + "/" + image + "_t1ce.npy"))).type(
                    torch.FloatTensor).unsqueeze(0))

    random.shuffle(source_list)
    test_list = source_list[-test_size:]
    source_list = source_list[:-test_size]

    return torch.stack(source_list), torch.stack(test_list), target_img

