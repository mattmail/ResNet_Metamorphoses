from utils import get_vnorm, get_znorm, plot_results, save_losses
from tqdm import tqdm
import time
import torch
import os
import random

def train_opt(model, source, target, optimizer, device, n_iter=1000, local_reg=False, debug=False, plot_iter=500, L2_weight=.5, v_weight=3e-6, z_weight=3e-6):
    l = model.l
    mu = model.mu
    if local_reg:
        source_img = source[0]
        source_seg = source[1]
    else:
        source_img = source
    for i in range(n_iter):
        if i == 5 and debug:
            break
        if local_reg:
            source_deformed, fields, residuals, grad = model(source_img, source_seg)
        else:
            source_deformed, fields, residuals, grad = model(source_img)
        v_norm = get_vnorm(residuals, fields, grad)
        residuals_norm = get_znorm(residuals)
        L2_norm = ((source_deformed[l] - target) ** 2).sum()
        total_loss = L2_weight * L2_norm + v_weight * (v_norm) + mu * z_weight * residuals_norm
        print("Iteration %d: Total loss: %f   L2 norm: %f   V_reg: %f   Z_reg: %f" % (i, total_loss, L2_norm, v_norm, residuals_norm))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if i == 500:
            for g in optimizer.param_groups:
                g['lr'] = 5e-4
        if i == 1000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4
        if (i + 1) % plot_iter == 0:
            plot_results(source_img, target, source_deformed, fields, residuals, model.l, model.mu, i)

    plot_results(source_img, target, source_deformed, fields, residuals, model.l, model.mu, n_iter)


def train_learning(model, train_list, test_list, target_img, optimizer, device, batch_size=4, n_epoch=50, local_reg=False, debug=False, plot_epoch=5, L2_weight=.5, v_weight=3e-6, z_weight=3e-6):
    indexes = list(range(train_list.shape[0]))
    n_iter = len(indexes) // batch_size
    target = torch.cat([target_img for _ in range(batch_size)], dim=0).to(device)
    l = model.l
    mu = model.mu
    L2_loss = []
    L2_val = []
    if local_reg:
        result_path = "../results/" + "meta_model_" + time.strftime("%m%d_%H%M", time.localtime())
    else:
        result_path = "../results/" + "meta_model_local_" + time.strftime("%m%d_%H%M", time.localtime())
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for e in range(n_epoch):
        """if e == 5 and debug:
            break"""
        model.train()
        total_loss_avg = 0
        L2_norm_avg = 0
        residuals_norm_avg = 0
        v_norm_avg = 0
        for i in tqdm(range(n_iter)):
            if i == 5 and debug:
                break
            if local_reg:
                source = train_list[indexes[i * batch_size:(i + 1) * batch_size]]
                source_img = source[:, 0]
                source_seg = source[:, 1]
                source_img = source_img.to(device)
                source_seg = source_seg.to(device)
                source_deformed, fields, residuals, grad = model(source_img, source_seg)
            else:
                source_img = train_list[indexes[i * batch_size:(i + 1) * batch_size]].to(device)
                source_deformed, fields, residuals, grad = model(source_img)

            v_norm = get_vnorm(residuals, fields, grad) / batch_size
            residuals_norm = get_znorm(residuals) / batch_size
            L2_norm = ((source_deformed[l] - target) ** 2).sum() / batch_size
            total_loss = (L2_weight * L2_norm + v_weight * v_norm + mu * z_weight * residuals_norm) / batch_size
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_avg += total_loss / n_iter
            L2_norm_avg += L2_norm / n_iter
            residuals_norm_avg += residuals_norm / n_iter
            v_norm_avg += v_norm / n_iter

        random.shuffle(indexes)
        L2_loss.append(L2_norm_avg.detach().cpu())

        print("Training: epoch %d, total loss: %f, L2 norm: %f, v_norm: %f, residuals: %f" % (
            e + 1, total_loss_avg, L2_norm_avg, v_norm_avg, residuals_norm_avg))

        if e == 10:
            for g in optimizer.param_groups:
                g['lr'] = 5e-4
        if e == 20:
            for g in optimizer.param_groups:
                g['lr'] = 1e-4

        ### Validation
        L2_val = eval(model, test_list, target_img, local_reg, e, plot_epoch, result_path, device, l, L2_val)

        save_losses(L2_loss, L2_val, e, result_path)


def eval(model, test_list, target, local_reg, e, plot_iter, result_path, device, l, L2_val, nb_img_plot=4):
    with torch.no_grad():
        model.eval()
        L2_norm_mean = 0
        for i in range(test_list.shape[0]):
            if local_reg:
                source_img = test_list[i, 0].to(device).unsqueeze(0)
                source_seg = test_list[i, 1].to(device).unsqueeze(0)
                source_deformed, fields, residuals, _ = model(source_img, source_seg)
            else:
                source_img = test_list[i].to(device).unsqueeze(0)
                source_deformed, fields, residuals, grad = model(source_img)
            L2_norm = ((source_deformed[l] - target.unsqueeze(0)) ** 2).sum()
            L2_norm_mean += L2_norm.detach().cpu()
            if (e + 1) % plot_iter == 0 and i < nb_img_plot:
                plot_results(source_img, target, source_deformed, fields, residuals, model.l, model.mu, e, mode="learning")

        if (e+1) % plot_iter == 0:
            torch.save(model, result_path + "/model.pt")

        print("Validation L2 loss: %f" %(L2_norm_mean /test_list.shape[0]))
        L2_val.append(L2_norm_mean / test_list.shape[0])
    return (L2_val)

