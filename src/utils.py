import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import kornia as K
import numpy as np

def deform_image(image, deformation, interpolation="bilinear"):
    _, _, H, W = image.shape
    mult = torch.tensor((2 / (W - 1), 2 / (H - 1))).unsqueeze(0).unsqueeze(0).to(image.device)
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

def get_vnorm(residuals, fields, grad):
    return torch.stack([(residuals[j] * grad[j].squeeze(1) * (-fields[j].permute(0, 3, 1, 2))).sum() for j in
                              range(len(fields))]).sum()

def get_znorm(residuals):
    return (torch.stack(residuals) ** 2).sum()

def plot_results(source, target, source_deformed, fields, residuals, l, mu, i, mode="opt"):
    _,_,h,w = source.shape
    fig, ax = plt.subplots(2, 3, figsize=(7.5, 5), constrained_layout=True)
    ax[0, 0].imshow(source.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
    ax[0, 0].set_title("Source")
    ax[0, 0].axis('off')
    ax[0, 1].imshow(target.squeeze().detach().cpu(), cmap='gray', vmin=0, vmax=1)
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
    id_grid = K.utils.grid.create_meshgrid(h, w, False, source.device)
    residuals_deform = residuals[l]
    for r in range(1, l):
        res_tmp = residuals[r]
        for f in range(r, l):
            res_tmp = deform_image(res_tmp, id_grid - fields[f] / l)
        residuals_deform += res_tmp
    residuals_deform = residuals_deform * mu ** 2 / l

    im = ax[1, 2].imshow(residuals_deform.squeeze().detach().cpu().numpy())
    cbar = ax[1, 2].figure.colorbar(im, ax=ax[1, 2])
    ax[1, 2].set_title("Residuals heatmap")
    ax[1, 2].axis('off')
    ax[0, 2].set_title('Shape deformation only')
    deformations = [source]
    for j in range(l):
        deformations.append(deform_image(deformations[j], id_grid - fields[j] / l))

    ax[0, 2].imshow(deformations[l].detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
    ax[0, 2].axis('off')
    if mode == "learning":
        fig.suptitle('Metamorphoses, epoch: %d' % (i + 1))
    else:
        fig.suptitle('Metamorphoses, iteration: %d' % (i + 1))

    plt.axis('off')
    plt.show()

def save_losses(L2_loss, L2_val, e, result_path):
    plt.figure()
    x = np.linspace(1, e + 1, e + 1)
    plt.plot(x, L2_loss, color='blue', label="Training")
    plt.plot(x, L2_val, color='red', label="Validation")
    plt.title('L2 norm during training and validation ')
    plt.xlabel('epoch')
    plt.ylabel('L2 norm')
    plt.legend()
    plt.savefig(result_path + '/loss.png')
    plt.clf()