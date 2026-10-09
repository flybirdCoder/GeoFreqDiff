import os
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

x_start = 0
x_end = 0
y_start = 0
y_end = 0


# noise power prior
def crop_roi(x, center_x, center_y, size_x, size_y):
    global x_start, x_end, y_start, y_end

    x_start = max(0, center_x - (size_x - 1) // 2)
    x_end = min(x.shape[1], center_x + (size_x - 1) // 2)
    y_start = max(0, center_y - (size_y - 1) // 2)
    y_end = min(x.shape[0], center_y + (size_y - 1) // 2)

    ROI = x[y_start:y_end, x_start:x_end]
    return ROI


def calculate_nps(noise_ROI, pixelSpacing=1.0):
    nps = torch.abs(torch.fft.fft2(noise_ROI))
    output_nps = pixelSpacing * pixelSpacing * torch.fft.fftshift(nps ** 2)
    return output_nps


def calculate_arp(lnps):
    height, width = lnps.shape

    center_x, center_y = width // 2, height // 2

    y, x = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='ij')
    distances = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    max_distance = min(height, width) // 2

    radial_mean = torch.zeros(max_distance, dtype=torch.float32)
    for radius in range(max_distance):
        mask = (distances >= radius) & (distances < radius + 1)
        radial_mean[radius] = lnps[mask].mean()

    return radial_mean.cpu().numpy()


def show_roi_fig(x, noise, trunc_min, trunc_max, center_x, center_y, size_x, size_y, save_path, file_name):
    original_ROI = crop_roi(noise[0], center_x, center_y, size_x, size_y)
    x_exchange_ROI = crop_roi(noise[1], center_x, center_y, size_x, size_y)
    y_exchange_ROI = crop_roi(noise[2], center_x, center_y, size_x, size_y)

    original_ROI_norm = original_ROI - original_ROI.mean()
    x_exchange_ROI_norm = x_exchange_ROI - x_exchange_ROI.mean()
    y_exchange_ROI_norm = y_exchange_ROI - y_exchange_ROI.mean()

    original_nps = calculate_nps(original_ROI_norm)
    x_exchange_nps = calculate_nps(x_exchange_ROI_norm)
    y_exchange_nps = calculate_nps(y_exchange_ROI_norm)

    energy_max = torch.max(original_nps)
    energy_min = torch.min(original_nps)
    original_nps = (original_nps - energy_min) / (energy_max - energy_min)
    x_exchange_nps = (x_exchange_nps - energy_min) / (energy_max - energy_min)
    y_exchange_nps = (y_exchange_nps - energy_min) / (energy_max - energy_min)

    original_arp = calculate_arp(original_nps)
    x_exchange_arp = calculate_arp(x_exchange_nps)
    y_exchange_arp = calculate_arp(y_exchange_nps)

    original_arp = gaussian_filter1d(original_arp, sigma=2.0)
    x_exchange_arp = gaussian_filter1d(x_exchange_arp, sigma=2.0)
    y_exchange_arp = gaussian_filter1d(y_exchange_arp, sigma=2.0)

    f, ax = plt.subplots(1, 5, figsize=(60, 10))
    plt.subplots_adjust(wspace=0.4, hspace=0.1)

    ax[0].imshow(x, cmap='gray', vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('original images', fontsize=30)
    ax[0].axis('off')

    rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                         linewidth=3, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)

    ax[1].imshow(original_nps, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title('original_nps', fontsize=30)
    ax[1].axis('off')

    ax[2].imshow(x_exchange_nps, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title('x_exchange_nps', fontsize=30)
    ax[2].axis('off')

    nps_plot = ax[3].imshow(y_exchange_nps, cmap='gray', vmin=0, vmax=1)
    ax[3].set_title('y_exchange_nps', fontsize=30)
    ax[3].axis('off')

    cbar = plt.colorbar(nps_plot, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=25)

    color_original = 'b'
    color_x_exchange = 'g'
    color_y_exchange = 'r'
    ax[4].plot(original_arp, color=color_original, linewidth=3, label='e-noise')
    ax[4].plot(x_exchange_arp, color=color_x_exchange, linewidth=3, label='f-noise')
    ax[4].plot(y_exchange_arp, color=color_y_exchange, linewidth=3, label='g-noise')
    ax[1].set_title('Average Radial Profile (ARP)', fontsize=30)
    ax[4].set_xlabel(r'frequency (mm$^{-1}$)', fontsize=20)
    ax[4].set_ylabel('noise power', fontsize=20)
    ax[4].tick_params(labelsize=20)
    ax[4].grid(False)
    ax[4].legend(loc='upper right', fontsize=20)

    f.savefig(os.path.join(save_path, file_name))
    plt.close()


def show_exchange_fig(x, y, x_exchange, y_exchange, trunc_min, trunc_max, save_path, file_name, original_result,
                      input_result, target_result):
    x, y, x_exchange, y_exchange = x.numpy(), y.numpy(), x_exchange.numpy(), y_exchange.numpy()
    f, ax = plt.subplots(1, 4, figsize=(30, 10))
    ax[0].imshow(x, cmap='gray', vmin=trunc_min, vmax=trunc_max)
    ax[0].set_title('input', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n"
                     "RMSE: {:.4f}".format(original_result[0],
                                           original_result[1],
                                           original_result[2]), fontsize=20)

    ax[1].imshow(x_exchange, cmap='gray', vmin=trunc_min, vmax=trunc_max)
    ax[1].set_title('x_exchange', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n"
                     "RMSE: {:.4f}".format(input_result[0],
                                           input_result[1],
                                           input_result[2]), fontsize=20)

    ax[2].imshow(y_exchange, cmap='gray', vmin=trunc_min, vmax=trunc_max)
    ax[2].set_title('y_exchange', fontsize=30)
    ax[2].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n"
                     "RMSE: {:.4f}".format(target_result[0],
                                           target_result[1],
                                           target_result[2]), fontsize=20)

    ax[3].imshow(y, cmap='gray', vmin=trunc_min, vmax=trunc_max)
    ax[3].set_title('target', fontsize=30)

    f.savefig(os.path.join(save_path, file_name))
    plt.close()
