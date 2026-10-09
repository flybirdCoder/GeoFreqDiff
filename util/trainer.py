import os

import lpips
import math
import numpy as np
import piq
import torch
from einops import rearrange
from ema_pytorch import EMA
from matplotlib import pyplot as plt
from pytorch_msssim import ssim
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from pathlib import Path
from model.diffusion.GeoFreqDiffusion import exists
from util.measure import compute_measure


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def patch_transform(x):
    if x.dim() == 3:
        return rearrange(x.float(), 'b h w -> b 1 h w')
    else:
        return rearrange(x.float(), 'b n h w -> (b n) 1 h w')


class GeoFreqDiffTrainer(object):
    def __init__(
            self,
            *,
            diffusion_model,
            data_loader=None,
            test_data_loader,
            val_data_loader=None,
            lr=1e-5,
            train_num_steps=100000,
            decay_iters=5000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_val_every=1000,
            results_folder='./results/model_save',
            test_folder='./results/test',
            norm_range_min=None,
            norm_range_max=None,
            trunc_min=None,
            trunc_max=None,
            accelerator=None,
            mode=None
    ):
        super().__init__()

        self.accelerator = accelerator
        self.diff_model = diffusion_model
        self.mode = mode
        self.norm_range_min = norm_range_min
        self.norm_range_max = norm_range_max
        self.trunc_min = trunc_min
        self.trunc_max = trunc_max
        self.lr = lr

        self.save_and_val_every = save_and_val_every

        self.train_num_steps = train_num_steps
        self.decay_iters = decay_iters
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader

        self.opt = Adam(self.diff_model.parameters(),
                        lr=lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(self.diff_model, beta=ema_decay,
                           update_every=ema_update_every)
            self.set_results_folder(results_folder)

        self.diff_model, self.opt = self.accelerator.prepare(
            self.diff_model, self.opt)

        device = self.accelerator.device
        self.device = device
        self.test_folder = test_folder

        self.l1_loss = nn.L1Loss()
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)

        self.step = 0

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def estimation_losses(self, data_dict):
        pred_low_res = data_dict['pred_res']
        truth_low_res = data_dict['truth_res']
        res_loss = self.l1_loss(pred_low_res, truth_low_res)

        pred_image = data_dict['pred_image']
        truth_image = data_dict['truth_image']
        l1_loss = self.l1_loss(pred_image, truth_image)
        ssim_loss = 1 - ssim(pred_image, truth_image, data_range=1.0).to(self.device)
        p_loss = self.lpips_fn.forward(pred_image, truth_image).mean()
        loss = res_loss + 0.35 * l1_loss + 0.70 * ssim_loss + 0.05 * p_loss
        return loss

    def train(self):
        total_losses = []
        accelerator = self.accelerator
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.diff_model.train()

                data = next(self.data_loader)

                data = [patch_transform(item).to(self.device) for item in data]

                with self.accelerator.autocast():
                    data_dict = self.diff_model(data, self.mode)
                    loss = self.estimation_losses(data_dict)
                    total_losses.append(loss.item())
                self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.diff_model.parameters(), 1.0)
                pbar.set_description(f'loss: {loss.item():.8f}', refresh=False)

                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_val_every == 0:
                        # for mayo and clinic
                        self.sample_val()

                        milestone = (self.step // self.save_and_val_every) * 10
                        self.save(milestone)
                        np.save(os.path.join(self.results_folder, 'loss_{}k_iter.npy'.format(milestone)),
                                np.array(total_losses))

                    if self.step != 0 and self.step % self.decay_iters == 0:
                        self.lr_decay()

                pbar.update(1)

        # for melt
        self.test()
        accelerator.print('training complete')

    @torch.no_grad()
    def sample_val(self):
        self.ema.to(self.device)
        self.ema.ema_model.eval()

        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        data_len = len(self.val_data_loader)

        val_step = 0

        with torch.no_grad():
            print("\n validate start")
            with tqdm(initial=val_step, total=data_len) as pbar:
                for i, data in enumerate(self.val_data_loader):
                    x_input_sample = [patch_transform(item).to(self.device) for item in data]
                    show_input_sample = x_input_sample

                    x_original = self.one_to_image(show_input_sample[1], test=True)
                    y_original = self.one_to_image(show_input_sample[0], test=True)

                    pred = self.diff_model(x_input_sample)
                    pred_image = self.one_to_image(pred, test=True)

                    data_range = self.trunc_max - self.trunc_min

                    original_result, pred_result = compute_measure(x_original, y_original, pred_image, data_range)

                    ori_psnr_avg += original_result[0]
                    ori_ssim_avg += original_result[1]
                    ori_rmse_avg += original_result[2]
                    pred_psnr_avg += pred_result[0]
                    pred_ssim_avg += pred_result[1]
                    pred_rmse_avg += pred_result[2]

                    val_step += 1
                    pbar.update(1)
            pbar.close()

            if self.accelerator.is_main_process:
                print('validate mode result: \n', flush=True)
                print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                    ori_psnr_avg / len(self.val_data_loader),
                    ori_ssim_avg / len(self.val_data_loader),
                    ori_rmse_avg / len(self.val_data_loader)), flush=True)
                print('\n', flush=True)
                print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                    pred_psnr_avg / len(self.val_data_loader),
                    pred_ssim_avg / len(self.val_data_loader),
                    pred_rmse_avg / len(self.val_data_loader)), flush=True)

    @torch.no_grad()
    def test(self):
        self.ema.to(self.device)
        self.ema.ema_model.eval()

        gen_img = self.test_folder
        self.set_results_folder(gen_img)

        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg, ori_vif_avg = 0, 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg, pred_vif_avg = 0, 0, 0, 0

        data_len = len(self.test_data_loader)

        self.step = 0

        print("test start")
        with torch.no_grad():
            with tqdm(initial=self.step, total=data_len) as pbar:
                for i, data in enumerate(self.test_data_loader):
                    x_input_sample = [patch_transform(item).to(self.device) for item in data]
                    show_input_sample = x_input_sample

                    x_original = self.one_to_image(show_input_sample[1], test=True)
                    y_original = self.one_to_image(show_input_sample[0], test=True)

                    pred = self.diff_model(x_input_sample)
                    pred_image = self.one_to_image(pred, test=True)

                    data_range = self.trunc_max - self.trunc_min

                    original_result, pred_result = compute_measure(x_original, y_original, pred_image, data_range)
                    original_vif = self.compute_VIF(x_original, y_original)
                    pred_vif = self.compute_VIF(pred_image, y_original)

                    ori_psnr_avg += original_result[0]
                    ori_ssim_avg += original_result[1]
                    ori_rmse_avg += original_result[2]
                    ori_vif_avg += original_vif.item()

                    pred_psnr_avg += pred_result[0]
                    pred_ssim_avg += pred_result[1]
                    pred_rmse_avg += pred_result[2]
                    pred_vif_avg += pred_vif.item()

                    file_name = f'pred-{i}.png'
                    self.show_test_fig(x_original, y_original, pred_image,
                                       original_result, pred_result, self.results_folder, file_name)
                    self.step += 1
                    pbar.update(1)

            # for test
            print('\n')
            print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nVIF avg: {:.4f}'.format(
                ori_psnr_avg / len(self.test_data_loader),
                ori_ssim_avg / len(self.test_data_loader),
                ori_rmse_avg / len(self.test_data_loader),
                ori_vif_avg / len(self.test_data_loader)))
            print('\n')
            print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f} \nVIF avg: {:.4f}'.format(
                pred_psnr_avg / len(self.test_data_loader),
                pred_ssim_avg / len(self.test_data_loader),
                pred_rmse_avg / len(self.test_data_loader),
                pred_vif_avg / len(self.test_data_loader)))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diff_model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}k.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}k.pt')

        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            model = self.accelerator.unwrap_model(self.diff_model)
            model.load_state_dict(data['model'])
            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            self.ema.load_state_dict(data['ema'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])
            print("load model - " + str(path))

    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)

    def show_test_fig(self, x, y, pred, original_result, pred_result, save_path, file_name):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap='gray', vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('input', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n"
                         "RMSE: {:.4f}".format(original_result[0],
                                               original_result[1],
                                               original_result[2]), fontsize=20)

        ax[1].imshow(pred, cmap='gray', vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\n"
                         "RMSE: {:.4f}".format(pred_result[0],
                                               pred_result[1],
                                               pred_result[2]), fontsize=20)

        ax[2].imshow(y, cmap='gray', vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('target', fontsize=30)

        f.savefig(os.path.join(save_path, file_name))
        plt.close()

    def one_to_image(self, image, test=False, display=False):
        shape_ = image.shape[-1]
        image = self.trunc(self.denormalize_(image.view(shape_, shape_).cpu().detach()))
        if display:
            image = 255 * (image - self.trunc_min) / (self.trunc_max - self.trunc_min)
        if test:
            if image.dim() > 2:
                image = image.squeeze()
            if image.dim() != 2:
                raise ValueError("Please check the dimensions")
            return image.float()
        return image

    def compute_VIF(self, x, y):
        x_tensor = torch.from_numpy(
            x.numpy() if isinstance(x, torch.Tensor) else x).float()
        x_tensor = (x_tensor - self.trunc_min) / (self.trunc_max - self.trunc_min)
        y_tensor = torch.from_numpy(
            y.numpy() if isinstance(y, torch.Tensor) else y).float()
        y_tensor = (y_tensor - self.trunc_min) / (self.trunc_max - self.trunc_min)

        x_tensor = x_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        y_tensor = y_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        return piq.vif_p(
            x_tensor.to(self.device),
            y_tensor.to(self.device),
            data_range=1.0,
            reduction='none'
        )
