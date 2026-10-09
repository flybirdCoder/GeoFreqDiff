import torch

from dataProcess.loader import get_loader
from model.diffusion.GeoFreqDiffusion import UnetRes, ResidualDiffusion
from util.trainer import GeoFreqDiffTrainer


class Solver(object):

    def __init__(self, args, accelerator):
        self.mode = args.mode
        self.accelerator = accelerator
        self.processed_data = args.processed_data
        self.test_patient = args.test_patient
        self.val_patient = args.val_patient

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.results_folder = args.results_folder
        self.test_folder = args.test_folder
        self.multi_gpu = args.multi_gpu
        self.num_workers = args.num_workers

        self.lr = args.lr
        self.decay_iters = args.decay_iters

        self.batch_size = args.batch_size
        self.patch_n = args.patch_n
        self.patch_size = args.patch_size
        self.transform_type = getattr(args, 'transform', 'gtf')

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))

        # diffusionModel args
        self.ema_decay = args.ema_decay
        self.timesteps = args.timesteps
        self.save_and_val_every = args.save_and_val_every
        self.sampling_timesteps = args.sampling_timesteps
        self.train_num_steps = args.train_num_steps
        self.data_loader = get_loader(mode='train',
                                      processed_data=self.processed_data,
                                      test_patient=self.test_patient,
                                      val_patient=self.val_patient,
                                      patch_n=self.patch_n,
                                      patch_size=self.patch_size,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      accerlator=self.accelerator)

        self.test_data_loader = get_loader(mode='test',
                                           processed_data=self.processed_data,
                                           test_patient=self.test_patient,
                                           val_patient=self.val_patient,
                                           patch_n=0,
                                           patch_size=0,
                                           batch_size=1,
                                           num_workers=self.num_workers)
        # for val if you need
        self.val_data_loader = get_loader(mode='val',
                                          processed_data=self.processed_data,
                                          test_patient=self.test_patient,
                                          val_patient=self.val_patient,
                                          patch_n=0,
                                          patch_size=0,
                                          batch_size=1,
                                          num_workers=self.num_workers)

    def get_diff_trainer(self):
        num_unet = 1
        objective = 'pred_res'
        test_res_or_noise = "res"
        sum_scale = 1

        condition = True
        input_condition = False
        input_condition_mask = False

        model = UnetRes(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            num_unet=num_unet,
            condition=True,
            input_condition=False,
            objective=objective,
            test_res_or_noise=test_res_or_noise
        )

        diffusion = ResidualDiffusion(
            model,
            image_size=512,
            timesteps=self.timesteps,  # number of steps
            # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
            sampling_timesteps=self.sampling_timesteps,
            objective=objective,
            condition=condition,
            sum_scale=sum_scale,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
            test_res_or_noise=test_res_or_noise,
            transform_type=self.transform_type,
        )

        trainer = GeoFreqDiffTrainer(
            diffusion_model=diffusion,
            data_loader=self.data_loader,
            test_data_loader=self.test_data_loader,
            val_data_loader=self.val_data_loader,
            lr=self.lr,
            train_num_steps=self.train_num_steps,
            ema_decay=self.ema_decay,
            save_and_val_every=self.save_and_val_every,
            results_folder=self.results_folder,
            norm_range_min=self.norm_range_min,
            norm_range_max=self.norm_range_max,
            trunc_min=self.trunc_min,
            trunc_max=self.trunc_max,
            accelerator=self.accelerator,
            test_folder=self.test_folder,
            mode=self.mode
        )

        return trainer, self.train_num_steps // self.save_and_val_every
