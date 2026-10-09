# The code can be referenced at https://github.com/SSinyu/RED-CNN

import argparse

from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.backends import cudnn
from util.solver import Solver


def main(args):
    cudnn.benchmark = True
    accelerator = Accelerator()
    set_seed(args.seed)

    solver = Solver(args, accelerator)
    diff_trainer, model_id = solver.get_diff_trainer()

    if args.mode == 'train':
        diff_trainer.train()
    else:
        diff_trainer.load(args.load_milestone)
        diff_trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeoFreqDiff for Low-dose CT Denoising')

    # train and test
    parser.add_argument('--mode', type=str, default='train')

    # mayo
    parser.add_argument('--processed_data', type=str,
                        default='./dataProcess/gen_data/mayo_npy_img')

    # piglet
    # parser.add_argument('--processed_data', type=str,
    #                     default='./dataProcess/gen_data/piglet_25_npy/test')

    parser.add_argument('--results_folder', type=str, default='./results/GeoFreqDiff/GeoFreqDiff_clinic_save')
    parser.add_argument('--test_folder', type=str, default='./results/GeoFreqDiff/test/cm_test')

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--decay_iters', type=int, default=10000)

    parser.add_argument('--lr', type=float, default=8e-5)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    # for mayo test or val
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--val_patient', type=str, default='L333')

    ### diffusion ###
    parser.add_argument('--ema_decay', type=int, default=0.995)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_and_val_every', type=int, default=10000)
    parser.add_argument('--sampling_timesteps', type=int, default=5)
    parser.add_argument('--train_num_steps', type=int, default=100000)
    parser.add_argument('--load_milestone', type=int, default=100)
    parser.add_argument('--seed', type=int, default='408')
    parser.add_argument('--transform', type=str, choices=('gtf', 'dwt'), default='gtf')

    args = parser.parse_args()

    main(args)
