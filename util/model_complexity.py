import argparse
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile as torch_profile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.diffusion.GeoFreqDiffusion import ResidualDiffusion, UnetRes


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def build_model(transform, input_size, sampling_timesteps):
    residual_network = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=1,
        condition=True,
        input_condition=False,
        objective='pred_res',
        test_res_or_noise='res',
    )
    return ResidualDiffusion(
        residual_network,
        image_size=input_size,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        objective='pred_res',
        condition=True,
        sum_scale=1,
        input_condition=False,
        input_condition_mask=False,
        test_res_or_noise='res',
        transform_type=transform,
    )


def estimate_flops(model, model_input, device):
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    synchronize(device)
    with torch.inference_mode():
        with torch_profile(activities=activities, with_flops=True) as profiler:
            model(model_input)
    synchronize(device)
    return sum(event.flops or 0 for event in profiler.key_averages())


def measure_runtime(model, model_input, device, warmup_runs, measured_runs):
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(model_input)

        synchronize(device)
        start = time.perf_counter()
        for _ in range(measured_runs):
            model(model_input)
        synchronize(device)

    total_time = time.perf_counter() - start
    return total_time, total_time / measured_runs


def collect_complexity(args):
    if args.input_size <= 0 or args.input_size % 2 != 0:
        raise ValueError('input_size must be a positive even integer.')
    if args.runs <= 0 or args.warmup_runs < 0:
        raise ValueError('runs must be positive and warmup_runs cannot be negative.')

    transform = 'gtf' if args.transform in ('cfb', 'gtf') else 'dwt'
    transform_name = 'GTF-CFB' if transform == 'gtf' else 'DWT'
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(transform, args.input_size, args.sampling_timesteps)
    model = model.to(device).eval()

    ndct = torch.randn(1, 1, args.input_size, args.input_size, device=device)
    ldct = torch.randn_like(ndct)
    model_input = [ndct, ldct]

    with torch.inference_mode():
        coefficients = model.analysis_transform(ldct)
    component_count = coefficients.shape[1]
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_time, average_time = measure_runtime(
        model,
        model_input,
        device,
        args.warmup_runs,
        args.runs,
    )
    flops = estimate_flops(model, model_input, device)

    print('Complexity ===')
    print('Transform: {}'.format(transform_name))
    print('Input shape: 1 x 1 x {0} x {0}'.format(args.input_size))
    print('Sampling steps: {}'.format(args.sampling_timesteps))
    print('Transform components: {}'.format(component_count))
    print('Total parameters: {:,} ({:.4f} M)'.format(
        total_parameters, total_parameters / 1e6
    ))
    print('Trainable parameters: {:,} ({:.4f} M)'.format(
        trainable_parameters, trainable_parameters / 1e6
    ))
    print('Measured runs: {}'.format(args.runs))
    print('Total measured time: {:.6f} s'.format(total_time))
    print('Average inference time: {:.6f} s/patch'.format(average_time))
    print('Estimated FLOPs: {:.4f} GFLOPs'.format(flops / 1e9))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure GeoFreqDiff complexity using a fixed-size synthetic input.'
    )
    parser.add_argument('--transform', choices=('cfb', 'gtf', 'dwt'), default='dwt')
    parser.add_argument('--input_size', type=int, default=64)
    parser.add_argument('--sampling_timesteps', type=int, default=5)
    parser.add_argument('--warmup_runs', type=int, default=5)
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=408)
    return parser.parse_args()


if __name__ == '__main__':
    collect_complexity(parse_args())
