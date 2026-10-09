import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GTF_IGTF.GTFTrans import GTF, IGTF
from util.measure import compute_PSNR, compute_SSIM


class PairedNpyDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        return np.load(target_path), np.load(input_path)


def find_pairs(folder, patient):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    pairs = []
    for input_path in sorted(folder.glob(f"{patient}_*_input.npy")):
        target_name = input_path.name.replace("_input.npy", "_target.npy")
        target_path = folder / target_name
        if not target_path.is_file():
            raise FileNotFoundError(f"Target file not found for {input_path.name}: {target_path}")
        pairs.append((input_path, target_path))

    if not pairs:
        raise ValueError(f"No paired data found for {patient} in {folder}")
    return pairs


def to_image(image, args):
    image = image * (args.norm_max - args.norm_min) + args.norm_min
    return image.clamp(args.trunc_min, args.trunc_max)


@torch.no_grad()
def evaluate_patient(patient, pairs, args, gtf, igtf, device):
    data_loader = DataLoader(
        PairedNpyDataset(pairs),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    data_range = args.trunc_max - args.trunc_min
    low_psnr = low_ssim = high_psnr = high_ssim = 0.0

    for target, low_dose in data_loader:
        target = target.float().unsqueeze(1).to(device)
        low_dose = low_dose.float().unsqueeze(1).to(device)
        target_coeff = gtf(target)
        low_dose_coeff = gtf(low_dose)

        low_frequency_degraded = igtf(
            torch.cat([low_dose_coeff[:, :1], target_coeff[:, 1:]], dim=1)
        )
        high_frequency_degraded = igtf(
            torch.cat([target_coeff[:, :1], low_dose_coeff[:, 1:]], dim=1)
        )

        reference = to_image(target, args)
        low_frequency_degraded = to_image(low_frequency_degraded, args)
        high_frequency_degraded = to_image(high_frequency_degraded, args)

        low_psnr += compute_PSNR(low_frequency_degraded, reference, data_range)
        low_ssim += compute_SSIM(low_frequency_degraded, reference, data_range)
        high_psnr += compute_PSNR(high_frequency_degraded, reference, data_range)
        high_ssim += compute_SSIM(high_frequency_degraded, reference, data_range)

    count = len(data_loader)
    low_result = low_psnr / count, low_ssim / count
    high_result = high_psnr / count, high_ssim / count

    print(f"\n{patient} slices: {count}")
    print("LDCT low frequency + NDCT high frequency:")
    print(f"PSNR: {low_result[0]:.4f}, SSIM: {low_result[1]:.4f}")
    print("NDCT low frequency + LDCT high frequency:")
    print(f"PSNR: {high_result[0]:.4f}, SSIM: {high_result[1]:.4f}")
    print("Same conclusion:", high_result[0] > low_result[0] and high_result[1] > low_result[1])


def test_prior(args):
    mayo_data = Path(args.mayo_data)
    if not mayo_data.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {mayo_data}")

    patients = sorted({path.name.split("_", 1)[0] for path in mayo_data.glob("*_input.npy")})
    if len(patients) < args.num_patients:
        raise ValueError(
            f"Only {len(patients)} Mayo patients were found, "
            f"but {args.num_patients} were requested."
        )
    selected_patients = random.Random(args.seed).sample(patients, args.num_patients)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gtf = GTF().to(device).eval()
    igtf = IGTF().to(device).eval()

    print("Randomly selected Mayo patients:", ", ".join(selected_patients))
    for patient in selected_patients:
        evaluate_patient(patient, find_pairs(mayo_data, patient), args, gtf, igtf, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mayo_data",
        type=str,
        default=str(PROJECT_ROOT / "dataProcess" / "gen_data" / "mayo_npy_img"),
    )
    parser.add_argument("--num_patients", type=int, default=3)
    parser.add_argument("--seed", type=int, default=408)
    parser.add_argument("--norm_min", type=float, default=-1024.0)
    parser.add_argument("--norm_max", type=float, default=3072.0)
    parser.add_argument("--trunc_min", type=float, default=-160.0)
    parser.add_argument("--trunc_max", type=float, default=240.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    test_prior(parser.parse_args())
