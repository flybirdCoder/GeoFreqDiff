# GeoFreqDiff

**GeoFreqDiff: Geometric Filter-Bank Guided Frequency Diffusion Model for Low-Dose CT Denoising**


## Run

Train:

```bash
python main.py --mode train
```

Test:

```bash
python main.py --mode test
```

## Switch Dataset

To use another dataset, change `--processed_data` to the target `.npy` directory.

The default data loader reads paired files named as:

- `*_input.npy`
- `*_target.npy`

If your dataset uses a different naming rule, update [dataProcess/loader.py](dataProcess/loader.py).

For data preprocessing examples, refer to:

- `dataProcess/gen_data/mayo_dataGen.py`
- `dataProcess/gen_data/piglet_dataGen.py`

