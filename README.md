# Parallel Hyper-Prior and Dual-Domain Autoregressive Transformer for Lossless JPEG Recompression of Pathology Images

The implementation is built on top of [CompressAI](https://github.com/InterDigitalInc/CompressAI) and extends it with a pathology-oriented DCT-domain recompression pipeline, including:

- a parallel luma/chroma recompression framework,
- a dual-domain autoregressive Transformer,
- fractal spatial autoregressive modeling,
- frequency-domain autoregressive modeling,
- pathology JPEG dataset loaders based on on-the-fly JPEG quantization.

## Repository Scope

This fork is cleaned for public release around the paper implementation. The focus is on the modules required to reproduce the pathology JPEG recompression experiments rather than the full upstream CompressAI distribution.

Main paper-related entry points:

- `train_lossless_jpeg_trans.py`: training script for the pathology JPEG recompression model.
- `train_lossless_jpeg_trans.sh`: minimal example launcher.
- `compressai/models/trans_eff.py`: main recompression model definitions.
- `compressai/models/trans_network.py`: fractal autoregressive backbone.
- `compressai/datasets/image.py`: pathology dataset and DCT/JPEG preprocessing utilities.
- `pathology.py`: optional pathology slide preprocessing helpers.

## Environment

Recommended environment:

- Python 3.8+
- PyTorch
- torchvision
- CompressAI dependencies from `setup.py`
- `torchjpeg`
- `jpegio`
- `range_coder`
- `openslide-python` for slide preprocessing utilities in `pathology.py`

Install the package in editable mode:

```bash
pip install -e .
pip install torchjpeg jpegio range_coder openslide-python
```

If the optional CUDA extensions under `compressai/latent_codecs` are needed in your setup, build them separately after the main environment is ready.

## Dataset Preparation

The training code expects pathology image patches stored under a dataset root. The current `PNGFolder_Trans` loader reads all image files under each immediate child directory of the dataset root and performs JPEG quantization on the fly.

Expected layout:

```text
DATASET_ROOT/
  patient_or_slide_group_001/
    patch_0001.png
    patch_0002.png
    ...
  patient_or_slide_group_002/
    patch_0001.png
    patch_0002.png
    ...
```

The split logic currently follows the original experiment code in `compressai/datasets/image.py`:

- `train`: folders whose numeric suffix is `<= 100`
- `test`: folders whose numeric suffix is `> 100`
- `val`: folders whose numeric suffix is `> 128`

If your local dataset naming convention differs, adjust `PNGFolder_Trans` accordingly before training.

## Training

Minimal example:

```bash
python train_lossless_jpeg_trans.py \
  --cuda \
  --dataset /path/to/pathology_dataset \
  --quality 75 \
  --batch-size 8 \
  --test-batch-size 32 \
  --num-workers 4 \
  --net B \
  --output-dir ./compress_output \
  --experiment-name fractral_tmi_q75
```

Or use the helper shell script:

```bash
DATASET_ROOT=/path/to/pathology_dataset bash train_lossless_jpeg_trans.sh
```

Checkpoints and logs are written to:

```text
compress_output/<experiment_name>/
```

## Notes on Cleanup

For public release, the repository has been lightly cleaned to:

- remove hard-coded personal file paths,
- replace internal cluster launch commands with local examples,
- remove local build artifacts and cache files from version control,
- add a top-level README and ignore rules.

This cleanup intentionally avoids large-scale refactoring so that the released code remains close to the experiment code used in the manuscript.

## Acknowledgment

This project is based on CompressAI. Please also cite the original CompressAI repository and any upstream components you use in your work.

## License

See [LICENSE](LICENSE).
