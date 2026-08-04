# Articulate That Object Part (ATOP): 3D Part Articulation via Text and Motion Personalization

Official implementation of **ATOP**
([arXiv:2502.07278](https://arxiv.org/abs/2502.07278)).

ATOP generates 3D articulation from static objects by finetuning a diffusion
model to produce spatially controllable multi-view articulation videos, which
are then lifted into an articulated 3D result. This repository currently
covers the finetuning and inference side of that pipeline.

## Repository structure

```
.
├── diffusers/                   # customized diffusers fork this repo runs against (see below)
├── docs/                        # project webpage (aditya-vora.github.io/atop)
├── src/                         # model, dataset, and pipeline code (includes src/ip_adapter/,
│                                 # which diffusers/ depends on)
├── finetune.py                  # training entry point
├── infer.py                     # inference entry point
├── requirements.txt             # core Python deps (training / inference)
├── .gitignore
└── README.md                    # you are here
```

`checkpoints/` and `datasetv0/` aren't part of the repo (they're `.gitignore`d) —
see [1.3](#13-get-the-pretrained-checkpoint) and [section 2](#2-training-data)
for how to populate them locally.

## Modules

| Module | Description | Documentation | Status |
|--------|-------------|---------------|--------|
| **Data preparation** | Convert PartNet-Mobility (SAPIEN) shapes into multi-view articulation videos and part-segmentation masks. | — | Not in this repo (see [Known issues](#known-issues)) |
| **Training** | Finetune the pretrained multi-view image checkpoint into a spatially-controllable multi-view video model. | [`finetune.py`](finetune.py) (`python finetune.py --help`) | Available |
| **Inference** | Generate multi-view articulation videos from a finetuned checkpoint. | [`infer.py`](infer.py) (`python infer.py --help`) | Available |
| **3D optimization** | Optimize an articulated 3D result from the generated videos. | — | Coming soon |

---

## 1. Environment setup

### 1.1 Clone the repository

```bash
git clone https://github.com/aditya-vora/atop.git
cd atop
```

### 1.2 Install dependencies

Requires Python 3.9+ and a CUDA-capable GPU.

```bash
conda create -n atop python=3.10 -y
conda activate atop
pip install -r requirements.txt
```

> `requirements.txt` pins a `cu128` PyTorch build (for Blackwell / RTX
> 50-series GPUs). On other hardware, edit the `--extra-index-url` line at
> the top of `requirements.txt` to match your CUDA version first.

`diffusers` is deliberately not in `requirements.txt` — this repo vendors its
own customized fork at [`diffusers/`](diffusers/), which anything run from
the repo root (`finetune.py`, `infer.py`) imports automatically instead of a
pip-installed version. No separate install step is needed for it, but see
[Known issues](#known-issues) for what that means if you ever move or copy
these scripts elsewhere.

### 1.3 Get the pretrained checkpoint

`finetune.py` starts from a pretrained multi-view *image* diffusion checkpoint
(ImageDream/MVDream-style). `checkpoints/` is `.gitignore`d (multi-GB
weights), so fetch it separately:

```bash
git lfs install
git clone https://huggingface.co/ashawkey/imagedream-ipmv-diffusers \
    checkpoints/imagedream-ipmv-diffusers
```

### 1.4 Get the training data

`datasetv0/` (or whatever you point `--data_root` at) is also `.gitignore`d
and needs to be populated separately — see [section 2](#2-training-data)
below for the layout `finetune.py`/`infer.py` expect.

---

## 2. Training data

The data-preparation pipeline that generates this layout from raw
PartNet-Mobility shapes isn't part of this repository yet (see
[Known issues](#known-issues)). `finetune.py`/`infer.py` expect the following
on disk, however you produce it, rooted at `--data_root` (default
`datasetv0/`):

```
<data_root>/<Category>/<shape_id>/
├── mobility_v2.json                          # PartNet-Mobility joint annotations
├── videos/video_r_<view>_<part_idx>_<joint>.mp4  # multi-view articulation videos
├── masks/mask_r_<view>_<part_idx>_<joint>.png    # per-view part-segmentation masks
│                                                   (the spatial control signal)
└── poses/pose_r_<view>_<part_idx>_<joint>.txt    # per-view "azimuth elevation distance"
```

`<part_idx>_<joint>` (e.g. `0_hinge`) identifies a movable part, in the same
order as the movable (`joint` + `jointData`) entries of that shape's
`mobility_v2.json`. `<view>` is a zero-padded azimuth in degrees (e.g. `000`,
`090`, `180`, `270` — the default `--views`). A shape only contributes a
training/inference example for a part once it has a video, mask, and pose for
*every* requested view; `finetune.py`/`infer.py` skip (with a warning)
anything incomplete, and raise a clear error if a split ends up empty.

Training/test splits are flat `Category,shape_id` text files, one shape per
line — see the format used by `datasetv0/train.txt` / `datasetv0/test.txt`.

---

## 3. Training

Finetune the pretrained checkpoint on your training split:

```bash
python finetune.py \
    --pretrained_model_path checkpoints/imagedream-ipmv-diffusers \
    --data_root datasetv0 \
    --train_split datasetv0/train.txt \
    --output_dir output/atop-finetune
```

Only the UNet's cross-view/IP-Adapter attention projections and the newly
added temporal + part-mask attention are trained (see
[`src/models/mv_unet.py`](src/models/mv_unet.py) for why); everything else —
VAE, text/image encoders, the rest of the UNet — stays frozen. Training
periodically checkpoints to `--output_dir`, and writes the final, standalone
checkpoint to `<output_dir>/final_checkpoint/`.

Run `python finetune.py --help` for the full option list (views, frame
count/rate, resolution, batch size, learning rate, guidance dropout, ...).

## 4. Inference

Generate multi-view articulation videos from a finetuned checkpoint:

```bash
python infer.py \
    --checkpoint output/atop-finetune/final_checkpoint \
    --data_root datasetv0 \
    --test_split datasetv0/test.txt \
    --output_dir output/atop-finetune/inference
```

For each shape/part in `--test_split`, this writes a tiled multi-view GIF,
per-view PNG frame sequences, and the conditioning image/mask used, to
`<output_dir>/<category>_<shape_id>_<motion_type>/`.

Run `python infer.py --help` for the full option list (guidance scale,
inference steps, resolution, seed, prompt override, ...).

---

## Known issues

- **The data-preparation pipeline isn't in this repository.** Section 2
  documents the on-disk layout `finetune.py`/`infer.py` expect, but the code
  that generates it from raw PartNet-Mobility shapes (organizing shapes,
  articulating meshes, rendering multi-view frames, assembling videos, and
  rendering the part-segmentation masks) is not included here yet.
- **Part masks aren't rendered for most shapes in `datasetv0/` yet** — only
  the `Door` category currently has them. `finetune.py`/`infer.py` skip any
  shape missing masks (with a warning) and error out if a split ends up empty
  after skipping.
- **`diffusers/` at the repo root shadows any pip-installed `diffusers`.**
  This is intentional (see [1.2](#12-install-dependencies)) — `diffusers/`
  is a customized fork this repo depends on, not a stray vendored copy. It
  in turn depends on `src/ip_adapter/` (`diffusers/models/attention_processor.py`
  imports `SparseCausalAttentionProcessor` from it). If you copy
  `finetune.py`/`infer.py` somewhere without both `diffusers/` and
  `src/ip_adapter/` alongside them, `import diffusers` will fail.
- `checkpoints/` and `datasetv0/` are intentionally untracked (`.gitignore`);
  see [1.3](#13-get-the-pretrained-checkpoint) and
  [section 2](#2-training-data) for how to populate them on a fresh clone.

## Citation

If you use this code, please cite:

```bibtex
@article{atop2025,
  title   = {ATOP},
  journal = {arXiv preprint arXiv:2502.07278},
  year    = {2025},
  url     = {https://arxiv.org/abs/2502.07278}
}
```

> **Note:** update the BibTeX entry above with the paper's full title and author
> list before public release.
