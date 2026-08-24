# MK-UNet

Official Pytorch implementation of the paper [MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Rahman_MK-UNet_Multi-kernel_Lightweight_CNN_for_Medical_Image_Segmentation_ICCVW_2025_paper.pdf) published in ICCV 2025 CVAMD
[Md Mostafijur Rahman](https://mostafij-rahman.github.io/), [Radu Marculescu](https://radum.ece.utexas.edu/)
<p>The University of Texas at Austin</p>

[ARXIV](https://arxiv.org/abs/2509.18493) | [PAPER](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Rahman_MK-UNet_Multi-kernel_Lightweight_CNN_for_Medical_Image_Segmentation_ICCVW_2025_paper.pdf) | [Code](https://github.com/SLDGroup/MK-UNet)

#### 🔍 **Check out our papers: [LoMix](https://github.com/SLDGroup/LoMix) [NeurIPS 2025], [EfficientMedNeXt](https://github.com/SLDGroup/EfficientMedNeXt) [MICCAI 2025], [EffiDec3D](https://github.com/SLDGroup/EffiDec3D) [CVPR 2025], [EMCAD](https://github.com/SLDGroup/EMCAD) [CVPR 2024], [PP-SAM](https://github.com/SLDGroup/PP-SAM) [CVPRW 2024], [G-CASCADE](https://github.com/SLDGroup/G-CASCADE) [WACV 2024], [MERIT](https://github.com/SLDGroup/MERIT) [MIDL 2023], [CASCADE](https://github.com/SLDGroup/CASCADE) [WACV 2023]**

## Local Reproduction and Path A Extension

This repository also contains a local two-step evaluation and extension study by **M. Mohaiminul Islam**. This added work is separate from the original MK-UNet authors' official release. The goal is to reproduce the released polyp pipeline, audit the code, and test a methodological extension called **MK-UNet-CC: Conditional and Sparse Multi-Kernel Computation**.

### Step 1: MK-UNet-T reproduction

The reproduction used the released polyp pipeline with `MK_UNet_T` on ClinicDB and ColonDB. Both runs used one seed, 200 epochs, repository-default hyperparameters, and CPU-only execution.

| Dataset | Paper DICE | Our DICE | Our IoU | Interpretation |
|---|---:|---:|---:|---|
| ClinicDB | 91.26 | 91.09 | 84.63 | Close to the published mean |
| ColonDB | 85.03 | 78.32 | 69.46 | Lower than the published mean |

The ClinicDB result is numerically consistent with the paper. The ColonDB result is not an exact reproduction of the paper result. The difference may come from the repository-default protocol, augmentation, preprocessing, random seed, or the lack of paper-exact multi-seed runs.

The main Step 1 outputs are in:

- `submission_texfile_pdf/MKUNet_All_Revised_v4/Step1_Deliverable1_Execution_Summary.pdf`
- `submission_texfile_pdf/MKUNet_All_Revised_v4/Step1_Deliverable2_Technical_Report.pdf`

### Step 2 / Path A: MK-UNet-CC extension

The original MK-UNet multi-kernel depth-wise convolution block computes the `1x1`, `3x3`, and `5x5` branches for every input and combines them with a fixed sum:

```text
F = F1 + F3 + F5
```

The extension asks whether every image and every stage needs all three kernel branches equally. MK-UNet-CC keeps the MK-UNet topology but changes the multi-kernel computation in controlled stages:

1. **Adaptive soft routing:** a small router predicts input-dependent weights for the three branches. This tests adaptivity, but all branches still run.
2. **Predictive-entropy routing:** decoder routers can also receive a difficulty signal from earlier coarse segmentation heads. This uses predictive entropy, not Bayesian epistemic uncertainty.
3. **Sparse top-1 routing:** at inference, only the selected branch is executed. This is the main efficiency experiment.
4. **Routing-guided pruning:** routing statistics are used to identify weak branches and design a smaller static model without router overhead.

#### MK-UNet-CC architecture (our extension)

<p align="center">
<img src="path%20A/results/mkunet_cc_architecture.png" width=100% class="center">
</p>

This figure shows the proposed extension on top of the original MK-UNet topology. The CMKDC-CC block adds routing to the multi-kernel depth-wise branches, the decoder can use detached predictive-entropy signals from coarse heads, and post-training routing statistics can be used for static pruning.

### Preliminary extension results

All extension results below are preliminary, single-seed ClinicDB results. Runs with different epoch budgets are not treated as directly comparable.

| Arm | Epochs | Params | Test DICE | FLOPs at 256 | Batch-1 CPU latency | Main finding |
|---|---:|---:|---:|---:|---:|---|
| Fixed MK-UNet-T baseline | 200 | 27,384 | 91.09 | 0.0668 G | 53.6 ms | Reproduction baseline |
| Adaptive soft routing | 200 | 30,588 | 91.53 | 0.0677 G | 57.5 ms | Small accuracy gain in one run, but not sparse |
| Fixed + auxiliary heads | 100 | 27,384 | 88.89 | 0.0668 G | - | Matched control for entropy experiment |
| Adaptive + auxiliary heads | 100 | 30,588 | 89.97 | 0.0677 G | - | Adaptivity helped over matched control |
| Predictive entropy + auxiliary heads | 100 | 30,618 | 88.99 | 0.0677 G | - | Entropy did not improve DICE in this run |
| Sparse top-1 | 60 fine-tune | 30,588 | 90.01 | 0.0307 G | 46.4 ms | Real branch skipping and lower latency |

The strongest efficiency result is sparse top-1 routing. Forward hooks verified that, for batch size 1, only **10 of 30** branch convolutions executed and **20 of 30** were skipped. This reduced FLOPs by about **54.1%** and measured batch-1 CPU latency by about **13.3%**. The latency reduction is smaller than the FLOP reduction because routing and dynamic dispatch have their own runtime cost.

The current result should not be described as final proof that MK-UNet-CC is better than MK-UNet. The safe claim is narrower: **conditional sparse branch execution can reduce computation and measured CPU latency while keeping competitive ClinicDB performance in this preliminary single-seed study**. Final claims require full-budget, multi-seed runs and target-device latency tests.

Key extension artifacts:

- `path A/results/all_runs.csv` — collected run table
- `path A/results/cost_table.json` — parameter and FLOP comparison
- `path A/results/sparse_verification.json` — executed/skipped branch counts and latency
- `path A/results/mkunet_cc_architecture.png` — extension architecture figure
- `submission_texfile_pdf/MKUNet_All_Revised_v4/Step2_PathA_Research_Proposal.pdf` — final Path A proposal with the method framing, literature positioning, and limitations

### How to reproduce the Path A extension

Run these commands from the repository root after preparing the ClinicDB and ColonDB folders under `data/polyp/target/`.

Install the environment as shown in the main usage section, then confirm these packages are available because Path A imports them directly:

```bash
pip install timm==0.6.12 thop
```

Quick verification before training:

```bash
python -W ignore step1/scripts/check_data.py
python -W ignore "path A/01_verify_head_mapping.py"
python -W ignore "path A/02_smoke_test.py"
python -W ignore "path A/03_measure_cost.py"
python -W ignore "path A/04_verify_sparse_execution.py"
```

Run the full Path A queue on CPU:

```powershell
powershell -ExecutionPolicy Bypass -File "path A/run_queue.ps1" -Python python -Device cpu
```

For CUDA, use:

```powershell
powershell -ExecutionPolicy Bypass -File "path A/run_queue.ps1" -Python python -Device cuda
```

The queue trains the adaptive stage first, finds its best checkpoint automatically, then fine-tunes the sparse top-1 model from that checkpoint. After training, it runs evaluation, reliability analysis, cost measurement, sparse-branch verification, and result collection through `path A/11_finalize.py`.

To refresh the final result table and figures after any new run:

```bash
python -W ignore "path A/11_finalize.py"
```

Main outputs are written to:

- `path A/runs/` for local checkpoints, run configs, histories, and per-run metrics
- `path A/results/all_runs.csv` for the consolidated metrics table
- `path A/results/summary_figure.png` for the result summary plot
- `path A/results/mkunet_cc_architecture.png` for the proposed architecture figure

## Architecture

<p align="center">
<img src="mkunet_architecture.png" width=100% height=40% 
class="center">
</p>

## Quantitative Results

## Qualitative Results

## Usage:
### Recommended environment:
**Please run the following commands.**
```
conda create -n mkunetenv python=3.8
conda activate mkunetenv

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

pip install -r requirements.txt

```

### Data preparation:

- **ClinicDB dataset:**
Download the splited ClinicDB dataset from [Google Drive](https://drive.google.com/drive/folders/1FPJr5f91uUCikxMvkwtZSEnYHemTZq1P?usp=share_link) and move into './data/polyp/' folder.

- **ColonDB dataset:**
Download the splited ColonDB dataset from [Google Drive](https://drive.google.com/drive/folders/1u4_8dMztnEBUaX-w3XfUR3jXLBhpccPA?usp=share_link) and move into './data/polyp/' folder.

### Training:
```
cd into MK-UNet
CUDA_VISIBLE_DEVICES=0 python -W ignore train_polyp.py --network MK_UNet

```

### Testing:
```
cd into MK-UNet 
CUDA_VISIBLE_DEVICES=0 python -W ignore test_polyp.py --network MK_UNet --run_id <your run_id>

```

## Acknowledgement
We are very grateful for these excellent works [EMCAD](https://github.com/SLDGroup/EMCAD), [CASCADE](https://github.com/SLDGroup/CASCADE), [MERIT](https://github.com/SLDGroup/MERIT), [G-CASCADE](https://github.com/SLDGroup/G-CASCADE), [PP-SAM](https://github.com/SLDGroup/PP-SAM), [PraNet](https://github.com/DengPingFan/PraNet), and [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT), which have provided the basis for our framework.

## Citations

``` 
@inproceedings{rahman2025mk,
  title={Mk-unet: Multi-kernel lightweight cnn for medical image segmentation},
  author={Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1042--1051},
  year={2025}
}
```
