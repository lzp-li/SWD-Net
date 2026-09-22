# SWD-Net: Lightweight Steel Defect Detection Method Based on Frequency Domain Reconstruction and Denoising Mechanism

**Status:** Currently Under Review in *Journal of Electronic Imaging*.

This repository provides the core implementation of SWD-Net. To address challenges such as the easy loss of subtle defect features and the interference of low-contrast complex background noise in steel surface defect detection, SWD-Net designs a symmetric frequency domain reconstruction mechanism and an adaptive semantic denoising module. It preserves the high-frequency topological edges of subtle defects and dynamically suppresses background noise without amplifying it during reconstruction.

## 1. Prerequisites

Our experiments are implemented and validated based on the following environment:

* **OS:** Ubuntu 24.04.3 LTS
* **GPU:** NVIDIA GeForce RTX 5070 Ti Laptop GPU (12GB VRAM)
* **Framework:** PyTorch v2.10.0, CUDA 13.0

Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/lzp-li/SWD-Net.git](https://github.com/lzp-li/SWD-Net.git)
cd SWD-Net
# pip install -r requirements.txt (If you upload it later)
2. Data Preparation
Our experiments are conducted on the publicly available NEU-DET and GC10-DET steel surface defect datasets.

Download the datasets from their official repositories.

Organize the images and labels in the standard YOLO format.

Modify the dataset path in your .yaml configuration file (e.g., gc10.yaml).

3. Quick Start Guide
Training
To train the SWD-Net model from scratch on the GC10-DET dataset, run the corresponding training script. The default input resolution is 640x640:
python train_GC10.py --weights '' --cfg models/swd_net.yaml --data gc10.yaml --epochs 300
(Note: Please replace gc10.yaml with your local dataset yaml file path).

Testing / Inference
To evaluate the model's performance on the test set, use the test script:

python test.py --weights runs/train/exp/weights/best.pt --data gc10.yaml --img-size 640
4. Main Results
SWD-Net achieves an optimal balance between detection performance and lightweight design on steel surface defect benchmarks:

NEU-DET: 80.2% mAP@0.5.

GC10-DET: 69.4% mAP@0.5.

Disclaimer & Code Availability
Note: This repository currently provides the core structural implementation (e.g., common.py) and step-by-step run scripts for peer-review validation. Full pre-trained weights and highly detailed ablation configurations will be fully released upon the formal acceptance of the manuscript.
