# Accurate baseball player pose refinement using motion prior guidance
---
[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-red?style=for-the-badge)](https://www.sciencedirect.com/science/article/pii/S2405959525000360)
[![Journal](https://img.shields.io/badge/Journal-Homepage-blue?style=for-the-badge)](https://www.sciencedirect.com/journal/ict-express)
[![JCR](https://img.shields.io/badge/JCR-ICT_Express-green?style=for-the-badge)](https://jcr.clarivate.com/jcr-jp/journal-profile?journal=ICT%20EXPRESS&year=2024&fromPage=%2Fjcr%2Fhome)

![The framework of BPPC](./assets/BPPC_framework.png)

## Contributions

- We propose Baseball Player Pose Corrector (BPPC), an optimization technique for refining keypoints in baseball batting, leveraging prior knowledge of the 3D swing motion.

- We introduce a 4D keypoint projection method that accurately matches 3D standard motions to 2D test
videos, regardless of the differences between the standard motion and test swing videos.

- We propose a loss function that adaptively optimizes poses based on keypoint confidence and movements.

- BPPC improves the quantitative and qualitative performance of state-of-the-art HPE models on benchmark datasets.

## Getting Started

### Environment Requirement

Clone the repo:

```bash
git clone https://github.com/OSHMOS/BPPC.git
```

Install the bppc requirements using `conda` and `pip`:
```bash
conda create -n bppc python=3.12 -y

# Install PyTorch
# The following command is an example and should be modified according to your CUDA version and system environment
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

cd grid_sample1d/
python setup.py install
cd ..

pip install opencv-python
pip install tabulate
pip install scipy
pip install tqdm
pip install yacs
pip install numba
pip install scikit-image
pip install filterpy
```

Prepare the [checkpoints](https://drive.google.com/drive/folders/1vXUerOenwrbp0clkALKPehKkvq5HWvQK?usp=drive_link):

```
${POSE_ROOT}
    `-- lib
        `-- checkpoint
            |-- darkpose
            |   |-- w32_384×288.pth
            |   `-- w48_384×288.pth
            |-- hrnet
            |   |-- pose_hrnet_w32_384x288.pth
            |   `-- pose_hrnet_w48_384x288.pth
            |-- resnet
            |    |-- pose_resnet_50_384x288.pth
            |    |-- pose_resnet_101_384x288.pth
            |    `-- pose_resnet_152_384x288.pth
            `-- yolo3.weights
```


### Test

Test the left-handed batter:
```bash
bash run_bppc.sh
(python fine-tuning_one_motion.py --handed left)

# for the left-handed batter
--handed left

# for the right-handed batter
--handed right
```

## Citation

```
@article{OH2025,
    title = {Accurate baseball player pose refinement using motion prior guidance},
    journal = {ICT Express},
    year = {2025},
    issn = {2405-9595},
    doi = {https://doi.org/10.1016/j.icte.2025.03.008},
    url = {https://www.sciencedirect.com/science/article/pii/S2405959525000360},
    author = {Seunghyun Oh and Heewon Kim},
    keywords = {Human pose estimation, Human pose refinement, Deep learning}
}
```

## Acknowledgement

- The repo (`grid_sample1d`) is based on [Grid Sample 1d](https://github.com/luo3300612/grid_sample1d.git). Thanks for their well-organized code!
