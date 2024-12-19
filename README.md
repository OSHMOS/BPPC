# Accurate Baseball Player Pose Refinement Using Motion Prior Guidance

![The framework of BPPC](./assets/BPPC_framework.png)

https://drive.google.com/file/d/1YeizctfSZQtzHbl8uBxJkT9PJnZxiOw9/view

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
git clone https://github.com/BPPE-BaseballPlayerPoseEstimation/BPPC.git
```

Install the bppc requirements using `conda`:
```bash
conda env create -f bppc.yaml

pip install yacs==0.1.8 filterpy

cd grid_sample1d/
python setup.py install
cd ..
```

### Test

Test the left-handed batter:
```bash
sh run_left.sh
```

Test the right-handed batter:
```bash
sh run_right.sh
```

## Citation

```
coming soon!
```

## Acknowledgement

- The repo (`grid_sample1d`) is based on [Grid Sample 1d](https://github.com/luo3300612/grid_sample1d.git). Thanks for their well-organized code!