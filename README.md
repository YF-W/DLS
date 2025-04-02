# Superpixel-Based Difference-Level Contrastive Learning Framework for Challenging Sample Processing in Medical Image Segmentation
## Paper Link: Under Review
## Usage:
1. Clone the repo
```
git clone https://github.com/YF-W/DLS.git
```
2. Put the data in './DLS/data';
3. Train the model (The training files will be released upon the acceptance of the paper.)
```
cd DLS/code
python train_DLS.py
```
4. Test the model
```
cd DLS/code
python test_3d.py
```

## Acknowledgements:
Our code is adapted from [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [MC-Net](https://github.com/ycwu1997/MC-Net) and [mutual-learning-with-reliable-pseudo-labels](https://github.com/Jiawei0o0/mutual-learning-with-reliable-pseudo-labels). Thanks for these authors for their valuable works and hope our model can promote the relevant research as well.