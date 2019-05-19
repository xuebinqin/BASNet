# BASNet
Code for paper '[BASNet: Boundary Aware Salient Object Detection](https://webdocs.cs.ualberta.ca/~xuebin/BASNet.pdf)', [Xuebin Qin](https://webdocs.cs.ualberta.ca/~xuebin/), Zichen Zhang, Chenyang Huang, Chao Gao, Masood Dehghan and Martin Jagersand.

## Required libraries

OS Ubuntu 18.04.2 LTS
Python 3.6
numpy 1.15.2
scikit-image 0.14.0
PIL 5.2.0
PyTorch 0.4.0
torchvision 0.2.1
glob

The SSIM loss is adapted from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py).

## Usage
1. Clone this repo
```
git clone https://github.com/NathanUA/BASNet.git
```
2. Download the pre-trained model [basnet.pth](https://drive.google.com/file/d/1qeKYOTLIOeSJGqIhFJOEch48tPyzrsZx/view?usp=sharing) and put it into the dirctory 'saved_models/basnet_bsi/'

3.  cd to 'BASNet'
run
```
python basnet_test.py
```
to test the inference
or
run
```
python basnet_train.py
```
to test the train.

 We also provid the predicted [saliency maps](https://drive.google.com/file/d/1K9y9HpupXT0RJ4U4OizJ_Uk5byUyCupK/view?usp=sharing) including SOD, ECSSD, DUT-OMRON, PASCAL-S, HKU-IS and DUTS-TE.

## Architecture

![BASNet architecture](figures/architecture.png)


## Quantitative Comparison

![Quantitative Comparison](figures/quan.png)

## Qualitative Comparison

![Qualitative Comparison](figures/qual.png)

## Citation
```
@InProceedings{Qin_2019,
  author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Gao, Chao and Dehghan, Masood and Jagersand, Martin},
  title = {BASNet: Boundary Aware Salient Object Detection},
  year = {2019}
}
```
