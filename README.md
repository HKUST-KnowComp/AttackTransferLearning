# White-box and Black-box Attacks for Transfer Learning

This repository contains the code of our KDD20 paper "Two Sides of the Same Coin: White-box and Black-box Attacks for Transfer Learning". 

## Usage
### Dependencies
- PyTorch >= 1.0
- Python3

### Prepare Datasets
Please specify your local path to store datasets in line 18 of `prep_data.py`. Some public datasets such as MNIST and SVHN can be automatically downloaded with torchvision APIs. More datasets can be added by implementing customized data loaders. 

### Pre-trained Models
A few pre-trained models are provided under the `ckpt` path. 
You can also train a model from scratch or fine-tune a model from a pre-trained model using the `train.py` script. 

### White-box Attacks
The `whitebox_attack.py` applies the FGSM attack towards a model. 

To attack an MNIST model that is trained from scratch, the following command can be used: 
```python whitebox_attack.py --dataset=mnist --arch=DTN --ckpt_file=./ckpt/white-box/mnist_scratch.pt --attack_method=FGSM --eps=4,8,16,32```

Similarly, to attack an MNIST model that is fine-tuned from a model that is pre-trained on SVHN, the following command can be used: 
```python whitebox_attack.py --dataset=mnist --arch=DTN --ckpt_file=./ckpt/white-box/mnist_ft_svhn.pt --attack_method=FGSM --eps=4,8,16,32```

### Black-box Attacks
The `blackbox_attack_by_transfer.py` attacks a target model with the adversarial examples produced by its source model. Two models can be trained with Scratch, FT or CommonInit strategies. 

The commands to reproduce the results of (MNIST, USPS, SVHN) are given in the following. 

When two models are trained independently: 
```
python blackbox_attack_by_transfer.py --dataset=usps --arch=DTN --ckpt_a=./ckpt/black-box/mnist_source.pt --ckpt_b=./ckpt/black-box/usps_scratch.pt --attack_method=FGSM --eps=4,8,16,32
```

When the USPS model is fine-tuned from an MNIST model: 
```
python blackbox_attack_by_transfer.py --dataset=usps --arch=DTN --ckpt_a=./ckpt/black-box/mnist_source.pt --ckpt_b=./ckpt/black-box/usps_ft_from_mnist.pt --attack_method=FGSM --eps=4,8,16,32
```

When the two models are commonly initialized from a model pre-trained on SVHN: 
```
python blackbox_attack_by_transfer.py --dataset=usps --arch=DTN --ckpt_a=./ckpt/black-box/mnist_commoninit.pt --ckpt_b=./ckpt/black-box/usps_commoninit.pt --attack_method=FGSM --eps=4,8,16,32
```

## Citation
If you use this code, please consider citing our paper: 

```
Yinghua Zhang, Yangqiu Song, Jian Liang, Kun Bai, and Qiang Yang. 2020. Two Sides of the Same Coin: White-box and Black-box Attacks for Transfer Learning. In Proceedings of the 26th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’20), August 23–27, 2020, Virtual Event, CA, USA. ACM, New York, NY, USA. https://doi.org/10.1145/3394486.34033491. 
```

## Contact
Yinghua Zhang (yzhangdx@cse.ust.hk)
