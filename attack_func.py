'''attack methods'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


np_dtype = np.dtype('float32')


# FGSM attack code
class FGSM(object):
    """FGSM attack"""
    def __init__(self, epsilon, clip_min=0., clip_max=1., targeted=False):
        super(FGSM, self).__init__()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = epsilon
        self.targeted = targeted

    def attack_batch(self, model, images, targets):
        model.eval()

        # Set requires_grad attribute of tensor. Important for Attack
        images.requires_grad = True

        # Forward pass the data through the model
        outputs = model(images)

        # Calculate the loss
        if self.targeted:
            loss = -1. * F.nll_loss(outputs, targets)
        else:
            loss = F.nll_loss(outputs, targets)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = images.grad.data

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_images = images + self.eps * sign_data_grad
        # Adding clipping to maintain [clip_min, clip_max] range
        perturbed_images = torch.clamp(perturbed_images, 
                                       self.clip_min, self.clip_max)
        # Return the perturbed image
        return perturbed_images, data_grad


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0., 1.)
    # Return the perturbed image
    return perturbed_image

