import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# For utilities
import time, sys, os
sys.path.insert(0, '../../')

# For conversion

import cv2
import opencv_transforms.transforms as TF
import opencv_transforms.functional as FF

# For everything
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.transforms as tt

# For our model

import torchvision.models
import itertools

# To ignore warning
import warnings
warnings.simplefilter("ignore", UserWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device=='cuda':
    print("The gpu to be used : {}".format(torch.cuda.get_device_name(0)))
else:
    print("No gpu detected")
