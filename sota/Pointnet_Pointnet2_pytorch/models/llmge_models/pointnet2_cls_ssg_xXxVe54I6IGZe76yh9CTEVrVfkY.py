# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.path as path
import matplotlib.patheffects as pe
import matplotlib.colors as colors
import matplotlib.cm as cm
from PIL import Image
import json
import argparse
import pdb
import pickle
import random
from tqdm import tqdm
from collections import namedtuple
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.activation import ReLU
from torch.nn.modules.container import Sequential
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.padding import ZeroPad2d
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.sampler import SubsetRandomSampler
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.modules.loss import KLDivLoss
from torch.nn.modules.loss import MSELoss
from torch.nn.modules.loss import NLLLoss
from torch.nn.modules.loss import PoissonNLLLoss
from torch.nn.modules.loss import SmoothL1Loss
from torch.nn.modules.loss import SoftMarginLoss
from torch.nn.modules.loss import MultiLabelSoftMarginLoss
from torch.nn.modules.loss import HingeEmbeddingLoss
from torch.nn.modules.loss import MultiTaskLoss
from torch.nn.modules.loss import TripletMarginLoss
from torch.nn.modules.loss import CosineEmbeddingLoss
from torch.nn.modules.loss import AdaptiveLogSoftmaxWithLoss
from torch.nn.modules.loss import AdaptiveMaxSoftmaxWithLoss
from torch.nn.modules.loss import HungarianMatcher
from torch.nn.modules.loss import Matcher
from torch.nn.modules.loss import LabelSmoothingCrossEntropyLoss
from torch.nn.modules.loss import SoftDiceLoss
from torch.nn.modules.loss import TverskyLoss
from torch.nn.modules.loss import LovaszHingeLoss
from torch.nn.modules.loss import LovaszSoftmaxLoss
from torch.nn.modules.loss import OhemCELoss
from torch.nn.modules.loss import SigmoidFocalLoss
from torch.nn.modules.loss import QFocalLoss
from torch.nn.modules.loss import DiceLoss
from torch.nn.modules.loss import IoULoss
from torch.nn.modules.loss import CIoULoss
from torch.nn.modules.loss import FocalLoss
from torch.nn.modules.loss import
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--