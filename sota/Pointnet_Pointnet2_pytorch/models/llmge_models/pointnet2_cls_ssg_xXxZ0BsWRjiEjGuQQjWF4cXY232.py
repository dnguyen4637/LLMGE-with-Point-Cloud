# --PROMPT LOG--
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction

# --OPTION--
import tensorflow as tf

def random_normal(shape, mean=0.0, stddev=1.0):
    return tf.truncated_normal(shape, mean=mean, stddev=stddev)

def neural_network(x):
    # Input Layer
    W1 = tf.Variable(random_normal([n_hidden_1, n_input]))
    b1 = tf.Variable(random_normal([n_hidden_1]))
    layer_1 = tf.nn.relu(tf.matmul(W1, x) + b1)

    # Hidden Layers
    W2 = tf.Variable(random_normal([n_hidden_2, n_hidden_1]))
    b2 = tf.Variable(random_normal([n_hidden_2]))
    layer_2 = tf.nn.relu(tf.matmul(W2, layer_1) + b2)

    # Output Layer
    W3 = tf.Variable(random_normal([n_output, n_hidden_2]))
    b3 = tf.Variable(random_normal([n_output]))
    output = tf.matmul(W3, layer_2) + b3

    return output
# --OPTION--

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
# --OPTION--
import numpy as np

def calculate_average(numbers):
    return sum(numbers) / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(calculate_average(numbers))
