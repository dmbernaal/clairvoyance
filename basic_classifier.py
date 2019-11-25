from fastai.vision import *
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from ranger import *
from functools import partial

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return x * ( torch.tanh(F.softplus(x)))

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

def conv_layer(ni, nf, act, ks=3, stride=2, bn=True, **kwargs):
    """
    A single conv layer will be: [Conv(ni x nf) + batchnorm + act]

    Will return Conv2d + Act

    ks = 3 == (3x3)
    """
    bias = False if bn else True # batch norm? if so remove bias
    if act=='mish': act_fn = Mish() # we will add more as we progress this builder
    else: act_fn = nn.ReLU(inplace=True)

    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=bias, **kwargs), act_fn]
    if bn: layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1)) # append bn if true

    return nn.Sequential(*layers)

def get_cnn_layers(data, nfs, layer, act, **kwargs):
    """
    data: databunch object
    nfs: number of filters = [32, 64, 64, etc]
    layer: type of layer: conv_layer by default
    """
    #input channel dimension
    c_in = data.train_ds[0][0].shape[0]
    c_out = data.c

    # appending first channel in
    nfs = [c_in] + nfs

    # creating our cnn body
    body = [layer(nfs[i], nfs[i+1], act, 5 if i==0 else 3, **kwargs) for i in range(len(nfs)-1)]

    # creating our model head: we should make this dynamic as well
    head = [
        nn.AdaptiveAvgPool2d(1),
        Flatten(), # add more linear layers here?
        nn.Linear(nfs[-1], c_out)
    ]

    return body + head

def get_cnn_model(data, nfs, layer, act, **kwargs):
    """
    Will create our dynamic
    Default:
        BatchNorm = True, Bias=False
        Activation = F.relu
    """
    return nn.Sequential(*get_cnn_layers(data, nfs, layer, act, **kwargs))

def init_cnn_(m, f):
    """
    Remove bias from model
    """
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, 'bias', None) is not None: m.bias.data.zero_() # zero bias if exist
    for l in m.children(): init_cnn_(l, f) # recursion

def init_cnn(m, uniform=False):
    """
    Initializing weights for our model. In the future we will append different init methods here
    """
    # kaiming norm: weight = weight * math.sqrt(1./fan_in)
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_cnn_(m, f)

def create_cnn_model(data, nfs, act, layer=conv_layer, uniform=False, **kwargs):
    """
    Main function to create a dynamic CNN Model
    """
    model = get_cnn_model(data, nfs, layer, act, **kwargs)
    init_cnn(model, uniform=uniform)
    return model