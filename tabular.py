## Inspired by FastAI
## Author: Diego Medina-Bernal
## github: https://github.com/dmbernaal

from fastai.layers import *
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from fastai.tabular import Module
import torch
import math

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return x * ( torch.tanh(F.softplus(x)))

class Selu(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.tensor(1.6732632423543772848170429916717)
        self.scale = torch.tensor(1.0507009873554804934193349852946)

    def forward(self, x):
        return self.scale * torch.where(x>=0.0, x, self.alpha * torch.exp(x) - self.alpha)

#export
def calc_ni_in(emb_szs, n_cont):
    """
    Will calculate our initial input dimension for initial linear layer in body layer
    """
    # setting our emb list for calculations
    sum_embs = emb_szs[:]
    
    # getting sum of emb(nf) for each emb but last
    sum_emb_ = 0
    for i in sum_embs:
        sum_emb_ += i[1]
        
    
    # creating ni for first linear layer
    ni_in = sum_emb_ + n_cont
    
    return ni_in

def embedding(ni, nf):
    emb = nn.Embedding(ni, nf)
    with torch.no_grad(): trunc_normal_(emb.weight, std=0.01)
    return emb

def get_model_embs(data):
    """
    Will return default model embeddings:

    TODO include custom embedding size for dimension per embedding layer
    """
    return data.get_emb_szs()

def tabular_layer(ni, nf, act, bn=True):
    """
    A single tabular layer is:
    linear(ni,nf), act_fn(), batchnorm if bn
    """
    # setting act_fun
    # TODO: Include others: Swish, SeLU, ETC
    if act == 'relu': act_fn = nn.ReLU(inplace=True)
    elif act == 'mish': act_fn = Mish()
    elif act == 'selu': act_fn, bn = Selu(), False

    # removing bias if batchnorm
    bias = False if bn else True

    # setting our layer: linear+act
    layers = [nn.Linear(ni, nf), act_fn]
    if bn: layers.append(nn.BatchNorm1d(nf, eps=1e-5, momentum=0.1))

    return nn.Sequential(*layers)

class Tabular(Module):
    def __init__(self, data, nfs, layer, act, emb_drop=0., **kwargs):
        super().__init__()
        """
        Will build our model layer by layer.
        """
        # getting our embedding sizes
        # TO DO: Have this dynamic, including user input for dim sz
        self.emb_szs = data.get_emb_szs()
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in self.emb_szs])
        self.n_emb = len(self.emb_szs)

        # emb drop and encoder batchnorm
        self.n_cont = len(data.cont_names) # number of cont vars
        self.emb_drop = nn.Dropout(emb_drop) # TO DO: make this dynamic
        self.bn_cont = nn.BatchNorm1d(self.n_cont)

        # BODY: Body of the model
        # creating initial ni
            # ni = sum(emb(nf) + last_emb(ni*nf))
        ni = calc_ni_in(self.emb_szs, self.n_cont)
        self.nfs = [ni] + nfs # APPENDING

        BODY = [layer(self.nfs[i], self.nfs[i+1], act, **kwargs) for i in range(len(self.nfs)-1)]

        # HEAD: replace this for other task
        # default: classification
        HEAD = nn.Linear(nfs[-1], data.c)

        self.layers = nn.Sequential(*BODY, HEAD)

    def forward(self, x_cat, x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        return x

# MODIFIED: INIT
# w = w / math.sqrt(1/.fan_in)
def selu_normal_(tensor, mode1='fan_in', mode2='fan_out'):
    fan_in = nn.init._calculate_correct_fan(tensor, mode1)
    fan_out = nn.init._calculate_correct_fan(tensor, mode2)
    with torch.no_grad():
        return torch.randn(fan_in, fan_out) / math.sqrt(1./fan_in)

nn.init.selu_normal_ = selu_normal_ # adding modified init

# MODIFIED: Will init weights: w = w / math.sqrt(1./fan_in)
def init_nn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0) # for batchnorm layers
    if isinstance(m, (nn.Linear)): nn.init.selu_normal_(m.weight)
    for l in m.children(): init_nn(l)
        
def create_tabular_model(data, nfs, layer=tabular_layer, act='relu', **kwargs):
    """
    Main function to create a dynamic tabular model

    TO DO: Add Init function. Experiment with deferent initialization methods such as no bias with BN, etc
    """
    model = Tabular(data, nfs, layer, act, **kwargs)
    if act=='selu': init_nn(model)
    return model