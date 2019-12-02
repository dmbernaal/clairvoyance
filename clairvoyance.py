# CLAIRVOYANCE TRAINING & AUTOENCODER
# Author: Diego Medina-Bernal
# Github: https://github.com/dmbernaal

from fastai.tabular import *
from ranger import *
from functools import partial
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn.functional as F

def normalize_data(df, cont_names):
    """
    Will create a new df that is normalized, therefor any adjustments will not effect the original dataframe data. This will be def into our autoencoder class to create an appropriate dataloader object
    """
    df['index'] = df.index

    df_new = df.copy()

    cont_names = cont_names + ['index']

    cols_to_drop = list(df_new.columns.values)
    for name in cont_names:
        cols_to_drop.remove(name)

    df_new.drop(cols_to_drop, axis=1, inplace=True)

     # grabbing columns to normalize
    pre_norm_cols = []
    for col in df_new.columns.values:
        if col == 'index': pass
        else: pre_norm_cols.append(col)

    # grabbing col means and stds
    col_means = []
    col_stds = []
    for i in pre_norm_cols:
        col_means.append(df_new[i].mean())
        col_stds.append(df_new[i].std())

    # normalizing
    for i, col in enumerate(pre_norm_cols):
        df_new[col] = ((df_new[col] - col_means[i]) / col_stds[i])

    return df_new

class TabularEncoderDataset(Dataset):
    """
    This class will return appropriate dataset for our encoder model to learn representations from
    """

    def __init__(self, df):

        self.index = 'index'

        self.X = df
        self.y = self.X[self.index]
        self.X = self.X.drop(self.index, axis=1)

#         self.X = self.X.drop(self.target, axis=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]

def train_sparse_autoencoder(dataloader, model, optimizer, num_epochs=10, sparse_reg=1e-3):
    
    model.optimizer_ = optimizer
    model.sparse_reg_ = sparse_reg

    if __name__=='__main__':

        loss_metric = nn.MSELoss()
        loss_kl = nn.KLDivLoss(reduction='batchmean')
        SPARSE_REG = sparse_reg

        for epoch in range(num_epochs):

            running_loss = 0.0

            for i, (inputs, labels) in enumerate(dataloader):

                inputs = inputs.to(device)

                outputs = model(inputs.float())
                mse_loss = loss_metric(outputs, inputs.float())
                kl_loss = loss_kl(mse_loss, inputs.float())
                loss = mse_loss + kl_loss * SPARSE_REG

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

def train_autoencoder(dataloader, model, optimizer, num_epochs=10):
    
    model.optimizer_ = optimizer

    loss_func = nn.MSELoss()

    for epoch in range(num_epochs):

        running_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):

            inputs = inputs.to(device)

            outputs = model(inputs.float())
            loss = loss_func(outputs, inputs.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

def get_enc(m): return m.encoder

def get_pred(dataloader, model, c=0, c_end=5):
    preds = []
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs.float())
            preds.append(outputs)

        c += 1
        if c >= c_end: break

    return preds

def f2s(p):
    ps = []
    for i in range(len(p)):
        ps.append(p[i].tolist())
    return ps

def return_vec(p):
    ps = []
    for i in range(len(p)):
        ps.append(p[i].squeeze(dim=0))
    return ps

def create_enc_dataset(dataloader, model, dl_full=True, print_=False):
    """
    Once you have a pre-trained autoencoder, this function will take in the model, grab the encoder an form a entirely new dataset based on encoder output.

    This is done by first creating a new dataloader class without shuffle and with batchsize as 1. Therefor every output of the encoder will map into the appropriate index.

    RETURN:
        enc_dataset: <list>
        dataloader: <dataloader>: our raw dataset without shuffle and bs = 1
    """
    # one-shot train: train for one epoch: this will slightly change the weights: will produce new output
    # will keep same parameters
    # creating initial dataloader
    bs = dataloader.batch_size
    dataset = dataloader.dataset
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=1)
    train_sparse_autoencoder_one_cycle(dataloader, model, optimizer=model.optimizer_, max_lr=model.max_lr_, num_epochs=1, sparse_reg=model.sparse_reg_, print_=print_);
    
    # creating new dataloader
    bs = 1
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

    # grab encoder
    enc = get_enc(model)

    # full dataset
    dll = dataloader.dataset.X.shape[0] if dl_full else 10

    # prediction from encoder
    p_en = get_pred(dataloader, enc, c_end=dll) # default 5 output
    p_en = return_vec(p_en)
    enc_dataset = f2s(p_en)

    return enc_dataset, dataloader

def create_augmented_df(enc_ds, dl, original_df, cat_names=False, dep_var=False):
    # combining our original dl with the index: its the same here, but this is for safe measure
    df1 = pd.DataFrame(dl.dataset.X)
    df1_index = pd.DataFrame(dl.dataset.y)
    df_concat = pd.concat([df1, df1_index], axis=1)

    # encoder df
    enc_df = pd.DataFrame(enc_ds)

    # grabbing old column and new column names
    old_col_names = []
    new_col_names = []
    for name in enc_df.columns:
        new_name = f'v{name}'
        new_col_names.append(new_name)
        old_col_names.append(name)

    # mapping old columns to new columsn as dictionary
    # will use this to create new column names
    otn_cols_dict = dict(zip(old_col_names, new_col_names))

    # renaming the encoder df columns
    enc_df.rename(columns=otn_cols_dict, inplace=True)

    # concatting enc_df with our old df
    df_w_enc = pd.concat([df_concat, enc_df], axis=1)
    df_w_enc.drop('index', axis=1, inplace=True)

    # concatting our categorical variables back to our new df
    if cat_names and not dep_var:
        df_cat = original_df[cat_names].copy()
    if cat_names and dep_var:
        cat_names = cat_names + [dep_var]
        df_cat = original_df[cat_names].copy()
    if dep_var and not cat_names: 
        df_cat = original_df[dep_var].copy()

    # forming our new df
    df_final = pd.concat([df_w_enc, df_cat], axis=1)

    return df_final

import torch.nn.functional as F

class Selu(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.tensor(1.6732632423543772848170429916717)
        self.scale = torch.tensor(1.0507009873554804934193349852946)

    def forward(self, x):
        return self.scale * torch.where(x>=0.0, x, self.alpha * torch.exp(x) - self.alpha)

from activations import *

# ORIGINAL
def enc_dec_layer(ni, nf, bn=True, act=False):
    # choosing act if provided
    if act=='selu':
        act_fn = Selu()
        bn = False
    elif act=='relu': act_fn = nn.ReLU(inplace=True)
    elif act=='elu': act_fn = nn.ELU(inplace=True)
    else: act_fn = act_(act)

    layers = [nn.Linear(ni, nf), act_fn]

    if bn: layers.append(nn.BatchNorm1d(nf, eps=1e-5, momentum=0.1))

    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, nfs, c_in, encoder_type, ps=0.1,**kwargs):
        super(AutoEncoder, self).__init__()

        if encoder_type=='over': enc_out = int(c_in * 8)
        elif encoder_type=='under': enc_out = int(c_in // 1.5)

        self.dropout = nn.Dropout(p=ps)

        # ENCODER
        nfs = [c_in] + nfs
        encoder = [enc_dec_layer(nfs[i], nfs[i+1], **kwargs) for i in range(len(nfs)-1)] + [nn.Linear(nfs[-1], enc_out)]
        self.encoder = nn.Sequential(*encoder)

        # DECODER
        # reverse nfs
        nfs = nfs + [enc_out]
        nfs.reverse()
        decoder = [enc_dec_layer(nfs[i], nfs[i+1], **kwargs) for i in range(len(nfs)-1)] + [nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):

        # encoder
        for l in self.encoder:
            x = self.dropout(l(x))

        # decoder
        for l in self.decoder:
            if not isinstance(l, nn.Tanh):
                x = self.dropout(l(x))

        return x

def create_tabular_dataset(df, cont_names):
    df_new = normalize_data(df, cont_names)
    dataset = TabularEncoderDataset(df_new)
    return dataset

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
        
def create_autoencoder(dataloader, nfs, encoder_type='over', cuda=True, **kwargs):
    """
    Creating a dynamic autoencoder
    TO DO: 
        Make more dynamic: Overcomplete, Undercomplete, Dropout, VAE, etc
    """
    c_in = dataloader.dataset.X.shape[1]
    model = AutoEncoder(nfs, c_in, encoder_type, **kwargs)
    init_nn(model) # SELU INIT: Change this if not using selu
    if cuda: model.cuda()
    return model

from onecyclelr import OneCycleLR

def train_sparse_autoencoder_one_cycle(dataloader, model, optimizer, max_lr, num_epochs=10, sparse_reg=1e-3, print_=True):
    
    # setting defaults: hyper
    model.optimizer_ = optimizer
    model.max_lr_ = max_lr
    model.sparse_reg_ = sparse_reg

    # ONE CYCLE PARAMS
    bs = dataloader.batch_size
    n_it = (len(dataloader.dataset.X)//bs)
    div_fact = 25.
    low_lr = max_lr / div_fact

    scheduler = OneCycleLR(optimizer, num_steps=n_it, lr_range=(low_lr, max_lr))

    loss_metric = nn.MSELoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    SPARSE_REG = sparse_reg

    for epoch in range(num_epochs):

        running_loss = 0.0

        for i, (inputs, labels) in enumerate(dataloader):

            inputs = inputs.to(device)

            outputs = model(inputs.float())
            mse_loss = loss_metric(outputs, inputs.float())
            kl_loss = loss_kl(mse_loss, inputs.float())
            loss = mse_loss + kl_loss * SPARSE_REG

            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            optimizer.step()

        if print_: print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

def get_nfs(dataloader):
    """
    Wide -> Narrow -> OverComplete Latent Representation -> Inverted
    """
    n_in = len(dataloader.dataset.X.columns)
    ni = n_in * 166.6
    nf = ni/2
    return int(ni), int(nf)

def augmented_mult(dataloader, model, original_df, mult=5, **kwargs):
    """
    Will create an augmented multiplier df
    """
    dfs = []
    for i in range(mult):
        enc_ds, dl = create_enc_dataset(dataloader, model)
        new_df = create_augmented_df(enc_ds, dl, original_df, **kwargs)
        dfs.append(new_df)
        
    df_final = pd.concat(dfs)
    return df_final