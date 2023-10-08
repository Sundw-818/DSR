#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
import numpy as np
from lib.utils import *

# lower value = smoother shapes
# higher value = sharper shapes (can cause breakup of connected components)
# default: 30
OMEGA = 30

class SirenActivation(nn.Module):
    def __init__(
        self
        #omega
    ):
        super().__init__()
        #self.omega = omega
        
        #print("omega =",omega)

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30 (omega)
        #print("omega2 =",self.omega)
        return torch.sin(OMEGA * input)

    @staticmethod
    def sine_init(m):
        #print("omega =",self.omega)
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See supplement Sec. 1.5 for discussion of factor 30 (omega)
                m.weight.uniform_(-np.sqrt(6 / num_input) / OMEGA, np.sqrt(6 / num_input) / OMEGA)

    @staticmethod
    def first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30 (omega)
                m.weight.uniform_(-1 / num_input, 1 / num_input)


class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh = False,
        latent_dropout=False,
        positional_encoding = False,
        fourier_degree = 1
        #omega = 30
    ):
        super(DeepSDF, self).__init__()

        def make_sequence():
            return []
        if positional_encoding is True:
            dims = [latent_size + 2*fourier_degree*3] + dims + [1]  # currently not used
        else:
            #dims = [latent_size + 3] + dims + [1]
            dims = [latent_size + 4] + dims + [1]  # (x, y, z) -> (x, y, z, t)

        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all # currently not used
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    #out_dim -= 3
                    out_dim -= 4

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.sine = SirenActivation()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # self.th = nn.Tanh()

        # Initialise appropriately for the SIREN activation functions
        self.apply(self.sine.sine_init)
        getattr(self, "lin0").apply(self.sine.first_layer_sine_init)

    # input: N x (L+3)
    #def forward(self, latent, xyz):
    def forward(self, latent, xyzt):

        if self.positional_encoding:
            #xyz = fourier_transform(xyz, self.fourier_degree)
            xyzt = fourier_transform(xyzt, self.fourier_degree)
        #input = torch.cat([latent, xyz.cuda()], dim=1)
        input = torch.cat([latent, xyzt.cuda()], dim=1)

        #if input.shape[1] > 3 and self.latent_dropout:
        if input.shape[1] > 4 and self.latent_dropout:
            #latent_vecs = input[:, :-3]
            latent_vecs = input[:, :-4]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            #x = torch.cat([latent_vecs, xyz], 1)
            x = torch.cat([latent_vecs, xyzt], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            # layers getting a latent code input
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyzt], 1)
            x = lin(x)
            # last layer Tanh (if use_tanh = True)
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            # hidden layers
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                #x = self.relu(x)
                x = self.sine(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

