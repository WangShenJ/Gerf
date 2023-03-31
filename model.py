# -*- coding: utf-8 -*-
"""RfNerf model
"""
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Misc
img2mse = lambda x, y: tr.mean((x - y) ** 2)
img2me = lambda x, y: tr.mean(abs(x - y))
sig2mse = lambda x, y: tr.mean(abs(x - y))


class Embedder():
    """positional encoding
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']  # input dimension of gamma
        out_dim = 0

        # why include input?
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']  # L-1, 10-1 by default
        N_freqs = self.kwargs['num_freqs']  # L

        # why not log sampling?
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** tr.linspace(0., max_freq, steps=N_freqs)  # 2^[0,1,...,L-1]
        else:
            freq_bands = tr.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        """return: gamma(input)
        """
        return tr.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """_summary_

    Parameters
    ----------
    multires : log2 of max freq for positional encoding, i.e., (L-1)
    i : set 0 for default positional encoding, -1 for none


    Returns
    -------
        embedding function; output_dims
    """
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [tr.sin, tr.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class RFNeRF(nn.Module):
    """
    """

    def __init__(self, D=8, W=256, input_pts_ch=3, input_view_ch=3, input_tag_ch=3,
                 output_ch=4, skips=[4]):
        """_summary_

        Parameters
        ----------
        D : Depth
        W : Dimension per layer
        input_pos_ch : postion in the space
        input_view_ch : view direction
        input_tag_ch : position of tag
        output_ch : int, optional
            _description_, by default 4
        skip : list, optional
            _description_, by default [4]
        """
        super().__init__()
        self.D, self.W = D, W
        self.input_pts_ch = input_pts_ch
        self.input_view_ch = input_view_ch
        self.input_tag_ch = input_tag_ch
        self.skips = skips

        self.alpha_linears = nn.ModuleList(
            [nn.Linear(input_pts_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_pts_ch, W)
             for i in range(D - 1)]
        )
        self.alpha_bn = nn.ModuleList([nn.BatchNorm1d(W) for i in range(D)])

        self.signal_linears = nn.ModuleList(
            [nn.Linear(input_view_ch + input_tag_ch + W, W)] +
            [nn.Linear(W, W // 2)]
        )
        self.signal_bn = nn.ModuleList([nn.BatchNorm1d(W), nn.BatchNorm1d(W // 2)])

        self.alpha_output = nn.Linear(W, 1)
        self.feature_layer = nn.Linear(W, W)
        self.signal_output = nn.Linear(W // 2, 1)  # amplitude, phase
        self.bn_layer1 = nn.BatchNorm1d(1)
        self.bn_layer2 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = x.to(tr.float32)
        """forward function of the model

        Parameters
        ----------
        x : [batchsize ,input_pts+input_views+input_tags], (after position encoding)

        Returns
        ----------
        outputs: [batchsize, 4].   alpha_amp, alpha_phase, signal_amp, signal_phase
        """

        """
        ws
        parameter
        ----------
        x : [batchsize, input_pts + input_views + input_tags]
        
        returns
        ----------
        outputs: [batchsize, 2]  alpha_c, signal_c,
        """
        input_pts, input_views, input_tags = tr.split(x, [self.input_pts_ch,
                                                          self.input_view_ch,
                                                          self.input_tag_ch], dim=-1)
        h = input_pts
        for i, l in enumerate(self.alpha_linears):
            h = self.alpha_linears[i](h)
            h = self.alpha_bn[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = tr.cat([input_pts, h], -1)

        alpha = self.alpha_output(h)  # (batch_size, 1)
        feature = self.feature_layer(h)
        alpha = self.bn_layer1(alpha)
        # alpha = F.relu(alpha)

        h = tr.cat([feature, input_views, input_tags], -1)

        for i, l in enumerate(self.signal_linears):
            h = self.signal_linears[i](h)
            h = self.signal_bn[i](h)
            h = F.relu(h)

        signal = self.signal_output(h)  # (batch_size, 1)
        signal = self.bn_layer2(signal)
        # signal = F.relu(signal)
        outputs = tr.cat([alpha, signal], -1)  # (batch_size, 1)

        return outputs
