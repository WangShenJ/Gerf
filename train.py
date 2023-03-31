# -*- coding: utf-8 -*-
"""
"""
import json
import math
import os
import random
import time

import configargparse
import matplotlib.image as plm
import numpy as np
import torch as tr
import torch.nn as nn
from tqdm import tqdm, trange

from batchLoader import RayDataset
from model import *
from utils.dataset import gen_dataset, update_gateway_postion

device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")

def generate_txt(base_path):
    """random suffle train/test set
    """

    data = np.load(base_path+"/gateway1.npy")
    datalen = data.shape[0]
    IDList = [x for x in range(datalen)]
    random.shuffle(IDList)
    train_number = int(len(IDList) * 0.2)
    train_list = IDList[:train_number]
    test_list = IDList[train_number:]
    train_list = np.array(train_list)
    test_list = np.array(test_list)

    train_txt = base_path + 'train_signal.txt'
    test_txt = base_path + 'test_signal.txt'
    np.savetxt(train_txt, train_list, fmt='%s')
    np.savetxt(test_txt, test_list, fmt='%s')


def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs',
                        help='config file path')
    parser.add_argument("--expname", type=str, default='s23-exp3',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/s23-exp3/',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--batchsize", type=int, default=1,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-3,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')

    # rendering options
    parser.add_argument("--dis", type=float, default=5.,
                        help='distance of the gateway relative to origin')
    parser.add_argument("--near", type=float, default=0.,
                        help='near boundary of the render area')
    parser.add_argument("--far", type=float, default=5.,
                        help='far boundary of the render area')
    parser.add_argument("--N_samples", type=int, default=32,
                        help='number of coarse samples per ray')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location.txt)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')

    # maybe add regulazation
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')

    ## blender flags
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=1,
                        help='frequency of weight ckpt saving')

    return parser


def run_network(pts, views, tags, network_fn, embed_pts_fn, embed_view_fn, embed_tag_fn):
    pts_shapes = list(pts.shape)    # [batchsize,chunk,n_samples, 3]
    pts = tr.reshape(pts, [-1, pts.shape[-1]])    # [batchsize*chunk*n_samples, 3]
    embedded = embed_pts_fn(pts)    # [batchsize*chunk*n_samples, 3*20]

    views = tr.reshape(views, [-1, views.shape[-1]])
    embedded = tr.cat([embedded, embed_view_fn(views)], -1)

    tags = tr.reshape(tags, [-1, tags.shape[-1]])
    embedded = tr.cat([embedded, embed_tag_fn(tags)], -1)

    outputs = network_fn(embedded)
    outputs = tr.reshape(outputs, pts_shapes[:-1] + [outputs.shape[-1]])  # [batchsize,chunk,n_samples,4]
    return outputs


def raw2outputs_signal(raw, r_vals, rays_d):
    """Transforms model's predictions to semantically meaningful values.

    Parameters
    ----------
    raw : [batchsize, chunks,n_samples,  4]. Prediction from model.
    r_vals : [batchsize, chunks, n_samples]. Integration distance.
    rays_d : [batchsize,chunks, 3]. Direction of each ray

    Return:
    ----------
    receive_signal : [batchsize]. abs(singal of each ray)
    """
    # raw2alpha = lambda raw, dists: 1.-tr.exp(-raw*dists)
    # raw2phase = lambda raw, dists: tr.exp(1j*raw*dists)
    raw2phase = lambda raw, dists: raw*dists
    raw2amp = lambda raw, dists: -raw*dists

    # Maybe distance between adjacent samples: np.diff
    dists = r_vals[...,1:] - r_vals[...,:-1]
    dists = tr.cat([dists, tr.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples]
    dists = dists * tr.norm(rays_d[...,None,:], dim=-1)  # [batchsize,chunks, n_samples, 3].

    att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [batchsize,chunks, N_samples]
    att_p, s_p = tr.sigmoid(att_p)*np.pi*2, tr.sigmoid(s_p)*np.pi*2
    # att_a, s_a = F.relu(att_a), F.relu(s_a)
    att_a, s_a = tr.sigmoid(att_a), tr.sigmoid(s_a)

    amp = raw2amp(att_a, dists)  # [batchsize,chunks, N_samples]
    phase = raw2phase(att_p, dists)

    # att_i = tr.cumprod(tr.cat([tr.ones((al_shape[:-1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    # phase_i = tr.cumprod(tr.cat([tr.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
    amp_i = tr.exp(tr.cumsum(amp, -1))            # [batchsize,chunks, N_samples]
    phase_i = tr.exp(1j*tr.cumsum(phase, -1))                # [batchsize,chunks, N_samples]

    recv_signal = tr.sum(s_a*tr.exp(1j*s_p)*amp_i*phase_i, -1)  # integral along line [batchsize,chunks]
    recv_signal = tr.sum(recv_signal, -1)   # integral along direction [batchsize,]

    return recv_signal


def get_points(rays_o, rays_d, near, far, n_samples):
    """get sample points along rays

    rays_o [batchsize, chunk, 3]
    Returns
    -----------
    pts: Sample points along rays. [batchsize, chunk, n_samples, 3]
    """
    near, far = near*tr.ones_like(rays_d[...,:1]), far*tr.ones_like(rays_d[...,:1])  # [batch, chunk, 1]
    near, far = near.to(device), far.to(device)
    t_vals = tr.linspace(0.1, 1., steps=n_samples)   # [n_samples,]
    t_vals = near * (1.-t_vals) + far * (t_vals)    # r = o + td, [batch, chunk, n_samples]?
    # r_vals = r_vals.expand([batchsize, n_samples])    # (batchsize, n_samples)

    # sample points
    pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # [batchsize, chunk, n_samples, 3]
    return pts, t_vals




def render_signal(tags, rays_o, rays_d, near, far, N_samples, **kwargs):
    """_summary_

    Parameters
    ----------
    tags: [batchsize, 3]. The position of tags
    rays_o : [batchsize, 3]. The origin of rays
    rays_d : [batchsize, 3*360*60]. The direction of rays
    near : float. The near bound of the rays
    far : float. The far bound of the rays
    n_samples: int. num of samples per ray
    """
    batchsize = tags.shape[0]
    rays_d = tr.reshape(rays_d, (batchsize, -1, 3))    # [batchsize, 360*60, 3]
    chunks = 180*30
    chunks_num = 180*30 // chunks
    rays_o_chunks = rays_o.expand(chunks, -1, -1).permute(1,0,2) #[bs, cks, 3]
    tags_chunk = tags.expand(chunks, -1, -1).permute(1,0,2)      #[bs, cks, 3]
    recv_signal = tr.zeros(batchsize)+ 1j*tr.zeros(batchsize)
    for i in range(chunks_num):
        rays_d_chunks = rays_d[:,i*chunks:(i+1)*chunks, :]  # [bs, cks, 3]
        pts, t_vals = get_points(rays_o_chunks, rays_d_chunks, near, far, N_samples) # [bs, cks, pts, 3]

        views_chunks = rays_d_chunks[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
        tags_chunks = tags_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]

        raw = run_network(pts, views_chunks, tags_chunks, **kwargs)  # [bs, cks, pts, 3]
        recv_signal_chunks = raw2outputs_signal(raw, t_vals, rays_d_chunks)  # [bs]
        recv_signal += recv_signal_chunks
    return recv_signal    # [batchsize,]



def create_nerf(args):
    """Instantiate NeRF's MLP model

    Parameters
    -----------

    """
    # position, direction, tag encoding
    embed_pts_fn, input_pts_ch = get_embedder(args.multires, args.i_embed)
    embed_view_fn, input_view_ch = get_embedder(args.multires_views, args.i_embed)
    embed_tag_fn, input_tag_ch = get_embedder(args.multires, args.i_embed)

    skips = [4]
    model = RFNeRF(D=args.netdepth, W=args.netwidth, input_pts_ch=input_pts_ch,
                   input_tag_ch=input_tag_ch, input_view_ch=input_view_ch,
                   skips=skips).to(device)
    # model = nn.DataParallel(model, device_ids=[0,1])
    grad_vars = list(model.parameters())  # ? why

    # Create optimizer
    optimizer = tr.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 1
    basedir = args.basedir
    expname = args.expname
    ckptsdir = os.path.join(basedir, expname, "ckpts")

    #### Load checkpoints
    ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reload from', ckpt_path)
        ckpt = tr.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['network_fn_state_dict'])

    # N_samples: number of coarse samples per ray
    render_kwargs_train = {
        'near': args.near,
        'far': args.far,
        'N_samples': args.N_samples,
        'network_fn': model,

        # embedding functions
        'embed_pts_fn': embed_pts_fn,
        'embed_view_fn': embed_view_fn,
        'embed_tag_fn': embed_tag_fn
    }
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def dataset_settings(args) -> None:
    """_summary_
    """
    basedir = args.basedir        # store models and log
    expname = args.expname        # experiment name
    datadir = args.datadir
    cfgname = os.path.join(args.config, expname)
    cfgname = cfgname+".json"
    dis = args.dis
    batchsize = args.batchsize

    os.makedirs(os.path.join(basedir, expname, "ckpts"), exist_ok=True)
    debugdir = os.path.join(basedir, expname, "debug")
    os.makedirs(debugdir, exist_ok=True)

    with open(cfgname, 'r') as f:
        configs = json.load(f)
        f.close()

    # generate dataset
    if not os.path.exists(datadir):
        dataname = configs["scenename"]
        gen_dataset(datadir, dataname)

    # random generate train/test set index
    files = os.listdir(datadir)
    if 'train_signal.txt' not in files:
        print('creating train.txt and test.txt')
        generate_txt(datadir)

    # generate gateway position
    if configs["gateway1"] == None:
        update_gateway_postion(expname, "gateway1", dis, debugdir)

    print("load datas")
    data = RayDataset(datadir, configs, batchsize, debugdir,output="signal")
    print("total data:%d" % len(data))

    return data



def train():

    parser = config_parser()
    args = parser.parse_args()

    basedir = args.basedir        # store models and log
    expname = args.expname        # experiment name
    data = dataset_settings(args)

    # save args
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    batchsize = args.batchsize
    epoches = 1200

    tr.set_default_tensor_type('torch.cuda.FloatTensor')

    ## train
    # e.g., ceil(10 / 4) = 3
    iter_per_epoch = int(math.ceil(len(data) / batchsize))
    global_step = start
    epoch_start = int(math.ceil(start / iter_per_epoch))
    print("Start training. Current Global step:%d. Epoch:%d"%(global_step, epoch_start))
    print(len(data))
    for epoch in range(1):
        loss_total = 0
        t1 = time.time()
        for i in trange(1):

            batchdata = tr.FloatTensor(data.Next().float())
            batchdata = batchdata.to(device)
            # print(batchdata)

    #
            amp_gt, phs_gt = batchdata[...,0], batchdata[...,1]

            amp_gt = amp_gt * 10  # have a try
            tags = batchdata[...,2:5]
            r_o, r_d = batchdata[...,5:8], batchdata[...,8:]
            print(r_o, r_d.shape)
    #         sig_gt = amp_gt*tr.exp(1j*phs_gt)
    #
    #         predict_ss = render_signal(tags, r_o, r_d, **render_kwargs_train)
    #         optimizer.zero_grad()
    #         loss = sig2mse(predict_ss, sig_gt)
    #         loss.backward()
    #         optimizer.step()
    #         loss_total += loss.item()
    #
    #         # NOTE: IMPORTANT!
    #         ###   update learning rate   ###
    #         decay_rate = 0.1
    #         decay_steps = args.lrate_decay * 1000
    #         new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = new_lrate
    #         ################################
    #
    #         global_step += 1
    #
    #     t2 = time.time()
    #     print("each epoch time comsuming:",t2-t1)
    #     loss_mean = float(loss_total) / iter_per_epoch
    #     tqdm.write("Epoch:%d, Loss:%f"%(epoch, loss_mean))
    #
    #     if epoch%args.i_weights==0:
    #         path = os.path.join(basedir, expname,"ckpts", '{:06d}.tar'.format(epoch))
    #         tr.save({
    #             'global_step': global_step,
    #             'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }, path)
    #         print('Saved checkpoints at', path)



if __name__ == '__main__':

    # tr.set_default_tensor_type('torch.cuda.FloatTensor')
    train()

