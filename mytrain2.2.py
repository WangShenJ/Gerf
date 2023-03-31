import copy
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
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm, trange
import pandas as pd

from batchLoader import RayDataset
from model import *
import matplotlib.pyplot as plt
import open3d as o3d

device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs',
                        help='config file path')
    parser.add_argument("--expname", type=str, default='my-exp7',
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/my-exp6/',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--batchsize", type=int, default=1,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=1e-3,
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
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=1,
                        help='frequency of weight ckpt saving')

    return parser


def generate_txt(base_path):
    """random suffle train/test set
    """

    f = open(base_path + "/traindata.csv", 'rt')

    datalen = len(f.readlines()) - 1
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


def create_nerf(args):
    """Instantiate NeRF's MLP model

    Parameters
    -----------

    """
    # position, direction, tag encoding 预测的位置，方向，卫星的位置
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
        ckpt = tr.load(ckpt_path, map_location='cuda:0')
        # ckpt = tr.load(ckpt_path)

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

    render_kwargs_train2 = {
        # embedding functions
        'network_fn': model,
        'embed_pts_fn': embed_pts_fn,
        'embed_view_fn': embed_view_fn,
        'embed_tag_fn': embed_tag_fn

    }
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    return render_kwargs_train, render_kwargs_train2, render_kwargs_test, start, grad_vars, optimizer


def get_points(rays_o, rays_d, near, far, n_samples):
    """get sample points along rays

    rays_o [batchsize, chunk, 3]
    Returns
    -----------
    pts: Sample points along rays. [batchsize, chunk, n_samples, 3]
    """
    near, far = near * tr.ones_like(rays_d[..., :1]), far * tr.ones_like(rays_d[..., :1])  # [batch, chunk, 1]
    near, far = near.to(device), far.to(device)
    t_vals = tr.linspace(0, 1., steps=n_samples).to(device)  # [n_samples,]
    t_vals = near * (1. - t_vals) + far * (t_vals).to(device)  # r = o + td, [batch, chunk, n_samples]?
    #
    # sample points
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]  # [batchsize, chunk, n_samples, 3]
    return pts, t_vals


def run_network(pts, views, tags, network_fn, embed_pts_fn, embed_view_fn, embed_tag_fn):
    pts_shapes = list(pts.shape)  # [batchsize,chunk,n_samples, 3]
    pts = tr.reshape(pts, [-1, pts.shape[-1]])  # [batchsize*chunk*n_samples, 3]
    embedded = embed_pts_fn(pts).to(device)  # [batchsize*chunk*n_samples, 3*20]

    views = tr.reshape(views, [-1, views.shape[-1]]).to(device)
    embedded = tr.cat([embedded, embed_view_fn(views)], -1)

    tags = tr.reshape(tags, [-1, tags.shape[-1]])
    embedded = tr.cat([embedded, embed_tag_fn(tags)], -1)

    outputs = network_fn(embedded)
    outputs = tr.reshape(outputs, pts_shapes[:-1] + [outputs.shape[-1]])  # [batchsize,chunk,n_samples,4]
    return outputs


def render_signal(rays_o, azs, els, near, far, N_samples, **kwargs):
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
    # print(az)
    # rays_d = [[math.sin(az) * math.cos(el)],[math.cos(az) * math.cos(el)],[math.sin(el)]]
    rays_d = []
    n = 0
    for az in azs:
        el = els[n]
        rays_d.append((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
        n += 1
    batchsize = rays_o.shape[0]
    # rays_d = tr.repeat_interleave(rays_d, batchsize)
    # rays_d = tags - rays_o
    # rays_dc = rays_d.cpu()
    # rays_norm = tr.tensor(np.linalg.norm(rays_dc, ord=None, axis=1))
    # rays_norm = tr.reshape(rays_norm, (batchsize, -1, 1))
    # rays_d = (rays_d / rays_norm).to(device)
    rays_d = tr.tensor(rays_d)
    rays_d = tr.reshape(rays_d, (batchsize, -1, 3))
    # print(rays_d)
    # rays_d = tr.reshape(rays_d, (batchsize, -1, 3))  # [batchsize, 180*30, 2]
    chunks = 1
    chunks_num = 1
    rays_o_chunks = rays_o.expand(chunks, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]
    rayy = tr.reshape(rays_d, (batchsize, 3))
    tags = rays_o[..., None, :] + rayy[..., None, :] * 0.8
    tags = tr.tensor(tags)
    tags = tr.reshape(tags, (batchsize, 3))
    # print(rays_o)
    # print(tags)
    tags_chunk = tags.expand(chunks, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]
    recv_signal = tr.zeros(batchsize).to(device)
    total = tr.ones(batchsize).to(device)
    nS = 200

    for i in range(chunks_num):
        rays_d_chunks = rays_d[:, i * chunks:(i + 1) * chunks, :]  # [bs, 180*30, 2]
        pts, t_vals = get_points(rays_o_chunks, rays_d_chunks, 0, 0.3, nS)  # [bs, cks, pts, 3]
        views_chunks = rays_d_chunks[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
        tags_chunks = tags_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]

        raw = run_network(pts, views_chunks, tags_chunks, **kwargs)  # [bs, cks, pts, 3]
        recv_signal_chunks, total_signal = raw2outputs_signal(raw, t_vals, rays_d_chunks, nS)  # [bs]

        recv_signal += recv_signal_chunks
        total += total_signal
    return recv_signal, total  # [batchsize,]


def raw2outputs_signal(raw, r_vals, rays_d, nS):
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
    raw2amp = lambda raw, dists: -raw * dists

    # Maybe distance between adjacent samples: np.diff
    dists = r_vals[..., 1:] - r_vals[..., :-1]
    # dists = tr.cat([dists[..., :-1], tr.Tensor([1e10]).expand(dists[..., 1:3].shape).to(device)],
    #                -1)  # [batchsize, chunks, n_samples]
    dists = tr.cat([dists, dists[..., 2:3].to(device)], -1)
    # dists = tr.cat([dists, dists[..., 1:2].to(device)], -1)
    # dists = dists * tr.norm(rays_d[..., None, :], dim=-1).to(device)  # [batchsize,chunks, n_samples, 3].
    # dists = dists / 100

    att_a, s_a = raw[..., 0], raw[..., 1]  # [batchsize,chunks, N_samples]
    att_a, s_a = F.relu(att_a), tr.sigmoid(s_a)
    # att_a, s_a = tr.sigmoid(att_a), tr.sigmoid(s_a)
    # att_a, s_a = tr.sigmoid(att_a), tr.sigmoid(s_a)
    # att_a = F.relu(att_a)
    # att_a = tr.flip(att_a, dims=[2])

    att = att_a.cpu()
    att = att.detach().numpy()
    att[att > 0.8] = 1
    att[att < 0.5] = 0
    att = tr.tensor(att)
    # s_a = tr.flip(s_a, dims=[2])
    amp = raw2amp(att_a, 1 / nS)  # [batchsize,chunks, N_samples]
    amp_i = tr.exp(tr.cumsum(amp, -1))  # [batchsize,chunks, N_samples]
    # att_a = 1-tr.exp(-att_a)

    amp_i_cpu = (1 - amp_i).cpu().detach().numpy()
    att_cpu = (1 - att).cpu().detach().numpy()
    att_min = np.minimum(amp_i_cpu, att_cpu)
    att_min = tr.tensor(att_min)

    recv_signal = tr.sum(amp_i*(1-tr.exp(-att_a*(1/nS))*s_a), -1)  # integral along line [batchsize,chunks]
    # print(amp_i*(1-tr.exp(-att_a)))
    # print(s_a)
    recv_signal = tr.sum(recv_signal, -1)  # integral along direction [batchsize,]
    # print(recv_signal)
    total_signal = tr.sum(s_a, -1)
    total_signal = tr.sum(total_signal, -1)
    # recv_signal = tr.prod((tr.exp(amp)), -1)
    return recv_signal, total_signal


def getAllAngle():
    r_d = []
    r_dp = []
    beta_r = [0, 90]
    beta_len = int(beta_r[1] - beta_r[0])
    frac = 10
    radius = 1
    alphas = np.linspace(0, 360 - 1, 360 // frac) / 180 * np.pi
    betas = np.linspace(beta_r[0], beta_r[1] - 1, beta_len // frac) / 180 * np.pi
    angleLen = len(alphas) * len(betas)
    for i in range(len(alphas)):
        for j in range(len(betas)):
            r_d.append([alphas[i], betas[j]])
    for k in range(angleLen):
        x = radius * np.cos(r_d[k][0]) * np.cos(r_d[k][1])  # (90*360)
        y = radius * np.sin(r_d[k][0]) * np.cos(r_d[k][1])
        z = radius * np.sin(r_d[k][1])
        r_dp.append([x, y, z])
    return tr.tensor(r_dp)


def getBoxArray(**kwargs):
    pts, alphac = getSPDensity(**kwargs)
    alphac = tr.sigmoid(alphac)
    ptsc = pts.cpu()
    ptsc = np.array(ptsc)
    n = 0
    pNum = 0
    fPosition = open("data/pointCloud/position.txt", "w+", encoding='utf-8')
    for pt in ptsc:
        a = alphac[n]
        # print(a)
        if a > 0.95:
            # print(pt[0])
            pNum += 1
            s1 = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
            # print(s1)
            fPosition.write(s1)
        n = n + 1
    fPosition.close()
    positionC = o3d.io.read_point_cloud("data/pointCloud/position.txt", format='xyz')
    positionC = positionC.remove_radius_outlier(int(pNum / 4), 0.5)[0]  # 统计方法剔除
    obb = positionC.get_axis_aligned_bounding_box()
    box_points = np.array(obb.get_box_points())
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25,
                                                          resolution=100)
    positionC = copy.deepcopy(positionC).translate((0, 0, -0.25))
    obb.color = (1, 0, 0)
    obb = copy.deepcopy(obb).translate((0, 0, -0.25))
    o3d.visualization.draw_geometries([positionC, obb, mesh_sphere], width=600, height=600)
    return box_points


def getSPDensity(**kwargs):
    xNum = 40
    yNum = 40
    zNum = 20
    x_nSamples = np.linspace(-1, 1, xNum)
    y_nSamples = np.linspace(-1, 1, yNum)
    z_nSamples = np.linspace(0.1, 1, zNum)
    SP = []
    for x in x_nSamples:
        for y in y_nSamples:
            for z in z_nSamples:
                SP.append([x, y, z])
    SP = tr.tensor(SP)
    tr.reshape(SP, (-1, 3))
    views_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    tags_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    raw = run_network(SP, views_chunks, tags_chunks, **kwargs)  # [bs, cks, pts, 3]
    alphac = raw[..., 0]
    s = raw[..., 1]
    # print(alphac.shape)
    return SP, alphac


def render_signal2(rays_o, rays_d, azs, els, near, far, N_samples, **kwargs):
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
    batchsize = rays_o.shape[0]
    rays_d = tr.repeat_interleave(rays_d, batchsize)
    rays_d = tr.reshape(rays_d, (batchsize, -1, 3))
    # print(rays_d)
    # rays_d = tr.reshape(rays_d, (batchsize, -1, 3))  # [batchsize, 180*30, 2]
    chunks = 36 * 9
    chunks_num = 36 * 9 // chunks
    rays_o_chunks = rays_o.expand(chunks, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]

    ray_d = []
    n = 0
    for az in azs:
        el = els[n]
        ray_d.append((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
        n += 1
    # batchsize = rays_o.shape[0]
    ray_d = tr.tensor(ray_d)
    ray_d = tr.reshape(ray_d, (batchsize, -1, 3))
    # rays_o_chunk = rays_o.expand(1, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]
    rayy = tr.reshape(ray_d, (batchsize, 3))
    tags = rays_o[..., None, :] + rayy[..., None, :] * 3
    tags = tr.tensor(tags)
    tags = tr.reshape(tags, (batchsize, 3))

    tags_chunk = tags.expand(chunks, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]
    recv_signal = tr.zeros(batchsize).to(device)
    nS = 201
    for i in range(chunks_num):
        rays_d_chunks = rays_d[:, i * chunks:(i + 1) * chunks, :]  # [bs, 180*30, 2]
        # print(rays_d_chunks.shape)
        pts, t_vals = get_points(rays_o_chunks, rays_d_chunks, 0, 0.3, nS)  # [bs, cks, pts, 3]
        views_chunks = rays_d_chunks[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
        tags_chunks = tags_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]

        raw = run_network(pts, views_chunks, tags_chunks, **kwargs)  # [bs, cks, pts, 3]
        recv_signal_chunks, xx = raw2outputs_signal(raw, t_vals, rays_d_chunks)  # [bs]

        recv_signal += recv_signal_chunks
    return recv_signal  # [batchsize,]


def train():
    # 调整坐标 地固坐标系，所在地面不是xoy面，360度角包含地面，干扰
    # 光线传播模型
    # 读取csv 格式文件
    dt = pd.read_csv("data/my-exp6/dataTrain.csv")
    dtV = pd.read_csv('data/my-exp6/dataEval.csv')
    dt.head()
    # print(dt.head())   #打印标记文件头
    data = dt.values
    dataV = dtV.values
    parser = config_parser()
    args = parser.parse_args()

    basedir = args.basedir  # store models and log
    expname = args.expname  # experiment name

    # create nerf model
    render_kwargs_train, render_kwargs_train2, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    batchsize = 64
    epoches = 120
    tr.set_default_tensor_type('torch.cuda.FloatTensor')

    ## train
    # e.g., ceil(10 / 4) = 3
    iter_per_epoch = int(math.ceil(len(data) // batchsize))
    Viter_per_epoch = int(math.ceil(len(dataV) // batchsize))
    global_step = start
    epoch_start = int(math.ceil(start / iter_per_epoch))
    print("Start training. Current Global step:%d. Epoch:%d" % (global_step, epoch_start))
    r_dd = getAllAngle()

    # scheduler = StepLR(optimizer, step_size = 20, gamma = 0.1)
    x = []
    train_loss_list = []
    val_loss_list = []
    plt.figure(figsize=(6, 4), dpi=100)
    plt.subplot(2, 1, 1)
    plt.ion()

    for epoch in range(epoch_start, epoches + 1):
        loss_total = 0
        loss_totalV = 0
        tloss_total = 0
        t1 = time.time()
        count = 0
        countV = 0
        plt.cla()
        a = 1
        for i in trange(iter_per_epoch):
            # 取空间中的采样点

            count = count + batchsize
            batchdata = tr.FloatTensor(data[count - batchsize:count].astype(float))
            batchdata = batchdata.to(device)
            if len(batchdata) != batchsize:
                continue
            cn0 = batchdata[..., -1]
            sPosition = batchdata[..., 0:3]
            # pPosition = batchdata[..., 3:6]
            az = batchdata[..., 3]
            el = batchdata[..., 4]
            # print(az)
            # predict_ss, predict_ts = render_signal(sPosition, az,el, **render_kwargs_train)
            predict_ss, to = render_signal(sPosition, az, el, **render_kwargs_train)

            optimizer.zero_grad()
            # print("predict: ", predict_ss, "GT: ", cn0)
            loss = a * sig2mse(predict_ss, cn0) + (1 - a) * sig2mse(to, 1 - cn0)
            loss = loss.requires_grad_()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            tloss_total += sig2mse(to, 1 - cn0)
            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ###############################

            global_step += 1
        with tr.no_grad():
            for i in trange(Viter_per_epoch):
                countV = countV + batchsize
                batchdataV = tr.FloatTensor(dataV[countV - batchsize:countV].astype(float))
                batchdataV = batchdataV.to(device)
                if len(batchdataV) != batchsize:
                    continue
                cn0 = batchdataV[..., -1]
                sPosition = batchdataV[..., 0:3]
                # pPosition = batchdata[..., 3:6]
                az = batchdataV[..., 3]
                el = batchdataV[..., 4]
                # print(az)
                predict_ss, to = render_signal(sPosition, az, el, **render_kwargs_train)
                loss = a * sig2mse(predict_ss, cn0) + (1 - a) * sig2mse(to, 1 - cn0)
                loss_totalV += loss.item()

        t2 = time.time()

        # scheduler.step()

        print("each epoch time comsuming:", t2 - t1)
        loss_mean = float(loss_total) / iter_per_epoch
        tloss_mean = float(tloss_total) / iter_per_epoch
        lossV = float(loss_totalV) / Viter_per_epoch
        tqdm.write("Epoch:%d, Loss:%f, Vloss;%f, T;%f" % (epoch, loss_mean, lossV,tloss_mean))

        x.append(epoch)
        train_loss_list.append(loss_mean)
        val_loss_list.append(lossV)
        # try:
        #     train_loss_lines.remove(train_loss_lines[0])
        #     val_loss_lines.remove(val_loss_lines[0])
        # except Exception:
        #     pass
        train_loss_lines = plt.plot(x, train_loss_list, 'r', lw=1)
        val_loss_lines = plt.plot(x, val_loss_list, 'b', lw=1)
        plt.title("loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["train_loss", "val_loss"])
        plt.pause(0.1)

        # get bouding box
        # bounding_boxArray = getBoxArray()

        if epoch % 2 == 0:
            path = os.path.join(basedir, expname, "ckpts", '{:06d}.tar'.format(epoch))
            tr.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    # generate_txt('data/my-exp')
    train()
    # parser = config_parser()
    # args = parser.parse_args()
    # render_kwargs_train, render_kwargs_train2,render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    # points = getBoxArray(**render_kwargs_train2)
    # print(points)
