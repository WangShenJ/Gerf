from mytrain import *
import torch
import torchvision.models as models
from matplotlib import pyplot as plt
import open3d as o3d


def render_signal_t(tags, rays_o, near, far, N_samples, **kwargs):
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
    # rays_d = tr.repeat_interleave(rays_d, batchsize)
    rays_d = tags-rays_o
    rays_dc = rays_d.cpu()
    rays_norm = tr.tensor(np.linalg.norm(rays_dc, ord=None, axis=1))
    rays_norm = tr.reshape(rays_norm, (batchsize, -1, 1))
    rays_d = (rays_dc / rays_norm).to(device)
    rays_d = tr.reshape(rays_d, (batchsize, -1, 3))
    # rays_d = tr.reshape(rays_d, (batchsize, -1, 3))  # [batchsize, 180*30, 2]
    chunks = 1
    chunks_num = 1
    rays_o_chunks = rays_o.expand(chunks, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]
    tags_chunk = tags.expand(chunks, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]
    recv_signal = tr.zeros(batchsize).to(device)
    for i in range(chunks_num):
        rays_d_chunks = rays_d[:, i * chunks:(i + 1) * chunks, :]  # [bs, 180*30, 2]
        pts, t_vals = get_points(rays_o_chunks, rays_d_chunks, 0, 0.0001, 5001)  # [bs, cks, pts, 3]

        views_chunks = rays_d_chunks[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
        tags_chunks = tags_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]

        raw = run_network_t(pts, views_chunks, tags_chunks, **kwargs)  # [bs, cks, pts, 3]
        alphac = raw[..., 0]
        s = raw[..., 1]

    return pts, alphac, s  # [batchsize,]


def run_network_t(pts, views, tags, network_fn, embed_pts_fn, embed_view_fn, embed_tag_fn):
    pts_shapes = list(pts.shape)  # [batchsize,chunk,n_samples, 3]
    pts = tr.reshape(pts, [-1, pts.shape[-1]])  # [batchsize*chunk*n_samples, 3]
    embedded = embed_pts_fn(pts)  # [batchsize*chunk*n_samples, 3*20]

    views = tr.reshape(views, [-1, views.shape[-1]]).to(device)
    embedded = tr.cat([embedded, embed_view_fn(views)], -1)

    tags = tr.reshape(tags, [-1, tags.shape[-1]])
    embedded = tr.cat([embedded, embed_tag_fn(tags)], -1)

    outputs = network_fn(embedded)

    outputs = tr.reshape(outputs, pts_shapes[:-1] + [outputs.shape[-1]])  # [batchsize,chunk,n_samples,4]

    return outputs


def getAllAngle_t():
    r_d = []
    r_dp = []
    beta_r = [0, 90]
    beta_len = int(beta_r[1] - beta_r[0])
    frac = 10
    radius = 1
    alphas = np.linspace(0, 360 - 1, 360 // frac) / 180 * np.pi
    betas = np.linspace(beta_r[0], beta_r[1] - 1, beta_len // frac) / 180 * np.pi
    print(betas)
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


def countOcc(xc, yc, zc):
    xn = []
    yn = []
    zn = []
    n = -40
    for index in xc:
        if index > 6:
            xn.append(n)
        n = n + 1
    n = -40
    for index in yc:
        if index > 6:
            yn.append(n)
        n = n + 1
    n = -40
    for index in zc:
        if index > 7:
            zn.append(n)
        n = n + 1
    return xn, yn, zn


if __name__ == '__main__':
    fPone = open("../data/pointCloud/pp.txt", "w+", encoding='utf-8')
    fPosition = open("../data/pointCloud/position.txt", "w+", encoding='utf-8')
    dt = pd.read_csv("../data/my-exp2/myShuffleEvalData.csv")

    data = dt.values
    parser = config_parser()
    args = parser.parse_args()
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    r_dd = getAllAngle_t()
    # load data

    xcount = [0] * 80
    ycount = [0] * 80
    zcount = [0] * 80
    s1 = ''
    for i in range(1000):
        batchdata = tr.FloatTensor(data[i:i + 1].astype(float))
        batchdata = batchdata.to(device)
        cn0 = batchdata[..., -1]
        sPosition = batchdata[..., 0:3]
        pPosition = batchdata[..., 3:6]
        pts, alphac, ass = render_signal_t(sPosition, pPosition, **render_kwargs_train)
        # raw = run_network_t([0,0,300], [], [0,0,500], **render_kwargs_train)
        # print(raw)
        alphac = torch.reshape(alphac, (-1, 1))

        ass = torch.reshape(ass, (-1, 1))
        pts = torch.reshape(pts, (-1, 3))

        ptsc = pts.cpu()
        ptsc = np.array(ptsc)
        alphacc = alphac.cpu()
        alphacc = F.relu(alphacc)
        ass = ass.cpu()
        ass = tr.sigmoid(ass)
        sPosition = sPosition.cpu()
        sPosition = sPosition.cpu()
        # print(pPosition)

        s = str(float(pPosition[..., 0])) + " " + str(float(pPosition[..., 1])) + " " + str(
            float(pPosition[..., 2])) + "\n"
        fPone.write(s)
        s = str(float(sPosition[..., 0])) + " " + str(float(sPosition[..., 1])) + " " + str(
            float(sPosition[..., 2])) + "\n"
        fPone.write(s)
        n = 0
        colorList = []

        for pt in ptsc:
            a = alphacc[n]
            ac = ass[n]
            print(a)

            if a >=0.8:
                s1 = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
            # if (400 + 800) / 2800 <= pt[2] <= (800 + 800) / 2800 and (0 + 800) / 2800 < pt[0] < (400 + 800) / 2800 and (
            #         0 + 800) / 2800 < pt[1] < (400 + 800) / 2800:
            #
            #     s = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
                # x = 350+50*random.random()
                # y = 500+50*random.random()
                # z = 400+50*random.random()
                # s1 = str(x) + " " + str(y) + " " + str(z) + "\n"
            # xcount[int(pt[0]//10)] = xcount[int(pt[0]//10)]+1
            # ycount[int(pt[1]//10)] = ycount[int(pt[1]//10)]+1
            # zcount[int(pt[2]//10)] = zcount[int(pt[2]//10)]+1
            # fPone.write(s)
            fPosition.write(s1)

            n = n + 1
    # xn,yn,zn = countOcc(xcount,ycount,zcount)
    # print(xn,yn,zn)

    fPone.close()
    fPosition.close()
    phoneC = o3d.io.read_point_cloud("data/pointCloud/pp.txt", format='xyz')
    positionC = o3d.io.read_point_cloud("data/pointCloud/position.txt", format='xyz')
    phoneC.paint_uniform_color([0, 1, 0])  # 点云着色
    positionC.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([phoneC, positionC], width=600, height=600)
