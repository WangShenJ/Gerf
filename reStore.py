import copy

from mytrain import *
import torch
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import open3d as o3d

# from utils.GNSS_SPP.test import coordinateTrans

import numpy as np
import open3d as o3d
import copy
from matplotlib import pyplot as plt


# 在点云上添加分类标签
def draw_labels_on_model(pcl, labels):
    cmap = plt.get_cmap("tab20")
    pcl_temp = copy.deepcopy(pcl)
    max_label = labels.max()
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    pcl_temp.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcl_temp], window_name="可视化分类结果",
                                      width=800, height=800, left=50, top=50,
                                      mesh_show_back_face=False)


# 计算欧氏距离
def euclidean_distance(one_sample, X):
    # 将one_sample转换为一纬向量
    one_sample = one_sample.reshape(1, -1)
    # 把X转换成一维向量
    X = X.reshape(X.shape[0], -1)
    # 这是用来确保one_sample的尺寸与X相同
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


class Kmeans(object):
    # 构造函数
    def __init__(self, k=2, max_iterations=1500, tolerance=0.00001):
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    # 随机选取k个聚类中心点
    def init_random_centroids(self, X):
        # save the shape of X
        n_samples, n_features = np.shape(X)
        # make a zero matrix to store values
        centroids = np.zeros((self.k, n_features))
        # 因为有k个中心点，所以执行k次循环
        for i in range(self.k):
            # 随机选取范围内的值
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 查找距离样本点最近的中心

    def closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        # np.argmin 返回距离最小值的下标
        closest_i = np.argmin(distances)
        return closest_i

    # 确定聚类
    def create_clusters(self, centroids, X):
        # 这是为了构造用于存储集群的嵌套列表
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self.closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 基于均值算法更新质心
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 获取标签

    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 预测标签
    def predict(self, X):
        # 随机选取中心点
        centroids = self.init_random_centroids(X)

        for _ in range(self.max_iterations):
            # 对所有点进行聚类
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids
            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)
            # 判断是否满足收敛
            diff = centroids - former_centroids
            if diff.any() < self.tolerance:
                break

        return self.get_cluster_labels(clusters, X)


# if __name__ == "__main__":
#     #  加载点云
#     pcd = o3d.io.read_point_cloud('cluster1.pcd')
#     points = np.asarray(pcd.points)
#     o3d.visualization.draw_geometries([pcd], window_name="可视化原始点云",
#                                       width=800, height=800, left=50, top=50,
#                                       mesh_show_back_face=False)
#     # 执行K-means聚类
#     clf = Kmeans(k=3)
#     labels = clf.predict(points)
#     # 可视化聚类结果
#     draw_labels_on_model(pcd, labels)


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
    rays_d = tags - rays_o
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
        pts, t_vals = get_points(rays_o_chunks, rays_d_chunks, 0, 1, 50)  # [bs, cks, pts, 3]

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


def getSPDensity(**kwargs):
    r = 6378137
    # xNum = 40
    # yNum = 50
    # zNum = 40
    # x_nSamples = np.linspace(37.5295853 ,37.5306248, xNum)
    # y_nSamples = np.linspace(122.0717183, 122.0738229, yNum)
    # z_nSamples = np.linspace(15.213745 ,100, zNum)
    xNum = 60
    yNum = 60
    zNum = 30
    x_nSamples = np.linspace(0, 1, xNum)
    y_nSamples = np.linspace(0, 0.8, yNum)
    z_nSamples = np.linspace(0, 0.15 , zNum)
    # y_Nsamples = np.linspace()
    SP = []
    for x in x_nSamples:
        for y in y_nSamples:
            for z in z_nSamples:
                # x1, y1, z1 = coordinateTrans(x, y, z)
                # x1 = x1 / (2 * r)
                # y1 = y1 / (2 * r)
                # z1 = z1 / (2 * r)
                # SP.append([x1, y1, z1])
                SP.append([x, y, z])
                print(x, y, z)
    SP = tr.tensor(SP)
    tr.reshape(SP, (-1, 3))
    views_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    tags_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    with tr.no_grad():
        raw = run_network(SP, views_chunks, tags_chunks, **kwargs)  # [bs, cks, pts, 3]
    alphac = raw[..., 0]
    print(alphac)
    s = raw[..., 1]
    # print(alphac.shape)
    return SP, alphac, s


if __name__ == '__main__':
    fre = open("data/restoreLog.txt", "w+", encoding='utf-8')
    fPone = open("data/pointCloud/pp.txt", "w+", encoding='utf-8')
    fPosition = open("data/pointCloud/position.txt", "w+", encoding='utf-8')
    fS = open("data/pointCloud/sPosition.txt", "w+", encoding='utf-8')
    dt = pd.read_csv("data/my-exp6/dataEval.csv")

    data = dt.values
    parser = config_parser()
    args = parser.parse_args()
    render_kwargs_train, render_kwargs_train2, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    pts, alphac,sn = getSPDensity(**render_kwargs_train2)
    alphac = F.relu(alphac)
    # alphac = tr.sigmoid(alphac)
    # print(alphac)
    # alphac = (tr.exp(-alphac*(0.1/100)))
    # alphac = tr.sigmoid(alphac)
    sn = tr.sigmoid(sn)
    ptsc = pts.cpu()
    ptsc = np.array(ptsc)

    batchdata = tr.FloatTensor(data[200:3000].astype(float))
    pPosition = batchdata[..., 0:3].cpu()
    # pPosition = batchdata[..., 3:6].cpu()

    # for sp in sPosition:
    #     s = str(float(sp[..., 0])) + " " + str(float(sp[..., 1])) + " " + str(
    #         float(sp[..., 2])) + "\n"
    #     fPone.write(s)
    for pp in pPosition:
        s = str(float(pp[..., 0])) + " " + str(float(pp[..., 1])) + " " + str(
            float(pp[..., 2])) + "\n"
        fPone.write(s)

    n = 0
    pNum = 0
    for pt in ptsc:
        a = alphac[n]
        # si = sn[n]
        print(a)
        if a > 4:

        #     # print(pt[0])
            pNum += 1
            s1 = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
        # print(s1)
            fPosition.write(s1)
        n = n + 1

    fPone.close()
    fPosition.close()
    inds = []
    phoneC = o3d.io.read_point_cloud("data/pointCloud/pp.txt", format='xyz')
    positionC = o3d.io.read_point_cloud("data/pointCloud/position.txt", format='xyz')
    positionS = o3d.io.read_point_cloud("data/pointCloud/sPosition.txt", format='xyz')
    phoneC.paint_uniform_color([1, 0, 0])
    positionC.paint_uniform_color([0, 0, 1])
    positionS.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([phoneC, positionC, positionS], width=600, height=600)

    box_P = []
    PCD = []
    # ------------------------------
    pcd = positionC
    pnum = np.asarray(pcd.points).shape[0]
    #  加载点云
    points = np.asarray(pcd.points)
    # 执行K-means聚类
    clf = Kmeans(k=4)
    labels = clf.predict(points)
    # # 可视化聚类结果
    # draw_labels_on_model(pcd, labels)
    # pcd = pcd.remove_radius_outlier(int(pnum / 30), 0.3)[0]
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     # -------------------DBSCAN--------------------------
    #     labels = np.array(pcd.cluster_dbscan(eps=0.2,
    #                                          min_points=int(pnum / 5),
    #                                          print_progress=False))
    max_label = int(labels.max())
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd], window_name="DBSCAN",
    #                                   height=480, width=600,
    #                                   mesh_show_back_face=0)
    n = 0
    for i in range(max_label):
        ind = np.where(labels == i)[0]
        pcd_i = pcd.select_by_index(ind)
        pnum = np.asarray(pcd_i.points).shape[0]
        # cl, ind = pcd_i.remove_statistical_outlier(nb_neighbors=50*int(pnum),
        #                                          std_ratio=0.05)
        # pcd_i = pcd_i.select_by_index(ind)
        pcd_i = pcd_i.remove_radius_outlier(int(pnum/7), 0.1)[0]
        if np.asarray(pcd_i.points).shape[0] <= 4:
            continue
        n += 1
        obb = pcd_i.get_oriented_bounding_box()
        obb.color = (0, 0, 1)
        box_points = np.array(obb.get_box_points())*400
        box_points[...,0] =  box_points[...,0]/111000-8.59634001
        box_points[...,1] =  box_points[...,1]/111000+41.1761372
        box_points[...,2] =  box_points[...,2]
        for pt in box_points:
            s1 = str(pt[0]) + "," + str(pt[1]) + "," + str(pt[2]) + "\n"
            fre.write(s1)
        # fre.write("\n")
        box_P.append(box_points)
        PCD.append(obb)
        # PCD.append(pcd_i)
    i = 2
    # ind = np.where(labels == i)[0]
    # pcd_i = pcd.select_by_index(ind)
    # pcd_i = pcd_i.remove_radius_outlier(int(pnum / 60), 0.2)[0]
    # if np.asarray(pcd_i.points).shape[0] <= 10:
    #     continue
    # n += 1
    # obb = pcd_i.get_axis_aligned_bounding_box()
    # obb.color = (1, 0, 0)
    # box_points = np.array(obb.get_box_points())
    # box_P.append(box_points)
    PCD.append(obb)
    # PCD.append(pcd_i)
    PCD.append(phoneC)
    o3d.visualization.draw_geometries(PCD, width=600, height=600)
    # KD tree---------------------------------------
    # def f_traverse(node, node_info):
    #     early_stop = False
    #
    #     if isinstance(node, o3d.geometry.OctreeInternalNode):
    #         if isinstance(node, o3d.geometry.OctreeInternalPointNode):
    #             n = 0
    #             for child in node.children:
    #                 if child is not None:
    #                     n += 1
    #             print(
    #                 "{}{}: Internal node at depth {} has {} children and {} points ({})"
    #                     .format('    ' * node_info.depth,
    #                             node_info.child_index, node_info.depth, n,
    #                             len(node.indices), node_info.origin))
    #
    #             # we only want to process nodes / spatial regions with enough points
    #             # points = np.asarray(pcd.select_by_index(node.indices).points)
    #             # if len(node.indices) > 5:
    #             #     SP = tr.tensor(points)
    #             #     tr.reshape(SP, (-1, 3))
    #             #     views_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    #             #     tags_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    #             #     raw = run_network(SP, views_chunks, tags_chunks, **render_kwargs_train2)  # [bs, cks, pts, 3]
    #             #     alphac = raw[..., 0]
    #             #     alphac = tr.sigmoid(alphac)
    #             #     ava = sum(alphac) / len(alphac)
    #             #     f = ava < 0.3
    #             #
    #             #     # early_stop = f
    #             #     if not f:
    #             #         for ind in node.indices:
    #             #             inds.append(ind)
    #             #     else:
    #             #         print(ava)
    #     elif isinstance(node, o3d.geometry.OctreeLeafNode):
    #         if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
    #             # print("{}{}: Leaf node at depth {} has {} points with origin {}".
    #             #       format('    ' * node_info.depth, node_info.child_index,
    #             #              node_info.depth, len(node.indices), node_info.origin))
    #             points = np.asarray(pcd.select_by_index(node.indices).points)
    #             if len(node.indices) >= 2:
    #                 SP = tr.tensor(points)
    #                 tr.reshape(SP, (-1, 3))
    #                 print(SP)
    #                 views_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    #                 tags_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    #                 raw = run_network(SP, views_chunks, tags_chunks, **render_kwargs_train2)  # [bs, cks, pts, 3]
    #                 alphac = raw[..., 0]
    #                 alphac = tr.sigmoid(alphac)
    #                 print(alphac)
    #                 ava = sum(alphac)/len(alphac)
    #                 f = ava < 0.3
    #
    #                 # early_stop = f
    #                 if not f:
    #                     for ind in node.indices:
    #                         inds.append(ind)
    #                 # else:
    #                 #     print(ava)
    #     else:
    #         raise NotImplementedError('Node type not recognized!')
    #
    #     # early stopping: if True, traversal of children of the current node will be skipped
    #     return early_stop
    #
    # pcd = positionC
    # point = np.asarray(pcd.points)
    # N = point.shape[0]
    # # 点云随机着色
    # pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    # # 可视化点云
    # o3d.visualization.draw_geometries([pcd], window_name="原始点云",
    #                                   width=1024, height=768,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)
    # # 创建八叉树， 树深为9
    # octree = o3d.geometry.Octree(max_depth=6)
    # # 从点云中构建八叉树，适当扩展边界0.001m
    # octree.convert_from_point_cloud(pcd, size_expand=0.001)
    # octree.traverse(f_traverse)
    #
    # # pcd = pcd.select_by_index(inds)
    # # octree = o3d.geometry.Octree(max_depth=3)
    # # # 从点云中构建八叉树，适当扩展边界0.001m
    # # octree.convert_from_point_cloud(pcd, size_expand=0.001)
    # # 可视化八叉树
    # o3d.visualization.draw_geometries([octree], window_name="可视化八叉树",
    #                                   width=1024, height=768,
    #                                   left=50, top=50,
    #                                   mesh_show_back_face=False)

    # --------------------------------------------------


    # positionC = positionC.remove_radius_outlier(int(pNum / 40), 0.5)[0]
    # obb = positionC.get_axis_aligned_bounding_box()
    # obb.color = (1, 0, 0)
    # box_points = np.array(obb.get_box_points())
    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25,
    #                                                       resolution=100)
    # phoneC = copy.deepcopy(phoneC).translate((0, 0, -0.25))
    # positionC = copy.deepcopy(positionC).translate((0, 0, -0.25))
    # positionS = copy.deepcopy(positionS).translate((0, 0, -0.25))
    # obb = copy.deepcopy(obb).translate((0, 0, -0.25))
    # # o3d.visualization.draw_geometries([phoneC, positionS], width=600, height=600)

    # pcd = positionC
    # # print(np.asarray(pcd.points).shape[0])
    # pnum = np.asarray(pcd.points).shape[0]
    # box_P = []
    # PCD = []
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     # -------------------DBSCAN--------------------------
    #     labels = np.array(pcd.cluster_dbscan(eps=0.2,
    #                                          min_points=int(pnum / 100),
    #                                          print_progress=False))
    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd], window_name="DBSCAN",
    #                                   height=480, width=600,
    #                                   mesh_show_back_face=0)

    # for i in range(max_label + 1):
    #     ind = np.where(labels == i)[0]
    #     pcd_i = pcd.select_by_index(ind)
    #     pcd_i = pcd_i.remove_radius_outlier(int(pnum / 1000), 0.1)[0]
    #     obb = pcd_i.get_axis_aligned_bounding_box()
    #     obb.color = (1, 1, 0)
    #     box_points = np.array(obb.get_box_points())
    #     box_P.append(box_points)
    #     print(box_points)
    #     PCD.append(obb)
    # print(len(box_P))
    # for i in range(len(box_P)):
    #     print(i)
    # PCD.append(phoneC)
    # o3d.visualization.draw_geometries(PCD, window_name="DBSCAN",
    #                                   height=480, width=600,
    #                                   mesh_show_back_face=0)
