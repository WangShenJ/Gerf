from utils.mytrain import config_parser, create_nerf
from utils.simulate.simulateData import Box, Point, LineSegment, insertAABB
import math
import torch as tr
import pandas as pd


import numpy as np
import open3d as o3d
import copy
from matplotlib import pyplot as plt
device = tr.device("cuda:0" if tr.cuda.is_available() else "cpu")

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

def getline(points):
    polygon_points = np.array(points)
    lines = [[0, 1]]  # 连接的顺序，封闭链接
    color = [[0, 1, 0] for i in range(len(lines))]
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    return lines_pcd


def polygon(points):
    # 绘制顶点
    polygon_points = np.array(points)
    lines = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [4, 6], [6, 7], [7, 5], [6, 2],
             [7, 3]]  # 连接的顺序，封闭链接
    color = [[0, 0, 1] for i in range(len(lines))]
    # 添加顶点，点云
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    points_pcd.paint_uniform_color([0, 0, 1])  # 点云颜色

    # 绘制线条
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color)  # 线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    return lines_pcd, points_pcd


class OctNode:
    # New Octnode Class, can be appended to as well i think
    def __init__(self, position, size, depth, parent):
        # OctNode Cubes have a position and size
        # position is related to, but not the same as the objects the node contains.
        self.position = position
        self.size = size
        self.isLeafNode = True
        self.isExtended = True
        # might as well give it some emtpy branches while we are here.
        self.branches = [None, None, None, None, None, None, None, None]
        self.points = [(self.position[0] + size / 2, self.position[1] + size / 2, self.position[2] + size / 2),
                       (self.position[0] + size / 2, self.position[1] + size / 2, self.position[2] - size / 2),
                       (self.position[0] + size / 2, self.position[1] - size / 2, self.position[2] + size / 2),
                       (self.position[0] + size / 2, self.position[1] - size / 2, self.position[2] - size / 2),
                       (self.position[0] - size / 2, self.position[1] + size / 2, self.position[2] + size / 2),
                       (self.position[0] - size / 2, self.position[1] + size / 2, self.position[2] - size / 2),
                       (self.position[0] - size / 2, self.position[1] - size / 2, self.position[2] + size / 2),
                       (self.position[0] - size / 2, self.position[1] - size / 2, self.position[2] - size / 2)]
        self.depth = depth
        self.parent = parent


class Octree:
    def __init__(self, worldSize):
        self.root = self.addNode((0.5, 0.5, 0.5), worldSize, 0, None)
        self.worldSize = worldSize
        self.leafNum = 0
        self.leaf = []

    def addNode(self, position, size, depth, parent):
        # 创建一个叶子节点
        return OctNode(position, size, depth, parent)

    def extendNode(self, node):
        if node.isExtended and node.isLeafNode:
            offset = node.size / 4
            pos = node.position
            for i in range(0, 8):
                if i == 0:
                    # 右上前
                    newCenter = (pos[0] + offset, pos[1] + offset, pos[2] + offset)
                elif i == 1:
                    # 右下前
                    newCenter = (pos[0] + offset, pos[1] + offset, pos[2] - offset)
                elif i == 2:
                    # 右上后
                    newCenter = (pos[0] + offset, pos[1] - offset, pos[2] + offset)
                elif i == 3:
                    # 右下后
                    newCenter = (pos[0] + offset, pos[1] - offset, pos[2] - offset)
                elif i == 4:
                    # 左上前
                    newCenter = (pos[0] - offset, pos[1] + offset, pos[2] + offset)
                elif i == 5:
                    # 左下前
                    newCenter = (pos[0] - offset, pos[1] + offset, pos[2] - offset)
                elif i == 6:
                    # 左上后
                    newCenter = (pos[0] - offset, pos[1] - offset, pos[2] + offset)
                elif i == 7:
                    # 左下后
                    newCenter = (pos[0] - offset, pos[1] - offset, pos[2] - offset)
                newNode = self.addNode(newCenter, node.size / 2, node.depth + 1, node)
                node.branches[i] = newNode
                node.isLeafNode = False
        elif (node.isLeafNode == False) and node.isExtended:
            for i in range(0, 8):
                self.extendNode(node.branches[i])
        else:
            return

    def findLeaf(self, node):
        if node == self.root:
            self.leafNum = 0
            self.leaf = []
        if node.isLeafNode and node.isExtended:
            # print(node.depth)
            # print(node.parent.position)
            self.leafNum += 1
            self.leaf.append(node)
        elif node.isExtended:
            for i in range(0, 8):
                self.findLeaf(node.branches[i])


def insertJP(node, pp, sp, nS, ar, all_d):
    # if node.isLeafNode and (node.isExtended != False):
    #     leaf = node
    #     box = Box(Point(leaf.points[-1][0], leaf.points[-1][1], leaf.points[-1][2]),
    #               Point(leaf.points[0][0], leaf.points[0][1], leaf.points[0][2]))
    #     tx = Point(sp[0], sp[1], sp[2])
    #     rx = Point(pp[0], pp[1], pp[2])
    #     l = LineSegment(tx, rx)
    #     F = insertAABB(pp,sp,leaf.points[-1],leaf.points[0])
    #     if F:
    #         p1, p2 = box.get_intersect_point(l)
    #         if p1 != None:
    #             # insert calculate points
    #             p1 = np.array(p1.coord)
    #             p2 = np.array(p2.coord)
    #             d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    #             t_vals = np.array(tr.linspace(0, 1., steps=int((d / all_d) * nS)).cpu())  # [n_samples,]
    #             for t in t_vals:
    #                 pts = p2 + (p1 - p2) * t
    #                 ar.append(pts)
    #     return ar

    for leaf in node.branches:
        if leaf.isExtended == False:
            continue
        # print(leaf.points[0], leaf.points[-1])
        F = insertAABB(pp, sp, leaf.points[-1], leaf.points[0])
        if F:
            # if p1 != None:
            if leaf.isLeafNode:
                box = Box(Point(leaf.points[-1][0], leaf.points[-1][1], leaf.points[-1][2]),
                          Point(leaf.points[0][0], leaf.points[0][1], leaf.points[0][2]))
                tx = Point(sp[0], sp[1], sp[2])
                rx = Point(pp[0], pp[1], pp[2])
                l = LineSegment(tx, rx)
                p1, p2 = box.get_intersect_point(l)
                # insert calculate points
                p1 = np.array(p1.coord)
                p2 = np.array(p2.coord)
                d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
                t_vals = np.array(tr.linspace(0, 1., steps=int(d / all_d * nS)).cpu())  # [n_samples,]
                for t in t_vals:
                    pts = p2 + (p1 - p2) * t
                    ar.append(pts)
            else:

                ar = insertJP(leaf, pp, sp, nS, ar, all_d)
    return ar


def insertJ(node, pp, sp):
    if node.isLeafNode:
        leaf = node
        box = Box(Point(leaf.points[-1][0], leaf.points[-1][1], leaf.points[-1][2]),
                  Point(leaf.points[0][0], leaf.points[0][1], leaf.points[0][2]))
        tx = Point(sp[0], sp[1], sp[2])
        rx = Point(pp[0], pp[1], pp[2])
        l = LineSegment(tx, rx)
        p1, p2 = box.get_intersect_point(l)
        if p1 != None:
            leaf.isExtended = False
        return

    for leaf in node.branches:
        # print(leaf.points[0], leaf.points[-1])
        box = Box(Point(leaf.points[-1][0], leaf.points[-1][1], leaf.points[-1][2]),
                  Point(leaf.points[0][0], leaf.points[0][1], leaf.points[0][2]))
        tx = Point(sp[0], sp[1], sp[2])
        rx = Point(pp[0], pp[1], pp[2])
        l = LineSegment(tx, rx)
        p1, p2 = box.get_intersect_point(l)
        if p1 != None:
            # print(p1.coord,p2.coord)
            if leaf.isLeafNode:
                leaf.isExtended = False
            else:
                insertJ(leaf, pp, sp)


def avDensity(points, **kwargs):
    d = points[0][0] - points[-1][0]
    Num = int(d*20)
    SP = []
    xNum = Num
    if xNum <4:
        xNum = 4
    yNum = Num
    if yNum <4:
        yNum = 4
    zNum = Num
    if zNum <4:
        zNum = 4

    x_nSamples = np.linspace(points[-1][0], points[0][0], xNum)
    y_nSamples = np.linspace(points[-1][1], points[0][1], yNum)
    z_nSamples = np.linspace(points[-1][2], points[0][2], zNum)
    for x in x_nSamples:
        for y in y_nSamples:
            for z in z_nSamples:
                SP.append([x, y, z])
                # print(x, y, z)
    SP = tr.tensor(SP)
    # print(SP)
    tr.reshape(SP, (-1, 3))
    views_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    tags_chunks = tr.tensor([0, 0, 0]).expand(SP.shape).to(device)
    with tr.no_grad():
        raw = run_network(SP, views_chunks, tags_chunks, **kwargs)  # [bs, cks, pts, 3]
    alphac = raw[..., 0]
    alphac = tr.sigmoid(alphac)
    alphac = tr.mean(alphac)
    return alphac

def ifSin(tleaf,tree):
    xt = tleaf.position[0]
    yt = tleaf.position[1]
    zt = tleaf.position[2]
    for leaf in tree.leaf:
        x = leaf.position[0]
        y = leaf.position[1]
        z = leaf.position[2]
        d = math.sqrt((x-xt)**2+(y-yt)**2+(z-zt)**2)
        if d == (tleaf.points[0][-1] - tleaf.points[-1][-1]):
            # 有相邻
            return False
    return True
if __name__ == '__main__':
    octree = Octree(1)
    octree.extendNode(octree.root)
    # octree.findLeaf(octree.root)
    # for leaf in octree.leaf:
    #     if leaf.points[-1][-1] >= 0.3:
    #         leaf.isExtended = False
    #         octree.findLeaf(octree.root)
    octree.extendNode(octree.root)
    octree.extendNode(octree.root)
    octree.extendNode(octree.root)
    # octree.extendNode(octree.root)
    octree.extendNode(octree.root)
    octree.findLeaf(octree.root)

    pcd = []
    fre = open("data/restoreLog.txt", "w+", encoding='utf-8')
    fPone = open("data/pointCloud/pp.txt", "w+", encoding='utf-8')
    fS = open("data/pointCloud/sPosition.txt", "w+", encoding='utf-8')
    dt = pd.read_csv("data/my-exp6/dataEval.csv")
    data = dt.values
    batchdata = tr.FloatTensor(data[0:500].astype(float))
    pPosition = batchdata[..., 0:3].cpu()
    azs = batchdata[..., 3]
    els = batchdata[..., 4]
    rays_d = []
    n = 0
    for az in azs:
        el = els[n]
        rays_d.append((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))
        n += 1
    batchsize = pPosition.shape[0]
    rays_d = tr.tensor(rays_d)
    rays_d = tr.reshape(rays_d, (batchsize, -1, 3))
    chunks = 1
    chunks_num = 1
    rays_o_chunks = pPosition.expand(chunks, -1, -1).permute(1, 0, 2)  # [bs, cks, 3]
    rayy = tr.reshape(rays_d, (batchsize, 3))
    sPosition = pPosition[..., None, :] + rayy[..., None, :] * 0.8
    sPosition = sPosition.reshape(pPosition.shape)

    pam = batchdata[..., -1].cpu()
    # print(len(octree.leaf))
    boxp = o3d.io.read_point_cloud("data/pointCloud/pp.txt", format='xyz')
    # print(sPosition.shape)
    for pp, sp, am in zip(pPosition, sPosition, pam):
        s = str(float(pp[..., 0])) + " " + str(float(pp[..., 1])) + " " + str(
            0) + "\n"
        fPone.write(s)
        print(am)
        if am > 0.95:
            points = []
            points.append(pp)
            points.append(sp)
            # line = getline(points)
            # pcd.append(line)
            # print(am)
            insertJ(octree.root, pp, sp)
            # pA = []
            # pA = insertJP(octree.root, pp, sp, 300, pA)
            # print(len(pA))
            octree.findLeaf(octree.root)
            # print(len(octree.leaf))

            #
            # for sp in pA:
            #     s = str(float(sp[..., 0])) + " " + str(float(sp[..., 1])) + " " + str(
            #         float(sp[..., 2])) + "\n"
            #     fPone.write(s)
    # parser = config_parser()
    # args = parser.parse_args()
    # render_kwargs_train, render_kwargs_train2, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    for leaf in octree.leaf:
        F = ifSin(leaf, octree)
        if F:
            leaf.isExtended = False
    octree.extendNode(octree.root)
    octree.findLeaf(octree.root)
    for leaf in octree.leaf:
        if leaf.isExtended == False:
            continue
        # if leaf.points[0][-1] <= 0.18 and leaf.points[-1][-1] >= 0:
        #     if leaf.points[0][1] <= 0.8 and leaf.points[-1][1] >= 0:
        #         if leaf.points[0][2] <= 0.7 and leaf.points[-1][1] >= 0:
        # alphac = avDensity(leaf.points,**render_kwargs_train2)
        # # alphac = tr.sigmoid(alphac)
        # print(alphac)
        # if alphac < 0.55:
        #     leaf.isExtended = False
        #     octree.findLeaf(octree.root)
        # else:

                # if not F:
                    # print(F)
        pP, lP = polygon(leaf.points)
        # R = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, -np.pi/8))
        # pP = pP.rotate(R)
        # lP = lP.rotate(R)
        # boxp = boxp +pP +lP
        pcd.append(pP)
        pcd.append(lP)
        # print(pcd)
    #         # if leaf.points[0][0] <= 0.65 and leaf.points[-1][0] >= 0.3:
    #         #     if leaf.points[0][1] <= 0.8 and leaf.points[-1][1] >= 0:

    phoneC = o3d.io.read_point_cloud("data/pointCloud/pp.txt", format='xyz')
    # R = o3d.geometry.get_rotation_matrix_from_axis_angle((0, 0, np.pi/8))
    # phoneC = phoneC.rotate(R)
    pcd.append(phoneC)
    phoneC.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries(pcd, width=600, height=600)
