import csv
import math
import random
from collections import namedtuple

import numpy as np
import open3d as o3d
from matplotlib import collections
import pandas as pd
import os
from sklearn.utils import shuffle


class Point:
    def __init__(self, point_x, point_y, point_z):
        self.coord = [point_x, point_y, point_z]


# self.origin 为线段起始点坐标，坐标等同于 point_start
# self.direction 可视为线段的方向向量
class LineSegment:
    def __init__(self, point_start, point_end):
        origin = []
        direction = []
        for index in range(3):
            origin.append(point_start.coord[index])
            direction.append(point_end.coord[index] - point_start.coord[index])

        self.origin = origin
        self.direction = direction

    # 通过系数 t 获得其对应的线段上的点
    # 0 <= t <= 1 意味着点在线段上
    def get_point(self, coefficient):
        point_coord = []
        for index in range(3):
            point_coord.append(self.origin[index] + coefficient * self.direction[index])
        return Point(point_coord[0], point_coord[1], point_coord[2])


# point_a, point_b 为平行于坐标轴的立方体处于对角位置的两个顶点
class Box:
    def __init__(self, point_a, point_b):
        self.pA = point_a
        self.pB = point_b

    # 获得立方体与线段 line_segment 的两个交点
    def get_intersect_point(self, line_segment):
        # 线段 direction 分量存在 0  预处理
        for index, direction in enumerate(line_segment.direction):
            if direction == 0:
                box_max = max(self.pA.coord[index], self.pB.coord[index])
                box_min = min(self.pA.coord[index], self.pB.coord[index])
                if line_segment.origin[index] > box_max or line_segment.origin[index] < box_min:
                    return None, None

        # 常规处理
        t0, t1 = 0., 1.
        for index in range(3):
            if line_segment.direction[index] != 0.:
                inv_dir = 1. / line_segment.direction[index]
                t_near = (self.pA.coord[index] - line_segment.origin[index]) * inv_dir
                t_far = (self.pB.coord[index] - line_segment.origin[index]) * inv_dir
                if t_near > t_far:
                    t_near, t_far = t_far, t_near
                t0 = max(t_near, t0)
                t1 = min(t_far, t1)
                if t0 > t1:
                    return None, None
        intersection_point_near = line_segment.get_point(t0)
        intersection_point_far = line_segment.get_point(t1)

        return intersection_point_near, intersection_point_far

    # 获得立方体与线段的交线长度
    def get_intersect_length(self, line_segment):
        near_point, far_point = self.get_intersect_point(line_segment)
        if near_point is None:
            return 0.
        length = 0.
        for index in range(3):
            length += (far_point.coord[index] - near_point.coord[index]) ** 2
        return length ** 0.5


def getTXPoints(minc, maxc, maxZ, size, rs):
    np.random.seed(rs)
    array_size = (size, 2)
    array = np.random.randint(minc-1000, maxc+200, size=array_size)
    array_size = (size, 1)
    array2 = np.random.randint(maxZ+400, maxZ + 600, size=array_size)
    array = np.hstack((array, array2))
    return array


def getRXPoints(minc, maxc, minZ, size, rs):
    np.random.seed(rs)
    array_size = (size, 2)
    array = np.random.randint(minc+200, maxc-1000, size=array_size)
    array_size = (size, 1)
    array2 = np.random.randint(minZ -1, minZ , size=array_size)
    array = np.hstack((array, array2))
    return array


def getxTXPoints(minc, maxc, maxX, size, rs):
    np.random.seed(rs)
    array_size = (size, 2)
    # y,z
    array = np.random.randint(minc, maxc, size=array_size)
    # x
    array_size = (size, 1)
    array2 = np.random.randint(maxZ + 100, maxZ + 600, size=array_size)
    array = np.hstack((array2, array))
    return array


def getxRXPoints(minc, maxc, maxX, size, rs):
    np.random.seed(rs)
    array_size = (size, 2)
    # y,z
    array = np.random.randint(minc, maxc, size=array_size)
    # x
    array_size = (size, 1)
    array2 = np.random.randint(minc, minc + 300, size=array_size)
    array = np.hstack((array2, array))
    return array


def drawRawPoints(cf, pf, cubeP, TP, RP):
    for pt in cubeP:
        s = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
        cf.write(s)

    for pt in TP:
        s = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
        pf.write(s)

    for pt in RP:
        s = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
        pf.write(s)

    cf.close()
    pf.close()
    phoneC = o3d.io.read_point_cloud("../../data/pointCloud/pp.txt", format='xyz')
    positionC = o3d.io.read_point_cloud("../../data/pointCloud/position.txt", format='xyz')
    phoneC.paint_uniform_color([1, 0, 0])  # 点云着色
    positionC.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([phoneC, positionC], width=600, height=600)


def insertAABB(sp, sq, amin, amax):
    EPS = 0.0000001
    d = [sq[0] - sp[0], sq[1] - sp[1], sq[2] - sp[2]]

    tmin = 0.0
    tmax = 1.0
    for i in range(3):

        if abs(d[i]) < EPS:
            if sp[i] < amin[i] or sp[i] > amax[i]:
                return False
        else:
            ood = 1.0 / d[i]
            t1 = (amin[i] - sp[i]) * ood
            t2 = (amax[i] - sp[i]) * ood
            if t1 > t2:
                tmp = t1
                t1 = t2
                t2 = tmp
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return False
    return True


if __name__ == '__main__':
    f = open('../../data/simu-exp/myTrainData.csv', 'w', newline='')
    header = ['sx', 'sy', 'sz', 'px', 'py', 'pz', 'Cn0DbHz']
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

    fo = open('../../data/simu-exp/ofile.csv', 'w', newline='')
    header = ['sx', 'sy', 'sz', 'px', 'py', 'pz', 'Cn0DbHz']
    writero = csv.DictWriter(fo, fieldnames=header)
    writero.writeheader()

    fv = open('../../data/simu-exp/vfile.csv', 'w', newline='')
    header = ['sx', 'sy', 'sz', 'px', 'py', 'pz', 'Cn0DbHz']
    writerv = csv.DictWriter(fv, fieldnames=header)


    cf = open("../../data/pointCloud/pp.txt", "w+", encoding='utf-8')
    pf = open("../../data/pointCloud/position.txt", "w+", encoding='utf-8')

    cubeP = [[400, 0, 400], [0, 0, 400], [400, 400, 400], [0, 400, 400], [400, 0, 800], [0, 0, 800],
             [400, 400, 800], [0, 400, 800]]
    maxp = max(cubeP)
    minp = min(cubeP)
    cubeP = np.array(cubeP)
    maxZ = max(cubeP[..., 2])
    minZ = min(cubeP[..., 2])
    minc = -800
    maxc = 2000

    tP = getTXPoints(minc, maxc , maxZ, 200, 55)
    # tP = [[200,200,1500]]
    tP = np.array(tP)
    rP = getRXPoints(minc, maxc, minZ, 400, 4)
    # rP = [[100,100,-500]]
    rP = np.array(rP)
    xtP = getxTXPoints(minc, maxc - 800, 100, 2000, 21)
    xrP = getxRXPoints(minc, maxc - 800, 100, 20, 51)
    n = 0
    j = 0
    n_dic = {}
    RP = []
    TP = []

    # normalization
    rP = (rP - (minc)) / (maxc - minc)
    tP = (tP - (minc)) / (maxc - minc)
    xrP = (xrP - (minc)) / (maxc - minc)
    xtP = (xtP - (minc)) / (maxc - minc)
    maxp = (np.array(maxp) - (minc)) / (maxc - minc)
    minp = (np.array(minp) - (minc)) / (maxc - minc)
    cubeP = (np.array(cubeP) - (minc)) / (maxc - minc)

    occC = 30000
    vacanC = 30000

    for rp in rP:
        if ((0 - minc) / (maxc - minc) <= rp[0] <= (400 - minc) / (maxc - minc) and (0 - minc) / (maxc - minc) <=
                rp[1] <= (400 - minc) / (maxc - minc) and (0 - minc) / (maxc - minc) <= rp[2] <= (1200 - minc) / (
                        maxc - minc)):
            continue
        for tp in tP:
            # if ((-800 - minc) / (maxc - minc) <= tp[0] <= (1200 - minc) / (maxc - minc) and (-800 - minc) / (maxc - minc) <= tp[
            #     1] <= (1200 - minc) / (maxc - minc) and (0 - minc) / (maxc - minc) <= tp[2] <= (1200 - minc) / (maxc - minc)):
            #     continue
            flag2 = insertAABB(rp, tp, minp, maxp)
            n_dic['sx'] = tp[0]
            n_dic['sy'] = tp[1]
            n_dic['sz'] = tp[2]
            n_dic['px'] = rp[0]
            n_dic['py'] = rp[1]
            n_dic['pz'] = rp[2]
            RP.append(rp)
            TP.append(tp)
            if flag2:
                # occ
                occC = occC-1
                n = n + 1
                Cn0bHz = 1 + random.random()
                Cn0bHz = (Cn0bHz - 1) / (45 - 1)
                n_dic['Cn0DbHz'] = Cn0bHz
                writero.writerow(n_dic)
            else:

                # vacan
                j = j+1
                vacanC = vacanC -1
                Cn0bHz = 40 + 5 * random.random()
                Cn0bHz = (Cn0bHz - 1) / (45 - 1)
                n_dic['Cn0DbHz'] = Cn0bHz
                writerv.writerow(n_dic)
                cubeP = list(cubeP)
            # writer.writerow(n_dic)

    occC = 8000
    vacanC = 16000
    data = pd.read_csv('../../data/simu-exp/ofile.csv', sep=',')
    data = shuffle(data)  # 打乱
    data.to_csv('../../data/simu-exp/ofile.csv', index=False, header=True)  # index索引不出现，header表头出现
    datao = pd.read_csv('../../data/simu-exp/ofile.csv', sep=',')
    datao = datao.loc[0:occC-1]
    datao.to_csv('../../data/simu-exp/myTrainData.csv', index=False)

    data = pd.read_csv('../../data/simu-exp/vfile.csv', sep=',')
    data = shuffle(data)  # 打乱
    data.to_csv('../../data/simu-exp/vfile.csv', index=False)  # index索引不出现，header表头出现
    datav = pd.read_csv('../../data/simu-exp/vfile.csv', sep=',')
    datav = datav.loc[1:vacanC - 1]
    datav.to_csv('../../data/simu-exp/myTrainData.csv', index=False, mode='a')
    print(occC, vacanC)
    print(n,j)
    drawRawPoints(cf, pf, cubeP, RP, TP)
