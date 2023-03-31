# -*- coding: utf-8 -*-
"""create simulate data for validation of GPS NeRF
"""
import os

import numpy as np
import pyrr
import scipy.io as scio
import open3d as o3d
import random
import copy


def drawRawPoints(cf, pf, TP, RP):
    for pt in TP:
        s = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
        cf.write(s)

    for pt in RP:
        s = str(pt[0]) + " " + str(pt[1]) + " " + str(pt[2]) + "\n"
        pf.write(s)

    cf.close()
    pf.close()
    phoneC = o3d.io.read_point_cloud("../../data/pointCloud/pp.txt", format='xyz')
    # phoneC = copy.deepcopy(phoneC).translate((-0.4, 0, -0.25))
    positionC = o3d.io.read_point_cloud("../../data/pointCloud/position.txt", format='xyz')
    # positionC = copy.deepcopy(positionC).translate((-0.4, 0, -0.25))
    phoneC.paint_uniform_color([1, 0, 0])  # 点云着色
    positionC.paint_uniform_color([0, 1, 0])
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.25,
                                                          resolution=100)
    mesh_sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.25,
                                                          resolution=100)
    mesh_sphere = copy.deepcopy(mesh_sphere).translate((0.6, 0, 0.25))
    mesh_sphere2 = copy.deepcopy(mesh_sphere2).translate((-0.6, 0, 0.25))
    mesh_sphere.compute_vertex_normals()
    mesh_sphere2.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_sphere2.paint_uniform_color([0.1, 0.1, 0.7])
    o3d.visualization.draw_geometries([mesh_sphere])
    o3d.visualization.draw_geometries([phoneC, positionC, mesh_sphere,mesh_sphere2], width=600, height=600)


if __name__ == '__main__':
    cf = open("../../data/pointCloud/pp.txt", "w+", encoding='utf-8')
    pf = open("../../data/pointCloud/position.txt", "w+", encoding='utf-8')

    expname = "exp-sim1"
    space_R = 1
    entity_R = 0.25
    entity_O = [0.4, 0, 0.25]
    entity_O2 = [-0.4, 0, 0.25]
    entity_R2 = 0.25
    tx_num = 250
    rx_num = 2000

    alphas_tx = np.random.uniform(0, 360, tx_num) / 180 * np.pi
    betas_tx = np.random.uniform(0, 90, tx_num) / 180 * np.pi
    # alphas_tx = np.linspace(0, 360 - 1, 36) / 180 * np.pi
    # betas_tx = np.linspace(0, 90, 20) / 180 * np.pi
    alphas_rx = np.random.uniform(0, 360, rx_num) / 180 * np.pi
    betas_rx = np.random.uniform(0, 0, rx_num) / 180 * np.pi
    # alphas_rx = np.linspace(0, 360 - 1, 36) / 180 * np.pi

    # r_d = []
    # r_dp = []
    # for i in range(len(alphas_tx)):
    #     for j in range(len(betas_tx)):
    #         r_d.append([alphas_tx[i], betas_tx[j]])
    # for k in range(tx_num):
    #     x = space_R * np.cos(r_d[k][0]) * np.cos(r_d[k][1])  # (90*360)
    #     y = space_R * np.sin(r_d[k][0]) * np.cos(r_d[k][1])
    #     z = space_R * np.sin(r_d[k][1])
    #     r_dp.append([x, y, z])
    # coor_tx = np.array(r_dp).reshape(-1, 3)

    x_tx = space_R * np.cos(alphas_tx) * np.cos(betas_tx)  # (100)
    y_tx = space_R * np.sin(alphas_tx) * np.cos(betas_tx)
    z_tx = space_R * np.sin(betas_tx)
    coor_tx = np.stack([x_tx, y_tx, z_tx], axis=0).T  # (100,3)

    # r_d = []
    # r_dp = []
    # rList = np.linspace(0.25, 0.9, 10)
    # for rx_i in alphas_rx:
    #     r_d.append([rx_i, 0])
    # for r_i in rList:
    #     for d in r_d:
    #         x = r_i * np.cos(d[0]) * np.cos(d[1])  # (90*360)
    #         y = r_i * np.sin(d[0]) * np.cos(d[1])
    #         z = r_i * np.sin(d[1])
    #         r_dp.append([x, y, z])
    # coor_rx = np.array(r_dp).reshape(-1, 3)

    x_rx = space_R * np.random.uniform(0, 0.9, rx_num) * np.cos(alphas_rx) * np.cos(betas_rx)  # (100)
    y_rx = space_R * np.random.uniform(0, 0.9, rx_num) * np.sin(alphas_rx) * np.cos(betas_rx)
    z_rx = space_R * np.sin(betas_rx)
    coor_rx = np.stack([x_rx, y_rx, z_rx], axis=0).T  # (100,3)

    entity_sphere = pyrr.sphere.create(entity_O, entity_R)
    entity_sphere2 = pyrr.sphere.create(entity_O2, entity_R2)

    num_insec1 = 0
    num_insec2 = 0
    amp = 1
    data = np.zeros((tx_num * rx_num, 7))
    for i in range(tx_num):
        for j in range(rx_num):
            ray = pyrr.ray.create(coor_tx[i], coor_rx[j] - coor_tx[i])
            insec = pyrr.geometric_tests.ray_intersect_sphere(ray, entity_sphere)
            insec2 = pyrr.geometric_tests.ray_intersect_sphere(ray, entity_sphere2)
            if len(insec) == 0:
                dis1 = 0
            else:
                dis1 = np.linalg.norm(insec[0] - insec[1], 2)
                num_insec1 += 1

            if len(insec2) == 0:
                dis2 = 0
            else:
                dis2 = np.linalg.norm(insec2[0] - insec2[1], 2)
                num_insec2 += 1


            dis = dis1+dis2



            if dis / (entity_R*2) > 1:
                dis = 1
            else:
                dis = dis / (entity_R*2)

            received_signal = (1 - dis) * amp
            data[i * rx_num + j, :] = np.concatenate((coor_tx[i], coor_rx[j], np.array([received_signal])), axis=-1)

    print("total num:", tx_num * rx_num, "insec1 num:", num_insec1, "insec2 num:", num_insec2)
    np.random.shuffle(data)
    train_data = data[0:int(tx_num * rx_num * 0.8), ...]
    test_data = data[int(tx_num * rx_num * 0.8):, ...]

    savepath = os.path.join("../../data", expname)
    os.makedirs(savepath, exist_ok=True)
    np.savetxt(os.path.join(savepath, "dataTrain.csv"), train_data, delimiter=',', fmt="%.4f",
               header="sx,sy,sz,px,py,pz,Cn0DbHz", comments="")
    np.savetxt(os.path.join(savepath, "dataEval.csv"), test_data, delimiter=',', fmt="%.4f",
               header="sx,sy,sz,px,py,pz,Cn0DbHz", comments="")
    scio.savemat(os.path.join(savepath, "simdata.mat"), {"train": train_data, "test": test_data})
    drawRawPoints(cf, pf, train_data[..., 0:3], train_data[..., 3:6])
