# -*- coding: utf-8 -*-
"""load config.json Parameters, kalman filter
"""
import json
import math

import numpy as np
import scipy.constants as sconst
import torch as tr
from cv2 import KalmanFilter

with open("utils/config.json") as json_file:
    config = json.load(json_file)

class ParamLoader():
    """load config.json
    """
    version = config['version']
    algorithm = config['algorithm']
    realtime = config['realtime']
    gateway_names = config['gateway_names']

    hardware = config['hardware']
    ant1offsets   = np.array(hardware['ant1offset'])
    ant2offsets   = np.array(hardware['ant2offset'])
    ant3offsets   = np.array(hardware['ant3offset'])
    atnseq = np.array(hardware['atnseq'])
    atnseq_ours = np.array(hardware['atnseq_ours'])

    server = config['server']
    ip = server['IP']
    port = server['Port']
    username = server['Username']
    password = server['Password']

    loc = config['loc']
    music_param = loc['music']
    freq = music_param['frequency']
    element_dis = music_param["element_dis"]
    array_length = music_param["array_length"]
    subarray_length = music_param["subarray_length"]
    xRange = music_param['xRange']
    yRange = music_param['yRange']
    zRange = music_param['zRange']
    coarse_resolution = music_param['coarse_resolution']
    fine_resolution = music_param['fine_resolution']


class Kalman2D(KalmanFilter):

    def __init__(self):
        super().__init__(2,2)
        self.measurementMatrix = np.array([[1,0],[0,1]],np.float32)
        self.transitionMatrix = np.array([[1,0],[0,1]], np.float32)
        self.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.01
        self.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.05

        self.count = 0

    def kalmanPredict(self, mes):

        mes = np.reshape(mes,(2,1))
        if self.count == 0:
            self.statePost = np.array(mes,np.float32)
        else:
            self.predict()                          # 预测
            self.correct(mes)                       # 用测量值纠正
        self.count += 1

        return self.statePost


def rigid_transform_3D(A, B):
    """transfrom from A coordinate system to B coordinate system
    B = R@A + t
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


class Bartlett():
    """Bartlett Algorithm Searching AoA space
    """

    def __init__(self):
        antenna_loc = [[0   , 0.16, 0.32, 0.48, 0   , 0.16, 0.32, 0.48, 0   , 0.16, 0.32, 0.48, 0  , 0.16, 0.32, 0.48],
                       [0.48, 0.48, 0.48, 0.48, 0.32, 0.32, 0.32, 0.32, 0.16, 0.16, 0.16, 0.16, 0  , 0   , 0   ,    0],
                       [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0  , 0   , 0   ,    0]]
        antenna_loc = tr.tensor(antenna_loc)
        x,y = antenna_loc[0,:],antenna_loc[1,:]
        antenna_num = 16                                     # the number of the antenna array element
        atn_polar   = tr.zeros((antenna_num,2))              # Polar Coordinate
        for i in range(antenna_num):
            atn_polar[i,0] = math.sqrt(x[i] * x[i] + y[i] * y[i])
            atn_polar[i,1] = math.atan2(y[i], x[i])

        self.theory_phase = self.get_theory_phase(atn_polar)


    def get_theory_phase(self, atn_polar):
        """get theory phase, return (360x90)x16 array
        """
        a_step = 1 * 360
        e_step = 1 * 90
        spacealpha = tr.linspace(0, np.pi*2*(1-1/360), a_step)  # 0-2pi
        spacebeta = tr.linspace(0, np.pi/2*(1-1/90), e_step)   # 0-pi/2

        alpha = spacealpha.expand(e_step,-1).flatten()    # alpha[0,1,..0,1..]
        beta = spacebeta.expand(a_step,-1).permute(1,0).flatten() #beta[0,0,..1,1..]

        theta_k = atn_polar[:,1].view(16,1)
        r = atn_polar[:,0].view(16,1)
        lamda = sconst.c / 920e6
        theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta) # (16, 360x90)
        print(theta_t.shape)

        return theta_t.T


    def get_aoa_heatmap(self, phase_m):
        """got aoa heatmap
        """
        delta_phase = self.theory_phase - phase_m.reshape(1,16)     #(360x90,16) - 1x16
        cosd = (tr.cos(delta_phase)).sum(1)
        sind = (tr.sin(delta_phase)).sum(1)
        p = tr.sqrt(cosd * cosd + sind * sind) / 16
        p = p.view(90,360)
        p = p.numpy()
        return p



