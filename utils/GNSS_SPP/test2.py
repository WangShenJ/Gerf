import numpy as np
import csv
import os
import sys
from math import cos, pi, sin, sqrt

import georinex as gr
import xarray
import netCDF4
from netCDF4 import Dataset
import numpy as np

import folium
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import re

if __name__ == '__main__':
    # f = open('pixel4_out.txt','r')
    # la = []
    # lo = []
    # al = []
    # for line in f.readlines():
    #     contains = line.split(',')
    #     latitude = float(contains[1])
    #     longtitude = float(contains[3])
    #     altitude = float(contains[5])
    #     la.append(latitude)
    #     lo.append(longtitude)
    #     al.append(altitude)
    # print(np.array(la).min(),np.array(la).max())
    # print(np.array(lo).min(),np.array(lo).max())
    # print(np.array(al).min(),np.array(al).max())
    # obs = gr.load('my_mi8/gnss_log_2022_11_18_19_13_44.22o')
    # print(obs)
    # for n in range(0, 200):
    #     print(obs.S1C[n])
    # print(obs)
    n = 0
    f2 = open("../GNSS_SPP/my_mi8/location.txt", "w+", encoding='utf-8')
    for filename in os.listdir('../GNSS_SPP/my_mi8'):
        if filename.find('gnss_log') != -1 and filename.find('txt') != -1:
            n += 1
            f = open('../GNSS_SPP/my_mi8/'+filename, "r", encoding='utf-8')
            s = f.readline()
            while s:
                if s.find('#') == -1 and s.find('Fix') != -1 and s.find('GPS') != -1:
                    f2.write(s)
                s = f.readline()
