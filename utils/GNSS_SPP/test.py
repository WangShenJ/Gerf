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

from utils.GNSS_SPP.readNav import readRinexNav, getSatXYZ


class GetGPS:
    def __init__(self, srcfile, dstfile):
        self.srcfile = srcfile
        self.dstfile = dstfile

    def run(self):
        print("GetGPS" + self.srcfile)
        print("run..." + '\n')
        print("SRC: " + self.srcfile + "\n")
        print("DST: " + self.dstfile + "\n")

        dstfd = open(self.dstfile, 'wt', encoding='UTF-8')
        if dstfd == None:
            return
        with open(self.srcfile, 'rt', encoding='UTF-8') as fd:
            for line in fd:
                line = line.strip('\n')
                mtype = re.findall(r"\$(.+?),", line, re.M)
                print(mtype[0])
                if len(mtype) == 0:
                    continue
                if mtype[0] != "GNGGA":
                    continue

                print(line)
                print(re.findall(r"[^ ]* [^,]*,[^,]*,(.+?),", line, re.M))

                reg = re.compile(
                    r'(?P<mtype>.+?),(?P<time_str>.+?),(?P<latitude_str>.+?),(?P<lathem_str>.+?),(?P<longtitude_str>.+?),(?P<longthem_str>.+?),[^,]*,[^,]*,[^,]*,(?P<altitude_str>.+?),(?P<altunit_str>.+?),')
                regMatch = reg.match(line)
                if regMatch == None:
                    print(line)
                    print("ERROR : regMatch == None")
                    continue
                linebits = regMatch.groupdict()
                # for k, v in linebits.items():
                #    print(k + ": " + v)
                latitude = self.str2latitude(linebits["latitude_str"])
                longtitude = self.str2longtitude(linebits["longtitude_str"])

                strtmp = linebits["time_str"] + ',' + str(latitude) + ',' + linebits["lathem_str"] + ',' + str(
                    longtitude) + ',' + linebits["longthem_str"] \
                         + ',' + linebits["altitude_str"] + ',' + linebits["altunit_str"]
                print(strtmp)
                dstfd.write(strtmp + '\n')

        fd.close()
        dstfd.close()

    def str2latitude(self, latitude_str):
        # print(latitude_str)
        degree = latitude_str[0:2]
        minute = latitude_str[2:]
        # print(degree+" "+minute)
        latitude = round(float(degree) + (float(minute) / 60), 6)
        # print(latitude)
        return latitude

    def str2longtitude(self, longtitude_str):
        # print(longtitude_str)
        degree = longtitude_str[0:3]
        minute = longtitude_str[3:]
        # print(degree+" "+minute)
        longtitude = round(float(degree) + (float(minute) / 60), 6)
        # print(longtitude)
        return longtitude


def myDate2JD(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: float = 0.0):
    """
	将年月日转换为儒略日
	:param y: 年
	:param m: 月
	:param d: 日
	:param hh: 时
	:param mm: 分
	:param ss: 秒
	:return: 儒略日
	"""
    # 日期转换
    if m in [1, 2]:
        m = m + 12
        y = y - 1
    B = 0
    if y > 1582 or (y == 1582 and m > 10) or (y == 1582 and m == 10 and d >= 15):
        B = 2 - int(y / 100) + int(y / 400)
    JD = int(365.25 * (y + 4712)) + int(30.61 * (m + 1)) + d - 63.5 + B
    JD = JD + hh / 24 + mm / 60 / 24 + ss / 60 / 60 / 24
    return JD


def getPPosition(timeP, s):
    f = open(s, 'r')
    latitude = -1
    longtitude = -1
    altitude = -1
    for line in f.readlines():
        contains = line.split(',')
        time = int(float(contains[0]) / 1)
        timeP = int(timeP)
        print(timeP,time)
        if (time - timeP) == 0:
            latitude = float(contains[1])
            longtitude = float(contains[3])
            altitude = float(contains[5])
            return latitude, longtitude, altitude
        if 0<(time - timeP) <=1:
            latitude = float(contains[1])
            longtitude = float(contains[3])
            altitude = float(contains[5])
            return latitude, longtitude, altitude
        if 1<(time - timeP) <=2:
            latitude = float(contains[1])
            longtitude = float(contains[3])
            altitude = float(contains[5])
            return latitude, longtitude, altitude
        if 2<(time - timeP) <=3:
            latitude = float(contains[1])
            longtitude = float(contains[3])
            altitude = float(contains[5])
            return latitude, longtitude, altitude
    return latitude, longtitude, altitude


def coordinateTrans(latitude, longitude, height):
    a = 6378137.0
    b = 6356752.31424518
    E = (a * a - b * b) / (a * a)
    COSLAT = cos(latitude * pi / 180)
    SINLAT = sin(latitude * pi / 180)
    COSLONG = cos(longitude * pi / 180)
    SINLONG = sin(longitude * pi / 180)
    N = a / (sqrt(1 - E * SINLAT * SINLAT))
    NH = N + height
    x = NH * COSLAT * COSLONG
    y = NH * COSLAT * SINLONG
    z = (b * b * N / (a * a) + height) * SINLAT
    return x, y, z


if __name__ == '__main__':
    r = 6378137
    data = []
    for filename in os.listdir('../GNSS_SPP/mi8'):
        if filename.find('obs') == -1:
            continue
        obs = gr.load('../GNSS_SPP/mi8/mi8.obs')
        # nav = readRinexNav('brdc3220.22n')
        n = 0
        pPositions = []
        for time in obs.time:
            time = str(time.values)
            print(time)
            year = float(time[0:4])
            month = float(time[5:7])
            day = float(time[8:10])
            hour = float(time[11:13])
            minute = float(time[14:16])
            second = float(time[17:19])
            timeP = int(time[11:13] + time[14:16] + time[17:19])

    #         la, lo, al = getPPosition(timeP, dstfile)
    #         x, y, z = coordinateTrans(la, lo, al)
    #         pPositions.append([x,y,z])
    #         # x = x / (2 * r)
    #         # y = y / (2 * r)
    #         # z = z / (2 * r)
    #         # print(la, lo, al)
    #         # print(x, y, z)
    #         # print('--------------------')
    #         if la == -1:
    #             break
    #
    # #         # calculate GPS time----------------------------
    #         UT = hour + (minute / 60.0) + (second / 3600)
    #         # 需要将当前需计算的时刻先转换到儒略日再转换到GPS时间
    #         JD = myDate2JD(year, month, day, hour, minute, second)
    #         WN = int((JD - 2444244.5) / 7)  # WN:GPS_week number 目标时刻的GPS周
    #         t_oc = (JD - 2444244.5 - 7.0 * WN) * 24 * 3600.0  # t_GPS:目标时刻的GPS秒
    #
    #         time = np.datetime64(time)
    #         svs = obs.S1C[n].sv
    #         cn0s = obs.S1C[n]
    #
    #         j = 0
    #         for cn0 in cn0s:
    #             if cn0.values >= 0:
    #                 if str(svs[j].values).find('G') != -1:
    #                     n_dic = {}
    #                     sv = str(svs[j].values).replace('G', '')
    #                     sv = int(sv)
    #                     xyz = getSatXYZ(nav, sv, [time], t_oc)
    #                     sx, sy, sz = xyz[0][0], xyz[0][1], xyz[0][2]
    #                     sx = sx / (2 * r)
    #                     sy = sy / (2 * r)
    #                     sz = sz / (2 * r)
    #                     cn0 = (cn0.values - 10) / 30
    #                     if cn0 > 1:
    #                         cn0 = 1
    #
    #                     data.append([sx, sy, sz, x, y, z, cn0])
    #                     # n_dic['sx'] = sx
    #                     # n_dic['sy'] = sy
    #                     # n_dic['sz'] = sz
    #                     # n_dic['px'] = x
    #                     # n_dic['py'] = y
    #                     # n_dic['pz'] = z
    #                     # n_dic['Cn0DbHz'] = cn0
    #                     # write to csv
    #                     # writer.writerow(n_dic)
    #
    #                 j += 1
    #         n += 1
    # len = len(data)
    # print(n)
    # data = np.array(data)
    # data = np.around(data, 17)
    # # data = data.reshape((-1,7))
    # np.random.shuffle(data)
    # # print(data)
    #
    # train_data = data[0:int(len* 0.8), ...]
    # test_data = data[int(len* 0.8):, ...]
    # savepath = os.path.join("../../data", expname)
    # os.makedirs(savepath, exist_ok=True)
    # np.savetxt(os.path.join(savepath, "dataTrain.csv"), train_data, delimiter=',', fmt="%.17f",
    #            header="sx,sy,sz,px,py,pz,Cn0DbHz", comments="")
    # np.savetxt(os.path.join(savepath, "dataEval.csv"), test_data, delimiter=',', fmt="%.17f",
    #            header="sx,sy,sz,px,py,pz,Cn0DbHz", comments="")
