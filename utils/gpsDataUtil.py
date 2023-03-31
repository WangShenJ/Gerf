import time
from math import cos, sin, sqrt, pi

import folium
import os
import numpy as np
from matplotlib import pyplot as plt

from utils.GNSS_SPP.readNav import getSatXYZ, readRinexNav
from utils.GNSS_SPP.test import myDate2JD
from utils.getStaPosition import getPosition


def draw_gps(locations1, color1):
    """
    绘制gps轨迹图
    :param locations: list, 需要绘制轨迹的经纬度信息，格式为[[lat1, lon1], [lat2, lon2], ...]
    :param output_path: str, 轨迹图保存路径
    :param file_name: str, 轨迹图保存文件名
    :return: None
    """
    m1 = folium.Map(locations1[0], zoom_start=15, attr='default')  # 中心区域的确定

    folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
        locations1,  # 将坐标点连接起来
        weight=3,  # 线的大小为3
        color=color1,  # 线的颜色为橙色
        opacity=0.8  # 线的透明度
    ).add_to(m1)  # 将这条线添加到刚才的区域m内

    # 起始点，结束点
    folium.Marker(locations1[0], popup='<b>Starting Point</b>').add_to(m1)

    m1.save(os.path.join('E://', '12.HTML'))  # 将结果以HTML形式保存到指定路径


def getLinearEquation(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]


def getAllPoints(p1, p2, ptype, n, a, b, c):
    pointsList = []
    pointsList.append(p1)
    if ptype == 0:
        x = p1[0]
        pstep = (p2[0] - p1[0]) / n
        while n > 0:
            n = n - 1
            x = x + pstep
            y = -(a * x + c) / b
            pointsList.append([x, y])
        pointsList.append(p2)
    else:
        y = p1[1]
        pstep = (p2[1] - p1[1]) / n
        # print(pstep)
        while n > 0:
            n = n - 1
            y = y + pstep
            x = -(b * y + c) / a
            pointsList.append([x, y])
        pointsList.append(p2)
    return pointsList


def ger_lL(filename):
    f = open("data/my-exp/DGPS_log_2020_07_31_09_42_56.txt", "r", encoding='utf-8')
    s = f.readline()
    while s:
        print(s)
        s = f.readline()


def getAllWalkingPosition():
    locationC1 = [[22.27969, 114.179115], [22.27903, 114.17671], [22.27856, 114.1721], [22.27883, 114.16896],
                  [22.27843, 114.16877], [22.27813, 114.17213], [22.27863, 114.17677], [22.2794, 114.17946],
                  [22.27974, 114.17928]]
    # tC1 = 0
    tC2 = 8 * 60 * 1000  # ms
    tC3 = 17 * 60 * 1000
    tC4 = 27 * 60 * 1000
    tC5 = 38 * 60 * 1000
    # locationC1 = [[22.27969, 114.179115]]
    # locationC2 = [22.27878, 114.1744]
    # locationC3 = [22.27879, 114.16891]
    # locationC4 = [22.27845, 114.17498]
    # locationC5 = [22.27968, 114.17932]
    all = []
    pointNum = 0
    for n in range(0, 8):
        if n == 0:
            pointNum = tC2 * 5 / 9
        elif n == 1:
            pointNum = tC2 * 4 / 9
        elif n == 2:
            pointNum = (tC3 - tC2) * 4 / 9
        elif n == 3:
            pointNum = (tC3 - tC2) * 5 / 9
        elif n == 4:
            pointNum = (tC4 - tC3) * 1 / 20
        elif n == 5:
            pointNum = (tC4 - tC3) * 19 / 20 + (tC5 - tC4) * 4 / 11
        elif n == 6:
            pointNum = (tC5 - tC4) * 6 / 11
        elif n == 7:
            pointNum = (tC5 - tC4) * 1 / 11
        a, b, c = getLinearEquation(locationC1[n][0], locationC1[n][1], locationC1[n + 1][0], locationC1[n + 1][1])
        psl = getAllPoints(locationC1[n], locationC1[n + 1], 1, pointNum, a, b, c)
        all = all + psl
    return all


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


# draw_gps(locations1, locations2, 'red', 'orange')
if __name__ == '__main__':
    # st = '2022-09-22T03:07:15.441921000'
    r = 6378137
    la = []
    lo = []
    al = []
    expname = 'my-exp2'
    # f = open('../data/my-exp2/gnss_log_2022_11_17_21_13_26.txt', 'r')
    # f2 = open('../data/my-exp2/gnss_my_data.txt', 'a')
    # s = f.readline()
    # while s:
    #     if s.find('Raw') != -1 or s.find('Fix') != -1:
    #         f2.write(s)
    #     s = f.readline()
    f = open("../data/my-exp2/gnss_my_data.txt", "r", encoding='utf-8')
    # f2 = open("GNSS_SPP/s10/s10_out.txt", "w+", encoding='utf-8')
    s = f.readline()
    # nav = readRinexNav('../data/my-exp2/brdc2650.22n')
    n = 200
    j = 0
    ff = True
    t2 = 0
    staNum = [0] * 33
    xd = []
    yd = []
    zd = []
    i = 0
    locations = []
    data = []
    while s:
        n = n - 1
        i = 0
        # raw data
        if s.find('# Raw') != -1:
            itemcontain = tuple(s.split(","))
            # print(itemcontain)
        if s.find('#') == -1 and s.find('Fix') != -1 and s.find('GPS') != -1:
            # print(s)
            locationContain = tuple(s.split(","))
            latitude = float(locationContain[2])
            longtitude = float(locationContain[3])
            altitude = float(locationContain[4])
            # print(longtitude, latitude, '-------------------------')
            locations.append([latitude, longtitude])
            la.append(latitude)
            lo.append(longtitude)
            al.append(altitude)
            ss = str(latitude)+','+str(longtitude)+','+str(altitude)+"\n"
            # f2.write(ss)
            x, y, z = coordinateTrans(latitude, longtitude, altitude)
            x = x / (2 * r)
            y = y / (2 * r)
            z = z / (2 * r)
        # draw_gps(locations,locations,'red','red')
        elif s.find('#') == -1 and s.find('Raw') != -1:
            contain = tuple(s.split(","))
            unixT = int(int(contain[1]) / 1000)
            timeX = time.gmtime(unixT)
            timeU = time.strftime('%Y-%m-%dT%H:%M:%S', timeX)
            year = int(time.strftime('%Y', timeX))
            month = int(time.strftime('%m', timeX))
            day = int(time.strftime('%d', timeX))
            hour = int(time.strftime('%H', timeX))
            minute = int(time.strftime('%M', timeX))
            second = int(time.strftime('%S', timeX))
            time2 = np.datetime64(timeU)
            svid = int(contain[11])
            cn0 = float(contain[16])
            phase = contain[24]
            timeNanos = float(contain[2])
            leapSecond = float(contain[3])
            fullBaisNanos = float(contain[5])
            baisNanos = float(contain[6])
            gpsT = int((timeNanos - fullBaisNanos - baisNanos) / 1000000000) % 604800
            print(phase)
    #         UT = hour + (minute / 60.0) + (second / 3600)
    #         # 需要将当前需计算的时刻先转换到儒略日再转换到GPS时间
    #         JD = myDate2JD(year, month, day, hour, minute, second)
    #         WN = int((JD - 2444244.5) / 7)  # WN:GPS_week number 目标时刻的GPS周
    #         t_oc = (JD - 2444244.5 - 7.0 * WN) * 24 * 3600.0  # t_GPS:目标时刻的GPS秒
    #         if svid == 11 or svid == 4 or svid == 5 or svid == 6 or svid == 9 or svid == 11 or svid == 17 or svid == 19 or svid == 20 or svid ==25:
    #             print(svid)
    #             n = n + 1
    #             xyz = getSatXYZ(nav, svid, [time2], t_oc)
    #             sx, sy, sz = xyz[0][0], xyz[0][1], xyz[0][2]
    #             # sx, sy, sz = getPosition(svid, gpsT, '../data/my-exp2/20200923.csv')
    #             if sx != 0:
    #                 sx = sx / (2 * r)
    #                 sy = sy / (2 * r)
    #                 sz = sz / (2 * r)
    #                 cn0 = (cn0 - 20) / 30
    #                 if cn0 < 0:
    #                     cn0 = 0
    #                 if cn0 > 1:
    #                     cn0 = 1
    #                 data.append([sx, sy, sz, x, y, z, cn0])
    #                 print(sx,sy,sz,'------------------------')
    #
    #         if int(contain[11]) <= 32:
    #             staNum[int(contain[11])] = staNum[int(contain[11])] + 1
        s = f.readline()
    # len = len(data)
    # data = np.array(data)
    # data = np.around(data, 17)
    # # data = data.reshape((-1,7))
    # np.random.shuffle(data)
    # # print(data)
    #
    # train_data = data[0:int(len * 0.8), ...]
    # test_data = data[int(len * 0.8):, ...]
    # savepath = os.path.join("../data", expname)
    # os.makedirs(savepath, exist_ok=True)
    # np.savetxt(os.path.join(savepath, "dataTrain.csv"), train_data, delimiter=',', fmt="%.17f",
    #            header="sx,sy,sz,px,py,pz,Cn0DbHz", comments="")
    # np.savetxt(os.path.join(savepath, "dataEval.csv"), test_data, delimiter=',', fmt="%.17f",
    #            header="sx,sy,sz,px,py,pz,Cn0DbHz", comments="")
    # print(np.array(la).min(), np.array(la).max())
    # print(np.array(lo).min(), np.array(lo).max())
    # print(np.array(al).min(), np.array(al).max())
