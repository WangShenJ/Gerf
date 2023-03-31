# 处理GPS数据(.txt)和星历数据(.n)，得到新的数据: GPST, LeapSecond, svid, state, satellitePosition, phonePosition, Cn0DbHz, CarrierPhase
import csv

from utils.gpsDataUtil import *


def getMyData():
    f = open("data/my-exp/gnss_my_data.txt", "r", encoding='utf-8')
    s = f.readline()
    f2 = open('data/my-exp/myTrainData.csv', 'w', newline='')
    header = ['GPST', 'LeapSecond', 'svid', 'state', 'sx', 'sy', 'sz', 'px', 'py', 'pz', 'Cn0DbHz']
    writer = csv.DictWriter(f2, fieldnames=header)
    writer.writeheader()
    while s:
        n_dic = {}
        if s.find('#') == -1:
            if s.find('Fix') != -1:
                # location.txt
                locationContain = tuple(s.split(","))
                latitude = float(locationContain[2])
                longtitude = float(locationContain[3])
                altitude = float(locationContain[4])
                s = f.readline()
                while s and s.find('Raw') != -1:
                    contain = tuple(s.split(","))
                    svid = int(contain[11])
                    ElapsedRealtimeMillis = float(contain[1])
                    timeNanos = float(contain[2])
                    leapSecond = float(contain[3])
                    fullBaisNanos = float(contain[5])
                    baisNanos = float(contain[6])
                    state = float(contain[13])
                    Cn0bHz = float(contain[16])
                    gpsT = int((timeNanos - fullBaisNanos - baisNanos) / 1000000000) % 604800
                    x, y, z = getPosition(svid, gpsT, 'data/my-exp/20200810.csv')
                    px, py, pz = coordinateTrans(latitude, longtitude, altitude)
                    n_dic['GPST'] = gpsT
                    n_dic['LeapSecond'] = leapSecond
                    n_dic['svid'] = svid
                    n_dic['state'] = state
                    n_dic['sx'] = x
                    n_dic['sy'] = y
                    n_dic['sz'] = z
                    n_dic['px'] = px
                    n_dic['py'] = py
                    n_dic['pz'] = pz

                    n_dic['Cn0DbHz'] = Cn0bHz
                    # write to csv
                    writer.writerow(n_dic)
                    s = f.readline()
                continue

        s = f.readline()


def dataFilter():
    f = open('data/my-exp/gnss_log_2022_08_10_19_39_51.txt', 'r')
    f2 = open('data/my-exp/gnss_my_data.txt', 'a')
    s = f.readline()
    while s:
        if s.find('OrientationDeg') == -1:
            f2.write(s)
        s = f.readline()


if __name__ == '__main__':
    # dataFilter()
    getMyData()

    # 获取所有经纬度数据
    # allPosition = getAllWalkingPosition()
    # f = open("data/my-exp/DGPS_log_2020_07_31_09_42_56.txt", "r", encoding='utf-8')
    # s = f.readline()
    # flag = 1
    # standerTime = 0
    # n = 200
    # with open('data/my-exp/traindata.csv', 'w', newline='') as csvF:
    #     header = ['GPST', 'LeapSecond', 'svid', 'state', 'satellitePosition', 'phonePosition', 'Cn0DbHz',
    #               'CarrierPhase']
    #     writer = csv.DictWriter(csvF, fieldnames=header)
    #     writer.writeheader()
    #     while s:
    #         n = n - 1
    #         n_dic = {}
    #         # 逐行读取RAW数据，获取卫星位置，phone位置
    #         if s.find('#') == -1:
    #             contain = tuple(s.split(","))
    #             if flag == 1:
    #                 standerTime = float(contain[1])
    #                 flag = 0
    #             svid = int(contain[11])
    #             ElapsedRealtimeMillis = float(contain[1])
    #             timeNanos = float(contain[2])
    #             leapSecond = float(contain[3])
    #             fullBaisNanos = float(contain[5])
    #             baisNanos = float(contain[6])
    #             state = float(contain[13])
    #             Cn0bHz = float(contain[16])
    #             CarrierPhase = contain[24]
    #             gpsT = int((timeNanos - fullBaisNanos - baisNanos) / 1000000000) % 604800
    #             x, y, z = getPosition(svid, gpsT, 'data/my-exp/20200731.csv')
    #             phonePosition = allPosition[int(ElapsedRealtimeMillis - standerTime)]
    #             px, py, pz = coordinateTrans(phonePosition[0], phonePosition[1], 5)
    #             n_dic['GPST'] = gpsT
    #             n_dic['LeapSecond'] = leapSecond
    #             n_dic['svid'] = svid
    #             n_dic['state'] = state
    #             n_dic['satellitePosition'] = str(x) + ' ' + str(y) + ' ' + str(z)
    #             n_dic['phonePosition'] = str(px) + ' ' + str(py) + ' ' + str(pz)
    #             n_dic['Cn0DbHz'] = Cn0bHz
    #             n_dic['CarrierPhase'] = 0
    #             # write to csv
    #             writer.writerow(n_dic)
    #         s = f.readline()
    # csvF.close()
