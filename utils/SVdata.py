import math
import os
from math import cos, sin, sqrt, pi

import pandas as pd
import numpy as np
import open3d as o3d


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


def cal_angle_of_vector(v0, v1, is_use_deg=True):
    dot_product = np.dot(v0, v1)
    v0_len = np.linalg.norm(v0)
    v1_len = np.linalg.norm(v1)
    try:
        angle_rad = np.arccos(dot_product / (v0_len * v1_len))
    except ZeroDivisionError as error:
        raise ZeroDivisionError("{}".format(error))

    if is_use_deg:
        return np.rad2deg(angle_rad)
    return angle_rad


def cal(lat1, lon1, brng):
    # print(lat1,lon1)
    R = 6378.1  # Radius of the Earth
    brng = brng / 180 * pi
    d = 1  # Distance in km

    # lat2  52.20444 - the lat result I'm hoping for
    # lon2  0.36056 - the long result I'm hoping for.
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    # lat1 = lat1 * (pi * 180)  # Current lat point converted to radians
    # lon1 = lon1 * (pi * 180)  # Current long point converted to radians

    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
                     math.cos(lat1) * math.sin(d / R) * math.cos(brng))

    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                             math.cos(d / R) - math.sin(lat1) * math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return lat2, lon2

def arcsin_and_arccos(pt1, pt2):
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    sin = delta_y/math.sqrt(delta_x**2 + delta_y**2)
    cos = delta_x/math.sqrt(delta_x**2 + delta_y**2)
    if sin>=0 and cos>=0:
        return math.asin(sin), math.acos(cos)
    elif sin>=0 and cos<0:
        return math.pi-math.asin(sin), math.acos(cos)
    elif sin<0 and cos<0:
        return math.pi-math.asin(sin), 2*math.pi-math.acos(cos)
    elif sin<0 and cos>=0:
        return 2*math.pi+math.asin(sin), 2*math.pi-math.acos(cos)



if __name__ == '__main__':
    expname = 'my-exp5'
    dt = pd.read_csv('../data/my-exp5/intersection_raw.csv',nrows=3000000)
    dts = dt.values
    f = open("../data/pointCloud/pp.txt", "w+", encoding='utf-8')
    allDataO = []
    allDataV = []
    i = 0
    no = 0
    nv = 0
    for data in dts:
        # print(i)
        lon = data[3]
        lat = data[4]
        alt = data[5]
        # print(alt)
        az = data[8]
        el = data[9]
        snr = data[10]
        # if el <= 15:
        #     continue
        if 165 >= alt >= 135:
            # print(alt)
            x, y, z = coordinateTrans(lat, lon, alt)
            lat2, lon2 = cal(lat, lon, az)
            # print(lon,lat)
            # print(lon2,lat2)
            lon = lon * 111000
            lat = lat * 111000
            lon2 = lon2 * 111000
            lat2 = lat2 * 111000
            # v1 = np.array([lon2 - lon, lat2 - lat])
            v1 = [lon2 - lon,lat2 - lat]
            # v1 = np.array([-lon, -lat, -alt])
            # print(v1)
            # v2 = np.array([1, 0])
            v2 = [1,0]
            # anglew = cal_angle_of_vector(v1, v2)
            angle = arcsin_and_arccos(v1,v2)[0]
            # angle = angle1[0]/pi*180
            # print(angle1[0]/pi*180)
            # angle = angle / 180 * pi
            if el <= 15:
                continue
            el = el / 180 * pi
            # angle = 360 - az - angle1
            # lon = lon*111000
            # lat = lat*111000
            # print(angle)
            if snr < 35:
                no += 1
                allDataO.append([lon, lat, alt, angle, el, snr])
            else:
                nv += 1
                allDataV.append([lon, lat, alt, angle, el, snr])

            i += 1
    lenO = len(allDataO)
    lenV = len(allDataV)

    allDataO = np.array(allDataO)
    for n in range(3):
        allDataO[..., n] = (allDataO[..., n] - min(allDataO[..., n]))
    for n in range(3):
        allDataO[..., n] = (allDataO[..., n] - min(allDataO[..., n])) / 350
    # allDataO[..., 2] = 0

    allDataV = np.array(allDataV)
    for n in range(3):
        allDataV[..., n] = (allDataV[..., n] - min(allDataV[..., n]))
    for n in range(3):
        allDataV[..., n] = (allDataV[..., n] - min(allDataV[..., n])) / 350
    # allDataV[..., 2] = 0

    # allData[..., 2] = (allData[..., 2] - 150/111000) / (400/111000 - 150/111000)
    allDataO[..., -1] = (allDataO[..., -1] - 10) / (45 - 10)
    allDataV[..., -1] = (allDataV[..., -1] - 10) / (45 - 10)

    np.random.shuffle(allDataO)
    # np.random.shuffle(allDataV)
    # allData = allDataO[0:int(lenO * 0.3), ...]
    allData = np.concatenate((allDataO[0:int(lenO * 0.6), ...], allDataV), axis=0)
    len = len(allData)
    np.random.shuffle(allData)
    # data = allData[..., -1]
    # data[data > 1] = 1
    # data[data < 0] = 0
    # allData[..., 2] = 0
    train_data = allData[0:int(len * 0.8), ...]
    test_data = allData[int(len * 0.8):, ...]
    savepath = os.path.join("../data", expname)
    os.makedirs(savepath, exist_ok=True)
    np.savetxt(os.path.join(savepath, "dataTrain.csv"), train_data, delimiter=',', fmt="%.4f",
               header="lon,lat,alt,az,el,snr", comments="")
    np.savetxt(os.path.join(savepath, "dataEval.csv"), test_data, delimiter=',', fmt="%.4f",
               header="lon,lat,alt,az,el,snr", comments="")

    dt = pd.read_csv('../data/my-exp5/dataTrain.csv')
    dts = dt.values
    n = 0
    print(lenV,lenO)
    for data in dts:
        lon = data[0]
        lat = data[1]
        alt = data[2]
        s = str(float(lon)) + " " + str(float(lat)) + " " + str(float(alt)) + "\n"
        # s = str(float(x)) + " " + str(float(y)) + " " + str(float(z)) + "\n"
        f.write(s)
        n += 1
    phoneC = o3d.io.read_point_cloud("../data/pointCloud/pp.txt", format='xyz')
    phoneC.paint_uniform_color([1, 0, 0])
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([phoneC, FOR1], width=600, height=600)
