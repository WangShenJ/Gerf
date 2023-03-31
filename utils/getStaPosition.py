import csv
import math  as m


# 获取卫星坐标
def getPosition(svid,t, s):
    # get parameters
    if svid <= 32:
        Prn = svid
    # compare with TEO
    with open(s, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if svid <= 22:
                Prn = svid
            else:
                continue
            if int(row['卫星PRN号']) == Prn:

                TEO = float(row["TEO"])
                # 整点数据
                # print(TEO, t, (TEO <= t <= TEO + 7200))
                if (TEO <= t <= TEO + 7200) and TEO % 7200 == 0:
                    PRN = float(row["卫星PRN号"])
                    TIME = row["历元"]
                    year = int(TIME.strip('\n')[0:3])
                    month = int(TIME.strip('\n')[3:6])
                    day = int(TIME.strip('\n')[6:9])
                    hour = int(TIME.strip('\n')[9:12])
                    minute = int(TIME.strip('\n')[12:15])
                    second = float(TIME.strip('\n')[15:20])
                    a_0 = float(row["卫星钟偏差(s)"])
                    a_1 = float(row["卫星钟漂移(s/s)"])
                    a_2 = float(row["卫星钟漂移速度(s/s*s)"])
                    IODE = float(row["IODE"])
                    C_rs = float(row["C_rs"])
                    δn = float(row["n"])
                    M0 = float(row["M0"])
                    C_uc = float(row["C_uc"])
                    e = float(row["e"])
                    C_us = float(row["C_us"])
                    sqrt_A = float(row["sqrt_A"])
                    C_ic = float(row["C_ic"])
                    OMEGA = float(row["OMEGA"])
                    C_is = float(row["C_is"])
                    I_0 = float(row["I_0"])
                    C_rc = float(row["C_rc"])
                    w = float(row["w"])
                    OMEGA_DOT = float(row["OMEGA_DOT"])
                    IDOT = float(row["IDOT"])
                    L2_code = float(row["L2_code"])
                    PS_week_num = float(row["PS_week_num"])
                    L2_P_code = float(row["L2_P_code"])
                    # 卫星精度(m)=2
                    # 卫星健康状态=0
                    TGD = float(row["TGD"])
                    IODC = float(row["IODC"])
                    x, y, z = CulLocation(year, month, day, hour, minute, second, t, IDOT, C_rs, δn, M0, C_uc, e, C_us,
                                          sqrt_A, TEO, C_ic, OMEGA, C_is,
                                          I_0, C_rc, w, OMEGA_DOT)
                    return x, y, z

    return 0, 0, 0


def CulLocation(year, month, day, hour, minute, second, t_oc,IDOT, C_rs, δn, M0, C_uc, e, C_us, sqrt_A,
                TEO, C_ic, OMEGA, C_is, I_0, C_rc, w, OMEGA_DOT):
    # 1.计算卫星运行平均角速度 GM:WGS84下的引力常数 =3.986005e14，a:长半径
    GM = 398600500000000
    n_0 = m.sqrt(GM) / m.pow(sqrt_A, 3)
    n = n_0 + δn

    # 2.计算归化时间t_k 计算t时刻的卫星位置  UT：世界时 此处以小时为单位
    UT = hour + (minute / 60.0) + (second / 3600)
    # GPS时起始时刻1980年1月6日0点   year是两位数 需要转换到四位数
    if year >= 80:
        if year == 80 and month == 1 and day < 6:
            year = year + 2000
        else:
            year = year + 1900
    else:
        year = year + 2000
    if month <= 2:
        year = year - 1
        month = month + 12  # 1，2月视为前一年13，14月

    # # 需要将当前需计算的时刻先转换到儒略日再转换到GPS时间
    # JD = (365.25 * year) + int(30.6001 * (month + 1)) + day + int(UT / 24 + 1720981.5)
    #
    # WN = int((JD - 2444244.5) / 7)  # WN:GPS_week number 目标时刻的GPS周
    # t_oc = (JD - 2444244.5 - 7.0 * WN) * 24 * 3600.0  # t_GPS:目标时刻的GPS秒
    # print('toc', t_oc)
    # print('teo', TEO)
    # 对观测时刻t1进行钟差改正,注意：t1应是由接收机接收到的时间
    # if t1 is None:
    #
    #     t_k = 0
    # else:
    #     δt = a_0 + a_1(t1 - t_oc) + a_2(t1 - t_oc) ^ 2
    #     t = t1 - δt
    print(t_oc,TEO)
    print('------------------------')
    t_k = t_oc - TEO

    # if t_k > 302400:
    #     t_k -= 604800
    # else :
    #     t_k += 604800

    # 3.平近点角计算M_k = M_0+n*t_k
    # print(M0)
    M_k = M0 + n * t_k  # 实际应该是乘t_k，但是没有接收机的观测时间，所以为了练手设t_k=0

    # 4.偏近点角计算 E_k  (迭代计算) E_k = M_k + e*sin(E_k)
    E = 0
    E1 = 1
    count = 0
    while abs(E1 - E) > 1e-10:
        count = count + 1
        E1 = E
        E = M_k + e * m.sin(E)
        if count > 1e8:
            print("计算偏近点角时未收敛！")
            break

            # 5.计算卫星的真近点角
    V_k = m.atan((m.sqrt(1 - e * e) * m.sin(E)) / (m.cos(E) - e));

    # 6.计算升交距角 u_0(φ_k), ω:卫星电文给出的近地点角距
    u_0 = V_k + w

    # 7.摄动改正项 δu、δr、δi :升交距角u、卫星矢径r和轨道倾角i的摄动量
    δu = C_uc * m.cos(2 * u_0) + C_us * m.sin(2 * u_0)
    δr = C_rc * m.cos(2 * u_0) + C_rs * m.sin(2 * u_0)
    δi = C_ic * m.cos(2 * u_0) + C_is * m.sin(2 * u_0)

    # 8.计算经过摄动改正的升交距角u_k、卫星矢径r_k和轨道倾角 i_k
    u = u_0 + δu
    r = m.pow(sqrt_A, 2) * (1 - e * m.cos(E)) + δr
    i = I_0 + δi + IDOT * (t_k)  # 实际乘t_k=t-t_oe

    # 9.计算卫星在轨道平面坐标系的坐标,卫星在轨道平面直角坐标系（X轴指向升交点）中的坐标为：
    x_k = r * m.cos(u)
    y_k = r * m.sin(u)

    # 10.观测时刻升交点经度Ω_k的计算，升交点经度Ω_k等于观测时刻升交点赤经Ω与格林尼治恒星时GAST之差  Ω_k=Ω_0+(ω_DOT-omega_e)*t_k-omega_e*t_oe
    omega_e = 7.292115e-5  # 地球自转角速度
    OMEGA_k = OMEGA + (OMEGA_DOT - omega_e) * t_k - omega_e * TEO  # 星历中给出的Omega即为Omega_o=Omega_t_oe-GAST_w

    # 11.计算卫星在地固系中的直角坐标l
    X_k = x_k * m.cos(OMEGA_k) - y_k * m.cos(i) * m.sin(OMEGA_k)
    Y_k = x_k * m.sin(OMEGA_k) + y_k * m.cos(i) * m.cos(OMEGA_k)
    Z_k = y_k * m.sin(i)

    # 12.计算卫星在协议地球坐标系中的坐标，考虑级移
    # [X,Y,Z]=[[1,0,X_P],[0,1,-Y_P],[-X_p,Y_P,1]]*[X_k,Y_k,Z_k]

    # if month > 12:  # 恢复历元
    #     year = year + 1
    #     month = month - 12

    # print("历元：", year, "年", month, "月", day, "日", hour, "时", minute, "分", second, "秒", "卫星PRN号:", PRN, "平均角速度:", n,
    #       "卫星平近点角:", M_k, "偏近点角:", E, "真近点角:", V_k, "升交距角:", u_0, "摄动改正项:", δu, δr, δi, "经摄动改正后的升交距角、卫星矢径和轨道倾角:", u, r,
    #       i, "轨道平面坐标X,Y:", x_k, y_k, "观测时刻升交点经度:", OMEGA_k, "地固直角坐标系X:", X_k, "地固直角坐标系Y:", Y_k, "地固直角坐标系Z:", Z_k)
    return X_k, Y_k, Z_k
