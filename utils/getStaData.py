import csv
import math  as m
# 处理星历数据，将其写进csv文件
with open('../data/my-exp2/brdc2660.22n', 'r') as f:
    if f == 0:
        print("不能打开文件！")
    else:
        print("导航文件打开成功！")
    nfile_lines = f.readlines()  # 按行读取N文件
    print(len(nfile_lines))
    f.close()


def start_num():  # 定义数据记录的起始行
    for i in range(len(nfile_lines)):
        if nfile_lines[i].find('END OF HEADER') != -1:
            start_num = i + 1
    return start_num


n_dic_list = []

n_data_lines_nums = int((len(nfile_lines) - start_num()) / 8)
print("一共%d组数据" % (n_data_lines_nums))

# 第j组，第i行
for j in range(n_data_lines_nums):
    n_dic = {}
    for i in range(8):
        data_content = nfile_lines[start_num() + 8 * j + i]
        n_dic['数据组数'] = j + 1
        if i == 0:
            n_dic['卫星PRN号'] = int(data_content.strip('\n')[0:2].strip(' '))
            n_dic['历元'] = data_content.strip('\n')[3:22]
            n_dic['卫星钟偏差(s)'] = float(
                (data_content.strip('\n')[23:41][0:-4] + 'e' + data_content.strip('\n')[23:41][-3:]).strip(
                    ' '))  # 利用字符串切片功能来进行字符串的修改
            n_dic['卫星钟漂移(s/s)'] = float(
                (data_content.strip('\n')[42:60][0:-4] + 'e' + data_content.strip('\n')[42:60][-3:]).strip(' '))
            n_dic['卫星钟漂移速度(s/s*s)'] = float(
                (data_content.strip('\n')[61:79][0:-4] + 'e' + data_content.strip('\n')[61:79][-3:]).strip(' '))

        if i == 1:
            n_dic['IODE'] = float(
                (data_content.strip('\n')[4:22][0:-4] + 'e' + data_content.strip('\n')[4:22][-3:]).strip(' '))
            n_dic['C_rs'] = float(
                (data_content.strip('\n')[22:41][0:-4] + 'e' + data_content.strip('\n')[23:41][-3:]).strip(' '))
            n_dic['n'] = float(
                (data_content.strip('\n')[41:60][0:-4] + 'e' + data_content.strip('\n')[42:60][-3:]).strip(' '))
            n_dic['M0'] = float(
                (data_content.strip('\n')[60:79][0:-4] + 'e' + data_content.strip('\n')[60:79][-3:]).strip(' '))
        if i == 2:
            n_dic['C_uc'] = float(
                (data_content.strip('\n')[4:22][0:-4] + 'e' + data_content.strip('\n')[4:22][-3:]).strip(' '))
            n_dic['e'] = float(
                (data_content.strip('\n')[22:41][0:-4] + 'e' + data_content.strip('\n')[23:41][-3:]).strip(' '))
            n_dic['C_us'] = float(
                (data_content.strip('\n')[41:60][0:-4] + 'e' + data_content.strip('\n')[42:60][-3:]).strip(' '))
            n_dic['sqrt_A'] = float(
                (data_content.strip('\n')[60:79][0:-4] + 'e' + data_content.strip('\n')[61:79][-3:]).strip(' '))
        if i == 3:
            n_dic['TEO'] = float(
                (data_content.strip('\n')[4:22][0:-4] + 'e' + data_content.strip('\n')[4:22][-3:]).strip(' '))
            n_dic['C_ic'] = float(
                (data_content.strip('\n')[22:41][0:-4] + 'e' + data_content.strip('\n')[23:41][-3:]).strip(' '))
            n_dic['OMEGA'] = float(
                (data_content.strip('\n')[41:60][0:-4] + 'e' + data_content.strip('\n')[42:60][-3:]).strip(' '))
            n_dic['C_is'] = float(
                (data_content.strip('\n')[60:79][0:-4] + 'e' + data_content.strip('\n')[61:79][-3:]).strip(' '))

        if i == 4:
            n_dic['I_0'] = float(
                (data_content.strip('\n')[4:22][0:-4] + 'e' + data_content.strip('\n')[4:22][-3:]).strip(' '))
            n_dic['C_rc'] = float(
                (data_content.strip('\n')[22:41][0:-4] + 'e' + data_content.strip('\n')[23:41][-3:]).strip(' '))
            n_dic['w'] = float(
                (data_content.strip('\n')[41:60][0:-4] + 'e' + data_content.strip('\n')[42:60][-3:]).strip(' '))
            n_dic['OMEGA_DOT'] = float(
                (data_content.strip('\n')[60:79][0:-4] + 'e' + data_content.strip('\n')[61:79][-3:]).strip(' '))
        if i == 5:
            n_dic['IDOT'] = float(
                (data_content.strip('\n')[4:22][0:-4] + 'e' + data_content.strip('\n')[4:22][-3:]).strip(' '))
            n_dic['L2_code'] = float(
                (data_content.strip('\n')[22:41][0:-4] + 'e' + data_content.strip('\n')[23:41][-3:]).strip(' '))
            n_dic['PS_week_num'] = float(
                (data_content.strip('\n')[41:60][0:-4] + 'e' + data_content.strip('\n')[42:60][-3:]).strip(' '))
            n_dic['L2_P_code'] = float(
                (data_content.strip('\n')[60:79][0:-4] + 'e' + data_content.strip('\n')[61:79][-3:]).strip(' '))
        if i == 6:
            n_dic['卫星精度(m)'] = float(
                (data_content.strip('\n')[4:22][0:-4] + 'e' + data_content.strip('\n')[4:22][-3:]).strip(' '))
            n_dic['卫星健康状态'] = float(
                (data_content.strip('\n')[22:41][0:-4] + 'e' + data_content.strip('\n')[23:41][-3:]).strip(' '))
            n_dic['TGD'] = float(
                (data_content.strip('\n')[41:60][0:-4] + 'e' + data_content.strip('\n')[42:60][-3:]).strip(' '))
            n_dic['IODC'] = float(
                (data_content.strip('\n')[60:79][0:-4] + 'e' + data_content.strip('\n')[61:79][-3:]).strip(' '))

    n_dic_list.append(n_dic)

with open('../data/my-exp2/20200923.csv', 'a+', newline='') as f:
    header = ['数据组数', '卫星PRN号', '历元', '卫星钟偏差(s)', '卫星钟漂移(s/s)', '卫星钟漂移速度(s/s*s)', 'IODE', 'C_rs', 'n', 'M0', 'C_uc',
              'e', 'C_us', 'sqrt_A', 'TEO', 'C_ic', 'OMEGA', 'C_is', 'I_0', 'C_rc', 'w', 'OMEGA_DOT', 'IDOT', 'L2_code',
              'PS_week_num', 'L2_P_code', '卫星精度(m)', '卫星健康状态', 'TGD', 'IODC']
    writer = csv.DictWriter(f, fieldnames=header)
    # writer.writeheader()
    writer.writerows(n_dic_list)
f.close()

with open('../data/my-exp2/20200923.csv', 'a+') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['数据组数'] == '1':
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
            TEO = float(row["TEO"])
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
            print(IODC)
            print(sqrt_A)

        # 卫星位置的计算




t1 = None
# CulLocation(PRN,t, a_0, a_1, a_2, IDOT, C_rs, δn, M0, C_uc, e, C_us, sqrt_A, TEO,
#             C_ic, OMEGA, C_is, I_0, C_rc, w, OMEGA_DOT)
