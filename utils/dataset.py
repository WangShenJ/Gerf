# -*- coding: utf-8 -*-
"""estimate the position and pose of array
"""
import json
import os
import sys
import time

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import matplotlib.image as plm
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sconst
import scipy.io as scio
import torch as tr
from pymongo import MongoClient
from tqdm import tqdm

from utils.msic import Bartlett, Kalman2D, rigid_transform_3D
from utils.msic import ParamLoader as PM

normalization = lambda data: (data-np.min(data)) / (np.max(data) - np.min(data))

def get_db():
    """Connect to MongoDB"""
    try:
        client = MongoClient('mongodb://tagsys:tagsys@127.0.0.1:27017/')
        # client = MongoClient('mongodb://tagsys:tagsys@158.132.255.205:27017/')
        db = client.LRT  # tabledatas
    except:
        raise Exception("can't connect to mongodb")
    return db



def gen_dataset(datadir, scene, mode="ss") -> None:
    """generate dataset. Contain two parts: heatmap, tag position.

    Parameters:
        mode: ss; signal
    """

    db = get_db()

    heatmap_path = os.path.join(datadir,"heatmap")
    os.makedirs(heatmap_path, exist_ok=True)

    kalman_workers = {'gateway1':[Kalman2D() for _ in range(16)], 'gateway2':[Kalman2D() for _ in range(16)],
                      'gateway3':[Kalman2D() for _ in range(16)]}

    gateway_names = PM.gateway_names
    atnoffsets = {'gateway1':PM.ant1offsets, 'gateway2':PM.ant2offsets,'gateway3':PM.ant3offsets}
    bartlett_workers = {gateway_name:Bartlett() for gateway_name in gateway_names}


    # phase1_before = np.array([])
    # phase1_after = np.array([])

    num_all, num_3= 0, 1
    last_pos = [10,10,10]

    col = db[scene]                  # tabledatas
    data = col.find({}, {'_id': 0}).sort([("phyTime",1)]).allow_disk_use(True)
    datalen = data.count()
    labels = np.zeros((datalen,3))
    if mode == "signal":
        signals = {"gateway%d"%i: np.zeros((datalen,16,2)) for i in range(1, 4)}
    for each in tqdm(data, total=datalen):
        num_all += 1
        # if num_all % 3 != 0:
        #     continue

        gateways = each['xServer'][0]['gateways']
        target_pos = each['truth']
        dis = np.linalg.norm((np.array(target_pos)-np.array(last_pos)))

        if dis < 0.005:
            continue

        last_pos = target_pos
        target_pos = np.array(target_pos).reshape(1,3)
        psp_dict = {}
        if len(gateways)==3:
            labels[num_3-1] = target_pos

            for name in gateways:
                gateway = gateways[name]
                atn_phase = -np.array(gateway['phase'])    # try to reverse
                rss = np.array(gateway['rss'])   # (16,)
                atn_amp = 10 ** ((rss + 17) / 20)

                kalman_worker = kalman_workers[name]       # list
                atn_iq = np.exp(1j * atn_phase)
                atn_phase = [kalman_worker[i].kalmanPredict(np.array([atn_iq[i].real, atn_iq[i].imag], np.float32))for i in range(16)]
                atn_phase = np.array(atn_phase, np.float64)
                atn_real,atn_imag = atn_phase[:,0,0],atn_phase[:,1,0]
                atn_phase = np.angle(atn_real + 1j*atn_imag)

                phase_cali = atn_phase.reshape(1, -1)
                # if name != "gateway3":
                #     dataoffset = dataoffset[:, PM.atnseq].transpose()
                # elif name == "gateway3":
                phase_cali = phase_cali[:, PM.atnseq_ours].transpose()    # (16,1)
                atn_amp = atn_amp[PM.atnseq_ours]

                if mode == "ss":
                    psp = bartlett_workers[name].get_aoa_heatmap(phase_cali)
                    psp_dict.update({name:normalization(psp)})
                elif mode == "signal":
                    # signal = atn_amp * np.exp(1j * phase_cali.flatten())  # np.complex128 (16,)
                    signals[name][num_3-1,:,0] = atn_amp
                    signals[name][num_3-1,:,1] = phase_cali.flatten()

            if mode == "ss":
                psp_all = np.concatenate((psp_dict['gateway1'].reshape(90,360,1),psp_dict['gateway2'].reshape(90,360,1),psp_dict['gateway3'].reshape(90,360,1)),axis=2)
                plm.imsave(heatmap_path + "/%05d.png"%num_3, psp_all)

            num_3 += 1

    # Y-up to Z-up
    labels = labels[:,[2,0,1]]
    labels = labels[:num_3-1,:]

    mat = {}
    if mode == "signal":
        for name, value in signals.items():
            savedir = os.path.join(datadir,name+".npy")
            np.save(savedir, value[:num_3-1,:])
            mat.update({name:value[:num_3-1,:]})
        scio.savemat(datadir+"/signals.mat",mat)

    np.savetxt(datadir + "/tagpos.txt", labels, delimiter=',',fmt='%.04f')




class ArrayEstimator():
    """
    """
    def __init__(self, dis:float, dataname:str) -> None:
        """_summary_

        Parameters
        ----------
        dis : float
            the distance of array relative to origin of global coordinate
        """
        self.radius = dis
        self.al_r = 360    # alpha resultion
        self.be_r = 30     # beta resultion

        _, scene, scenename = dataname.split("/")
        self.dataname = dataname
        self.scene = scene
        self.scenename = scenename[:-4]

        if not os.path.exists(dataname):
            os.makedirs("data/"+scene, exist_ok=True)
            self.dataloader(self.scenename)
        self.datadict = np.load(dataname, allow_pickle=True).item()


    def dataloader(self, scenename) -> None:
        """save phase data and position as npy
        """
        def get_db():
            """Connect to MongoDB"""
            try:
                # client = MongoClient('mongodb://tagsys:tagsys@127.0.0.1:27017/')
                client = MongoClient('mongodb://tagsys:tagsys@158.132.255.118:27017/')
                db = client.LRT  # tabledatas
            except:
                raise Exception("can't connect to mongodb")
            return db

        kalman_workers = {'gateway1':[Kalman2D() for _ in range(16)], 'gateway2':[Kalman2D() for _ in range(16)],
                        'gateway3':[Kalman2D() for _ in range(16)]}

        atnoffsets = {'gateway1':PM.ant1offsets, 'gateway2':PM.ant2offsets,'gateway3':PM.ant3offsets}

        db = get_db()
        # collist = db.list_collection_names()
        # collist.sort()

        num_all, num_3= 0, 1
        last_pos = [10,10,10]
        # for i,element in enumerate(collist):
        #     print('[%d]:'%i, element)
        # a = input("please input the collection index:")
        # col = db[collist[int(a)]]
        col = db[scenename]
        data = col.find({}, {'_id': 0}).sort([("phyTime",1)]).allow_disk_use(True)
        datalen = data.count()
        tag_pos = np.zeros((0,3))
        phase_dict = {'gateway%d'%i:np.zeros((0,16)) for i in range(1,4)}
        for each in tqdm(data, total=datalen):
            gateways = each['xServer'][0]['gateways']
            target_pos = each['truth']
            dis = np.linalg.norm((np.array(target_pos)-np.array(last_pos)))
            if dis < 0.005:
                continue
            last_pos = target_pos
            target_pos = np.array(target_pos).reshape(1,3)
            if len(gateways) == 3:
                tag_pos = np.vstack((tag_pos, target_pos))

                for name in gateways:
                    gateway = gateways[name]
                    atn_phase = gateway['phase']
                    kalman_worker = kalman_workers[name]
                    atn_iq = np.exp(1j * np.array(atn_phase))
                    atn_phase = [kalman_worker[i].kalmanPredict(np.array([atn_iq[i].real, atn_iq[i].imag], np.float32))for i in range(16)]
                    atn_phase = np.array(atn_phase, np.float64)
                    atn_real = atn_phase[:,0,0]
                    atn_imag = atn_phase[:,1,0]
                    atn_phase = np.angle(atn_real + 1j*atn_imag)
                    dataoffset = atn_phase.reshape(1, -1)

                    # if name != "gateway3":
                        # dataoffset = dataoffset[:, PM.atnseq]
                    # elif name == "gateway3":
                    dataoffset = dataoffset[:, PM.atnseq_ours]

                    phase_dict[name] = np.vstack((phase_dict[name], dataoffset))

        tag_pos = tag_pos[:,[2,0,1]]
        phase_dict.update({'tag':tag_pos})
        np.save(self.dataname, phase_dict)


    def sch_ps_fast(self,gateway:str) -> np.ndarray:
        """calculate the position of antenna element

        Parameters
        ----------
        gateway : str
            gateway name

        Returns
        -------
        e_coor:np.ndarray
            # (3,16) element coor
        """
        t1 = time.time()

        alphas = tr.linspace(0, 360-360/self.al_r, self.al_r) / 180 * np.pi
        betas = tr.linspace(0,30-30/self.be_r,self.be_r) / 180 * np.pi
        alphas = alphas.repeat_interleave(self.be_r)
        betas = betas.tile(self.al_r)

        x = self.radius * tr.cos(alphas) * tr.cos(betas)  # (10800,)
        y = self.radius * tr.sin(alphas) * tr.cos(betas)
        z = self.radius * tr.sin(betas)


        l1 = tr.tensor([-0.24, -0.08, 0.08, 0.24])
        l2 = tr.tensor([0.24, 0.08, -0.08, -0.24])
        l1 = l1.repeat_interleave(4)                           # (16,)
        l2 = l2.tile(4)

        alphas = alphas.expand(16,self.al_r*self.be_r).T     # (10800,16)
        betas = betas.expand(16,self.al_r*self.be_r).T

        ex = x + (l1*tr.sin(betas)*tr.cos(alphas) - l2*tr.sin(alphas)).T
        ey = y + (l1*tr.sin(betas)*tr.sin(alphas) + l2*tr.cos(alphas)).T    # (16,10800)
        ez = z - (l1*tr.cos(betas)).T

        ex, ey, ez = ex.reshape(16,1,self.al_r*self.be_r), ey.reshape(16,1,self.al_r*self.be_r),\
                     ez.reshape(16,1,self.al_r*self.be_r) # (16,1,10800)
        e_coor = tr.cat((ex,ey,ez),dim=1).permute(2,0,1) # (10800,16,3)

        # calculate theory phase
        lamda = sconst.c / 915e6
        batchsize = 400
        lens = self.datadict['tag'].shape[0]
        iteration = lens // batchsize
        remain = lens % batchsize
        vec_sum = tr.zeros(self.al_r*self.be_r, 16, dtype=tr.complex128)

        for i in range(iteration+1):
            if i == iteration:
                tagpos = self.datadict['tag'][i*batchsize:i*batchsize+remain]         # (bs,3)
                g1_phase = -self.datadict[gateway][i*batchsize:i*batchsize+remain]  #(bs,16)
                batchsize = remain
            else:
                tagpos = self.datadict['tag'][i*batchsize:i*batchsize+batchsize]         # (bs,3)
                g1_phase = -self.datadict[gateway][i*batchsize:i*batchsize+batchsize]  #(bs,16)
            e_coor1 = e_coor.expand(batchsize,360*30,16,3).permute(1,0,2,3)    #(10800,bs,16,3)
            tagpos = tr.from_numpy(tagpos).expand(16,batchsize,3).permute(1,0,2)          #(bs,16,3)
            g1_phase = tr.from_numpy(g1_phase).expand(360*30,batchsize,16)

            if tr.cuda.is_available():
                vec_sum = vec_sum.cuda()
                e_coor1 = e_coor1.cuda()
                tagpos = tagpos.cuda()
                g1_phase = g1_phase.cuda()
            temp = tr.norm(e_coor1-tagpos, dim=3)         # dis (10800,bs,16)
            temp = (temp / lamda * 2*np.pi) % (2*np.pi)   # theory_phase (10800,bs,16)
            temp -= temp[...,:1]                          # theory_phase_diff (10800,bs,16)
            g1_phase -= g1_phase[...,:1]
            temp = tr.exp(1j*(g1_phase - temp))   # vecs (10800,bs,16)
            vec_sum += tr.sum(temp,dim=1)         #(10800,16)

        # temp = tr.abs(vec_sum)
        totals = tr.sum(tr.abs(vec_sum),dim=1)   #(10800,)
        totals = totals.cpu()
        ind = tr.where(totals==tr.max(totals))[0]
        hat_alpha = ind // 30
        hat_beta = ind % 30
        print('estimated alpha:%f, beta:%f'%(hat_alpha,hat_beta))
        t2 = time.time()
        print('time: %f'%(t2-t1))

        # save element position
        hat_alpha = hat_alpha / 180 * np.pi
        hat_beta = hat_beta / 180 * np.pi


        x = self.radius * tr.cos(hat_alpha) * tr.cos(hat_beta)  # (10800,)
        y = self.radius * tr.sin(hat_alpha) * tr.cos(hat_beta)
        z = self.radius * tr.sin(hat_beta)

        ex = x + l1*tr.sin(hat_beta)*tr.cos(hat_alpha) - l2*tr.sin(hat_alpha)
        ey = y + l1*tr.sin(hat_beta)*tr.sin(hat_alpha) + l2*tr.cos(hat_alpha)    # (16,10800)
        ez = z - l1*tr.cos(hat_beta)

        e_coor = tr.vstack((ex,ey,ez)).numpy()    # (3,16)
        tr.cuda.empty_cache()

        return e_coor


def update_gateway_postion(scene, gateway, dis, debugdir) -> None:
    """Update gateway position

    Parameters
    ----------
    scene : str. experiment data config filename
    gateway : str. gateway1|gateway2|gateway3
    dis : float. The distance of antenna array
    """
    jsonname = "./configs/"+scene+".json"
    with open(jsonname,'r') as f:

        configs = json.load(f)
        scenename = configs["scenename"]
        filename = "data/" + scene + "/" + scenename + ".npy"
        worker = ArrayEstimator(dis,filename)
        print("estimate array postion...")
        gcs_coor = worker.sch_ps_fast(gateway)    # (3,16)
        lcs_coor = np.array([[-0.24, -0.08, 0.08, 0.24, -0.24, -0.08, 0.08, 0.24,
                            -0.24, -0.08, 0.08, 0.24, -0.24, -0.08, 0.08, 0.24],
                            [0.24, 0.24, 0.24, 0.24, 0.08, 0.08, 0.08, 0.08,
                            -0.08, -0.08, -0.08, -0.08, -0.24, -0.24, -0.24, -0.24],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            ])

        # position of the center of the array
        gcs_coor = gcs_coor.astype(np.float64)
        # scio.savemat('data/test2.mat',{"pos":element_coor})

        r_o = np.mean(gcs_coor, axis=1)
        r_o = np.round(r_o, 4)
        R,t = rigid_transform_3D(lcs_coor, gcs_coor)
        R, t = np.round(R, 4), np.round(t, 4)

        element_coor = R@lcs_coor + t

        scio.savemat(debugdir+"/gateway1_pos.mat",{"pos":element_coor})

        configs["gateway1"] = {"dis": dis, "r_o": r_o.tolist(), "R": R.tolist(), "t": t.tolist()}
        f.close()

    with open(jsonname,'w') as f:
        json.dump(configs, f, indent=2)
        f.close()



if __name__ == '__main__':

    # gateways = ['gateway1', 'gateway2', 'gateway3']
    # filename = 'data/gf_05m_0710.npy'
    # worker = ArrayEstimator(5,filename)
    # data_dict = {}
    # for gateway in gateways:
    #     ans = worker.sch_ps_fast(gateway)
    #     data_dict[gateway] = ans
    # scio.savemat(filename[:-4]+'.mat',data_dict)
    # update_gateway_postion("s31-exp2","gateway1", 5.)
    gen_dataset("data/s23-exp3", "pq504_05m_0622","signal")

