import json
import os

import numpy as np
import scipy.io as scio
import torch
import torch.utils.data
from tqdm import tqdm
import matplotlib.image as plm

normalization = lambda data: (data-np.min(data)) / (np.max(data) - np.min(data))

class RayDataset(torch.utils.data.Dataset):

    def __init__(self, datadir, config, batchsize, debugdir=None, mode='train', output="ss"):

        self.load_dataset_info(datadir, config, batchsize, debugdir, mode, output)
        datalen = len(self.IDList)

        if output == "ss":
            self.heatmap_path = os.path.join(datadir,"heatmap")
            self.datasets = torch.zeros((datalen, 90*360, 10))
            current_point = 0
            for i,_ in tqdm(enumerate(self.IDList), total=datalen):
                if output == "ss":
                    ind = int(self.IDList[i])
                    name = datadir+"/heatmap/%s.png"%self.IDList[i]
                    img = plm.imread(name)[...,:3]    # (90,360,3)
                    gateway1 = img[:,:,0]             # R channel
                    gateway1 = np.reshape(normalization(gateway1), (-1,1))    # (90*360, 1)
                    gateway1 = torch.from_numpy(gateway1)
                    tag = self.tagpos[ind-1].expand(gateway1.shape[0], 3)
                    # temp = torch.cat([gateway1, tag], -1)[None,...]    # [1, 90*360, 4]
                    self.datasets[current_point:current_point+1,:,0:4] = torch.cat([gateway1, tag], -1)[None,...]
                    current_point += 1

            # self.datasets = torch.cat([self.datasets, self.ray.expand(self.datasets.shape[0], 90*360, 6)], -1) #[num,360*90,4+6]
            self.datasets[...,4:10] = self.ray.expand(self.datasets.shape[0], 90*360, 6)  #[num,360*90,4+6]
            self.datasets = torch.reshape(self.datasets, (-1, 10))  #[num*360*90,4+6]

        elif output == "signal":
            datapath = os.path.join(datadir, "gateway1.npy")
            signals = np.load(datapath)[self.IDList]  # (IDlist.len, 16, amp+phase)
            signals = torch.from_numpy(signals)
            signals = torch.reshape(signals, (-1,2))  # [IDlist.len * 16, amp+phase]
            tag = self.tagpos[self.IDList].expand(16,-1, 3).permute(1,0,2)# [IDlist.len*16, 3]
            tag = torch.reshape(tag,(signals.shape[0], 3))
            self.ray = self.ray.expand(datalen,16,-1).reshape((datalen*16, -1)) # [IDList.len*16, r_o+r_d]
            self.datasets = torch.cat([signals, tag, self.ray,],-1) # [IDlist.len * 16, amp+phase+3+r_o+r_d]
        if mode == 'train':
            ray_index = torch.randperm(self.datasets.shape[0])
            self.datasets = self.datasets[ray_index]    # random shuffle rays. [num*360*90,1+3+r_o+r_d]

        self.current_index = 0


    def load_dataset_info(self, datadir, config, batchsize, debugdir, mode, output):

        train_txt = datadir + "/train_" + output + ".txt"
        test_txt = datadir + "/test_" + output + ".txt"

        self.batchsize = batchsize
        self.train_index = np.loadtxt(train_txt, dtype=int)
        self.test_index = np.loadtxt(test_txt, dtype=int)
        self.all_index = np.hstack((self.train_index, self.test_index))
        self.tagpos = torch.from_numpy(np.loadtxt(os.path.join(datadir, 'tagpos.txt'),delimiter=','))

        if mode == "train" or mode == "test":
            r_o = np.array(config["gateway1"]["r_o"])    # (3,)
            R, t = np.array(config["gateway1"]["R"]), np.array(config["gateway1"]["t"])
            self.ray = self.get_ray(r_o, R, t, output)    # [90*360, r_o+r_d] or [16, r_o+r_d]
        elif mode == "scene_sphere":
            r_o = np.array([0,0,0])
            R = np.identity(3)
            t = np.array([[0.], [0.], [0.]])
            self.ray = torch.from_numpy(self.get_ray(r_o, R, t), output)    # (90*360, r_o+r_d)

        if debugdir is not None:
            scio.savemat(debugdir+"/ray.mat", {"rays":self.ray.numpy()})

        # all index ID
        if output == "ss":
            if mode == "train":
                self.IDList = [str(int(x)).zfill(5) for x in self.train_index]
            elif mode == "test":
                self.IDList = [str(int(x)).zfill(5) for x in self.test_index]
            elif mode == "scene_sphere":
                self.IDList = [str(int(x)).zfill(5) for x in self.test_index[0:1]]
        elif output == "signal":
            if mode == "train":
                self.IDList = self.train_index
            elif mode == "test":
                self.IDList = self.test_index


    @staticmethod
    def get_ray(r_o, R, t, mode="ss"):
        """get r_o,r_d

        Parameters
        ----------
        r_o: origin. (3,)

        Returns
        ----------
        ray: ss -> [90*360, r_o+r_d];  signal -> [16, r_o+3*360*60]
        """
        radius = 1
        if mode == "ss":
            beta_r = [0, 90]
        elif mode == "signal":
            beta_r = [30, 90]
        beta_len = int(beta_r[1] - beta_r[0])
        frac = 2

        alphas = np.linspace(0, 360-1, 360//frac) / 180 * np.pi
        betas = np.linspace(beta_r[0],beta_r[1]-1, beta_len//frac) / 180 * np.pi
        alphas = np.tile(alphas,beta_len//frac)    # [0,1,2,3,....]
        betas = np.repeat(betas,360//frac)    # [0,0,0,0,...]

        x = radius * np.cos(alphas) * np.cos(betas)  # (90*360)
        y = radius * np.sin(alphas) * np.cos(betas)
        z = radius * np.sin(betas)

        r_d = np.stack([x,y,z], axis=0)    # (3,90*360)
        r_d = (R@r_d + t).T               # (90*360, 3)
        r_o = np.tile(r_o, ((360//frac)*(beta_len//frac) ,1))   # (90*360, 3)
        r_d = torch.from_numpy(r_d - r_o)  # [90*360, 3]


        # if mode == "ss":
        #     ray = np.concatenate((r_o, r_d), axis = -1)
        #     return ray    # (90*360, r_o+r_d)
        if mode == "signal":
            r_d = r_d.flatten()    #[3*360*60,]
            r_d = r_d.expand([16,-1])    #[16, 3*360*60]
            lcs_coor = np.array([[-0.24, -0.08, 0.08, 0.24, -0.24, -0.08, 0.08, 0.24,
                    -0.24, -0.08, 0.08, 0.24, -0.24, -0.08, 0.08, 0.24],
                    [0.24, 0.24, 0.24, 0.24, 0.08, 0.08, 0.08, 0.08,
                    -0.08, -0.08, -0.08, -0.08, -0.24, -0.24, -0.24, -0.24],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    ])
            r_o = (R@lcs_coor + t).T    #(16, 3)

        r_o = torch.from_numpy(r_o)
        ray = torch.cat([r_o, r_d], axis= -1)
        return ray


    def Next(self):
        """_summary_

        Returns
        -------
        batchdata: [batchsize,1+3+r_o+r_d]
        """

        batchdata = self.datasets[self.current_index: self.current_index+self.batchsize]

        self.current_index += self.batchsize
        if self.current_index >= self.datasets.shape[0]:
            print("Shuffle data after an epoch!")
            ray_index = torch.randperm(self.datasets.shape[0])
            self.datasets = self.datasets[ray_index]
            self.current_index = 0

        return batchdata


    def EvalBatch(self):
        """_summary_

        Returns
        -------
        batchdata: [batchsize,1+3+r_o+r_d]
        """

        evalbatch = 120*90
        batchdata = self.datasets[self.current_index: self.current_index+evalbatch]

        self.current_index += evalbatch
        if self.current_index > self.datasets.shape[0]:
            self.current_index = 0
            raise Exception("Out of data!")

        return batchdata



    def __getitem__(self, index):
        return self.datasets[index]


    def __len__(self):
        return self.datasets.shape[0]



if __name__ == '__main__':
    with open("configs/s23.json") as f:
        configs = json.load(f)
        f.close()

    d = RayDataset("s23",configs)
    rays = d.get_ray(configs)
    scio.savemat("data/test.mat", {"rays": rays})
