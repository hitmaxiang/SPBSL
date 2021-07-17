'''
Author: your name
Date: 2021-07-16 14:46:36
LastEditTime: 2021-07-16 20:43:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Public_SPBSL/openpose/motion_models.py
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import math
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from skimage.measure import label
from torch.utils.data import Dataset, DataLoader

from openpose.models import handpose_model, bodypose_model
from openpose import utils as pose_util


class VideoDataset(Dataset):
    def __init__(self, videopath, cropRec=None, transform=None):
        super().__init__()
        self.video = cv2.VideoCapture()
        self.cropRec = cropRec
        self.transform = transform
    
    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __getitem__(self, index):
        ''' In order to make the imgs read fast, the frames are read as stream
        the shuffle of dataloader should be set to false
        '''
        _, img = self.video.read()
        if self.cropRec is not None:
            Rec = self.cropRec
            img = img[Rec[0][1]:Rec[1][1], Rec[0][0]:Rec[1][0], :]
        if self.transform is not None:
            img = self.transform(img)


class HandImageDataset(Dataset):
    def __init__(self, videopath, posedata, boxsize, recpoints=None):
        super().__init__()
        self.video = cv2.VideoCapture(videopath)
        self.posedata = posedata
        self.boxsize = boxsize
        self.rec = recpoints

        self.framescount = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        assert self.framescount == len(self.posedata)
        
    def __len__(self):
        return self.framescount
    
    def __getitem__(self, index):
        _, img = self.video.read()
        if self.rec is not None:
            img = img[self.rec[0][1]:self.rec[1][1], self.rec[0][0]:self.rec[1][0], :]
        
        # construct the subset according to posedata
        pose = self.posedata[index]
        subset = np.zeros((1, 20))
        candidate = np.zeros((20, 4))

        for i in range(18):
            subset[0, i] = -1 if sum(pose[i]) == 0 else i
            candidate[i, :2] = pose[i, :2]
        
        hands_list = pose_util.handDetect(candidate, subset, img.shape[:2])

        # the default hands
        lefthand = np.zeros((self.boxsize, self.boxsize, 3), dtype=np.uint8) + 128
        righthand = np.zeros_like(lefthand) + 128
        leftparams, rightparams = np.zeros(3), np.zeros(3)

        for x, y, w, is_left in hands_list:
            if not is_left:  # right hand
                tempimg = img[y:y+w, x:x+w]
                righthand = cv2.resize(tempimg, (self.boxsize, self.boxsize), interpolation=cv2.INTER_CUBIC)
                rightparams = np.array([x, y, w])
            else:
                tempimg = cv2.flip(img[y:y+w, x:x+w], 1)
                lefthand = cv2.resize(tempimg, (self.boxsize, self.boxsize), interpolation=cv2.INTER_CUBIC)
                leftparams = np.array([x, y, w])
        
        righthand = torchvision.transforms.ToTensor()(righthand)
        lefthand = torchvision.transforms.ToTensor()(lefthand)

        return lefthand, leftparams, righthand, rightparams


class Body_Estimation_Model():
    def __init__(self, model_path):
        self.model = bodypose_model()
        if torch.has_cuda:
            torch.cuda.empty_cache()
            self.model = self.model.cuda()
        model_dict = pose_util.modeltransfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

        # define the global variables
        self.scale_search = 0.5
        self.boxsize = 368
        self.stride = 8
        self.padvalue = 128
        self.thre1, self.thre2 = 0.1, 0.05

        # find connection in the specified sequence, center 29 is in the position 15
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
                        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
                        [1, 16], [16, 18], [3, 17], [6, 18]]
        # the middle joints heatmap correpondence
        self.mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
                       [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
                       [55, 56], [37, 38], [45, 46]]
        
        self.gaussian_filter_conv = pose_util.GaussianBlurConv(19)
        if torch.has_cuda:
            self.gaussian_filter_conv = self.gaussian_filter_conv.cuda()
    
    def __call__(self, imgs):
        B, C, H, W = imgs.size()
        scale, new_h, new_w, pad_h, pad_w = self.calculate_size_pad(self.scale_search, H, W)

        with torch.no_grad():
            imgs = F.interpolate(imgs, scale_factor=scale, mode='bicubic')
            imgs -= 0.5
            imgs = F.pad(imgs, [0, pad_w, 0, pad_h], mode='constant', value=0)
            paf, heatmap = self.model(imgs)

            heatmap = F.interpolate(heatmap, scale_factor=self.stride, mode='bicubic')
            heatmap = heatmap[:, :, :new_h, :new_w]
            heatmap = F.interpolate(heatmap, size=(H, W), mode='bicubic')

            paf = F.interpolate(paf, scale_factor=self.stride, mode='bicubic')
            paf = paf[:, :, :new_h, :new_w]
            paf = F.interpolate(paf, size=(H, W), mode='bicubic')

            heatmap = self.gaussian_filter_conv(heatmap)
            peaks = pose_util.findpeaks_torch(heatmap[:, :-1, :, :], self.thre1)
        
            heatmap = heatmap.cpu().numpy()
            peaks = peaks.cpu().numpy()
        
        heatmap = np.transpose(heatmap, [0, 2, 3, 1])
        paf = np.transpose(paf, [0, 2, 3, 1])

        all_peaks = [[[] for i in range(18)] for j in range(B)]
        b_num = None
        counter = 0
        for b_i, c_i, h_i, w_i in peaks:
            if b_i != b_num:
                b_num = b_i
                counter = 0
            else:
                counter += 1
            all_peaks[b_i][c_i].append((w_i, h_i, heatmap[b_i, h_i, w_i, c_i], counter))
        
        results = []
        for batch_id in range(len(heatmap)):
            candidates, subset = self.FindBody_frame(heatmap, paf, all_peaks)
            results.append((candidates, subset))
        
        return results
    
    def FindBody_frame(self, heatmap, paf, all_peaks):
        connection_all = []
        special_k = []
        mid_num = 10
        mapIdx = self.mapIdx
        limbSeq = self.limbSeq
        for k in range(len(mapIdx)):
            score_mid = paf[:, :, [x - 19 for x in mapIdx[k]]]
            candA = all_peaks[limbSeq[k][0] - 1]
            candB = all_peaks[limbSeq[k][1] - 1]
            nA = len(candA)
            nB = len(candB)
            indexA, indexB = limbSeq[k]
            if (nA != 0 and nB != 0):
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        # add a small value to avoid divide error
                        norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]) + 1e-10  
                        vec = np.divide(vec, norm)

                        startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num),
                                            np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(startend[k][1])), int(round(startend[k][0])), 0]
                                          for k in range(len(startend))])
                        vec_y = np.array([score_mid[int(round(startend[k][1])), int(round(startend[k][0])), 1]
                                          for k in range(len(startend))])

                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                            0.5 * heatmap.shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_midpts > self.thre2)[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if (i not in connection[:, 3] and j not in connection[:, 4]):
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if (len(connection) >= min(nA, nB)):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        # print('the find-2 cost %f seconds' % (time.time()-begin_time))

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])

        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        # print('the find-3 cost %f seconds' % (time.time()-begin_time))

        # subset: n*20 array, 0-17 is the index in candidate, 18 is the total score, 19 is the total parts
        # candidate: x, y, score, id
        return candidate, subset
    
    def calculate_size_pad(self, g_scale, height, width):
        scale = self.boxsize * g_scale/height
        h, w = int(height*scale), int(width*scale)
        pad_h = (self.stride - (h % self.stride)) % self.stride
        pad_w = (self.stride - (w % self.stride)) % self.stride
        
        return (scale, h, w, pad_h, pad_w)


class Hand_Estimation_Model():
    def __init__(self, model_path):
        self.model = handpose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        model_dict = pose_util.modeltransfer(self.model, torch.load(model_path))
        self.model.load_state_dict(model_dict)
        self.model.eval()

        # self.scale_search = [0.5, 1.0, 1.5, 2.0]
        self.scale_search = [1.0]
        self.boxsize = 368
        self.stride = 8
        self.padValue = 128
        self.thre = 0.035

        self.guassian_filter_conv = pose_util.GaussianBlurConv(22)
        if torch.cuda.is_available():
            self.guassian_filter_conv.cuda()

    def __call__(self, batch_imgs):

        batch_size, c, h, w = batch_imgs.size()

        if torch.cuda.is_available():
            batch_imgs = batch_imgs.cuda()

        with torch.no_grad():
            batch_imgs = batch_imgs - 0.5
            heatmap = self.model(batch_imgs)
            del batch_imgs

            heatmap = F.interpolate(heatmap, scale_factor=self.stride, mode='bicubic')
            heatmap = self.guassian_filter_conv(heatmap)
            
            heatmap = heatmap.cpu().numpy()
            torch.cuda.empty_cache()
            heatmap = np.transpose(heatmap, (0, 2, 3, 1))
        
        Batch_peaks = []
        for i in range(batch_size):
            all_peaks = []
            for part in range(21):
                map_ori = heatmap[i, :, :, part]
                binary = map_ori > self.thre
                # 全部小于阈值
                if np.sum(binary) == 0:
                    all_peaks.append([0, 0, 0])
                    continue

                label_img, label_numbers = label(binary, return_num=True, connectivity=binary.ndim)
                max_index = np.argmax([np.sum(map_ori[label_img == i]) for i in range(1, label_numbers + 1)]) + 1
                map_ori[label_img != max_index] = 0

                y, x = pose_util.npmax(map_ori)
                all_peaks.append([x, y, np.max(map_ori)])
            Batch_peaks.append(all_peaks)
        return np.array(Batch_peaks)