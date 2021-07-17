'''
Description: the user-defined shapelets function
Version: 2.0
Autor: mario
Date: 2020-09-24 16:25:13
LastEditors: Please set LastEditors
LastEditTime: 2021-07-17 18:04:02
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import re
import time

import joblib
import torch
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from src import utils
from shapelets.shapelet_reader import ShapeletReader, ShapeletWriter

from shapelets import shapeletSearchModel as SearchModel
from shapelets import shapeletNetModel as NetModel


class ShapeletsLearning():
    def __init__(self, premotionpath, samples_filepath):
        self.samples_filepath = samples_filepath
        self.premotionfilepath = premotionpath
        self.premotionfile = h5py.File(premotionpath, 'r')

        # 数据特征类型
        self.datamode = 'posehand'
        self.featuremode = 1
        self.normmode = False
        self.process_id = 0

        # m_len level
        self.min_m, self.max_m, self.stride_m = 15, 40, 2

        # max sample 
        self.max_samplenum = 100
        # 选取样本时候的拓宽范围
        self.delayfx = 1.5

    def Getsamples(self, word):
        ''' get the samples from the instance file
        args:
            word: str, the given word
        '''
        # sample_indexes 的格式为：[videokey(str), begin, end, label]
        with h5py.File(self.samples_filepath, mode='r') as samplefile:
            sample_indexes = samplefile[word][:].astype(np.int64)
        
        # remove the too short instance
        sample_indexes = [x for x in sample_indexes if (x[2] - x[1]) > self.max_m]
        
        pos_indexes = [['%03d' % x[0]] + list(x[1:]) for x in sample_indexes if x[-1] == 1]
        neg_indexes = [['%03d' % x[0]] + list(x[1:]) for x in sample_indexes if x[-1] == 0]

        # make sure the positive instances have the same num as negative instances
        pos_indexes = pos_indexes[:min(self.max_samplenum, len(pos_indexes))]
        neg_indexes = neg_indexes[:min(self.max_samplenum, len(neg_indexes))]
        
        print(f'the word {word} has {len(pos_indexes)} positive instance samples')
        sample_indexes = pos_indexes + neg_indexes
        
        # get the motiondata from the premotionfile
        samples = []
        for i in range(len(sample_indexes)):
            videokey, beginindex, endindex = sample_indexes[i][:3]
            clip_data = self.ExtractMotionClips(videokey, beginindex, endindex)
            samples.append(clip_data)

        return samples, sample_indexes
    
    def ExtractMotionClips(self, videokey, beginindex, endindex):
        datakey = '%s-%d/%s' % (self.datamode, self.featuremode, videokey)
        clip_data = self.premotionfile[datakey][beginindex:endindex]
        return clip_data

    def train(self, words, method, outputpath, ref_file_path=None, rewrite=False):
        # get the wanted m_levels
        write_mlen_list = list(range(self.min_m, self.max_m, self.stride_m))
        writeshapeleter = ShapeletWriter(outputpath)

        # iterate for all the words
        for idx, word in enumerate(words):
            # 当“不重写” 以及 “都全部存在“ 的时候进入下一个 word
            if (not rewrite) and writeshapeleter.checkout_level_exists(word, write_mlen_list):
                continue
            
            print('start to train the word: %s (%d/%d)' % (word, idx, len(words)))
            samples, sample_indexes = self.Getsamples(word)
            labels = torch.tensor([x[-1] for x in sample_indexes])

            # 为了保证完整的信息，这里将会把 pos_samples 准确的位置信息进行记录
            writeshapeleter.WriteInstanceInfo(word, sample_indexes, self.delayfx,
                                              self.datamode, self.featuremode)

            # Do the training loop
            X = torch.cat([torch.from_numpy(x).permute(1, 0).unsqueeze(0) for x in samples], dim=-1)
            Y = torch.clone(labels)
            lengths = [len(x) for x in samples]
            
            if torch.has_cuda:
                X = X.cuda()
                Y = Y.cuda()

            for i, m_len in enumerate(write_mlen_list):
                if method == 1:
                    self.FindShaplets_Net_ED(word, X, Y, lengths, m_len, ref_file_path, outputpath)
                elif method == 2:
                    self.FindShaplets_brute_force_ED(word, X, Y, lengths, m_len, outputpath)

    # 使用蛮力 matrix profile 的方式进行 shapelet 的 finding
    def FindShaplets_brute_force_ED(self, word, X, Y, lengths, m_len, outputpath):
        begin_time = time.time()
        
        model = SearchModel.ShapeletMatrixModel()
        model.train(X, Y, lengths, m_len)

        # get the training results
        locs = model.locs[Y == 1].astype(np.int16)
        dists = model.dists[Y == 1].astype(np.float32)
        shapelet = np.array([model.shapeindex]).astype(np.int16)
        score = np.array([model.score]).astype(np.float32)

        ShapeletWriter(outputpath).WriteShapeletInfo(word, m_len, shapelet, locs, dists, score)

        print('the %d word %s of %d length cost %f seconds with score %f' % (
            len(Y)//2, word, m_len, time.time()-begin_time, score[0]))

    # 使用 shapeletlearning net 的方式进行 shapelet 的 learning
    def FindShaplets_Net_ED(self, word, X, Y, lengths, m_len, ref_file_path, outputpath):
        begtime = time.time()

        # Set the default query
        default_query = self.GetInitShapelet(ref_file_path, word, m_len)
        default_query = torch.from_numpy(np.transpose(default_query)).unsqueeze(0)
        Net = NetModel.ShapeletNet(default_query)

        if torch.has_cuda:
            Net = Net.cuda()
        
        temp_model_path = './temp_shapelet_net_model_%d.pt' % self.process_id
        subdis, score = NetModel.shapeletNetlearning(Net, X, Y, lengths, temp_model_path)

        subdis = np.array([x for i, x in enumerate(subdis) if Y[i] == 1])
        dists = subdis[:, 0].astype(np.float32)
        locs = subdis[:, 1].astype(np.int16)
        shapelet = Net.query.detach().cpu().numpy()
        score = np.array([score], dtype=np.float32)

        ShapeletWriter(outputpath).WriteShapeletInfo(word, m_len, shapelet, locs, dists, score)
        print('the sl of word: {} with {} cost time {:.4f} with acc {:.2f}'.format(
            word, m_len, time.time()-begtime, score))

    def GetInitShapelet(self, ref_file_path, word, m_len):
        # 从 之前 SF 训练的结果中 提取 shapelet
        shapeleter = ShapeletReader(ref_file_path, self.premotionfilepath)
        if ('%s/%d' % (word, m_len)) in shapeleter.shapeletfile:
            shapeleter.GetInfos(word)
            shapeletdata = shapeleter.GetShapelet(str(m_len))
        else:
            shapeleter.ReleaseFile()
            print(f"the word {word}'s pretrain is not exist, it will be trained now")
            current_max = self.max_samplenum
            self.max_samplenum = 50
            self.train([word], method=2, outputpath=ref_file_path, rewrite=True)
            self.max_samplenum = current_max
            shapeletdata = self.GetInitShapelet(ref_file_path, word, m_len)
        return shapeletdata

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rewrite', action='store_true')
    parser.add_argument('--featuremode', type=int, default=2)
    parser.add_argument('-m', '--method', choices=[1, 2], type=int, default=2)
    parser.add_argument('-n', '--clsnum', type=int, default=100)

    args = parser.parse_args()
    
    rewrite = args.rewrite
    method = args.method
    featuremode = args.featuremode
    first_mul = args.firstmul
    clsnum = args.clsnum
    
    prefeaturefilepath = '../data/prefeaturedata.hdf5'
    sample_filepath = '../data/word_instance_samples_all.hdf5'
    wordsourcepath = '../data/word_statistics.txt'

    method_key = {1: 'net', 2: 'search'}
    words = utils.GetWantedWords(wordsourcepath, clsnum)
    clsnum = len(words)

    max_items = 50 if method == 2 else 300
    
    outputpath = f'./output/shapelet_{method_key[method]}_{clsnum}.hdf5'
    sf_file_path = f'./output/shapelet_{method_key[2]}_{clsnum}.hdf5'

    shapelearner = ShapeletsLearning(prefeaturefilepath, sample_filepath)
    shapelearner.featuremode = featuremode
    shapelearner.max_samplenum = max_items
    shapelearner.train(words, method, outputpath, sf_file_path, rewrite)