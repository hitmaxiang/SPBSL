import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import torch
import h5py
import numpy as np

from src import utils
from shapelets.shapelet_reader import ShapeletReader, ShapeletWriter


def Get_all_target_signs(shapelet_file_path, premotion_path, instancefile_path, outputpath, rewrite=False):
    ''' Find all the target signs from all positive samples
    it equal to expand the shapeletfile with more data
    '''
    
    shapeleter = ShapeletReader(shapelet_file_path, premotion_path)
    shapeletfile = h5py.File(shapelet_file_path, 'r')
    premotionfile = h5py.File(premotion_path, 'r')
    instancefile = h5py.File(instancefile_path, 'r')

    words = list(shapeletfile.keys())
    
    for ite, word in enumerate(words):
        # 首先获取shapelet 的相关信息
        shapeleter.GetInfos(word)
        if rewrite is False and os.path.exists(outputpath):
            target_shapeleter = ShapeletReader(outputpath)
            if word in target_shapeleter.shapeletfile.keys():
                target_shapeleter.GetInfos(word)
                if target_shapeleter.featurekey == shapeleter.featurekey:
                    if target_shapeleter.levelkeys == shapeleter.levelkeys:
                        continue
            # if need to write, output handle should be release first
            target_shapeleter.ReleaseFile()
            
        if len(shapeleter.levelkeys) == 0:
            print('the shapelet of the word %s is empty' % word)

        featurekey = shapeleter.featurekey
        levelkeys = shapeleter.levelkeys
        datamode, featuremode = shapeleter.datamode, shapeleter.featuremode

        # get all samples
        sample_indexes = instancefile[word][:].astype(np.int64)
        pos_indexes = [['%03d' % x[0]] + list(x[1:]) for x in sample_indexes if x[-1] == 1]

        # 录入正样本的信息
        ShapeletWriter(outputpath).WriteInstanceInfo(word, pos_indexes, 1.5, datamode, int(featuremode))

        # 根据索引提取所有的样本数据
        samples = []
        for i in range(len(pos_indexes)):
            vidkey, begidx, endidx = pos_indexes[i][:3]
            clip_data = premotionfile[featurekey][vidkey][begidx:endidx]
            samples.append(clip_data)

        # transfer to the tensor format
        cumlens = np.insert(np.cumsum([len(x) for x in samples]), 0, 0)
        catsamples = torch.cat([torch.from_numpy(x).float() for x in samples], dim=0)

        for level in levelkeys:
            shrink = 1 - int(level)
            # 获取 shapelet 的数据
            shapelet = shapeleter.GetShapelet(level)
            # 因为后面需要使用 shapelet 这里采用不共享内存的方式进行 tensor 转换
            shapeletdata = torch.tensor(shapelet).float()

            if torch.has_cuda:
                shapeletdata = shapeletdata.cuda()
                catsamples = catsamples.cuda()

            subdis = torch.zeros(len(samples), 2, dtype=torch.float32, device=shapeletdata.device)

            with torch.no_grad():
                if len(shapeletdata.shape) == 2:
                    sliding_dis = utils.SlidingDistance_torch(shapeletdata, catsamples)
                elif len(shapeletdata.shape) == 3:
                    sliding_dis = utils.SlidingDistanceSquare(shapeletdata, catsamples)
                
                for i in range(len(samples)):
                    clipdis = sliding_dis[cumlens[i]:cumlens[i+1]+shrink]
                    subdis[i] = torch.tensor(torch.min(clipdis, dim=0))
            subdis = subdis.cpu().numpy()

            # store the shapelet infomation
            dists = subdis[:, 0].astype(np.float32)
            locs = subdis[:, 1].astype(np.int16)
            score = np.array([shapeleter.score], dtype=np.float32)
            ShapeletWriter(outputpath).WriteShapeletInfo(word, int(level), shapelet, locs, dists, score)
            
        print('finish the identification of word [%d/%d] %s' % (ite, len(words), word))


if __name__ == '__main__':
    origin_names = ['net', 'search']
    origin = 'net'
    rewrite = True
    shapelet_path = {'net': '../data/shapelet_net_all.hdf5'}
    shapelet_path['search'] = '../data/shapelet_search_all.hdf5'

    premotionfilepath = '../data/prefeaturedata.hdf5'
    instancefilepath = '../data/word_instance_samples_all.hdf5'

    outputpath = 'output/target_signs_%s.hdf5' % origin

    Get_all_target_signs(shapelet_path[origin], premotionfilepath, instancefilepath, outputpath, rewrite)