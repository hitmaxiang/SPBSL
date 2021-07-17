import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import torch
import time
import argparse
import numpy as np

from src import utils
from shapelets.shapelet_reader import ShapeletReader


def GetAugSigns(shapelet_file_path, premotionfilepath, outputpath, scale, rewrite=False):
    '''Get the augmented signs from all level based on the scale fraction
    
    args:
        shapelet_file_path: the hdf5 file of the shapeletrecord
        premotionfilepath: the premotion file stored in hdf5 file
    '''
    print('start to augthedata of %s' % shapelet_file_path)
    premotionfile = h5py.File(premotionfilepath, 'r')
    shapletpatternsdict = GetAllShapeletpatterns(shapelet_file_path, premotionfilepath, scale)

    for featurekey in shapletpatternsdict.keys():
        # 获取在当前 feature 下的所有 patterns items
        shapeletpatterns = shapletpatternsdict[featurekey]

        # 根据当前的数据和选项判断是否需要写的数据（这个主要用来解决中断续问题）
        wantvideokeys = []
        if rewrite is True:
            # 重写的情况下,所有的video都需要
            wantvideokeys = list(premotionfile[featurekey].keys())
        elif os.path.exists(outputpath):
            with h5py.File(outputpath, mode='r') as targetfile:
                for vdk in premotionfile[featurekey].keys():
                    for shapelet in shapeletpatterns:
                        word, level, min_dist, sigma, shapeletdata = shapelet
                        datakey = '%s-%.2f/%s/%s/%s' % (featurekey, scale, word, level, vdk)
                        if datakey not in targetfile:
                            wantvideokeys.append(vdk)
                            break
        else:
            wantvideokeys = list(premotionfile[featurekey].keys())
        
        begtime = time.time()
        # 如果想要在GPU上面运行的话， 相比较于频繁转移premotion数据，操作 shapelet 数据更加经济一些
        for ite, videokey in enumerate(wantvideokeys):
            videodata = premotionfile[featurekey][videokey][:]
            videodata = torch.from_numpy(videodata)

            if torch.has_cuda:
                videodata = videodata.cuda()
            
            for j, shapelet in enumerate(shapeletpatterns):
                word, level, mindis, sigma, shapeletdata = shapelet
                datakey = '%s-%.2f/%s/%s/%s' % (featurekey, scale, word, level, videokey)

                # no overwrite
                if rewrite is False and os.path.exists(outputpath):
                    with h5py.File(outputpath, 'r') as f:
                        if datakey in f:
                            # print('%s is already exist!' % datakey)
                            continue
                
                shapeletdata = torch.from_numpy(shapeletdata)
                if torch.has_cuda:
                    shapeletdata = shapeletdata.cuda()
                
                # calcualting the sliding distance between shapelet and videodata
                with torch.no_grad():
                    if len(shapeletdata.shape) == 2:
                        dists = utils.SlidingDistance_torch(shapeletdata, videodata)
                    elif len(shapeletdata.shape) == 3:
                        dists = utils.SlidingDistanceSquare(shapeletdata, videodata)
                    else:
                        raise ValueError('the shape of shapeletdata is wrong!')
                    
                    indices = torch.nonzero(dists <= sigma, as_tuple=False)
                    dists = dists[indices]
                    data = torch.cat((dists, indices), dim=1).cpu().numpy().astype(np.float32)
                    data = PurgeData(data, int(level), tau=1)
                utils.WriteRecords2File(outputpath, datakey, data)
                # print('complete the datakey: %s' % datakey)

            # print the processbar information
            timecost = time.time() - begtime
            print('the %s (%d/%d) cost %.3f seconds' % (videokey, ite, len(wantvideokeys), timecost))
            begtime = time.time()


def PurgeData(data, m_len, tau=2):
    ''' 确保 data 中的每一个值都是 [- m_len * tau, m_len * tau]范围内的最小值

    '''
    validdata = []
    stride_tau = m_len * tau
    for dist, idx in data:
        if len(validdata) == 0:
            validdata.append([dist, idx])
        elif idx - validdata[-1][-1] > stride_tau:
            validdata.append([dist, idx])
        elif dist < validdata[-1][0]:
            validdata[-1] = [dist, idx]
    
    validdata = np.array(validdata, dtype=np.float32)

    return validdata


def GetAllShapeletpatterns(shapeletfilepath, premotionfilepath, scale):
    '''collects all patterns that used in augmendation

    '''
    shapeletfile = h5py.File(shapeletfilepath, 'r')
    shapeletInfoer = ShapeletReader(shapeletfilepath, premotionfilepath)

    # define the final output dict
    ShapeletPatternsDict = {}

    words = list(shapeletfile.keys())

    for word in words:
        shapeletInfoer.GetInfos(word)
        featurekey = shapeletInfoer.featurekey
        if featurekey not in ShapeletPatternsDict.keys():
            ShapeletPatternsDict[featurekey] = []
        
        levelkeys = shapeletInfoer.levelkeys

        # collect shapeletpatterns from all levels
        for level in levelkeys:
            shapeletdata = shapeletInfoer.GetShapelet(level)
            dists = np.sort(shapeletInfoer.dists)
            sigma = dists[int(len(dists)*scale)]
            # the information shoulde be include four components
            item = [word, level, dists[0], sigma, shapeletdata]
            ShapeletPatternsDict[featurekey].append(item)
    
    return ShapeletPatternsDict


if __name__ == '__main__':

    arger = argparse.ArgumentParser()
    arger.add_argument('-r', '--rewrite', action='store_true')
    arger.add_argument('-o', '--origin', choices=['search-tar', 'net-tar'], default='net-tar')

    args = arger.parse_args()
    testcode = args.testcode
    origin = args.origin
    rewrite = args.rewrite

    # origin = 'net-pad-tar'
    shapelet_path = {}
    shapelet_path['search-tar'] = 'output/target_signs_search.hdf5'
    shapelet_path['net-tar'] = 'output/target_signs_net.hdf5'

    premotionfilepath = '../data/prefeaturedata.hdf5'

    scale = 0.3
    outputpath = 'output/augsigns_all_level_%s.hdf5' % origin

    GetAugSigns(shapelet_path[origin], premotionfilepath, outputpath, scale, rewrite)