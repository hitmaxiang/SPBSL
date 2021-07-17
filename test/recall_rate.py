import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import re
import h5py
import numpy as np

from shapelets.shapelet_reader import ShapeletReader


def OverlapRate(annorange, locrange):
    '''Judge the overlap between annorange and locrange whether larger than sigma

    args:
        annorange: the range of groundtruth
        locrange: the identified range
    '''

    a, b = annorange
    x, y = locrange

    length = min(b-a, y-x)

    if y < a or x > b:  # no overlap
        overlap = 0
    elif x >= a and y <= b:  # locrange is in annorange
        overlap = y - x
    elif a >= x and b <= y:  # annorange is in locrange
        overlap = b - a
    elif x <= a and y <= b:
        overlap = y - a
    elif x >= a and y >= b:
        overlap = b - x
    
    ratio = overlap/length
    assert ratio <= 1
    return ratio


def GetInstanceIdxofAnnotation(annogroup, sampleidxs, videokeys):
    ''' match the annotation sample with the instances, return its id
    
    args:
        annogroup: the hdf5 group of annotation, format: videokey/Insbegidx/: [lable, begidx, endidx]
        sampleidx: array, each instance:(Insbegidx, Instendidx)
        videokeys: array, each instance '%03d'
    
    return:
        IDX [[idx, label, begidx, endidx]], idx represent the mathched instance id, -1 means missmathced 
    '''
    IDX = []

    for vidkey in annogroup.keys():
        for Insbegidx in annogroup[vidkey].keys():
            # intialize the idx of this annotaion item
            label, begidx, endidx = annogroup[vidkey][Insbegidx]
            IDX.append([-1, label, begidx, endidx])
            for i in range(len(sampleidxs)):
                if vidkey == videokeys[i] and int(Insbegidx) == sampleidxs[i][0]:
                    # if it is mathced
                    IDX[-1][0] = i
                    break
    
    return IDX


def IdentificationRate(shapelet_path, annot_filepath, words=None, sigma=0.5, theta=0.7):
    '''calculate the Identification of the shapeletfile for words
    
    args:
        shapelet_path: the hdf5 shapelet file
        annot_filepath: the mannully annotated signs file
        words: the wanted words, if it is None, the common words will be thought
        sigma: the overlap ratio threshold 
    '''

    shapeletfile = h5py.File(shapelet_path, 'r')
    annotationfile = h5py.File(annot_filepath, 'r')

    shapeleter = ShapeletReader(shapelet_path)

    if words is None:
        words = [w for w in shapeletfile.keys() if w in annotationfile.keys()]
    else:
        words = [w for w in words if (w in shapeletfile.keys() and w in annotationfile.keys())]
    
    IdentiDict = {}

    for word in words:

        shapeleter.GetInfos(word)
        annogroup = annotationfile[word]

        sampleidxes = shapeleter.sampleidxes
        videokeys = shapeleter.videokeys

        # the format of IDX is: [idx, label, begidx, endidx]
        IDX = GetInstanceIdxofAnnotation(annogroup, sampleidxes, videokeys)
        
        valid_idxs = [x for x in IDX if (x[0] >= 0 and x[0] < len(sampleidxes)*theta)]
        pos_idxs = [x for x in valid_idxs if x[1] == 1]

        if len(pos_idxs) == 0:
            continue

        IdentiDict[word] = {'pos_rate': len(pos_idxs)/len(valid_idxs)}

        gl_identi = np.zeros(len(pos_idxs))
        for level in shapeleter.levelkeys:
            shapeleter.Get_level_info(level)
            locs = shapeleter.locs
            Iden_num, ratio_sum = 0, 0
            for i, item in enumerate(pos_idxs):
                idx, _, begidx, endidx = item
                annorange = (begidx, endidx)
                locrange = (locs[idx], locs[idx]+int(level))
                ratio = OverlapRate(annorange, locrange)
                ratio_sum += ratio
                if ratio >= sigma:
                    Iden_num += 1
                    gl_identi[i] = 1
            # save the identi_rate of each level
            IdentiDict[word][level] = [Iden_num/len(pos_idxs), ratio_sum, len(pos_idxs)]
        
        IdentiDict[word]['gl_id_rate'] = np.sum(gl_identi)/len(pos_idxs)
    
    return IdentiDict


def FindBestLevels(shapeleter, word, method, scale=0.3):
    shapeleter.GetInfos(word)
    levelkeys = shapeleter.levelkeys
    
    if method in [0, 1]:  # choose the highest score and minimum std
        # 分别考虑 half 和 scale 的样本距离
        ratio = scale if method == 1 else 0.5

        max_score, min_std = 0, float('inf')
        for level in levelkeys:
            shapeleter.Get_level_info(level)
            score = shapeleter.score
            sort_dists = np.sort(shapeleter.dists)
            if score > max_score:
                max_score = score
                best_level = level
            elif score == max_score:
                # remove the zeros values (否则求解方差会偏向 level 较小的值)
                for i in range(len(sort_dists)):
                    if sort_dists[i] > 0:
                        break
                i = 0
                scale_std = np.std(sort_dists[i:int(len(sort_dists)*ratio)])
                if scale_std <= min_std:
                    min_std = scale_std
                    best_level = level
    
    return best_level


def RecallRateTest(shapelet_path, annot_filepath, words=None, alpha=0.5, sigma=0.5, theta=0.7, num=0):
    '''calculate the recall rate of the shapelet file based on annotation file 

    Args:
        shapelet_path: the hdf5 file path of the shapelet fiel
        annot_filepath: the hdf5 file path of the annotation file path
        alpah: the threshold of the pos_rate
        sigma: the overlap ratio threshold
    '''

    IdentiDict = IdentificationRate(shapelet_path, annot_filepath, sigma=sigma, theta=theta)

    # 这里可以根据 alpha 和 num 去分析不同筛选条件下的recall rate
    all_words = list(IdentiDict.keys())

    alpha_major_words = [w for w in all_words if IdentiDict[w]['pos_rate'] >= alpha]
    alpha_minor_words = [w for w in all_words if IdentiDict[w]['pos_rate'] < alpha]

    # get the samples nums of all word:
    num_major_words, num_minor_words = [], []
    shapeleter = ShapeletReader(shapelet_path)
    for word in all_words:
        shapeleter.GetInfos(word)
        if len(shapeleter.sampleidxes) >= num:
            num_major_words.append(word)
        else:
            num_minor_words.append(word)
    
    words_list = [alpha_major_words, alpha_minor_words, num_major_words, num_minor_words]
    key_names = ['alpha_major', 'alpha_minor', 'num_major', 'num_minor']

    for i, keyname in enumerate(key_names):
        ratio_identi_num = np.zeros(3)
        ratio_identi_word = np.zeros(3)

        recall_num = np.zeros(3)
        if len(words_list[i]) == 0:
            continue
        for word in words_list[i]:
            shapeleter.GetInfos(word)

            # 全局综合的得分
            gl_score = IdentiDict[word]['gl_id_rate']

            # 理论上最好的得分
            best_score = max([IdentiDict[word][level][0] for level in shapeleter.levelkeys])

            # 推导得到的最好得分
            infer_score, ratio, pos_num = IdentiDict[word][FindBestLevels(shapeleter, word, method=0)]

            for j, score in enumerate([gl_score, best_score, infer_score]):
                if score > 0.5:
                    recall_num[j] += 1
            
            ratio_identi_num += np.array([ratio, infer_score*pos_num, pos_num])
            ratio_identi_word += np.array([ratio/pos_num, infer_score, 1])
            
        recall_rate = recall_num/len(words_list[i])
        print(f'\nthis is the recall rate of {keyname}\n')
        for k, key in enumerate(['global', 'theory', 'infer']):
            print('the recall rate of {} is {}/{} = {:.4f}'.format(key, recall_num[k],
                  len(words_list[i]), recall_rate[k]))
        print('ratio-identi-num:', ratio_identi_num, ratio_identi_num[:2]/ratio_identi_num[-1])
        print('ratio-identi-word:', ratio_identi_word, ratio_identi_word[:2]/ratio_identi_word[-1])


if __name__ == '__main__':
    
    origin = 'search'

    shapeletfilepath = {'search': 'shapelet_search_all.hdf5'}
    shapeletfilepath['net'] = 'shapelet_net_all.hdf5'

    shapeletfilepath = os.path.join('../data/output', shapeletfilepath[origin])

    annotationfilepath = '../gui/annotation.hdf5'
        
    alpha, sigma, theta = 0.5, 0.5, 1.0
    RecallRateTest(shapeletfilepath, annotationfilepath, alpha=alpha, theta=theta, sigma=sigma, num=50)