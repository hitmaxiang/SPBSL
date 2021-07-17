import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import argparse
import numpy as np

from srcmx import utilmx
from shapelets.shapelet_reader import ShapeletReader


def GenerateAugSamples(shapeletpath, augdatapath, outputpath, method, 
                       scale=0.3, theta=0.3, max_num=200, chosenmode='random'):
    ''' Genererate the samples of the dataset for test and validation

    Args:
        shapeletpath: the corresponding shapelet file， can be used for the method options
        augdatapath: the augdata of all leveles
        method: the best level chosen method
        scale: the first level threshold
        theta: the second level threshold
        max_num: the maximum of each samples
        chosenmode: the samples take mode for the samples
    '''
    # augdatafile 的结构是：
    # datamode-featuremode-scale/word/levelkey/videokey: [[dist, indices]]
    
    shapeletfile = h5py.File(shapeletpath, 'r')
    augdatafile = h5py.File(augdatapath, 'r')

    shapeleter = ShapeletReader(shapeletpath)

    words = list(shapeletfile.keys())
    # words = ['blackpool']
    for ite, word in enumerate(words):
        best_level = FindBestLevels(shapeleter, word, method, scale)
        shapeleter.Get_level_info(best_level)

        # construct the basekey of the chosen level
        featurekey = shapeleter.featurekey
        basekey = f'{featurekey}-{scale:.2f}/{word}/{best_level}'

        # set the threshold
        dists = np.sort(shapeleter.dists)
        # get the first non-zero dist
        for i in range(len(dists)):
            if dists[i] > 0.1:
                break
        min_sigma = dists[i]
        dists = dists[i:]
        max_sigma = dists[int(scale * len(dists)) - 1]
        # min_sigma = dists[i]

        dist_tau = min_sigma + (max_sigma - min_sigma) * theta

        # collects the samples
        all_samples = []
        for vidkey in augdatafile[basekey].keys():
            items = augdatafile[basekey][vidkey]
            for dist, loc in items:
                if dist <= dist_tau:
                    # the final format: videokey, begidx, m_level, dist
                    all_samples.append([int(vidkey), loc, int(best_level), dist])
        
        # construct the training and validing samples
        split_samples = utilmx.RandomSplitSamples(all_samples, split_tuple=(0.6, 0.2, 0.2), max_num=max_num)[0]
        split_samples = [np.array(x, dtype=np.float32) for x in split_samples]
        train_samples, val_samples, test_samples = split_samples

        if len(train_samples) == 0:
            print()
        # save the samples into outputpath
        utilmx.WriteRecords2File(outputpath, f'{word}/train', train_samples)
        utilmx.WriteRecords2File(outputpath, f'{word}/val', val_samples)
        utilmx.WriteRecords2File(outputpath, f'{word}/test', test_samples)

        print(f'complete the genereation of word:[{ite}/{len(words)}]{word}')


def GenerateNoAugsamples(shapeletfilepath, outputpath, scale=0.3, method=0, max_num=200):
    ''' Generate the no augmented samples
    
    args:
        shapeletfilepath: the hdf5 file of the shapelet file
        outputpath: the output samples file path
        scale: the first level threshold parameters
    '''

    shapeletfile = h5py.File(shapeletfilepath, 'r')
    shapeleter = ShapeletReader(shapeletfilepath)

    words = shapeletfile.keys()

    for ite, word in enumerate(words):
        best_level = FindBestLevels(shapeleter, word, method, scale)
        shapeleter.Get_level_info(best_level)

        sampleidxes = shapeleter.sampleidxes
        videokeys = shapeleter.videokeys
        locs = shapeleter.locs
        dists = shapeleter.dists

        indexes = np.argsort(dists)[:int(scale*len(dists))]

        all_samples = []
        for idx in indexes:
            begidx = sampleidxes[idx][0] + locs[idx]
            item = [int(videokeys[idx]), begidx, int(best_level), dists[idx]]
            all_samples.append(item)

        # construct the training and validing samples
        split_samples = utilmx.RandomSplitSamples(all_samples, max_num=max_num, split_tuple=(0.6, 0.2, 0.2))[0]
        split_samples = [np.array(x, dtype=np.float32) for x in split_samples]
        train_samples, val_samples, test_samples = split_samples
       
        # save the samples into outputpath
        utilmx.WriteRecords2File(outputpath, f'{word}/train', train_samples)
        utilmx.WriteRecords2File(outputpath, f'{word}/val', val_samples)
        utilmx.WriteRecords2File(outputpath, f'{word}/test', test_samples)

        print(f'complete the genereation of word:[{ite}/{len(words)}]{word}')


def GetVerfiedTestSet(annotationfilepath, outputpath):
    # annotation 结构： word/videokey/offset:[label, begin, end]
    annotationfile = h5py.File(annotationfilepath, 'r')
    for word in annotationfile.keys():
        samples = []
        for videokey in annotationfile[word].keys():
            for offset in annotationfile[word][videokey].keys():
                label, begin, end = annotationfile[word][videokey][offset][:]
                if label == 1:
                    item = [int(videokey), int(offset)+begin, end-begin]
                    samples.append(item)
        samples = np.array(samples, dtype=np.float32)
        utilmx.WriteRecords2File(outputpath, f'{word}/test', samples)


def FindBestLevels(shapeleter, word, method, scale):
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
                scale_std = np.std(sort_dists[i:int(len(sort_dists)*ratio)])
                if scale_std <= min_std:
                    min_std = scale_std
                    best_level = level
    return best_level


def GetTestsamples(shapeletpath, outputpath, scale, theta):
    shapeleter = ShapeletReader(shapeletpath)
    words = list(shapeleter.shapeletfile.keys())

    for ite, word in enumerate(words):
        best_level = utilmx.FindBestLevels(shapeleter, word, 0, scale)
        shapeleter.GetInfos(word)
        shapeleter.Get_level_info(best_level)

        # set the threshold
        dists = np.sort(shapeleter.dists)
        # get the first non-zero dist
        for i in range(len(dists)):
            if dists[i] > 0.1:
                break
        min_sigma = dists[i]
        dists = dists[i:]
        max_sigma = dists[int(scale * len(dists)) - 1]
        dist_tau = min_sigma + (max_sigma - min_sigma) * theta

        all_samples = []
        for idx, dist in enumerate(shapeleter.dists):
            if dist <= dist_tau:
                vidkey = int(shapeleter.videokeys[idx])
                begidx = shapeleter.sampleidxes[idx][0] + shapeleter.locs[idx]
                all_samples.append([vidkey, begidx, int(best_level), dist])
        all_samples = np.array(all_samples, dtype=np.float32)
        
        if len(all_samples) == 0:
            print()
        utilmx.WriteRecords2File(outputpath, f'{word}/test', all_samples)

        print(f'complete the genereation of word:[{ite}/{len(words)}]{word}')


if __name__ == '__main__':

    arger = argparse.ArgumentParser()
    arger.add_argument('-t', '--testcode', type=int, default=0)
    arger.add_argument('-o', '--origin', default='net-tar')
    arger.add_argument('-a', '--augment', action='store_true')
    arger.add_argument('-m', '--method', type=int, default=0)

    args = arger.parse_args()
    origin = args.origin
    aug = args.augment
    method = args.method
    testcode = args.testcode

    # origin = 'net-tar'
    # testcode = 0
    # aug = True

    premotionfilepath = '../data/prefeaturedata.hdf5'

    shapelet_path_dict = {}
    shapelet_path_dict['search-tar'] = 'output/target_signs_search.hdf5'

    shapelet_path_dict['net-tar'] = 'output/target_signs_net.hdf5'

    shapelet_path_dict['notion'] = '../gui/annotation.hdf5'

    shapeletpath = shapelet_path_dict[origin]

    augdatafilepath = 'output/augsigns_all_level_%s.hdf5' % origin

    max_num = 200
    if testcode == 0:  # generate the samples 
        assert origin != 'notion'

        scale = 0.7 if aug is False else 0.3
        theta = 0.3
        aug_str = '-aug' if aug else ''
        outputpath = f'samples-{origin}{aug_str}-m-{method}-st-{scale:.2f}-{theta:.2f}-all.hdf5'
        outputpath = os.path.join('./output', outputpath)

        if aug is True:
            GenerateAugSamples(shapeletpath, augdatafilepath, outputpath, method, scale, theta, max_num)
        else:
            GenerateNoAugsamples(shapeletpath, outputpath, scale, method, max_num)
    
    elif testcode == 1:  # generate the test samples
        scale, theta = 0.3, 0.3
        if origin == 'notion':
            outputpath = 'output/verified_test_set.hdf5'
            GetVerfiedTestSet(shapeletpath, outputpath)
        else:
            outputpath = f'output/test-{origin}-m-{method}-st-{scale:.2f}-{theta:.2f}-all.hdf5'
            GetTestsamples(shapeletpath, outputpath, scale, theta)