'''
Description: 
Version: 2.0
Autor: mario
Date: 2021-06-24 23:24:18
LastEditors: Please set LastEditors
LastEditTime: 2021-07-16 23:50:32
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import re
import h5py
import numpy as np

from src import utils


class ShapeletReader():
    def __init__(self, shapeletfilepath, premotionpath=None):
        self.shapeletfile = h5py.File(shapeletfilepath, 'r')

        # construct the premotionfilepath
        if premotionpath is not None:
            self.premotionfile = h5py.File(premotionpath, 'r')
        else:
            self.premotionfile = None

        # pattern
        self.infopattern = r'datamode:(.+)-featuremode:(\d+)$'
        self.levelpattern = r'^\d+$'

    def GetInfos(self, word):
        self.word = word

        info = utils.Encode(self.shapeletfile[word]['loginfo'][0])
        sampleidxes = self.shapeletfile[word]['sampleidxs'][:]
        videokeys = self.shapeletfile[word]['videokeys'][:]
        videokeys = [utils.Encode(vidkey) for vidkey in videokeys]

        self.sampleidxes = sampleidxes
        self.videokeys = videokeys

        datamode, featuremode = re.findall(self.infopattern, info)[0]
        self.featurekey = '%s-%s' % (datamode, featuremode)
        self.datamode = datamode
        self.featuremode = featuremode

        levelkeys = []
        for levelkey in self.shapeletfile[word].keys():
            if re.match(self.levelpattern, levelkey):
                levelkeys.append(levelkey)
        
        self.levelkeys = levelkeys
    
    def GetShapelet(self, m_level):

        self.Get_level_info(m_level)
        shapelet = self.shapelet
        locs = self.locs

        if len(shapelet.shape) == 1:  # shapelet finding
            index = shapelet[0]
            begidx = self.sampleidxes[index][0] + locs[index]
            endidx = begidx + int(m_level)
            vidkey = self.videokeys[index]

            # get the clipdata
            if self.premotionfile is not None:
                shapeletdata = self.premotionfile[self.featurekey][vidkey][begidx:endidx]
            else:
                raise ValueError('the ori-sf shapelet should be provided with a valid premotionpath')
        
        elif len(shapelet.shape) == 3:  # shapelet Net learning
            # shapeletdata = np.transpose(shapelet[0])
            shapeletdata = shapelet
        
        elif len(shapelet.shape) == 2:  # target signs
            shapeletdata = shapelet
            
        else:
            raise ValueError('the shapelet of the shapeletfile is wrong!')
        
        self.shapeletdata = shapeletdata
        return self.shapeletdata
    
    def Get_level_info(self, m_level):
        assert m_level in self.levelkeys
        word = self.word
        self.m_level = m_level
        self.shapelet = self.shapeletfile[word][m_level]['shapelet'][:]
        self.locs = self.shapeletfile[word][m_level]['locs'][:]
        self.dists = self.shapeletfile[word][m_level]['dists'][:]
        self.score = self.shapeletfile[word][m_level]['score'][0]
    
    def ReleaseFile(self):
        self.shapeletfile.close()
        if self.premotionfile is not None:
            self.premotionfile.close()


class ShapeletWriter():
    def __init__(self, outputpath):
        self.outputpath = outputpath

    def checkout_level_exists(self, word, levels):
        '''check out whether word/levels exists in the outputpath

        args:
            outputpath: the outputpath of the shapelet file
            word: the target word
            levels: list(int), the wanted levels
        '''
        state = True
        if os.path.exists(self.outputpath):
            with h5py.File(self.outputpath, 'r') as f:
                for level in levels:
                    groupkey = '%s/%d' % (word, level)
                    if groupkey not in f:
                        state = False
                        break
        else:
            state = False
        return state
    
    def WriteInstanceInfo(self, word, sample_indexes, delayfx, datamode, featuremode):
        ''' write the training instance information to the outputpath

        args:
            word: the target word
            sample_indexes: list, format: [vidkey(str), begidx(int), endidx(int), label(int)]
            delayfx: float, the instance choose parameter
            datamode: str, represent the keypoints are used
            featuremode: int, the feature mode used
        '''
        # only the positive instances need to save
        vidkeys = [x[0] for x in sample_indexes if x[-1] == 1]
        sampleidxs = [x[1:3] for x in sample_indexes if x[-1] == 1]
        sampleidxs = np.array(sampleidxs, dtype=np.int32)
        pos_num = len(vidkeys)

        # define the strtype of the hdf5 file
        strdt = h5py.string_dtype(encoding='utf-8')

        # write the (begidx, endidx) for each instance
        idxkey = '%s/sampleidxs' % word
        utils.WriteRecords2File(self.outputpath, idxkey, sampleidxs)
        # write the videokeys of each instance
        vdokey = '%s/videokeys' % word
        utils.WriteRecords2File(self.outputpath, vdokey, vidkeys, (pos_num, ), dtype=strdt)
        # write the log inforamtion of the instance
        infokey = '%s/loginfo' % word
        infomsg = 'fx:%.2f-datamode:%s-featuremode:%d' % (delayfx, datamode, featuremode)
        utils.WriteRecords2File(self.outputpath, infokey, infomsg, (1, ), strdt)
    
    def WriteShapeletInfo(self, word, m_len, shapelet, locs, dists, score):
        basekey = '%s/%d' % (word, m_len)
        outputpath = self.outputpath
        utils.WriteRecords2File(outputpath, f'{basekey}/locs', locs)
        utils.WriteRecords2File(outputpath, f'{basekey}/dists', dists)
        utils.WriteRecords2File(outputpath, f'{basekey}/shapelet', shapelet)
        utils.WriteRecords2File(outputpath, f'{basekey}/score', score)