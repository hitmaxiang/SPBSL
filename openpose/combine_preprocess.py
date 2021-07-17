'''
Author: your name
Date: 2021-07-16 23:18:23
LastEditTime: 2021-07-17 14:45:16
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Public_SPBSL/openpose/combine_preprocess.py
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import joblib
import h5py
import numpy as np

from src import utils
from src import datapreprocess as DP


def combine_pkl_hdf5(datadir, outputpath):
    # the file pattern:
    # video-num-body.pkl/ video-num-hand.pkl

    filepattern = r'video-(.+)-(.{4}).pkl'
    for filename in os.listdir(datadir):
        results = re.findall(filepattern, filename)
        if len(results) == 0:
            continue
        videokey, mode = results[0]
        
        data = joblib.load(os.path.join(datadir, outputpath))
        data = data.astype(np.float32)
        datakey = f'{videokey}/{mode}'
        utils.WriteRecords2File(outputpath, datakey, data)


def preprocess_hdf5_motion(motionfilepath, datamode, featuremode, outputpath):
    motionfile = h5py.File(motionfilepath, 'r')
    videokeys = motionfile.keys()

    for videokey in videokeys:
        posedata = motionfile[videokey]['pose'][:, :2].astype(np.float32)
        handdata = motionfile[videokey]['hand'][:, :2].astype(np.float32)

        data = np.concatenate((posedata, handdata), axis=1)
        data = DP.MotionJointFeatures(data, datamode, featuremode)
        data = np.reshape(data, (data.shape[0], -1))

        key = f'{datamode}-{featuremode}/{videokey}'
        utils.WriteRecords2File(outputpath, key, data)
    
    motionfile.close()

    
if __name__ == '__main__':
    combine_pkl_hdf5('../data', 'motiondata.hdf5')
    preprocess_hdf5_motion('motiondata.hdf5', 'posehand', 2, 'premotiondata.hdf5')