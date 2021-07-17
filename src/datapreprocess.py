'''
Description: the preprocessing module for the data and filter
Version: 2.0
Autor: mario
Date: 2020-09-24 16:34:31
LastEditors: Please set LastEditors
LastEditTime: 2021-07-16 23:15:58
'''
import numpy as np
from numba import jit
from copy import deepcopy


def MotionJointFeatures(motiondata, datamode=None, featuremode=0):
    # TxCxD time-len, key-num, fea-dim 
    T, C, D = motiondata.shape

    # if the datamode is not given, default datamode can be infer by keypoints
    if datamode is None:
        if C == 18:
            datamode = 'pose'
        elif C == 60:
            datamode = 'posehand'
        else:
            raise Exception('please input correct datamode')
    
    # calculate the relative coordinates
    if featuremode == 0:
        paddingmode = 0
        InvalidMode = False
        rescale = False
        wantposechannel = [2, 3, 4, 5, 6, 7]
    # calculate relative position and rescale
    elif featuremode == 1:
        paddingmode = 0
        InvalidMode = False
        rescale = True
        wantposechannel = [2, 3, 4, 5, 6, 7]
    # calculate relative position (remove the invalid value), and rescale
    elif featuremode == 2:
        paddingmode = 1
        InvalidMode = True
        rescale = True
        wantposechannel = [2, 3, 4, 5, 6, 7]
    
    # replace invalid value with neighbor points
    motiondata = ProcessingTheNonValue(motiondata.reshape(T, -1), mode=paddingmode).reshape(T, C, D)
    # calculate the relative positions
    posedata = CorrespondingValue(motiondata[:, :18, :2], c=1, processInvalid=InvalidMode)
    # take the wanted keypoints
    posedata = posedata[:, wantposechannel]

    if datamode == 'posehand':
        # left hand
        lefthanddata = CorrespondingValue(motiondata[:, 18:18+21, :2], c=0, processInvalid=InvalidMode)
        # delete the original keypoint
        lefthanddata = lefthanddata[:, 1:]

        # righ hand
        righthanddata = CorrespondingValue(motiondata[:, 18+21:, :2], c=0, processInvalid=InvalidMode)
        # delete the original keypoint
        righthanddata = righthanddata[:, 1:]

        data = np.concatenate((posedata, lefthanddata, righthanddata), axis=1)
    else:
        data = posedata
    
    if rescale is True:
        # the rescale length is the verticle distance between (heart and nose)
        scale = motiondata[:, 0, 1] - motiondata[:, 1, 1] + 1e-5
        scale = scale[:, np.newaxis, np.newaxis]
        data = data/scale

    return data
    

def ProcessingTheNonValue(datamat, mode=0, sigma=2):
    
    if mode == 0:
        # replace the invalid with former points
        for c in range(datamat.shape[1]):
            for r in range(1, datamat.shape[0]):
                if datamat[r, c] == 0:
                    datamat[r, c] = datamat[r-1, c]
    elif mode == 1:
        # replace the invalid with closest valid points in +- sigma window
        copymat = deepcopy(datamat)
        for c in range(datamat.shape[1]):
            for r in range(datamat.shape[0]):
                if datamat[r, c] == 0:
                    for s in range(1, sigma+1):
                        if r-s >= 0 and copymat[r-s, c] != 0:
                            datamat[r, c] = copymat[r-s, c]
                            break
                        elif r+s < datamat.shape[0] and copymat[r+s, c] != 0:
                            datamat[r, c] = copymat[r+s, c]
                            break

    return datamat


# calculate the relative cooridinate
def CorrespondingValue(datamat, c, processInvalid=False):
    T, C, D = datamat.shape
    # do not process the invalid point
    if processInvalid is True:
        for i in range(T):
            for j in range(C):
                if j == c:
                    continue
                for k in range(D):
                    if datamat[i, j, k] == 0:
                        datamat[i, j, k] = datamat[i, c, k]
                        
    datamat -= datamat[:, c].reshape(T, 1, -1)
    return datamat