'''
Author: your name
Date: 2021-07-16 20:39:47
LastEditTime: 2021-07-17 14:32:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Public_SPBSL/openpose/motion_extraction.py
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import joblib
import argparse
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

from openpose import motion_extraction_model as MEM 


def Body_Estimation(model, videopath, outputpath, recpoints, batch_size=16):

    video = cv2.VideoCapture(videopath)
    if not video.isOpened():
        print('the file {} is not exist!'.format(videopath))
        return
    COUNTS = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    videodataset = MEM.VideoDataset(videopath, recpoints, transforms.ToTensor())
    video_dataloader = DataLoader(videodataset, batch_size, shuffle=False)

    MotionMat = np.zeros((COUNTS, 18, 3))
    count = 0
    for imgs in video_dataloader:
        results = model(imgs)

        for i in range(len(results)):
            PoseMat = np.zeros((18, 3))
            candidate, subset = results[i]
            # find the most right person in the screen
            max_index = None
            if len(subset) >= 1:
                leftshoulderX = np.zeros(len(subset))
                for person in range(len(subset)):
                    index = int(subset[person][5])
                    leftshoulderX[person] = candidate[index][0]
                max_index = np.argmax(leftshoulderX)
            
            # get the choosen person's keypoint
            if max_index is not None:
                for keyindex in range(18):
                    valueindex = int(subset[max_index][keyindex])
                    if valueindex != -1:
                        PoseMat[keyindex] = candidate[valueindex][:3]
            
            MotionMat[count] = PoseMat
            count += 1
    
    joblib.dump(MotionMat, outputpath)
    print('the file {} is saved!'.format(outputpath))
            
    
def Hand_Extraction(model, videopath, posedata, recpoints, outputpath):
    boxsize = 368
    batch_size = 32

    Handdataset = MEM.HandImageDataset(videopath, posedata, boxsize, recpoints)
    HandDataloader = DataLoader(Handdataset, batch_size, shuffle=False)

    Counts = len(posedata)
    HandMat = np.zeros((Counts, 42, 3))
    count = 0

    for samples in HandDataloader:
        lefthand, leftparams, righthand, rightparams = samples
        leftpeaks = model(lefthand)
        rightpeaks = model(righthand)

        leftparams = leftparams.numpy()
        rightparams = rightparams.numpy()

        for i in range(len(leftpeaks)):
            # right peaks
            rpeaks = rightpeaks[i]
            rx, ry, rw = rightparams[i]
            rpeaks[:, :2] = rpeaks[:, :2] * rw / boxsize

            rpeaks[:, 0] = np.where(rpeaks[:, 0] == 0, rpeaks[:, 0], rpeaks[:, 0]+rx)
            rpeaks[:, 1] = np.where(rpeaks[:, 1] == 0, rpeaks[:, 1], rpeaks[:, 1]+ry)
            HandMat[count, 21:, :] = rpeaks

            # left peaks
            lpeaks = leftpeaks[i]
            lx, ly, lw = leftparams[i]
            lpeaks[:, :2] = lpeaks[:, :2] * lw / boxsize
            lpeaks[:, 0] = np.where(lpeaks[:, 0] == 0, lpeaks[:, 0], lw-lpeaks[:, 0]-1+lx)
            lpeaks[:, 1] = np.where(lpeaks[:, 1] == 0, lpeaks[:, 1], lpeaks[:, 1]+ly)
            HandMat[count, :21, :] = lpeaks
    
    joblib.dump(HandMat, outputpath)
    print('the %s file is saved' % outputpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['pose', 'hand'], default='pose')
    args = parser.parse_args()
    mode = args.mode 

    # the global variable setting
    videodir = '/home/mario/sda/signdata/spbsl/normal'
    datadir = '../data'
    
    handmodelpath = '../data/models/hand_pose_model.pth'
    bodymodelpath = '../data/models/body_pose_model.pth'

    recpoints = [(700, 100), (1280, 720)]
    batch_size = 32
    
    if mode == 'pose':
        model = MEM.Body_Estimation_Model(bodymodelpath)
        for filename in os.listdir(videodir):
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                outname = 'video-%s-pose.pkl' % filename[:3]
            filepath = os.path.join(videodir, filename)
            outputpath = os.path.join(datadir, outname)
            Body_Estimation(model, filepath, outputpath, recpoints)
    
    else:
        mode = MEM.Hand_Estimation_Model(handmodelpath)
        for filename in os.listdir(videodir):
            ext = os.path.splitext(filename)[1]
            if ext in ['.mp4', '.mkv', '.rmvb', '.avi']:
                posename = 'video-%s-pose.pkl' % filename[:3]
                handname = 'video-%s-hand.pkl' % filename[:3]
            posefile = os.path.join(datadir, posename)
            if os.path.exists(posefile):
                posedata = joblib.load(posefile)
                outputpath = os.path.join(datadir, handname)
                Hand_Extraction(model, videodir, posedata, recpoints, outputpath)