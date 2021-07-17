'''
Author: your name
Date: 2021-07-16 15:38:05
LastEditTime: 2021-07-16 20:37:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Public_SPBSL/openpose/util.py
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# draw the body keypoint and lims
def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], 
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], 
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], 
              [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
              [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
              
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
            
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    # plt.imsave("preview.jpg", canvas[:, :, [2, 1, 0]])
    # plt.imshow(canvas[:, :, [2, 1, 0]])
    return canvas


# handpose image drawed by opencv
def draw_handpose(canvas, peaks):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9],
             [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17],
             [17, 18], [18, 19], [19, 20]]
    
    for i in range(len(peaks)):
        for j in range(len(edges)):
            x1, y1 = peaks[i][edges[j][0]]
            x2, y2 = peaks[i][edges[j][1]]

            if (x1+y1 != 0) and (x2+y2 != 0):
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color=[255, 0, 0], thickness=2)
        for k in range(len(peaks[i])):
            x, y = peaks[i][k]
            cv2.circle(canvas, (int(x), int(y)), 4, (0, 0, 255), thickness=-1)
    return canvas


def DrawPose(img, bodypose=None, handpeaks=None):
    '''
    description: display the posedata in the img
    param:
        img: BGR image data
        posed: the joint position of the skeleton or hands
        mode: the type of the motion data
    return {type} 
    author: mario
    '''
    if bodypose is not None:
        subset = np.zeros((1, 20))
        candidate = np.zeros((20, 4))
        for i in range(18):
            if sum(bodypose[i, :]) == 0:
                subset[0, i] = -1
            else:
                subset[0, i] = i
            candidate[i, :2] = bodypose[i, :2]
        img = draw_bodypose(img, candidate, subset)
        
    if handpeaks is not None:
        handpeaks = handpeaks[:, :2]
        handpeaks = np.reshape(handpeaks, (2, -1, 2))
        img = draw_handpose(img, handpeaks)

    return img


# detect hand according to body pose keypoints
def handDetect(candidate, subset, img_shape):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = img_shape
    for person in subset.astype(int):
        # if any of three not detected
        has_left = np.sum(person[[5, 6, 7]] == -1) == 0
        has_right = np.sum(person[[2, 3, 4]] == -1) == 0
        if not (has_left or has_right):
            continue
        hands = []
        # left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = person[[5, 6, 7]]
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        # right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = person[[2, 3, 4]]
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])

        for x1, y1, x2, y2, x3, y3, is_left in hands:

            x = x3 + ratioWristElbow * (x3 - x2)
            y = y3 + ratioWristElbow * (y3 - y2)
            distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
            
            # x-y refers to the center --> offset to topLeft point
            x -= width / 2
            y -= width / 2  # width = height
            # overflow the image
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            width1 = width
            width2 = width
            if x + width > image_width:
                width1 = image_width - x
            if y + width > image_height:
                width2 = image_height - y
            width = min(width1, width2)
            # the max hand box value is 20 pixels
            detect_result.append([int(x), int(y), int(width), is_left])

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result


# transfer caffe model to pytorch which will match the layer name
def modeltransfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        # print("channels: ", channels.shape)
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(channels), 1, 5, 5))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), self.weight, groups=self.channels)
        return x


def findpeaks_torch(data, thre):
    with torch.no_grad():
        peaks_binary = data > thre
        torch.logical_and(peaks_binary, data >= F.pad(data, (1, 0))[:, :, :, :-1], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 1))[:, :, :, 1:], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 0, 1, 0))[:, :, :-1, :], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 0, 0, 1))[:, :, 1:, :], out=peaks_binary)
        peaks_binary = torch.nonzero(peaks_binary, as_tuple=False)
    
    return peaks_binary


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j