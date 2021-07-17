'''
Description: the common used utilization functions
Version: 2.0
Autor: mario
Date: 2020-08-27 20:41:43
LastEditors: Please set LastEditors
LastEditTime: 2021-07-17 17:50:23
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import cv2
import math
import h5py
import bisect
import torch
import scipy
import matplotlib
import numpy as np
import torch.nn.functional as F

from src import utils
from torch import nn
from scipy.io import loadmat


'''
之前使用的函数文件， 目前不太需要了
'''


class Best_K_Items():
    # 注意 bisect 只能沿着一个方向（increase）进行排序
    def __init__(self, K, preferlarge=True):
        self.capacity = K
        self.scores = []
        self.keys = []
        self.data = {}
    
    def insert(self, score, items):
        if items[0] in self.keys:
            return
        index = bisect.bisect_left(self.scores, score)
        if index == 0:
            if len(self.scores) >= self.capacity:
                return
        self.scores.insert(index, score)
        self.keys.insert(index, items[0])
        self.data[items[0]] = items[1]

        if len(self.scores) > self.capacity:
            rmkey = self.keys[0]
            self.scores = self.scores[1:]
            self.keys = self.keys[1:]
            self.data.pop(rmkey)
    
    def clear(self):
        self.scores = []
        self.keys = []
        self.data = {}
    
    def wirteinfo(self, header, outpath, mode='a'):
        with open(outpath, mode) as f:
            f.write('header:%s\n\n' % header)
            for i in range(len(self.scores)):
                f.write('%dth:shapelet\t%s--%f\n' % (i, self.keys[i], self.scores[i]))
                for loc in self.data[self.keys[i]]:
                    f.write('\t%d' % loc)
                f.write('\n\n')
        

class Records_Read_Write():
    def __init__(self):
        pass

    def Read_shaplets_cls_Records(self, filepath):
        '''
        description: extract the data (test data) from the record data
        param {type} 
        return {type} 
        author: mario
        '''
        RecordsDict = {}
        with open(filepath, 'r') as f:
            lines = f.readlines()
        params_pattern = r'word:\s*(\w+),.*lenth:\s*(\d+),\s*iters:\s*(\d+),\s*feature:\s*(\d+)'
        score_pattern = r'score:\s*(\d*.\d+)$'
        locs_pattern = r'Locs:(.*)$'
        for line in lines:
            params = re.findall(params_pattern, line)
            
            if len(params) != 0:
                word = params[0][0]
                args = params[0][1:]
                key = '-'.join(args)
                if word not in RecordsDict.keys():
                    RecordsDict[word] = {}
                    temp_dict = {}
                else:
                    if key not in RecordsDict[word].keys():
                        temp_dict = {}
                    else:
                        temp_dict = RecordsDict[word][key]

                if temp_dict == {}:
                    temp_dict = {'num': 1}
                    temp_dict['score'] = []
                    temp_dict['location'] = []
                else:
                    temp_dict['num'] += 1
                
                RecordsDict[word][key] = temp_dict
                continue

            score = re.findall(score_pattern, line)
            if len(score) != 0:
                RecordsDict[word][key]['score'].append(float(score[0]))
                continue
            
            locs = re.findall(locs_pattern, line)
            if len(locs) != 0:
                locs = locs[0].split()
                locs = [int(d) for d in locs]
                RecordsDict[word][key]['location'].append(locs)
            
        return RecordsDict
    
    def Write_shaplets_cls_Records(self, filepath, word, m_len, iters, featuremode, score, locs):

        with open(filepath, 'a+') as f:
            f.write('\nthe test of word: %s, with lenth: %d, iters: %d, feature: %d\n' %
                    (word, m_len, iters, featuremode))
            f.write('score:\t%f\n' % score)
            f.write('Locs:')
            for loc in locs:
                f.write('\t%d' % loc[0])
            f.write('\n\n')
    
    def Get_extract_ed_ing_files(self, datadir, init=False):
        filelists = []
        if init is False:
            with open(os.path.join(datadir, 'extract_ed_ing.txt'), 'r') as f:
                lines = f.readlines()
            for line in lines:
                filelists.append(line)
        else:
            datafiles = os.listdir(datadir)
            with open(os.path.join(datadir, 'extract_ed_ing.txt'), 'w') as f:
                for datafile in datafiles:
                    if os.path.splitext(datafile)[1] in ['.npy', '.pkl']:
                        f.write('%s\n' % datafile)
                        filelists.append(datafile)
        return filelists
    
    def Add_extract_ed_ing_files(self, datadir, addfilename):
        with open(os.path.join(datadir, 'extract_ed_ing.txt'), 'a') as f:
            f.write('%s\n' % addfilename)


'''
Body and hand detect
'''

        
def FaceDetect(frame):
    '''
    description: detect the face with the opencv trained model
    param: 
        frame: the BGR picture
    return: the facecenter and the face-marked frame
    author: mario
    '''
    face_cascade = cv2.CascadeClassifier()
    profileface_cascade = cv2.CascadeClassifier()

    # load the cascades
    face_cascade.load(cv2.samples.findFile('./model/haarcascade_frontalface_alt.xml'))
    profileface_cascade.load(cv2.samples.findFile('./model/haarcascade_profileface.xml'))

    # convert the frame into grayhist
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    if len(faces) == 0:
        faces = profileface_cascade.detectMultiScale(frame_gray)
    
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), [0, 255, 0], 2)
    return faces, frame


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
            # width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
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


def findpeaks_torch(data, thre):
    # data = torch.from_numpy(data)
    # if torch.cuda.is_available():
    #     data = data.cuda()
    with torch.no_grad():
        peaks_binary = data > thre
        torch.logical_and(peaks_binary, data >= F.pad(data, (1, 0))[:, :, :, :-1], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 1))[:, :, :, 1:], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 0, 1, 0))[:, :, :-1, :], out=peaks_binary)
        torch.logical_and(peaks_binary, data >= F.pad(data, (0, 0, 0, 1))[:, :, 1:, :], out=peaks_binary)
        peaks_binary = torch.nonzero(peaks_binary, as_tuple=False)
        # peaks_binary = peaks_binary.cpu().numpy()
    
    return peaks_binary


'''
Image draw and display
'''


# handpose image drawed by opencv
def draw_handpose_by_opencv(canvas, peaks):
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    for i in range(len(peaks)):
        for j in range(len(edges)):
            x1, y1 = peaks[i][edges[j][0]]
            x2, y2 = peaks[i][edges[j][1]]

            if (x1+y1 != 0) and (x2+y2 != 0):
                # cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([j/float(len(edges)), 1.0, 1.0])*255, thickness=2)
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
        img = utils.draw_bodypose(img, candidate, subset)
        
    if handpeaks is not None:
        handpeaks = handpeaks[:, :2]
        handpeaks = np.reshape(handpeaks, (2, -1, 2))
        img = draw_handpose_by_opencv(img, handpeaks)

    return img


'''
sliding distance calculation
'''


def Calculate_shapelet_distance(query, timeseries):
    m_len = len(query)
    least_distance = np.linalg.norm(query - timeseries[:m_len])
    distance = least_distance
    best_loc = 0
    for loc in range(1, len(timeseries)-m_len+1):
        distance = np.linalg.norm(query - timeseries[loc:loc+m_len])
        if distance < least_distance:
            least_distance = distance
            best_loc = loc
    return least_distance, best_loc


def SlidingDistance(pattern, sequence):
    '''
    calculate the distance between pattern with all the candidate patterns in sequence
    the pattern has the shape of (m, d), and sequence has the shape of (n, d). the d is
    the dimention of the time series date.
    '''
    m = len(pattern)
    n = len(sequence)
    _len = n - m + 1
    dist = np.square(pattern[0] - sequence[:_len])
    # dist = dist.astype(np.float32)
    for i in range(1, m):
        dist += np.square(pattern[i] - sequence[i:i+_len])
    if len(dist.shape) == 2:
        dist = np.sum(dist, axis=-1)
    return np.sqrt(dist)


def matrixprofile(sequenceA, sequenceB, m):
    l_1 = len(sequenceA)
    l_2 = len(sequenceB)
    # DisMat = np.zeros((l_1-m+1, l_2-m+1), dtype=np.float16)
    DisMat = np.zeros((l_1-m+1, l_2-m+1))
    DisMat[0, :] = SlidingDistance(sequenceA[:m], sequenceB)
    DisMat[:, 0] = SlidingDistance(sequenceB[:m], sequenceA)
    for r in range(1, DisMat.shape[0]):
        offset = np.square(sequenceA[r+m-1]-sequenceB[m:])
        offset -= np.square(sequenceA[r-1]-sequenceB[:-m])
        offset = np.sum(offset, axis=-1)
        DisMat[r, 1:] = np.sqrt(DisMat[r-1, :-1]**2+offset)
    return DisMat


def SlidingDistance_torch(pattern, sequence):
    m = len(pattern)
    n = len(sequence)
    _len = n - m + 1
    dist = torch.square(pattern[0] - sequence[:_len])
    for i in range(1, m):
        dist += torch.square(pattern[i] - sequence[i:i+_len])
    if len(dist.shape) == 2:
        dist = torch.sum(dist, axis=-1)
    return torch.sqrt(dist)


def matrixprofile_torch(sequenceA, sequenceB, m, DisMat=None):
    l_1, l_2 = len(sequenceA), len(sequenceB)
    
    if DisMat is None: 
        DisMat = torch.zeros(l_1-m+1, l_2-m+1, dtype=torch.float32, requires_grad=False)
        DisMat = DisMat.to(sequenceA.device)
    else:
        DisMat.zero_()

    DisMat[0, :] = SlidingDistance_torch(sequenceA[:m], sequenceB)
    DisMat[:, 0] = SlidingDistance_torch(sequenceB[:m], sequenceA)
    for r in range(1, DisMat.shape[0]):
        offset = torch.square(sequenceA[r+m-1]-sequenceB[m:])
        offset -= torch.square(sequenceA[r-1]-sequenceB[:-m])
        offset = torch.sum(offset, axis=-1)
        DisMat[r, 1:] = torch.sqrt(torch.square(DisMat[r-1, :-1])+offset)
    return DisMat


def SlidingDistanceSquare(query, X):
    # query: (m, D), X: (n, D) should be reshape into (1, D, m) and (1, D, N)
    # query = query.permute(1, 0).unsqueeze(0)
    X = X.permute(1, 0).unsqueeze(0)
    const_query = torch.ones_like(query)

    QQ = torch.sum(torch.square(query))
    XX = F.conv1d(torch.square(X), weight=const_query)
    QX = F.conv1d(X, weight=query)
    DIS = QQ + XX - 2 * QX
    DIS = DIS[0, 0]
    return DIS


'''
Hdf5 File operation
'''


#  向一个 hdf5 写入一个关键词的数据
def WriteRecords2File(h5filepath, key, data, shape=None, dtype=None):

    shape = data.shape if shape is None else shape
    dtype = data.dtype if dtype is None else dtype
    # data = data.astype(dtype)
    with h5py.File(h5filepath, 'a') as f:
        if key in f.keys():
            if f[key][:].shape != shape or f[key][:].dtype != dtype:
                del f[key]
                f.create_dataset(key, shape, dtype=dtype)
        else:
            f.create_dataset(key, shape, dtype=dtype)
        f[key][:] = data


def CopyFromH5dfFile(srcfilepath, destfilepath, overwrite=False):
    srcfile = h5py.File(srcfilepath, mode='r')
    destfile = h5py.File(destfilepath, mode='a')

    groupkeys = list(srcfile.keys())
    for key in groupkeys:
        if key not in destfile.keys():
            srcfile.copy(key, destfile, key)
            print('coping the group :%s' % key)
        else:
            print('the group %s is exist' % key)
            if overwrite is True:
                del destfile[key]
                srcfile.copy(key, destfile, key)
                print('overwirte coping the group: %s' % key)
    srcfile.close()
    destfile.close()


'''
Others
'''


def Write_frames_counts(dirname):
    '''
    description: get the infomation of the video framnumbers, write the file with the format
        ['videoname, framcounts, maximumframeinsubtitle']
    param: dirname, the filepath of the videofile
    return {type} 
    author: mario
    '''
    # extract the startframe information
    starfile = '../data/bbcposestartinfo.txt'
    startdict = {}
    # define the re pattern of the video index and startframe
    pattern1 = r'of\s*(\d+)\s*'
    pattern2 = r'is\s*(\d+).*$'
    with open(starfile) as f:
        lines = f.readlines()
        for line in lines:
            videoindex = int(re.findall(pattern1, line)[0])
            startindex = int(re.findall(pattern2, line)[0])
            startdict[videoindex] = startindex

    # extract the framecount info
    countdict = {}
    for i in range(1, 93):
        videopath = os.path.join(dirname, 'e%d.avi' % i)
        video = cv2.VideoCapture(videopath)
        if video.isOpened():
            count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            countdict[i] = count
        video.release()
    
    # get the maximaum frameindex from the dictinfo
    maxframdict = {}
    matfile = '../data/bbc_subtitles.mat'
    mat = loadmat(matfile)
    datas = mat['bbc_subtitles']

    for index, data in enumerate(datas[0]):
        if len(data[1][0]) == 0:
            maxframdict[index+1] = 0
        else:
            lastrecord = data[1][0][-1]
            maxframdict[index+1] = lastrecord[1][0][0]

    with open('../data/videoframeinfo.txt', 'w') as f:
        for i in list(countdict.keys()):
            f.write('e%d.avi\t%6d\t%6d\t%6d\n' % (i, startdict[i], countdict[i], maxframdict[i]))


# 将二进制字符串表示为Unicode字符串
def Encode(strs):
    if isinstance(strs, bytes):
        strs = str(strs, encoding='utf8')
    return strs


def GetPlatformInfo():
    # get the name of the platform
    platform = sys.platform
    existfile = '/home/mario/sda'
    gpu_num = torch.cuda.device_count()

    if platform == 'win32':
        name = 'laptop'
    elif gpu_num > 0:
        if os.path.exists(existfile):
            name = 'pc'
        elif gpu_num == 2:
            name = 'server_hit605'
        else:
            name = 'server_k1320'
    else:
        name = None
    
    return name


'''
Train and Test
'''


def RandomSplitSamples(all_samples, labelist=None, split_tuple=(0.6, 0.2, 0.2), max_num=None, ):
    ''' randomly split the samples into different subsets for training, validation, and testing

    args:
        all_samples: list, the source data
        max_num: the maximum number of each class
        labels: list, the label of the samples in all_samples
        split_tuple: tuple, the split ratio of the subsets
    '''
    labelist = np.zeros(len(all_samples), dtype=np.int32) if labelist is None else np.array(labelist)
    labels = set(labelist)

    assert np.sum(np.array(split_tuple)) == 1
    assert len(all_samples) == len(labelist)

    split_samples = [[] for i in range(len(split_tuple))]
    split_labels = [[] for i in range(len(split_tuple))]

    for label in labels:
        samples = [x for i, x in enumerate(all_samples) if labelist[i] == label]
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        max_num = len(samples) if max_num is None else max_num

        indices = indices[:max_num]
        split_indices = np.int64(np.cumsum(split_tuple) * len(indices))
        split_indices = np.insert(split_indices, 0, 0)
   
        for i in range(len(split_tuple)):
            split_samples[i] += [samples[indices[j]] for j in range(split_indices[i], split_indices[i+1])]
            split_labels[i] += [label] * (split_indices[i+1] - split_indices[i])
    
    return split_samples, split_labels


def GetWantedWords(sourcepath, num=None):
    if sourcepath.endswith('.hdf5'):
        with h5py.File(sourcepath, mode='r') as h5file:
            words = list(h5file.keys())
    elif sourcepath.endswith('.txt'):
        with open(sourcepath, 'r') as f:
            lines = f.readlines()
        words = []
        for line in lines:
            word = line.split('\t')[0].strip()
            if word != '':
                words.append(word)
    if num is not None:
        words = words[:num]
    return words


def make_dataset(split_tuple, splitmode):
    # 对于 sample 文件来说，一般的结构为：word/train: [vidokey, idx, level, dist]
    # split_tuple = [cls_num, word_path, sample_datapath]
    cls_num, words_path, sample_file = split_tuple

    words = GetWantedWords(words_path, cls_num)
    cls_num = len(words)
    recordfile = h5py.File(sample_file, 'r')
    dataset = []
    num = 0
    for word in words:
        key = '%s/%s' % (word, splitmode)
        if key not in recordfile:
            continue

        num += 1
        candi_samples = recordfile[key][:]
        if len(candi_samples) == 0:
            print(f'the data of key: {key} is empty')
        for sample in candi_samples:
            videonum, begidx, level = np.int32(sample[:3])
            endidx = begidx + level
            dataset.append([word, videonum, begidx, endidx])
    
    print('there are actual %d/%d words' % (num, cls_num))
    # 这里返回的是名义上的words，这样可以使得clsnum一致的 posedataset 具有一致的label 映射
    return words, dataset


def Topk_Accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # ououtput: (batch_size, numcls)
    # target = (batch_size,)
    maxk = max(topk)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.sum(correct[:k].float())
        # correct_k1 = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        # assert correct_k == correct_k1
        res.append(correct_k)
    return res
    

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


if __name__ == "__main__":
    import time

    Testcode = 2
    if Testcode == 0:
        Records_Read_Write().Read_shaplets_cls_Records('../data/record.txt')
    elif Testcode == 1:
        A = np.random.rand(10, 3)
        B = np.random.rand(1000, 3)
        x = 1000
        t = time.time()

        for _ in range(x):
            d1, l1 = Calculate_shapelet_distance(A, B)
        t1 = time.time()

        for _ in range(x):
            d2 = SlidingDistance(A, B)
            l2 = np.argmin(d2)
            d2 = d2[l2]
        t2 = time.time()

        print(np.allclose(d1, d2))
        print('Orig %0.3f ms, second approach %0.3f ms' % ((t1 - t) * 1000., (t2 - t1) * 1000.))
        print('Speedup ', (t1 - t) / (t2 - t1))
    elif Testcode == 2:
        outputdir = './output'
        method = 1
        dataname = {1: 'sl', 2: 'sf'}
        numcls = 1000
        namepat = f'shapelet_{dataname[method]}_{numcls}_mul_\\d+.hdf5'

        destpath = f'shapelet_{dataname[method]}_{numcls}.hdf5'
        destpath = os.path.join(outputdir, destpath)
        filenames = os.listdir(outputdir)
        for filename in filenames:
            if re.match(namepat, filename):
                srcpath = os.path.join(outputdir, filename)
                CopyFromH5dfFile(srcpath, destpath)        