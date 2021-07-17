import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src import utils


class ShapeletMatrixModel():
    def __init__(self):
        pass
    
    def train(self, catsamples, labels, lengths, m_len):
        ''' 
        args:
            X: catsamples Ts x D
            Y: (samnum, )
            cumlens: [0] + cumlens
        '''
        
        bestscore = 0
        bestloss = float('inf')
        cumlens = np.insert(np.cumsum(lengths), 0, 0)

        shrink = - m_len + 1
        N, T = len(lengths), np.max(lengths)
        offset = [x + shrink for x in lengths]

        # 为了避免每次都要重新构建数据库，所以这里提前按照最大的sample构建距离矩阵
        DISMAT_pre = torch.zeros(T-m_len+1, cumlens[-1]+shrink, dtype=torch.float32)

        # 同样也会准备存放处理结果数据的
        MinDis = torch.zeros(N, T-m_len+1, dtype=torch.float32)
        MinLoc = torch.zeros(N, T-m_len+1, dtype=torch.int16)

        if torch.cuda.is_available():
            catsamples = catsamples.cuda()
            DISMAT_pre = DISMAT_pre.cuda()
            MinDis = MinDis.cuda()
            MinLoc = MinLoc.cuda()

        Dis_sample = np.zeros((N, T-m_len+1))
        Dis_loc = np.zeros(Dis_sample.shape, dtype=np.int16)
        dis = np.zeros((N,))
        locs = np.zeros((N,), dtype=np.int16)
        for i in range(N):
            if labels[i] == 0:
                continue

            # time_0 = time.time()
            begin, end = cumlens[i:i+2]
            # end = T * (i+1)
            with torch.no_grad():
                DISMAT_pre[:offset[i]] = utils.matrixprofile_torch(
                    catsamples[begin:end], 
                    catsamples, 
                    m_len, 
                    DISMAT_pre[:offset[i]])

                # time_1 = time.time()

                for j in range(N):
                    b_loc = cumlens[j]
                    e_loc = cumlens[j+1] + shrink
                    tempdis = DISMAT_pre[:offset[i], b_loc:e_loc]
                    MinDis[j, :offset[i]], MinLoc[j, :offset[i]] = torch.min(tempdis, dim=-1)

                Dis_sample[:] = MinDis.cpu().numpy()
                Dis_loc[:] = MinLoc.cpu().numpy()

            for candin_index in range(lengths[i]-m_len+1):
                
                score = self.Bipartition_score(Dis_sample[:, candin_index], labels.numpy(), bestscore)
                loss = np.mean(Dis_sample[:, candin_index][labels == 1])

                if score > bestscore:
                    bestscore = score
                    shapeindex = i
                    bestloss = loss
                    dis[:] = Dis_sample[:, candin_index]
                    locs[:] = Dis_loc[:, candin_index]
                elif score == bestscore and loss < bestloss:
                    shapeindex = i
                    bestloss = loss
                    dis[:] = Dis_sample[:, candin_index]
                    locs[:] = Dis_loc[:, candin_index]
            # time_2 = time.time()
            # print('%f----%f' % (time_1-time_0, time_2-time_1))
            # print('%d/%d--->loss: %f, accuracy: %f' % (i, N, bestloss, bestscore))
        self.shapeindex = shapeindex
        self.locs = locs
        self.dis = dis
        self.score = bestscore
    
    def Bipartition_score(self, distances, labels, bestscore):
        '''
        description: 针对一个 distances 的 二分类的最大分类精度
        param: 
            pos_num: 其中 distance 的前 pos_num 个 的标签为 1, 其余都为 0
        return: 最高的分类精度, 以及对应的分割位置
        author: mario
        '''     
        dis_sort_index = np.argsort(distances)
        correct = len(distances) - labels.sum()
        Bound_correct = len(distances)
        maxcorrect = correct

        for i, index in enumerate(dis_sort_index):
            if labels[index] == 1:  # 分对的
                correct += 1
                if correct > maxcorrect:
                    maxcorrect = correct
            else:
                correct -= 1
                Bound_correct -= 1
            if correct == Bound_correct:
                break
            if (Bound_correct/len(distances)) < bestscore:
                break
        
        score = maxcorrect/len(distances)

        return score