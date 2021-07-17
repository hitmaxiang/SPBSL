import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import h5py

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src import utils


class ShapeletNet(nn.Module):
    def __init__(self, query=None):
        ''' initialize the shapelet Net

        args:
            query: the initial shapelet, with the shape of (1, D, m)

        '''
        super().__init__()
        d, m = query.shape[1:]
        self.m_len = m
        self.query = nn.Parameter(data=query, requires_grad=True)
        self.onesquery = nn.Parameter(data=torch.ones(1, d, m), requires_grad=False)
        self.Linear = nn.Linear(1, 2)
        self.min_dis, self.max_dis = None, None

    def forward(self, X, cumlens, UpdateMinMax=True):
        ''' the forward function of the Net model

        args:
            X: the train data, with shape of (1,D,T)
            Y: the label, with shape (N,)
            cumlens: the cumsum of [[0] + lens(X)]
        '''
        dist = self.slidingdistance(X)[0]
        dist = self.SubDis(dist, cumlens)[:, 0]
        dist = self.MinMaxscaler(dist, UpdateMinMax).unsqueeze(-1)
        Y = self.Linear(dist)
        return Y
    
    def MinMaxscaler(self, dis, UpdateMinMax=True):
        if UpdateMinMax or self.min_dis is None:
            self.min_dis, self.max_dis = torch.min(dis), torch.max(dis)
        dis = (dis - self.min_dis)/(self.max_dis - self.min_dis + 1e-16)
        return dis

    def SubDis(self, Dist, cumlens):
        subdis = torch.zeros(len(cumlens)-1, 2, device=self.query.device)
        for i in range(len(cumlens)-1):
            subdis[i] = torch.tensor(torch.min(Dist[cumlens[i]:cumlens[i+1]-self.m_len+1], dim=0))
        return subdis

    def slidingdistance(self, X):
        # X shape (1, D, T)
        QQ = torch.sum(torch.square(self.query))
        XX = F.conv1d(torch.square(X), weight=self.onesquery)
        QX = F.conv1d(X, weight=self.query)
        DIS = QQ + XX - 2 * QX
        DIS = DIS.squeeze(dim=1)
        return DIS
    
    def Evaluate(self, catX, cumlens, minscale=None, maxscale=None):
        update = True
        if minscale and maxscale:
            self.min_dis, self.max_dis = minscale, maxscale
            update = False
        with torch.no_grad():
            dist = self.slidingdistance(catX)[0]
            subdis = self.SubDis(dist, cumlens)
            dist = self.MinMaxscaler(subdis[:, 0], UpdateMinMax=update).unsqueeze(-1)
            Y = self.Linear(dist)
        return subdis, Y


def shapeletNetlearning(model, X, Y, lengths, savemodelpath, max_epochs=10000, verbose=False):
    optimier = torch.optim.Adam(model.parameters(), lr=0.03)
    model.train()

    best_score = 0
    Ite_tau = 200
    num = 0
    cumlens = np.insert(np.cumsum(lengths), 0, 0)

    for ite in range(max_epochs):
        Y_pre = model.forward(X, cumlens)
        loss = F.cross_entropy(Y_pre, Y)
        score = utils.Topk_Accuracy(Y_pre, Y)[0]/len(Y)

        if score > best_score:
            torch.save(model.state_dict(), savemodelpath)
            best_score = score
            num = 0
        else:
            num += 1
            if num > Ite_tau:
                break

        optimier.zero_grad()
        loss.backward()
        optimier.step()

        if verbose and ite % 10 == 0:
            print('Ite: %d with loss: %.4f and Acc: %.4f' % (ite, loss, score))
            print('the best score is %.4f with num: %d' % (best_score, num))
    
    model.load_state_dict(torch.load(savemodelpath))
    model.eval()
    subdis, Y_pre = model.Evaluate(X, cumlens)
    score = utils.Topk_Accuracy(Y_pre, Y)[0]/len(Y)
    if verbose:
        print('after %d epoches, the score is: train: %.4f, best: %.4f' %
              (ite, best_score, score))
    return subdis.cpu().numpy(), score.cpu().item()