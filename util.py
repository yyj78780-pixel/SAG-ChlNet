
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import math
from net import *

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))
    
def split_nodes_into_regions(num_nodes, num_regions):
    nodes_per_region = num_nodes // num_regions
    region_groups = []
    for i in range(num_regions):
        start = i * nodes_per_region
        end = (i + 1) * nodes_per_region if i != num_regions - 1 else num_nodes
        region_groups.append(list(range(start, end)))
    return region_groups
class DataLoaderS(object):      
    # train and valid are ratios of dataset, test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, 
                 normalize=2, time_steps=2350, num_nodes=300, num_regions=70):
        self.P = window
        self.h = horizon
        self.device = device

        import pandas as pd
        df = pd.read_csv(file_name, skiprows=1, header=None)
        data = df.iloc[:, 3:].values  
        data_matrix = data.T  
        print("shape:", data_matrix.shape)
        assert data_matrix.shape[0] == time_steps, f"Number of time steps does not match, expected {time_steps}, got {data_matrix.shape[0]}"
        assert data_matrix.shape[1] == num_nodes, f"Number of nodes does not match, expected {num_nodes}, got {data_matrix.shape[1]}"
        self.raw_data = data_matrix.copy()
        self.dat = data_matrix.copy() 
        #self._remove_outliers(method="iqr", threshold=1.5)
        #cleaned_data = self._remove_outliers(self.dat, method="zscore", threshold=3.0)
        #self.dat = cleaned_data 


        #before = self.dat.copy()
       # self._remove_outliers()
        #print("Outlier ratio:", np.sum(before != self.dat) / before.size)
        self.n, self.m = self.dat.shape
        self.num_regions = num_regions
        self.region_groups = split_nodes_into_regions(self.m, self.num_regions)
        self.normalize = normalize
        self.scale = np.ones(self.m)
        train_len = int(train * self.n)
        valid_len = int((train + valid) * self.n)
        self._normalized(normalize, train_len)
        self._split(train_len, valid_len, self.n)
        print(f"train: 0 ~ {train_len}, valid: {train_len} ~ {valid_len}, test: {valid_len} ~ {self.n}")
        print(f"{self.n - 45} ~ {self.n}")
        self.scale = torch.from_numpy(self.scale).float().to(device)
        tmp = self.test[1] * self.scale.view(1, 1, -1)
        tmp = tmp.reshape(-1, self.m)

        self.scale = Variable(self.scale)
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
        self.rmse = torch.sum(torch.abs(tmp))

    def _normalized(self, normalize, train_len):
        # Scale from the training period only, then apply to all splits.
        train_dat = self.dat[:train_len]
        eps = 1e-6
        if normalize == 0:
            pass
        elif normalize == 1:
            max_val = np.max(np.abs(train_dat))
            self.scale[:] = max_val
            self.dat = self.dat / (max_val + eps)
        elif normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(train_dat[:, i]))
                self.dat[:, i] = self.dat[:, i] / (self.scale[i] + eps)

    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))  
        Y = torch.zeros((n, horizon, self.m)) 

        for i in range(n):
            end = idx_set[i] - horizon + 1
            start = end - self.P
            X[i] = torch.from_numpy(self.dat[start:end, :])
            Y[i] = torch.from_numpy(self.dat[end:end + horizon, :])


        X = X.permute(0, 2, 1).unsqueeze(1)

        X = X.to(self.device)
        Y = Y.to(self.device)
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt].to(self.device)
            Y = targets[excerpt].to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class Optim(object):
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr, weight_decay=self.lr_decay)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, weight_decay=self.lr_decay)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, clip, lr_decay=1, start_decay_at=None):
        self.params = params  # careful: params may be a generator
        self.last_ppl = None
        self.lr = lr
        self.clip = clip
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        grad_norm = 0
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.optimizer.step()
        return grad_norm

    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True
        if self.start_decay:
            self.lr = self.lr * self.lr_decay
        self.start_decay = False
        self.last_ppl = ppl
        self._makeOptimizer()
