import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from tqdm import trange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from datetime import timedelta



def train(model, df_log, cost = nn.MSELoss(), opt = torch.optim.Adam, pars = {'lr' : 0.0001} , epoch = 500, timestamp = 5, return_loss = False):
    losses = [np.nan]*epoch
    opt = opt(model.parameters(), **pars)
    for i in trange(epoch):
        total_loss = 0
        for k in range(0, df_log.shape[0] - 1, timestamp):
            index = min(k + timestamp, df_log.shape[0] - 1)
            batch_x = np.expand_dims(
                 df_log.iloc[index - timestamp : index, :].values, axis = 0
            )
            batch_y = df_log.iloc[index - timestamp + 1 : index + 1, :].values
            X = Variable(torch.tensor(batch_x, dtype = torch.float32))
            Y = Variable(torch.tensor(batch_y, dtype = torch.float32))
            prediction = model.forward(X)
            loss = cost(Y, prediction)

            loss.backward()
            opt.step()
            opt.zero_grad()

            total_loss += loss
        total_loss /= df_log.shape[0] // timestamp
        losses[i] = total_loss.detach().detach().numpy()
        if (i + 1) % 100 == 0:
            print('epoch:', i + 1, 'avg loss:', total_loss.detach().numpy())
        if return_loss:
            return losses
            
def predict(model, df_log, date_ori, future_day = 50, timestamp = 5):
    output_predict = np.zeros((df_log.shape[0] + future_day, df_log.shape[1]))
    output_predict[0, :] = df_log.iloc[0, :]
    upper_b = (df_log.shape[0] // timestamp) * timestamp
    model.train(False)
    for k in range(0, (df_log.shape[0] // timestamp) * timestamp, timestamp):
        index = min(k + timestamp, df_log.shape[0] -1)
        batch_x = np.expand_dims(
                df_log.iloc[index - timestamp : index, :].values, axis = 0
           )
        X = Variable(torch.tensor(batch_x, dtype = torch.float32))
        out_logits = model.forward(X)
        output_predict[k + 1 : k + timestamp + 1, :] = out_logits.detach().numpy()
    batch_x = np.expand_dims(df_log.iloc[-timestamp:, :], axis = 0)
    X = Variable(torch.tensor(batch_x, dtype = torch.float32))
    out_logits = model.forward(X)
    output_predict[ df_log.shape[0] + 1-5 : df_log.shape[0] + 1, :] = out_logits.detach().numpy()
    df_log.loc[df_log.shape[0]] = out_logits[0,-1, :].detach().numpy()
    date_ori.append(date_ori[-1] + timedelta(days = 1))
    for i in range(future_day - 1):
        batch_x = np.expand_dims(df_log.iloc[-timestamp:, :], axis = 0)
        X = Variable(torch.tensor(batch_x, dtype = torch.float32))
        out_logits = model.forward(X).detach().numpy()
        output_predict[df_log.shape[0], :] = out_logits[0,-1, :]
        df_log.loc[df_log.shape[0]] = out_logits[0,-1, :]
        date_ori.append(date_ori[-1] + timedelta(days = 1))
    return output_predict, date_ori