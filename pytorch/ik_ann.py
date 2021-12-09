import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import math
## length of the robot arms 
l_1 = 2
l_2 = 2
l_3 = 2

## joint constraints
q1_bound = (0, math.pi)
q2_bound = (-math.pi, 0)
q3_bound = (-math.pi, math.pi)

## End effector coordinates in cartesian space
def x_e(q1, q2, q3):
    return l_1 * np.cos(q1) + l_2 * np.cos(q1+q2) + l_3 * np.cos(q1+q2+q3)

def y_e(q1, q2, q3):
    return l_1 * np.sin(q1) + l_2 * np.sin(q1+q2) + l_3 * np.sin(q1+q2+q3)

def orn_e(q1, q2, q3):
    return q1 + q2 + q3

def gen_data(dataset_size):
    ## sample random q1, q2 and q3
    q1_train = np.random.uniform(low=q1_bound[0], high=q1_bound[1], size=(dataset_size, 1))
    q2_train = np.random.uniform(low=q2_bound[0], high=q2_bound[1], size=(dataset_size, 1))
    q3_train = np.random.uniform(low=q3_bound[0], high=q3_bound[1], size=(dataset_size, 1))

    ## generate random data
    xe_train = x_e(q1_train, q2_train, q3_train)
    ye_train = y_e(q1_train, q2_train, q3_train)
    orn_train = orn_e(q1_train, q2_train, q3_train)

    targets = np.concatenate((q1_train, q2_train, q3_train), axis=1)
    inputs = np.concatenate((xe_train, ye_train, orn_train), axis=1)

    return inputs, targets

inputs, targets = gen_data(1000)
t_inputs, t_targets = gen_data(100)

## define a neural net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc_in = nn.Linear(3, 100)
        self.fc_out = nn.Linear(100, 3)
    def forward(self, x):
        x = self.fc_in(x)
        x = torch.tanh(x)
        x = self.fc_out(x)
        
        return x

net = Net()

## define a cost function and an optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

def val_model(net, t_inputs, t_targets, loss_fn):
    net.eval()
    loss_list = list()
    for t_inp, t_tgt in zip(t_inputs, t_targets):
        t_inp = torch.tensor(t_inp, dtype=torch.float32)
        t_tgt = torch.tensor(t_tgt)
        out = net(t_inp)
        loss = loss_fn(out, t_tgt)
        loss_list.append(loss.item())
    return np.mean(np.array(loss_list))

for e in range(1000):
    epoch_train_loss = list()
    for inp, tgt in zip(inputs, targets):
        inp = torch.tensor(inp, dtype=torch.float32)
        tgt = torch.tensor(tgt)
        optimizer.zero_grad()
        out = net(inp)
        loss = loss_fn(out.double(), tgt)
        loss.backward()
        optimizer.step()
        epoch_train_loss.append(loss.item())
    ## validate
    with torch.no_grad():
        val_error = val_model(net, t_inputs, t_targets, loss_fn)

    print("Train Error: {}, val Error: {}".format(np.mean(np.array(epoch_train_loss)), val_error))



