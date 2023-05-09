import os
import sys
import torch
import numpy as np
from loss import CtdetLoss
from torch.utils.data import DataLoader
from dataset import ctDataset
from models import DlaNet
from torch.utils.tensorboard import SummaryWriter
from eval import pre_recall

writer = SummaryWriter(log_dir='logs', flush_secs=60)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

use_gpu = torch.cuda.is_available()

model = ResNet(34)

print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

loss_weight = {'hm_weight': 1, 'wh_weight': 0.1, 'ang_weight': 0.1, 'reg_weight': 0.1}
criterion = CtdetLoss(loss_weight)

device = torch.device("cuda")
if use_gpu:
    model.cuda()
model.train()

learning_rate = 1.25e-4
num_epochs = 20

params = []
params_dict = dict(model.named_parameters())
for key, value in params_dict.items():
    params += [{'params': [value], 'lr': learning_rate}]

optimizer = torch.optim.AdamW(params, lr=7, weight_decay=1e-4)
train_dataset = ctDataset(split='train')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=0)

test_dataset = ctDataset(split='val')
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
print('the dataset has %d images' % (len(train_dataset)))
num_iter = 0
best_test_loss = np.inf
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)


for epoch in range(num_epochs):
    model.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    total_loss = 0.

    for i, sample in enumerate(train_loader):
        for k in sample:
            sample[k] = sample[k].to(device=device, non_blocking=True)
        pred = model(sample['input'])
        loss = criterion(pred, sample)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            writer.add_scalar('training loss',
                              total_loss / 1000,
                              epoch)
        if (i + 1) % 5 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data, total_loss / (i + 1)))
            num_iter += 1

    validation_loss = 0.0
    model.eval()
    for i, sample in enumerate(test_loader):
        if use_gpu:
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)

        pred = model(sample['input'])
        loss = criterion(pred, sample)
        validation_loss += loss.item()

        if i % 10 == 9:
            writer.add_scalar('vaidation loss',
                              validation_loss / 1000,
                              epoch)
    validation_loss /= len(test_loader)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(model.state_dict(),'best.pth')

