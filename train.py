import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from einops import rearrange
# import pytorch_lightling as pl # TODO

from unified_detector import UnifiedDetector
from modeling.loss import UnifiedDetectorLoss
from datasets import HiertextDataset

import sys

json_file = sys.argv[1]
root_dir = sys.argv[2]
batch_size = int(sys.argv[3])

data = HiertextDataset(json_file, root_dir)
loader = DataLoader(data, batch_size, shuffle=True)
device = torch.device('cuda:0')

model = UnifiedDetector(img_size=1024, n_classes=3, n_masks=384).cuda()
criterion = UnifiedDetectorLoss().cuda()
optimizer = Adam(model.parameters(), lr=3e-4)

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    for _i, _d in enumerate(loader):

        P = _d['image'].to(device)
        masks = _d['masks'].to(device)
        labels = _d['labels'].to(device)
        semantic = _d['semantic_mask'].to(device)
        grouping = _d['grouping'].to(device)
        tgt_size = _d['num_mask']
        target_tuple = (masks, labels, semantic, grouping, tgt_size)

        optimizer.zero_grad()
        out = model(P)

        loss, loss_item = criterion(out, target_tuple)
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss
        if _i % 10 == 9:
            last_loss = running_loss / (i+1)
            print(f'epoch: {epoch_index} step: {_i + 1} running loss: {last_loss}')

    return last_loss, loss, loss_item

for i in range(100):
    last_loss, loss, loss_item = train_one_epoch(i+1)
    loss_string = ','.join(["{}: {}".format(k,v) for k,v in loss_item.items()])
    print(f'epoch {i+1}: running loss: {last_loss} current loss: {loss} \n loss items: {loss_string}')
    torch.save(model.state_dict(), f'epoch{i+1}.pt')

