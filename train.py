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


target_tuple = (masks, labels, semantic, grouping, tgt_size)

model = UnifiedDetector(img_size=1024, n_classes=3, n_masks=384).cuda()
criterion = UnifiedDetectorLoss().cuda()
optimizer = Adam(model.parameters(), lr=3e-4)

for i in range(100):
    optimizer.zero_grad()
    batch = next(iter(loader))
    P = batch['image'].to(device)

    masks = batch['masks'].to(device)
    labels = batch['labels'].to(device)
    semantic = batch['semantic_mask'].to(device)
    grouping = batch['grouping'].to(device)
    tgt_size = batch['num_mask']

    out = model(P)
    loss, loss_item = criterion(out, target_tuple)
    loss.backward()
    optimizer.step()

    if (i+1) % 20 == 0:
        loss_string = ','.join(["{}: {}".format(k,v) for k,v in loss_item.items()])
        print(f'epoch {i+1}: loss.item()')
        print(loss_string)
        torch.save(model.state_dict(), f'epoch{i+1}.pt')