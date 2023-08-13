import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
import cv2

def truncate_or_pad(tsr, N, v):
    if tsr.size(0) >= N:
        return tsr[:N]
    else:
        num_to_pad = N - tsr.size(0)
        padding = [0] * 2 * len(tsr.size())
        padding[-1] = num_to_pad
        tsr = F.pad(tsr, padding, 'constant', v)
        return tsr
        

class HiertextDataset(Dataset):
    """HierText dataset"""

    def __init__(self, json_file, root_dir, transforms= None):
        self.anno = json.load(open(json_file,'r'))['annotations']
        self.root_dir = root_dir
        self.transforms = transforms
        self.output_size = 1024 # 暂时Padding Resize不放在Transforms里，mask在这里直接处理比较方便
        self.num_masks = 384 # 不单独写collatFn了

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.anno[idx]['image_id']+'.jpg')
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h,w,c  = img.shape[:3]
        max_side = max(h,w)
        ratio = float(self.output_size/max_side)
        output={}
        groups = []
        masks= []
        labels = []

        full_mask = np.ones((self.output_size, self.output_size)) * 2
        cnts = 0
        # detection unit 默认为 "Line"
        for groupid, para in enumerate(self.anno[idx]['paragraphs']):
            for line in para['lines']:
                if cnts >= self.num_masks - 2:
                    for word in line['words']:
                        box = np.array(word['vertices']).astype(np.int32)
                        cv2.fillPoly(img, [box], (255,255,255)) # 在图像中擦除未被计入的区域
                    continue
                mask = np.zeros((self.output_size,self.output_size))
                for word in line['words']:
                    box = (np.array(word['vertices']) * ratio).astype(np.int32)
                    cv2.fillPoly(mask, [box], 1)
                    cv2.fillPoly(full_mask, [box], 0)
                masks.append(mask)
                groups.append(groupid)
                if line['legible'] == True:
                    labels.append(1)
                else:
                    labels.append(0)
                cnts += 1
        masks.append(full_mask)
        labels.append(2)
        cnts += 1

        masks = np.stack(masks, axis=0)
        groups = np.array(groups)
        labels = np.array(labels)

        output['labels'] = truncate_or_pad(torch.tensor(labels, dtype=torch.long), self.num_masks, 0) # [K]
        output['masks'] = truncate_or_pad(torch.tensor(masks), self.num_masks, 0).float() # [K, 1024, 1024]
        output['semantic_mask'] = torch.einsum('nhw,n->hw', output['masks'].long(), output['labels']) # [1024, 1024]
        output['semantic_mask'] = (output['semantic_mask'] > 0).long()
        output['grouping'] = truncate_or_pad(torch.tensor(groups), self.num_masks, -1).float() # [K]
        output['num_mask'] = cnts

        image_padded = np.ones((max_side, max_side, c)) * 127
        image_padded[:h, :w] = img
        image_padded = cv2.resize(image_padded, (self.output_size, self.output_size)).astype(np.float32)
        image_padded = image_padded / (255 - 0) * 2 - 1.0
        image = torch.tensor(image_padded)
        image = image.permute(2,0,1)
        output['image'] = image

        return output
