import torch
import argparse
import os
import cv2
import numpy as np
import json

from unified_detector import UnifiedDetector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./image/', help="Path of Image to Infer")
    parser.add_argument('--model', type=str, default='./model.pt', help="Path of model checkpoint")
    parser.add_argument('--output_dir', type=str, default='./outp/', help="Path to save result")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--img_size', type=int, default=1024)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--n_masks', type=int, default=384)
    parser.add_argument('--para_thres', type=float, default=0.5)
    parser.add_argument('--visualize', type=bool, default=True)
    return parser.parse_args()

def load_model(model_path, device, img_size, n_classes, n_masks):
    model = UnifiedDetector(img_size=img_size, n_classes=n_classes,n_masks=n_masks).to(device)
    model.load_state_dict(torch.load(model_path))
    return model

def pre_process(img, img_size):
    h,w,c = img.shape
    max_side = max(h,w)
    image_padded = np.ones((max_side, max_side, c)) * 127
    image_padded[:h, :w] = img
    ratio = max_side / img_size
    image_padded = cv2.resize(image_padded, (img_size, img_size)).astype(np.float32)
    image_padded = image_padded / (255 - 0) * 2 - 1.0
    return image_padded.transpose(2,0,1), ratio

def main(args):
    if os.path.isdir(args.image_dir):
        inputs = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir)]
    else:
        inputs = [args.image_dir]
    assert os.path.exists(args.model)
    net = load_model(args.model, args.device, args.img_size, args.n_classes, args.n_masks)
    with torch.no_grad():
        for img_path in inputs:
            img = cv2.imread(img_path)
            img, ratio = pre_process(img, args.img_size)
            img = torch.tensor(img).unsqueeze(0)
            out_tuple = net(img)
            masks, _, _, groups, _, clss = out_tuple
            masks, groups, clss = masks.numpy(), groups.numpy(), clss.numpy()
            indices = np.where(clss[0]==1)[0]
            # does masks need to softmax-argmax on dim 1 instead of taking (mask > 0.) to get non-overlapping instance? 
            mask_list = [masks[0, idx, :, :] for idx in indices]
            lines = []
            line_indices = []
            for index, mask in zip(indices, mask_list):
                line = {
                    'words': [],
                    'text': '',
                }

                cnts, _ = cv2.findContours((mask > 0.).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
                for cnt in cnts:
                    if (isinstance(cnt, np.ndarray)) and len(cnt.shape) == 3 and cnt.shape[0]>2 and cnt.shape[1] == 1 and cnt.shape[2] == 2:
                        cnt_list = (cnt[:,0] * ratio).astype(np.int32).tolist()
                        line['words'].append({'text':'', 'vertices': cnt_list})
                    else:
                        continue
                if line['words']:
                    lines.append(line)
                    line_indices.append(index)
            # line_grouping 
            affinity = groups[0][line_indices][:,line_indices]
            grps = list(range(len(line_indices)))
            for i1, i2 in zip(*np.where(affinity > args.para_thres)):
                grps[i1] = min(grps[i1], grps[i2])
                grps[i2] = min(grps[i1], grps[i2])
            grp_ids = list(set(grps))
            grps = np.array(grps)
            paragraphs = []
            for _distinct_id in grp_ids:
                paragraph = {'lines':[]}
                for _idx in np.argwhere(grps == _distinct_id):
                    paragraph['lines'].append(lines[_idx[0]])
                if paragraph:
                    paragraphs.append(paragraph)
            out_anno_path = os.path.join(args.output_dir, os.path.basename(img_path)[:-3]+'json')
            with open(out_anno_path, 'w') as f:
                f.write(json.dumps(paragraphs))



if __name__ == "__main__":
    args = parse_args()
    main(args)
