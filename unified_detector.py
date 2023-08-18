import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from max_deeplab.model import MaXDeepLabS, MaXDeepLabSEncoder, MaXDeepLabSDecoder, SemanticHead
from modeling.modules import *

class UnifiedDetector(nn.Module):
    def __init__(self, img_size = 1024, n_masks= 384, n_classes=3, dimension=256):
        super(UnifiedDetector, self).__init__()
        self.global_memory = nn.Parameter(torch.randn((n_masks, 256)), requires_grad=True)
        self.MaXDeepLabBackbone = MaXDeepLabSEncoder(im_size = img_size)
        self.TextClusterHead =  TextClusterHead(dimension=dimension)  # Groups Output Head
        # self.TextDetectionHead =    # Mask Output Head
        # self.TextnessHead =     # Classes Output Head
        self.MaXDeepLabDecoder = MaXDeepLabSDecoder(im_size = img_size, n_classes=n_classes, n_masks=n_masks)
        self.SemanticHead = SemanticHead(n_classes=3)

    def forward(self, P):
        M = repeat(self.global_memory, 'n k -> n b k', b=P.size(0))
        pixel_feature, memory = self.MaXDeepLabBackbone(P, M)
        cluster_feature, group_feature = self.TextClusterHead(memory) # [N, B, C]
        group_feature = group_feature.permute(1,0,2) # [B, N, C]
        group_feature_t = torch.transpose(group_feature, 1, 2)
        semantic = self.SemanticHead(pixel_feature[-1]) # last stage feature
        affinity = torch.matmul(group_feature, group_feature_t)
        mask_out, classes, mask_feature = self.MaXDeepLabDecoder(pixel_feature, memory)
        cls_prob = F.softmax(classes, dim=-1)
        cls = torch.argmax(cls_prob, dim=-1)
        return mask_out, classes, semantic, affinity, mask_feature, cls
