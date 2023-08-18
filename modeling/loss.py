import torch
import torch.nn as nn
import torch.nn.functional as F

from max_deeplab.losses import *

def semantic_loss():
    pass

def mask_loss():
    pass

def instance_loss():
    pass

def grouping_loss():
    pass

def mask_id_loss():
    pass

class ParagraphGroupingLoss(nn.Module):
    def __init__(self, tau=0.3, alpha=0.25, gamma=2.0, eps=1e-6):
        super(ParagraphGroupingLoss, self).__init__()
        self.alpha = alpha
        self.eps=eps
        self.gamma=gamma
        self.tau=tau
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input_masks, input_grouping, target_masks, target_grouping, matched_mask):
        """
        Loss = sum(sum(y_match(i)*y_match(j){alpha*A_match(i)_match(j)[-log(A_i_j)]+(1-alpha)*(1-A_match(i)_match(j))[-log(1-A_i_j)]}))

        matched_mask: (inp_pos_idx, tgt_pos_idx, neg_idx)
        input_mask: [B, N, H, W]
        input_grouping: [B, N, N]
        target_masks: [B, K, H, W]
        target_grouping: [B, K] 

        inp_pos_idx = 
        """
        (inp_pos_idx, tgt_pos_idx, neg_idx) = matched_mask
        b, n = input_masks.size()[:2]

        # step 1： calculate gt affinity
        has_group_id = (torch.sum(target_grouping + 1, axis=1) > 0).to(torch.float32) # [B]
        has_para_lable_gt = has_group_id.unsqueeze(1).unsqueeze(1) # [B, 1, 1]
        _device = target_grouping.device
        matching =  torch.zeros(b, n, n).to(_device)
        _combined_pos_idx = (inp_pos_idx[0], inp_pos_idx[1], tgt_pos_idx[1])
        matching[_combined_pos_idx] = 1
        pred_label_gt = torch.einsum('bij,bj -> bi', matching, target_grouping + 1) # B, N
        pred_label_gt_pad_col = pred_label_gt.unsqueeze(-1)
        pred_label_gt_pad_row = pred_label_gt.unsqueeze(1)
        gt_affinity = torch.eq(pred_label_gt_pad_col, pred_label_gt_pad_row).to(torch.float32) # [B, N, N]
        gt_affinity_mask = (has_para_lable_gt * pred_label_gt_pad_col * pred_label_gt_pad_row) # [B, N, N]
        gt_affinity_mask = torch.not_equal(gt_affinity_mask, 0.0).to(torch.float32)

        # step 2: predict affinity
        affinity = input_grouping

        # step 3: compute loss
        affinity = affinity.reshape(-1,1)
        gt_affinity = gt_affinity.reshape(-1,1)
        gt_affinity_mask = gt_affinity_mask.reshape(-1,1)
        pointwise_loss = self.bce(affinity/self.tau, gt_affinity) # [B * N * N,]

        # vanilla
        # loss = torch.sum(pointwise_loss * gt_affinity_mask) / (torch.sum(gt_affinity_mask) + self.eps)

        # balanced
        pos_mask = gt_affinity_mask * gt_affinity # [B, ]
        pos_loss = torch.sum(pointwise_loss * pos_mask) / (torch.sum(pos_mask) + self.eps) 
        neg_mask = gt_affinity_mask * (1. - gt_affinity) # [B, ]
        neg_loss = torch.sum(pointwise_loss * neg_mask) / (torch.sum(neg_mask) + self.eps)
        loss = 0.25 * pos_loss + 0.75 * neg_loss
        return loss

class UnifiedDetectorLoss(nn.Module):
    def __init__(
        self,
        pq_loss_weight=3,
        instance_loss_weight=1,
        maskid_loss_weight=1e-4,
        semantic_loss_weight=1,
        grouping_loss_weight=1,
        alpha=0.75,
        tau=0.3,
        eps=1e-6,
        gamma=2.0
    ):
        super(UnifiedDetectorLoss, self).__init__()
        self.pqw = pq_loss_weight
        self.idw = instance_loss_weight
        self.miw = maskid_loss_weight
        self.ssw = semantic_loss_weight
        self.grw = grouping_loss_weight
        self.pq_loss = PQLoss(alpha, eps)
        self.instance_loss = InstanceDiscLoss(tau, eps)
        self.maskid_loss = MaskIDLoss()
        self.semantic_loss = SemanticSegmentationLoss()
        self.grouping_loss = ParagraphGroupingLoss(tau, alpha, gamma, eps)

    def forward(self, input_tuple, target_tuple):
        """
        input_tuple: (input_masks, input_classes, input_semantic_segmentation, input_grouping, cls) Tensors
        target_tuple: (gt_masks, gt_classes, gt_semantic_segmentation, gt_grouping) NestedTensors
        """
        input_masks, input_classes, input_ss, input_grouping, mask_feature, cls = input_tuple
        gt_masks, gt_classes, gt_ss, gt_grouping, target_sizes = target_tuple

        pq, matched_mask = self.pq_loss(input_masks, input_classes, gt_masks, gt_classes, target_sizes)
        instdisc = self.instance_loss(mask_feature, gt_masks, target_sizes)

        #create the mask for maskid loss using argmax on ground truth
        maskid = self.maskid_loss(input_masks, gt_masks.argmax(1))
        semantic = self.semantic_loss(input_ss, gt_ss)
        grouping = self.grouping_loss(input_masks, input_grouping, gt_masks, gt_grouping, matched_mask)

        loss_items = {'pq': pq.item(), 'semantic': semantic.item(), 
                      'maskid': maskid.item(), 'instdisc': instdisc.item(), 'grouping': grouping.item()}

        total_loss = self.pqw * pq + self.ssw * semantic + self.miw * maskid + self.idw * instdisc + self.grw * grouping

        return total_loss, loss_items