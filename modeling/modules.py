import torch
import torch.nn as nn

class linear_bn_relu(nn.Module):
    def __init__(self, dimension):
        super(linear_bn_relu, self).__init__()
        self.linear = nn.Linear(256, dimension, bias=False)
        self.bn = nn.BatchNorm1d(dimension)
        self.activation = nn.ReLU(inplace=False)

    def forward(self, feature):
        out = self.linear(feature) 
        out = out.permute(1,2,0) # [384, b, 256] -> [b, 256, 384]
        out = self.bn(out)
        out = out.permute(2,0,1) # [b, 256, 384] -> [384, b, 256]
        out = self.activation(out)
        return out

class conv_bn_relu(nn.Module):
    def __init__(self, dimension):
        super(conv_bn_relu, self).__init__()
        self.Conv1d = nn.Conv1d(256, dimension, bias=False, kernel_size=1)
        self.Bn = nn.BatchNorm1d(dimension)
        self.activation = nn.ReLU(inplace=False)


    def forward(self, feature):
        feature = feature.permute(1,2,0) # [384, b, 256] - > [b, 256, 384]  tensorflow is chanel last, pytorch is chanel first
        feature = self.Conv1d(feature) # [b, 256, 384]
        feature = self.Bn(feature) # [b, 256, 384]
        feature = feature.permute(2,0,1) # [b, 384, 256]
        feature = self.activation(feature)
        return feature

class transformer_stack(nn.Module):
    def __init__(self, dimension, num_layers):
        super(transformer_stack, self).__init__()
        self.net = nn.Sequential()
        for i in range(num_layers):
            self.net.append(transformer_block(dimension, dropout_prob=0.1))
        
        self.net.append(nn.LayerNorm(256, 1e-6))

    def forward(self, group_feature):
        return self.net(group_feature)

class transformer_block(nn.Module):
    def __init__(self, dimension, dropout_prob=0.1):
        super(transformer_block, self).__init__()
        self.layernorm = nn.LayerNorm(dimension, 1e-6)
        self.multiheadattn = nn.MultiheadAttention(embed_dim=dimension, num_heads=4, dropout=0.1)
        self.drop = nn.Dropout(dropout_prob)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(dimension, 1e-6),
            nn.Linear(dimension, 512, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(512, dimension, bias=True),
            nn.Dropout(dropout_prob)
        )

    def forward(self, group_feature):
        s1 = self.layernorm(group_feature)
        attn, _ = self.multiheadattn(s1,s1,s1)
        s1 = self.drop(attn)
        s1 = s1 + group_feature
        s2 = s1 + self.feedforward(s1)
        return s2


class TextClusterHead(nn.Module):
    def __init__(self, dimension):
        super(TextClusterHead, self).__init__()
        self.class_extra = conv_bn_relu(dimension)
        self.class_embed = linear_bn_relu(dimension)
        self.group_embed = linear_bn_relu(dimension)
        self.para_head = transformer_stack(dimension, 3)
        self.para_proj = nn.Linear(256, dimension)

    def forward(self, memory):
        """dimension可能还是有点问题

        Args:
            memory: (N, B, C)
        """
        class_feature = self.class_extra(memory)
        class_feature = self.class_embed(class_feature)
        group_feature = self.group_embed(class_feature)
        # atten_bias = torch.tensor
        group_feature = self.para_head(group_feature)
        group_feature = self.para_proj(group_feature)
        group_feature = nn.functional.normalize(group_feature, dim=-1)
        return class_feature, group_feature