from torch.nn import functional as F
import torch.nn as nn
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
import numpy as np
# from ..modules import BasicConv2d

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x

block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}


class DeepGaitV2_2D(ResNet):
    def __init__(self, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(DeepGaitV2_2D, self).__init__(block, layers)

        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def fm(self, x):
        n, c, s, h, w = x.size()
        embed_x = x.float()
        embed_x = embed_x.reshape(n, c, s, h * w)
        embed_x = torch.mean(embed_x, dim=-1)
        embed_x = F.normalize(embed_x, dim=1)
        x_fm = embed_x.transpose(1, 2).matmul(embed_x)
        x_att = torch.sum(x_fm, dim=2).view(n,s,1)/s
        x_att_min = torch.min(x_att, dim=1)[0].view(n,1,1)
        x_att_max = torch.max(x_att, dim=1)[0].view(n,1,1)
        x_att = (x_att-x_att_min+0.025)/(x_att_max-x_att_min+0.025) #n,s,1

        return x_att
    def sqq(self, x):
        n, c, s, h, w = x.size()
        x1 = x.permute(0,2,1,3,4).reshape(n,s,-1)
        sum_x = torch.sum(x1,dim=-1)
        sum_x_max = torch.max(sum_x,dim=-1,keepdim=True)[0]
        sum_x_min = torch.min(sum_x,dim=-1,keepdim=True)[0]
        a2 = (sum_x_max-sum_x_min)/2
        a0 = torch.mean(sum_x,dim=-1,keepdim=True)
        f = (sum_x-a0)/a2
        f_max = torch.max(f,dim=-1,keepdim=True)[0]
        f_min = torch.min(f,dim=-1,keepdim=True)[0]
        f = (f-f_min)/(f_max-f_min)
        W=14
        L=s-W+1
        q=torch.zeros(n,L)
        for u in range(n):
            for i in range(L):
                r=[]
                for j in range(int(W/2)):
                    r_j = 0
                    for k in range(W-j):
                        r_j += f[u][i+k]*f[u][i+k+j]
                    r_j/=(W-j)
                    r.append(r_j)
                R = np.zeros((len(r),len(r)))
                for j in range(len(r)):
                    R[0,j]=r[j]
                    R[j,0]=r[j]
                for j in range(1,len(r)):
                    for k in range(1,len(r)):
                        R[j,k] = R[j-1][k-1]
                ev = np.linalg.eigvals(R)
                q[u][i]=np.sum(ev[:2])-np.sum(ev[4:])
        x_att = torch.zeros(n,s)
        num = torch.zeros(n,s)
        for i in range(n):
            for k in range(L):
                for j in range(int(W)):
                    x_att[i][k+j]+=q[i][k]
                    num[i][k+j]+=1
        x_att = x_att/num
        x_att_max = torch.max(x_att,dim=-1,keepdim=True)[0]
        x_att_min = torch.min(x_att,dim=-1,keepdim=True)[0]
        x_att = (x_att-x_att_min+0.1)/(x_att_max-x_att_min+0.1)
        x_att = x_att.unsqueeze(-1)
        return x_att

    def attention(self, x):
        n, c, s, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(n,s,c*h*w)
        x = x*self.x_att
        x = x.reshape(n,s,c,h,w).permute(0, 2, 1, 3, 4)
        return x


    def forward(self, x , n, s):
        bs = x.shape[0] // s
        x = x.view(bs, s, x.shape[1], x.shape[2], x.shape[3])
        x = x.permute(0, 2, 1, 3, 4)
        #self.x_att = self.sqq(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # bs = x.shape[0] // s
        # x = x.view(bs, s, x.shape[1], x.shape[2], x.shape[3])
        # x = x.permute(0, 2, 1, 3, 4)
        # self.x_att = self.fm(x)
        # x = self.attention(x)
        # x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        x = self.layer3(x)
        
        # x = x.view(bs, s, x.shape[1], x.shape[2], x.shape[3])
        # x = x.permute(0, 2, 1, 3, 4)
        # x = self.attention(x)
        # x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        x = self.layer4(x)

        # x = x.view(bs, s, x.shape[1], x.shape[2], x.shape[3])
        # x = x.permute(0, 2, 1, 3, 4)
        # x = self.attention(x)
        # x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])


        return x, None
