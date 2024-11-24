import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, SpatialPoolingPyramid, HorizentalTemporalPoolingPyramid


class Baseline_spp(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.SP = SpatialPoolingPyramid(torch.mean)
        self.HSP = HorizentalTemporalPoolingPyramid(max_num=model_cfg['max_num'])
    def forward(self, inputs):

        ipts, labs, _, _, _, _, seqL = inputs
        #print(inputs[4][0],inputs[2][0],inputs[3][0],inputs[5][0])
        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        
        if sils.size()[2]==1:
            sils = torch.cat((sils,sils,sils),dim=2)
        if sils.size()[2]==2:
            sils = torch.cat((sils,sils[:,:,-1,:,:].unsqueeze(2)),dim=2)
        if sils.size()[2]%3!=0:
            num = sils.size()[2]//3
            sils = sils[:,:,:num*3,:,:]



        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        seqL[0] = sils.size()[2]

        outs_tp, indice = self.TP(outs, seqL, options={"dim": 2})  # [n, c, h, w]


        outs_sp = self.SP(outs)
        feat_tp = self.HPP(outs_tp)  # [n, c, p]
        feat_sp = self.HSP(outs_sp)
        feat = torch.cat((feat_tp, feat_sp), dim=-1)


        #feat = self.HPP(outs_tp)  # [n, c, p]


        embed_1 = self.FCs(feat)  # [n, c, p]
        embed = embed_1

        n, _, s, h, w = sils.size()

        bnn = self.BNNecks.fc_bin[:, :, labs].permute(2, 1, 0).contiguous().float()  # [n,c,p] [p,c,class]

        if self.training:
            embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embed_1, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs, },
                },
                'visual_summary': {
                    'image/sils': sils.reshape(n*s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }
        else:
            retval = {
                'visual_summary': {
                    'image/sils': sils.view(n * s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }
        return retval
