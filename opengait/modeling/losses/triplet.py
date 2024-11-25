import torch
import torch.nn.functional as F
import numpy as np
from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]

        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        mean_ap_dist = ap_dist.mean((1,2,3))
        mean_an_dist = an_dist.mean((1,2,3))

        emb_norm = torch.norm(embeddings, p=2, dim=2)
        #emb_var = 0.1*torch.var(emb_norm)
        mean_norm = torch.mean(emb_norm)

        # max_ap_dist = ap_dist.reshape(ap_dist.size()[0],ap_dist.size()[1]*ap_dist.size()[2]*ap_dist.size()[3]).max(1)[0]
        # min_an_dist = an_dist.reshape(ap_dist.size()[0],ap_dist.size()[1]*ap_dist.size()[2]*ap_dist.size()[3]).min(1)[0]
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone(),
            'mean_ap_dist': mean_ap_dist.detach().clone(),
            'mean_an_dist': mean_an_dist.detach().clone(),
            'mean_norm': mean_norm.detach().clone(),})
            #'max_ap_dist': max_ap_dist.detach().clone(),
            #'min_an_dist': min_an_dist.detach().clone(),})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)  # [n, p, postive, 1]
        an_dist = dist[:, diffenc].view(p, n, 1, -1)  # [n, p ,1, negative]
        #print(torch.max(an_dist),torch.max(ap_dist))
        return ap_dist, an_dist

    
class m_simce(BaseLoss):
    def __init__(self, start = 30000, loss_term_weight=1.0):
        super(m_simce, self).__init__(loss_term_weight)
        self.start = start
    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]

        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)

        ap_dist_exp = torch.exp(ap_dist/5)
        an_dist_exp = torch.exp(an_dist/5)
        an_dist_exp_sum = torch.sum(an_dist_exp, dim=-1, keepdim=True)

        simce_loss = ap_dist_exp+an_dist_exp_sum
        simce_loss = ap_dist_exp/simce_loss
        loss = -0.2*torch.log(simce_loss)


        self.info.update({
            'loss': loss.detach().clone()})

        return loss, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1)  # [n, p, postive]
        an_dist = dist[:, diffenc].view(p, n, -1)  # [n, p, negative]
        #print(torch.max(an_dist),torch.max(ap_dist))
        return ap_dist, an_dist
