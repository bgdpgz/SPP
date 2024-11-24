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



class TripletDouble8124Loss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0, start=30000,):
        super(TripletDouble8124Loss, self).__init__(loss_term_weight)
        self.margin = margin
        self.count = 0
        self.start = start
        self.eps = 1e-3

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels, bnn):
        # embeddings: [n, c, p], label: [n], bnn: [p, n, c]
        self.centers = bnn.permute(2, 0, 1).contiguous().float()
        embeddings = embeddings.permute(2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels

        distmat = torch.pow(embeddings, 2).sum(dim=2) + torch.pow(self.centers, 2).sum(dim=2) - (
                    2 * embeddings * self.centers).sum(dim=2)  # [p,n]
        distmat = torch.sqrt(F.relu(distmat))
        dist_d = distmat.mean()

        # embeddings ?
        if self.count >= self.start:
            dist = self.ComputeDistance(embeddings, ref_embed)
            mean_dist = dist.mean((1, 2))
            ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist, distmat)  # 8, 124

        else:
            self.count += 1
            if self.count == self.start:
                print("---------------------------starting!---------------------------")
            dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
            mean_dist = dist.mean((1, 2))  # [p]
            ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)

        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)  # dist_diff
        dist_d /= 128
        loss = F.relu(dist_diff + self.margin) #+ dist_d)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'center_loss': dist_d.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

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
            x: [p, n_x, c] embeddings
            y: [p, n_y, c]
        """

        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist
    def Convert2Triplets(self, row_labels, clo_label, dist, dist_d=None):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        #
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c] [128, 128]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()  # [p, n, n]
        ap_dist = dist[:, matches].view(p, n, -1, 1)  # [p, n, 4, 1]
        an_dist = dist[:, diffenc].view(p, n, 1, -1)  # [p, n, 1, 124]
        if dist_d != None:
            ap_dist = torch.cat([ap_dist, dist_d.reshape(p,n,1,1)], dim=2)  # # [p, n, 5, 1]
        return ap_dist, an_dist

class TripletLoss_Max(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss_Max, self).__init__(loss_term_weight)
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
        max_ap_dist = torch.max(ap_dist,dim=2)[0]
        min_an_dist = torch.min(an_dist,dim=2)[0]
        # max_ap_dist = ap_dist.reshape(ap_dist.size()[0],ap_dist.size()[1]*ap_dist.size()[2]*ap_dist.size()[3]).max(1)[0]
        # min_an_dist = an_dist.reshape(ap_dist.size()[0],ap_dist.size()[1]*ap_dist.size()[2]*ap_dist.size()[3]).min(1)[0]
        dist_diff = (max_ap_dist - min_an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.MaxNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone(),
            'max_ap_dist': max_ap_dist.detach().clone(),
            'min_an_dist': min_an_dist.detach().clone(),})

        return loss_avg, self.info

    def MaxNonZeroReducer(self, loss):
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
        ap_dist = dist[:, matches].view(p, n, -1)  # [n, p, postive, 1]
        an_dist = dist[:, diffenc].view(p, n, -1)  # [n, p ,1, negative]
        #print(torch.max(an_dist),torch.max(ap_dist))
        return ap_dist, an_dist



class TripletMLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletMLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.iter = 0
    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]

        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff+0.1) #+ self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone(),
            'iter': 1+0.5*self.iter/120000},)
        self.iter+=1
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
        #a = ap_dist
        ap_dist = torch.pow(ap_dist, 2) + self.margin
        ap_dist = (1+0.5*self.iter/120000)*torch.sqrt(ap_dist)
        #ap_dist = (1 + 0.5 * self.iter / 120000) * ap_dist

        #ap_dist = torch.sqrt(ap_dist)
        #ap_dist = torch.pow(ap_dist,2)+self.margin
        #ap_dist = torch.sqrt(ap_dist)
        #print(a[0][0])

        an_dist = dist[:, diffenc].view(p, n, 1, -1)  # [n, p ,1, negative]
        #a = an_dist
        an_dist = torch.pow(an_dist,2)-self.margin
        an_dist = torch.clip(an_dist, min=0.01, max=100)
        an_dist = (1-0.5*self.iter/120000)*torch.sqrt(an_dist)
        an_dist = (1 - 0.5 * self.iter / 120000) * an_dist


        #print(ap_dist[0][0],an_dist[0][0])
        #print(a[0][0])
        # print(a[0][0]-an_dist[0][0])

        return ap_dist, an_dist



class TripletAttLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletAttLoss, self).__init__(loss_term_weight)
        self.margin = margin
        # self.register_buffer('batch_mean', torch.zeros(16,1))
        # self.register_buffer('batch_std', torch.zeros(16,1))
        # self.batch_std = 0
        # self.batch_mean = 0
        # self.flag = 0
        self.iter=0
        self.t_alpha = 0.01
        self.eps = 1e-3
    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]

        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
        p, n, c = embeddings.size()
        # with torch.no_grad():
        #     embeddings_norm = torch.norm(embeddings, p=2, dim=2)
        #     embeddings_norm = torch.clip(embeddings_norm, min=0.001, max=100)
        #     embeddings_norm = embeddings_norm.clone().detach()
        #     #print(embeddings_norm[0])
        #     mean = torch.mean(embeddings_norm, dim = 1, keepdim = True).detach()
        #     std = torch.std(embeddings_norm, dim = 1, keepdim = True).detach()
        #     # if self.flag==0:
        #     #     self.flag = 1
        #     #     self.batch_mean += mean
        #     #     self.batch_std += std
        #     # else:
        #     #     self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
        #     #     self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std
        #     margin_scaler = (embeddings_norm - mean) / (std + self.eps)  # 66% between -1, 1
        #     #margin_scaler=margin_scaler*0.1
        #     #print((n*p-torch.sum(margin_scaler<=-0.2)-torch.sum(margin_scaler>=0.2))/(n*p))
        #     margin_scaler = margin_scaler.view(p, n, 1, 1)
        #     margin_scaler = margin_scaler*0.004+2
        #
        #     #print((n * p - torch.sum(margin_scaler >= 2.01) - torch.sum(margin_scaler <= 1.99)) / (n * p))
        #     margin_scaler = self.margin * torch.clip(margin_scaler, min=1.99, max=2.01).clone().detach()
        #     #print(torch.mean(margin_scaler))
        #     #print(torch.mean(margin_scaler))



        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        #n1, n2 = ap_dist.size()[2], an_dist.size()[3]

        #print(an_dist[0][0])
        #margin = torch.rand(size=(p, n, 4, 124),device=dist.device)*0.04+0.18
        # m = margin_scaler.cpu().detach().numpy()
        # m1 = np.zeros((p,n,1,1))
        # m1 += m
        # margin = torch.from_numpy(m1).cuda()
        # margin_scaler = margin.repeat(1, 1, n1, n2).view(dist.size(0), -1)
        dist_diff = (1.5*ap_dist - an_dist).view(dist.size(0), -1)
        # print((ap_dist - an_dist).size())
        # print(margin_scaler.size())

        #margin = margin.view(dist.size(0), -1)
        loss = F.relu(dist_diff+self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})
        self.iter+=1
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
        #print(torch.mean(ap_dist))
        return ap_dist, an_dist



class TripletCenterLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0, start = 0):
        super(TripletCenterLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.start = start
        self.count = 0
    @gather_and_scale_wrapper
    def forward(self, embeddings, bnn, labels):
        # embeddings: [n, c, p], label: [n], bnn: [n,c,p]

        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
        p, n, c = embeddings.size()
        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        if self.count <= self.start:
            ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
            # mean_ap_dist = ap_dist.mean((1,2,3))
            # mean_an_dist = an_dist.mean((1,2,3))
            # max_ap_dist = ap_dist.reshape(ap_dist.size()[0],ap_dist.size()[1]*ap_dist.size()[2]*ap_dist.size()[3]).max(1)[0]
            # min_an_dist = an_dist.reshape(ap_dist.size()[0],ap_dist.size()[1]*ap_dist.size()[2]*ap_dist.size()[3]).min(1)[0]
            dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        else:
            self.centers = bnn.permute(2, 0, 1).contiguous().float() #p,n,c
            distmat_ap = torch.pow(embeddings, 2).sum(dim=2) + torch.pow(self.centers, 2).sum(dim=2) - (
                    2 * embeddings * self.centers).sum(dim=2)  # [p,n]
            distmat_ap = torch.sqrt(F.relu(distmat_ap)) # [p,n]
            for i in range(n):
                labels_i = labels!=labels[i]
                center_i = self.centers[:,labels_i,:] #p,n,c
                distmat_i = self.ComputeDistance(embeddings[:, i, :].unsqueeze(1), center_i)
                if i==0:
                    distmat = distmat_i
                else:
                    distmat = torch.cat([distmat,distmat_i],dim=1)
            #print(distmat.size())
            #distmat = self.ComputeDistance(embeddings, self.centers)  # [p,n,n]

            distmat_an = torch.min(distmat, dim=-1)[0]
            #print(distmat_an.size())
            dist_diff = (distmat_ap - distmat_an).view(dist.size(0), -1)

        self.count+=1
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone(),})
            # 'mean_ap_dist': mean_ap_dist.detach().clone(),
            # 'mean_an_dist': mean_an_dist.detach().clone(),})
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