import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from speechbrain.processing.features import STFT, Filterbank
from models.resnet18_encoder import resnet18
from utils.utils import count_acc,Averager
import tqdm,os
from models.feature_enhancer import EnhancedLocalFeature
from models.AttnClassifier import Classifier
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import numpy as np
from models.resnet_enhancer import LocalFeatureCluster
class MYNET(nn.Module):

    def __init__(self,args, mode=None):
        super().__init__()
        self.mode = mode# 模式设置，例如'encoder'或'openmeta' 
        self.args = args# 存储超参数的对象 
        self.encoder = resnet18(True, args)  # pretrained=False # 使用预训练的ResNet18作为特征提取器 
        self.num_features = 512# 特征数量，ResNet18的输出特征维度
        self.fc = nn.Linear(self.num_features, 100, bias=True)# 分类层，将特征映射到100个类  
        hdim=self.num_features# 隐藏状态维度  
        self.beta = 1.0 # 一个超参数 
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5) # 初始化多头注意力机制  
        self.transatt_proto = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
        self.cls_classifier = Classifier(args, self.num_features,args.train_weight_base) # 分类器 
        self.set_module_for_audio(args) # 音频模块的设置（可能涉及音频特征）
        # self.cluster_loss_weight = 0.5  # 可配置参数
        self.feature_enhance = LocalFeatureCluster(feat_dim=64, k_ratio=0.3)
    def forward(self,input,labels=None, conj_ids=None, base_ids=None, test=False):
        # 定义前向传播过程 
        if self.mode == 'encoder':# 如果模式是'encoder'，则只进行编码
            input = self.encode(input)
            return input
        elif self.mode == 'openmeta':# 进入openmeta模式的前向传播
            return self.open_forward(input, labels, conj_ids, base_ids, test)
        # elif self.mode == 'cluster':  # 新增cluster模式
        #     return self.cluster_forward(input)
        else:
            support_idx, query_idx = input# 获取支持集和查询集索引 
            logits = self._forward(support_idx, query_idx)# 执行另一个前向传播 
            return logits

    def open_forward(self, the_input, labels, conj_ids, base_ids, test):
         # Data Preparation
        # print(type(the_input))
        # combined_data=torch.cat(the_input,dim=0) #NS
        #LS FS
        combined_data=the_input.squeeze() # 将输入数据沿着第一个维度拼接
        (support_label,query_label,supopen_label,openset_label) = labels# 标签解构 
        # 将标签移到GPU（CUDA）  
        (support_label,query_label,supopen_label,openset_label)=(support_label.squeeze().cuda(),query_label.squeeze().cuda(),supopen_label.squeeze().cuda(),openset_label.squeeze().cuda())  
        combined_feat = self.encode(combined_data.cuda())#.detach()# 对拼接后的输入数据进行编码  
        # 将特征拆分为支持特征、查询特征、开放集特征 
        support_feat,query_feat,supopen_feat,openset_feat = torch.split(combined_feat,[len(support_label),len(query_label),len(supopen_label),len(openset_label)],dim=0)
       # 重新整理支持和开放特征的形状 
        support_feat,supopen_feat = support_feat.view(self.args.n_ways,self.args.n_shots,-1),supopen_feat.view(self.args.n_ways,self.args.n_shots,-1)
        #query_feat,openset_feat = query_feat.view(self.args.n_ways,self.args.n_queries,-1),openset_feat.view(self.args.n_ways,self.args.n_queries,-1)
        # 获取支持集和开放集的索引
        (supp_idx, open_idx) = conj_ids
        cls_label = torch.cat([query_label, openset_label])# 合并查询标签和开放集标签  
        test_feats = (support_feat, query_feat, openset_feat)# 测试特征的组合


        ### First Task
        # 计算余弦分数、支持原型、伪类原型和损失 
        test_cosine_scores, supp_protos, fakeclass_protos, loss_cls, loss_funit = self.task_proto((support_feat,query_feat,openset_feat), (supp_idx,base_ids), cls_label,query_label, test)
        cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1) # 合并支持原型和伪类原型
        test_cls_probs = self.task_pred(test_cosine_scores[0], test_cosine_scores[1])# 预测类概率

        if test:
            test_feats = (support_feat, query_feat, openset_feat)# 返回特征
            return test_feats, cls_protos, test_cls_probs# 返回特征、原型和类概率 

        

        #loss_open_hinge = 0.0
        loss_open_hinge = F.mse_loss(fakeclass_protos.repeat(1,self.args.n_ways, 1), supp_protos)# 计算开放集的hinge损失（这里使用均方误差） 
        #loss_open_hinge_2 = F.mse_loss(fakeclass_protos_aug.repeat(1,self.args.n_ways, 1), supp_protos_aug) 
        #loss_open_hinge = loss_open_hinge_1 + loss_open_hinge_2
        
        
        loss = (loss_cls, loss_open_hinge, loss_funit)
        return test_feats, cls_protos, test_cls_probs, loss
    
    
    def task_proto(self, features, cls_ids, cls_label,query_label,test=False):
        test_cosine_scores, supp_protos, fakeclass_protos, funit_distance = self.cls_classifier(features, cls_ids,test)
        (query_cls_scores,openset_cls_scores) = test_cosine_scores
        cls_scores = torch.cat([query_cls_scores,openset_cls_scores], dim=1)
        fakeunit_loss = self.fakeunit_compare(funit_distance,query_label)
        loss_cls = F.cross_entropy(cls_scores.squeeze(), cls_label)
        return test_cosine_scores, supp_protos, fakeclass_protos, loss_cls, fakeunit_loss
    
    def fakeunit_compare(self,funit_distance,cls_label):
        cls_label_binary = F.one_hot(cls_label).float()
        loss = torch.sum(F.binary_cross_entropy_with_logits(input=funit_distance.squeeze(), target=cls_label_binary))
        return loss
    
    def task_pred(self, query_cls_scores, openset_cls_scores, many_cls_scores=None):
        query_cls_probs = F.softmax(query_cls_scores.detach(), dim=-1)
        openset_cls_probs = F.softmax(openset_cls_scores.detach(), dim=-1)
        if many_cls_scores is None:
            return (query_cls_probs, openset_cls_probs)
        else:
            many_cls_probs = F.softmax(many_cls_scores.detach(), dim=-1)
            return (query_cls_probs, openset_cls_probs, many_cls_probs, query_cls_scores, openset_cls_scores)

    def _forward(self,support, query, pqa=False, sup_emb=None, novel_ids=None):  # support and query are 4-d tensor, shape(num_batch, 1, num_proto, emb_dim)
       
        emb_dim = support.size(-1)
        num_query = query.shape[1]*query.shape[2]#num of query*way
        query = query.view(-1, emb_dim).unsqueeze(1)  # shape(num_query, 1, emb_dim)

        # get mean of the support of shape(batch_size, shot, way, dim)
        mean_proto = support.mean(dim=1, keepdim=True)  # calculate the mean of each class's prototype without keeping the dim
        num_batch = mean_proto.shape[0]
        num_proto = mean_proto.shape[2]  # num_proto = num of support class

        # the shape of proto is different from query, so make them same by coping (num_proto, emb_dim)
        mean_proto_expand = mean_proto.expand(num_batch, num_query, num_proto, emb_dim).contiguous()  # can be regard as copying num_query(int) proto
        if sup_emb is not None:
            att_proto = self.get_att_proto(sup_emb, query, num_query, emb_dim)
            mean_proto_expand.data[:, :, novel_ids, :] = self.beta * att_proto.unsqueeze(0) \
                                                    + (1-self.beta) * mean_proto_expand[:, :, novel_ids, :]
        proto = mean_proto_expand.view(num_batch*num_query, num_proto, emb_dim)
      
        if pqa:
            combined = torch.cat([proto, query], 1)  # Nk x (N + 1) or (N + 1 + 1) x d, batch_size = NK
            combined, _ = self.slf_attn(combined, combined, combined)
            proto, query = combined.split(num_proto, dim=1)
        else:
            combined = proto
            combined, _ = self.slf_attn(combined, combined, combined)
            proto = combined
        
        logits=F.cosine_similarity(query, proto, dim=-1)
        logits=logits*self.args.network.temperature
        return logits, query, proto
        
    def get_att_proto_shot_score(self,sup_emb, num_query, emb_dim):
        sup_emb = sup_emb.view(self.args.episode.episode_shot, -1, emb_dim).permute(1, 0, 2)
        att_emb, att_logit = self.inneratt_proto(sup_emb, sup_emb, sup_emb)
        # att_proto = att_emb.mean(dim=1)

        shot_logit = att_logit.mean(dim=1)
        shot_score = F.softmax(shot_logit, dim=1)
        shot_score = shot_score.unsqueeze(-1)
        att_proto = shot_score * sup_emb
        att_proto = att_proto.sum(1)
        att_proto_expand = att_proto.unsqueeze(0).expand(num_query, -1, emb_dim)
        return att_proto_expand

    def get_att_proto(self,sup_emb, query, num_query, emb_dim):
        sup_emb = sup_emb.unsqueeze(0).expand(num_query, sup_emb.shape[0], sup_emb.shape[-1])
        cat_emb = torch.cat([sup_emb, query], dim=1)
        att_pq, att_logit = self.transatt_proto(cat_emb, cat_emb, cat_emb)
        att_logit = att_logit[:, :, -1][:, :-1] # 选取最后一列的前shot*way个logits
        att_score = torch.softmax(att_logit.view(num_query, args.episode.episode_shot, -1), dim=1)
        att_proto, _ = att_pq.split(sup_emb.shape[1], dim=1)
        att_proto = att_proto.view(num_query, args.episode.episode_shot, -1, emb_dim) * att_score.unsqueeze(-1) # args.episode_way+args.low_way
        att_proto = att_proto.sum(1)
        return att_proto

    def get_featmap(self,input):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        feat_map = self.encoder(input)
        return feat_map
    def enhance_encode(self,x):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(x)
        x = self.feature_enhance(x) 
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)               # (B, C, H, W) <- 保留空间维度  

        x = self.encoder.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        if self.mode=="encoder":
            x = self.fc(x)
        return x       
    def pre_encode(self, x):
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        return x

    def set_module_for_audio(self, args):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=self.args.extractor.window_size, hop_length=self.args.extractor.hop_size, 
            win_length=self.args.extractor.window_size, window=self.args.extractor.window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=self.args.extractor.sample_rate, n_fft=self.args.extractor.window_size, 
            n_mels=self.args.extractor.mel_bins, fmin=self.args.extractor.fmin, fmax=self.args.extractor.fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(self.args.extractor.mel_bins)

        # speechbrain tools 
        self.compute_STFT = STFT(sample_rate=self.args.extractor.sample_rate, 
                            win_length=int(self.args.extractor.window_size / self.args.extractor.sample_rate * 1000), 
                            hop_length=int(self.args.extractor.hop_size / self.args.extractor.sample_rate * 1000), 
                            n_fft=self.args.extractor.window_size)
        self.compute_fbanks = Filterbank(n_mels=self.args.extractor.mel_bins)
    
    def set_fea_extractor_for_s2s(self):
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        fs_sample_rate = 44100
        fs_window_size = 2048
        fs_hop_size = 1024
        fs_mel_bins = 128
        fs_fmax = 22050
        fs_spectrogram_extractor = Spectrogram(n_fft=fs_window_size, hop_length=fs_hop_size, 
            win_length=fs_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        fs_logmel_extractor = LogmelFilterBank(sr=fs_sample_rate, n_fft=fs_window_size, 
            n_mels=fs_mel_bins, fmin=0, fmax=fs_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)


        ns_sample_rate = 16000
        ns_window_size = 2048
        ns_hop_size = 1024
        ns_mel_bins = 128
        ns_fmax = 8000
        ns_spectrogram_extractor = Spectrogram(n_fft=ns_window_size, hop_length=ns_hop_size, 
            win_length=ns_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        ns_logmel_extractor = LogmelFilterBank(sr=ns_sample_rate, n_fft=ns_window_size, 
            n_mels=ns_mel_bins, fmin=0, fmax=ns_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        ls_sample_rate = 16000
        ls_window_size = 400
        ls_hop_size = 160
        ls_mel_bins = 128
        ls_fmax = 8000
        ls_spectrogram_extractor = Spectrogram(n_fft=ls_window_size, hop_length=ls_hop_size, 
            win_length=ls_window_size, window="hann", center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        ls_logmel_extractor = LogmelFilterBank(sr=ls_sample_rate, n_fft=ls_window_size, 
            n_mels=ls_mel_bins, fmin=0, fmax=ls_fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        bn0 = nn.BatchNorm2d(128) 

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        assert len(data) == self.args.episode.episode_way * self.args.episode.episode_shot
        
        if not self.args.strategy.data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.network.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            #print(class_index)
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
            #print(proto)
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.network.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.network.new_mode:
            return self.args.network.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        num_base = self.args.stdu.num_tmpb if self.args.tmp_train else self.args.num_base
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs.epochs_new):
                old_fc = self.fc.weight[:num_base + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[num_base + self.args.way * (session - 1):num_base + self.args.way * session, :].copy_(new_fc.data)
    def get_spectrogram(self,x):
        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)
        return x


    def encode(self, x):
        
        x = self.spectrogram_extractor(x)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = x.repeat(1, 3, 1, 1)#[128 3 201 128]
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        if self.mode=="encoder":
            x = self.fc(x)
        return x
    def hgnn_encode(self, x):
        x = self.spectrogram_extractor(x)  # (B, 1, T, F)
        x = self.logmel_extractor(x)      # (B, 1, T, M)
        x = x.transpose(1, 3)             # (B, M, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)             # (B, 1, T, M)
        x = x.repeat(1, 3, 1, 1)          # (B, 3, T, M)
        x = self.encoder(x)               # (B, C, H, W) <- 保留空间维度
        if self.mode == "encoder":
            x = self.fc(x)  # 如果fc需要全局特征，可保留
        return x
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn_logit = torch.bmm(q, k.transpose(1, 2))
        attn = attn_logit / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn_logit, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn_logit, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn_logit

def get_optimizer_standard(model,args,criterion):

    optimizer = torch.optim.SGD([{'params': model.encoder.parameters()},{'params': criterion.parameters()}], lr=args.lr.lr_std,
                                momentum=0.9, nesterov=True, weight_decay=args.optimizer.decay)

    if args.scheduler.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler.step, gamma=args.scheduler.gamma)
    elif args.scheduler.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler.milestones,
                                                            gamma=args.scheduler.gamma)
    return optimizer, scheduler

def get_optimizer(model,args):

    optimizer = torch.optim.SGD([{'params':filter(lambda p: p.requires_grad, model.parameters())}], lr=args.lr.lr_std,
                                momentum=0.9, nesterov=True, weight_decay=args.optimizer.decay)
    #model.encoder.pa
    if args.scheduler.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler.step, gamma=args.scheduler.gamma)
    elif args.scheduler.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler.milestones,
                                                            gamma=args.scheduler.gamma)
    return optimizer, scheduler

def standard_base_train(args, model, net_dict,save_model_path,trainloader, optimizer, scheduler, epoch, temp):
    num_base = args.stdu.num_tmpb if temp else args.num_base
    model = model.train()
    model.module.mode = 'encoder'
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    max_acc=0
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        logits = logits[:, :num_base]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        if acc>max_acc:
            torch.save(dict(params=model.state_dict()), save_model_path)
            torch.save(net_dict['optimizer'].state_dict(), os.path.join(args.model_dir, 'optimizer_best.pth'))
            max_acc = acc
            print("model saved to {}.".format(save_model_path))
        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        
        tqdm_gen.set_description(
                'base train, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      

def replace_base_fc(args,trainset, model):
    num_base_class =  args.num_base
    model = model.eval()
    assert len(set(trainset.targets)) == num_base_class
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                                num_workers=8, pin_memory=True, shuffle=False)
    # trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'incre'
            embedding = model.encode(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    for class_index in range(num_base_class):#args.known:#range(num_base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
    proto_list = torch.stack(proto_list, dim=0)
    #model.fc.weight.data[args.known] = proto_list.cuda()
    model.fc.weight.data[:num_base_class] = proto_list

    return model

def get_negapoint(args,model,criterion,epoch,trainloader):
    tl = Averager()
    ta = Averager()
    optimizer = torch.optim.SGD([{'params': criterion.parameters(),'lr': args.lr.lrg},
                                    ], momentum=0.9, nesterov=True, weight_decay=args.optimizer.decay)
    if args.scheduler.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler.step, gamma=args.scheduler.gamma)
    elif args.scheduler.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler.milestones,
                                                            gamma=args.scheduler.gamma)
    
    for i, batch in enumerate(trainloader, 1):
        data, label = [_.cuda() for _ in batch]
        feature = model.encode(data)
        proto = model.fc.weight[:args.num_labeled_classes,:]
        logits1=F.cosine_similarity(feature.unsqueeze(1), proto.detach(), dim=-1)
        logits2,loss_neg=criterion(feature,label)
        P_neg=F.softmax(logits2,dim=-1)
        P_all = torch.mul(P_neg,logits1)
        loss = loss_neg#+F.cross_entropy()
        acc = count_acc(logits2, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ta.add(acc)
        tl.add(loss.item())
    print('Session 0, epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, tl.item(), ta.item()))
    return criterion

def meta_update_model(model, optimizer, loss, meta_grads):
    # 创建模型参数的副本
    model_params_copy = {n: p.clone().requires_grad_(True) for n, p in model.named_parameters()}

    # 将梯度应用于模型参数的副本
    for n, p in model_params_copy.items():
        if n in meta_grads:
            p.grad = meta_grads[n]

    # 使用副本参数进行优化
    optimizer.zero_grad()
    optimizer.step()

    # 将优化后的副本参数应用于原始模型
    for n, p in model.named_parameters():
        if n in model_params_copy:
            p.data = model_params_copy[n].data


if __name__ == "__main__":
    proto = torch.randn(25, 512, 2, 4)
    query = torch.randn(75, 512, 2, 4)
