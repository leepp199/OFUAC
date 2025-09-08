import os
import time
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
from data.dataloader import get_pretrain_dataloader,get_dataloader, get_dataset_for_data_init, get_testloader
from utils.utils import Averager, count_acc
from loss.Dist import Dist

def get_optimizer_incremental(model, criterion,args):

    optimizer = torch.optim.SGD([{'params': model.encoder.parameters(), 'lr': args.lr.lr_stdu_base},
                                    {'params': model.slf_attn.parameters(), 'lr': args.lr.lrg}, 
                                    {'params': model.transatt_proto.parameters(), 'lr': args.lr.lrg}, 
                                    {'params': criterion.parameters(),'lr': args.lr.lrg},
                                    ], momentum=0.9, nesterov=True, weight_decay=args.optimizer.decay)

    if args.scheduler.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler.step, gamma=args.scheduler.gamma)
    elif args.scheduler.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler.milestones,
                                                            gamma=args.scheduler.gamma)

    return optimizer, scheduler


def base_incre_train(model, criterion,trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    tqdm_gen = tqdm(trainloader)

    label = torch.arange(args.episode.episode_way + args.episode.low_way).repeat(args.episode.episode_query)
    label = label.type(torch.cuda.LongTensor)

    for i, batch in enumerate(tqdm_gen, 1):
        data, true_label = [_.cuda() for _ in batch]

        bse_idx = args.episode.low_way * args.episode.low_shot # base support end
        bqe_idx = args.episode.low_way * (args.episode.low_shot + args.episode.episode_query) # base query end
        nse_idx = args.episode.episode_way * args.episode.episode_shot + args.episode.low_way * (args.episode.low_shot + args.episode.episode_query) # novel support end
        nqe_idx = args.episode.episode_way * (args.episode.episode_shot + args.episode.episode_query) + args.episode.low_way * (args.episode.low_shot + args.episode.episode_query) # novel query end
        base_labels = true_label[ : bqe_idx]
        bs_labels = true_label[ : bse_idx]
        bq_labels = true_label[bse_idx : bqe_idx]
        novel_labels = true_label[bqe_idx : ]
        ns_labels = true_label[bqe_idx : nse_idx]
        nq_labels = true_label[nse_idx : nqe_idx]

        base_ids, novel_ids =  base_labels.unique(), novel_labels.unique()
        print(f'\nBase classes:{base_ids}, novel classes:{novel_ids}')
        
        # sample low_way data
        # k = args.episode_way * args.episode_shot
        proto, query = data[:bse_idx], data[bse_idx:bqe_idx]
        proto_novel = data[bqe_idx:nse_idx]
        query_novel = data[nse_idx: nqe_idx]

        # encoder data to get embeddings
        model.mode = 'encoder'
        data = model(data[:bqe_idx])
        proto_novel = model(proto_novel)
        query_novel = model(query_novel)

        # split embeddins
        proto, query = data[:bse_idx], data[bse_idx:bqe_idx]
        if args.stdu.ap.use_ap:
            sup_emb = torch.cat([proto, proto_novel], dim=0)
            novel_ids = label.unique()
        else:
            sup_emb = None


        #get ARPLoss
        logits2,anchor_loss=criterion(logits,train_label)

        # reshape embeddings to shape(shot, way, embedding_dim)
        proto = proto.view(args.episode.low_shot, args.episode.low_way, proto.shape[-1])
        query = query.view(args.episode.episode_query, args.episode.low_way, query.shape[-1])
        proto_novel = proto_novel.view(args.episode.episode_shot, args.episode.episode_way, proto_novel.shape[-1])
        query_novel = query_novel.view(args.episode.episode_query, args.episode.episode_way, query_novel.shape[-1])

        # calculate the mean of shot axis to get prototype
        proto = proto.mean(0, keepdim=True)
        proto_novel = proto_novel.mean(0, keepdim=True)
        proto = torch.cat([proto, proto_novel], dim=1)
        query = torch.cat([query, query_novel], dim=1)

        proto = proto.unsqueeze(0)
        query = query.unsqueeze(0)

        logits, pqa_query, pqa_proto  = model._forward(proto, query, args.stdu.pqa, sup_emb=sup_emb, novel_ids=novel_ids)

        total_loss = F.cross_entropy(logits, label)

        total_loss = total_loss + anchor_loss
        acc = count_acc(logits, label)
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        tl.add(total_loss.item())
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), ta.item()))

    tl = tl.item()
    ta = ta.item()
    return tl, ta

def online_proto_adapt(args,model,criterion,epoch):
    tl = Averager()
    ta = Averager()
    optimizer = torch.optim.SGD([{'params': model.fc.parameters(), 'lr': args.lr.lrg}, 
                                    {'params': criterion.parameters(),'lr': args.lr.lrg},
                                    ], momentum=0.9, nesterov=True, weight_decay=args.optimizer.decay)
    if args.scheduler.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler.step, gamma=args.scheduler.gamma)
    elif args.scheduler.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler.milestones,
                                                            gamma=args.scheduler.gamma)
    _,trainloader= get_pretrain_dataloader(args) 
    distance = Dist(num_classes=80, feat_dim=512)
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, label,_ = [_.cuda() for _ in batch]
        feature = model.encode(data)
        proto = model.fc.weight[:args.num_labeled_classes,:]
        logits1=F.cosine_similarity(feature.unsqueeze(1), proto.detach().unsqueeze(0).unsqueeze(0), dim=-1)

        nepoint = criterion.Dist.centers[label,:]
        l2_distance = 2 - 2 * (model.fc.weight[label,:]* nepoint).sum(-1) + 1e-4
        loss_to = 2 - torch.sqrt(l2_distance)

        dist_l2_p = distance(feature,criterion.Dist.centers)
        dist_dot_p = distance(feature,criterion.Dist.centers,metric='dot')
        logits2 = dist_l2_p - dist_dot_p
        P_neg=F.softmax(logits2,dim=-1)

        P_all = torch.mul(P_neg,logits1.squeeze(0))
        loss = F.cross_entropy(P_all,label)#+F.cross_entropy(logits1.squeeze(0),label)#+loss_to.mean()
        acc = count_acc(P_all, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ta.add(acc)
        tl.add(loss.item())
        tqdm_gen.set_description(
            'Session 0, epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, loss.item(), ta.item()))
    tl = tl.item()
    ta = ta.item()
    print('acc:{},loss:{}'.format(ta,tl))
    return model,criterion