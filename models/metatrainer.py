from __future__ import print_function

import os
import numpy as np
import time
from tqdm import tqdm


import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
from utils.utils import  AverageMeter
from models.FSEval import run_test_fsl


def meta_train(args, model,train_loader, eval_loader=True):
    params = torch.load(args.pretrained_model_path)['params']
    cls_params = {k: v for k, v in params.items() if 'fc' in k}
    feat_params = {k: v for k, v in params.items() if 'encoder' in k}
    model.cls_classifier.init_representation(cls_params)
    ##### Load Pretrained Weights for Feature Extractor
    model_dict = model.state_dict()
    model_dict.update(feat_params)
    model.load_state_dict(model_dict)
    model.train()
    optim_param = [{'params': model.cls_classifier.parameters()}]
    optimizer = optim.SGD(optim_param, lr=args.learning_rate, momentum=args.optimizer.momentum, weight_decay=args.optimizer.decay, nesterov=True)
    if args.cosine:
        print("==> training with plateau scheduler ...")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    else:
        print("==> training with MultiStep scheduler ... gamma {} step {}".format(args.lr_decay_rate, args.lr_decay_epochs))

    trlog = {}
    trlog['args'] = vars(args)
    trlog['maxmeta_acc'] = 0.0
    trlog['maxmeta_acc_epoch'] = 0
    trlog['maxmeta_auroc'] = 0.0
    trlog['maxmeta_auroc_epoch'] = 0

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    for epoch in range(1, args.epochs.epochs_meta + 1):
        if args.cosine:
            scheduler.step(trlog['maxmeta_acc'])
        else:
            adjust_learning_rate(epoch, args, optimizer, 0.0001)
            
        train_acc, train_auroc, train_loss, train_msg = train_episode(epoch, train_loader, model, optimizer, args)

        model.eval()

        #evaluate
        if eval_loader is not None:
            start = time.time()
            config = {'auroc_type':['prob']}
            result = run_test_fsl(model, eval_loader, config)
            meta_test_acc = result['data']['acc']
            open_score_auroc = result['data']['auroc_prob']

            test_time = time.time() - start
            meta_msg = 'Meta Test Acc: {:.4f}, Test std: {:.4f}, AUROC: {:.4f}, Time: {:.1f}'.format(meta_test_acc[0], meta_test_acc[1], open_score_auroc[0], test_time)
            train_msg = train_msg + ' | ' + meta_msg
                
            if trlog['maxmeta_acc'] < train_acc:#meta_test_acc[0]:
                trlog['maxmeta_acc'] = train_acc#meta_test_acc[0]
                trlog['maxmeta_acc_epoch'] = epoch
                acc_auroc = (train_acc,train_auroc)#(meta_test_acc[0], open_score_auroc[0])
                save_model(epoch, 'max_acc', acc_auroc)
            if trlog['maxmeta_auroc'] < train_auroc:#open_score_auroc[0]:
                trlog['maxmeta_auroc'] = train_auroc#open_score_auroc[0]
                trlog['maxmeta_auroc_epoch'] = epoch
                acc_auroc = (train_acc,train_auroc)#(meta_test_acc[0], open_score_auroc[0])
            save_model(epoch, 'max_auroc', acc_auroc)
                
        print(train_msg)
        # # print(meta_test_acc[0])
        # print(trlog['maxmeta_acc'],trlog['maxmeta_acc_epoch'])

        # regular saving
        if epoch % 5 == 0:
            save_model(model,epoch,args)
            print('The Best Meta Acc {:.4f} in Epoch {}, Best Meta AUROC {:.4f} in Epoch {}'.format(trlog['maxmeta_acc'],trlog['maxmeta_acc_epoch'],trlog['maxmeta_auroc'],trlog['maxmeta_auroc_epoch']))


def train_episode(epoch, train_loader, model, optimizer, args):
    """One epoch training"""
    model.train()
    model.encoder.eval()


    batch_time = AverageMeter()
    losses_cls = AverageMeter()
    losses_funit = AverageMeter()
    acc = AverageMeter()
    auroc = AverageMeter()
    end = time.time()

    with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
        for idx, data in enumerate(pbar):
            support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label, supp_idx, open_idx,base_ids= data
            # Data Conversion & Packaging

            supp_idx, open_idx,base_ids = supp_idx.long(), open_idx.long(),base_ids.long()
            openset_label = args.n_ways * torch.ones_like(openset_label)
            # print(support_data.shape)  
            # print(query_data.shape)  
            # print(suppopen_data.shape)  
            # print(openset_data.shape)
            # the_img = torch.cat([support_data, query_data, suppopen_data, openset_data], dim=1)'
            # print(type(support_data))
            # print(type(query_data))
            # print(type(suppopen_data))
            # print(type(openset_data))
            # support_data=[support_data]
            # query_data=[query_data]
            # suppopen_data=[suppopen_data]
            # openset_data=[openset_data]
            # the_img     = support_data+query_data+suppopen_data+openset_data
            # 在新维度（如 0 维）上堆叠  
            # support_data=torch.squeeze(support_data,0)
            # query_data=torch.squeeze(query_data,0)
            # suppopen_data=torch.squeeze(suppopen_data,0)
            # # openset_data=torch.squeeze(openset_data,0)
            # print(support_data.shape)  
            # print(query_data.shape)  
            # print(suppopen_data.shape)  
            # print(openset_data.shape)
            # the_img     = support_data+query_data+suppopen_data+openset_data #NS
            #LS FS训练时
            the_img = torch.cat((support_data, query_data, suppopen_data, openset_data), dim=1)  
            # 这样可以保持各自的样本数量，并用相同的特征长度
            the_label   = (support_label,query_label,suppopen_label,openset_label)
            the_conj    = (supp_idx, open_idx)
            model.mode = 'openmeta'
            _, _, probs, loss = model(the_img,the_label,the_conj,base_ids)
            query_cls_probs, openset_cls_probs = probs
            (loss_cls, loss_open_hinge, loss_funit) = loss
            loss_open = args.gamma * loss_open_hinge + args.funit * loss_funit

            loss = loss_open + loss_cls
                
            ### Closed Set Accuracy
            close_pred = np.argmax(probs[0][:,:,:args.n_ways].view(-1,args.n_ways).cpu().numpy(),-1)
            close_label = query_label.view(-1).cpu().numpy()
            acc.update(metrics.accuracy_score(close_label, close_pred),1)

            ### Open Set AUROC
            open_label_binary = np.concatenate((np.ones(close_pred.shape),np.zeros(close_pred.shape)))
            query_cls_probs = query_cls_probs.view(-1, args.n_ways+1)
            openset_cls_probs = openset_cls_probs.view(-1, args.n_ways+1)
            open_scores = torch.cat([query_cls_probs,openset_cls_probs], dim=0).cpu().numpy()[:,-1]
            auroc.update(metrics.roc_auc_score(1-open_label_binary,open_scores),1)
                
                
            losses_cls.update(loss_cls.item(), 1)
            losses_funit.update(loss_funit.item(), 1)

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()
                
                
            pbar.set_postfix({"Acc":'{0:.2f}'.format(acc.avg), 
                                "Auroc":'{0:.2f}'.format(auroc.avg), 
                                "cls_ce" :'{0:.2f}'.format(losses_cls.avg), 
                                "funit" :'{0:.4f}'.format(losses_funit.avg), 
                                })

    message = 'Epoch {} Train_Acc {acc.avg:.3f} Train_Auroc {auroc.avg:.3f}'.format(epoch, acc=acc, auroc=auroc)

    return acc.avg, auroc.avg, (losses_cls.avg, losses_funit.avg), message

def save_model(model,epoch, args,name=None, acc_auroc=None):
    state = {
        'epoch': epoch,
        'cls_params': model.state_dict() ,
        'acc_auroc': acc_auroc
    }
    # 'optimizer': self.optimizer.state_dict()['param_groups'],
                 
    file_name = 'epoch_'+str(epoch)+'.pth' if name is None else name + '.pth'
    print('==> Saving', file_name)
    torch.save(state, args.save_dir+file_name)


    
    
def adjust_learning_rate(epoch, opt, optimizer, threshold=1e-6):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0 and opt.learning_rate > threshold:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr          