import torch.nn.functional as F
import torch
from loss.Dist import Dist
from utils.utils import Averager, count_acc
from enhance_module import LocalFeatureCluster
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from utils.util import cluster_acc,calc
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
def divide_unknown(args,model,test_loader,criterion):
    proto = model.fc.weight[:args.num_labeled_classes,:].detach()
    nepoint = criterion.Dist.centers[:args.num_labeled_classes,:]
    distance = Dist(num_classes=80, feat_dim=512)
    unknowns,knowns,p_k,p_u,unlabels,klabels=[],[],[],[],[],[]
    if args.mode=='compete':
        for i, batch in enumerate(test_loader, 1):
            data, label = [_.cuda() for _ in batch]
            feature = model.encode(data)
            logits1=F.cosine_similarity(feature.unsqueeze(1), proto, dim=-1)
            pred1,index= torch.max(logits1, dim=1)
            dist_l2_p = distance(feature,nepoint)
            dist_dot_p = distance(feature,nepoint,metric='dot')
            logits2 = dist_l2_p- dist_dot_p
            P_neg=F.softmax(logits2/args.temp,dim=-1)
            pred2,index2 = torch.max(P_neg, dim=1)
            P_all = torch.mul(P_neg,logits1)
            Pred,index3=torch.max(P_all,dim=1)
            p_sum = torch.sum(P_all,dim=1)
            acc = count_acc(P_all, label)
            
            for j in range(len(p_sum)):
                if Pred[j]<0.5:
                    unknowns+=[data[j].view(1,-1)]
                    unlabels.append(label[j].item())
                else:
                    knowns+=[data[j].view(1,-1)]
                    klabels.append(label[j].item())
                l=label[j].item()
                if l<args.num_base:
                    p_k.append(Pred[j])
                else:
                    p_u.append(Pred[j])
        acc_know = sum(klabels[i]<args.num_base for i in range(len(klabels)))/float(len(klabels))
        return unknowns,knowns,unlabels,klabels

# def run_test_fsl(model, args, test_loader):
#     # 初始化局部特征聚类模块
#     local_cluster = LocalFeatureCluster(k_ratio=0.5).cuda()  # k_ratio可根据任务调整
    
#     unknowns, unlabels, knowns, klabels = [], [], [], []
#     proto = model.fc.weight[:args.num_labeled_classes, :].detach()
    
#     with tqdm(test_loader, total=len(test_loader), leave=False) as pbar:
#         for idx, batch in enumerate(pbar):
#             data, label = [_.cuda() for _ in batch]
#             data, label = data.squeeze(), label.squeeze()
            
#             # ===== 原始特征提取 =====
#             raw_feature = model.hgnn_encode(data)  # [B, C, H, W]
            
#             # ===== 局部特征增强 =====
#             with torch.no_grad():
#                 # 如果特征维度是2D（如[B, D]），先reshape为4D（假设H=W=√D）
#                 if raw_feature.dim() == 2:
#                     feat_dim = raw_feature.size(1)
#                     h = w = int(feat_dim**0.5)
#                     raw_feature = raw_feature.view(-1, 1, h, w)  # [B, 1, H, W]
                
#                 # 执行局部聚类增强
#                 enhanced_feature, _ = local_cluster(raw_feature)  # [B, C, H, W]
#                 enhanced_feature = enhanced_feature.mean(dim=[2,3])  # [B, C]
#                 # # 还原为原始形状（如需）
#                 # if raw_feature.dim() == 2:
#                 #     enhanced_feature = enhanced_feature.flatten(1)  # [B, D]
            
#             # ===== 使用增强特征计算概率 =====
#             probs = compute_feats(model, label[:args.n_ways*5], enhanced_feature, proto)
            
#             # ===== 后续逻辑保持不变 =====
#             query_probs = probs[:, :args.num_labeled_classes]
#             thr = np.sort(np.max(query_probs.cpu().detach().numpy(), 1))[int(label.shape[0]*(1-0.5))-1]
            
#             for j in range(len(data)):
#                 query = torch.max(query_probs[j])
#                 if query < probs[j][args.num_labeled_classes]:
#                     unknowns += [data[j].view(1, -1)]
#                     unlabels.append(label[j].item())
#                 else:
#                     knowns += [data[j].view(1, -1)]
#                     klabels.append(label[j].item())
    
#     return unknowns, unlabels, knowns, klabels
#baseline1 
def run_test_fsl(model,args,test_loader):
    unknowns,unlabels,knowns,klabels = [],[],[],[]
    proto = model.fc.weight[:args.num_labeled_classes,:].detach()
    result={}
    result['fscore'] = []
    result['ak'] = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with tqdm(test_loader, total=len(test_loader), leave=False) as pbar:  
        for idx, batch in enumerate(pbar):
            data, label = [_.cuda() for _ in batch]
            data,label = data.squeeze(),label.squeeze()
            feature = model.hgnn_encode(data)
            feature,_ = LocalFeatureCluster(k_ratio=0.3)(feature) 
            feature=feature.to(device)
            probs= compute_feats(model, label[:args.n_ways*5],feature,proto)
            query_probs = probs[:,:args.num_labeled_classes]
            thr = np.sort(np.max(query_probs.cpu().detach().numpy(),1))[int(label.shape[0]*(1-0.5))-1]
            for j in range(len(data)):
                query=torch.max(query_probs[j])
                #index = torch.argmax(probs_closed[j])
                if query<probs[j][args.num_labeled_classes]:#probs[j][args.num_labeled_classes]:
                    unknowns+=[data[j].view(1,-1)]
                    unlabels.append(label[j].item())
                else:
                    knowns+=[data[j].view(1,-1)]
                    klabels.append(label[j].item())
            label,probs = label.cpu().numpy(),probs.cpu().numpy()
            # open_label = np.where(np.isin(label, args.known), 0, 1)
            open_label = np.where(label<[args.num_labeled_classes], 0, 1) 
            p=probs[:,-1]
            # auroc_result = metrics.roc_auc_score(open_label,probs[:,-1])
            # acc_know = sum(klabels[i]<args.num_base for i in range(len(klabels)))/float(len(klabels))
            acc_known,_ = known_test(args,model,knowns,klabels)
            return unknowns,unlabels,knowns,klabels
        #     acc_known,_ = known_test(args,model,knowns,klabels)
        #     fscore=calc(args,klabels,unlabels)
        #     result['fscore']+=[fscore]
        #     result['ak']+=[acc_known]
        #     unknowns,unlabels,knowns,klabels = [],[],[],[]
        # return result
    
# def run_test_fsl(model,args,test_loader):
#     unknowns,unlabels,knowns,klabels = [],[],[],[]
#     proto = model.fc.weight[:args.num_labeled_classes,:].detach()
#     result={}
#     result['fscore'] = []
#     result['ak'] = []
#     with tqdm(test_loader, total=len(test_loader), leave=False) as pbar:  
#         for idx, batch in enumerate(pbar):
#             data, label = [_.cuda() for _ in batch]
#             data,label = data.squeeze(),label.squeeze()
#             feature = model.encode(data)   
#             probs= compute_feats(model, label[:args.n_ways*5],feature,proto)
#             query_probs = probs[:,:args.num_labeled_classes]
#             thr = np.sort(np.max(query_probs.cpu().detach().numpy(),1))[int(label.shape[0]*(1-0.5))-1]
#             for j in range(len(data)):
#                 query=torch.max(query_probs[j])
#                 #index = torch.argmax(probs_closed[j])
#                 if query<probs[j][args.num_labeled_classes]:#probs[j][args.num_labeled_classes]:
#                     unknowns+=[data[j].view(1,-1)]
#                     unlabels.append(label[j].item())
#                 else:
#                     knowns+=[data[j].view(1,-1)]
#                     klabels.append(label[j].item())
#             label,probs = label.cpu().numpy(),probs.cpu().numpy()
#             # open_label = np.where(np.isin(label, args.known), 0, 1)
#             open_label = np.where(label<[args.num_labeled_classes], 0, 1) 
#             p=probs[:,-1]
#             # auroc_result = metrics.roc_auc_score(open_label,probs[:,-1])
#             # acc_know = sum(klabels[i]<args.num_base for i in range(len(klabels)))/float(len(klabels))
#             acc_known,_ = known_test(args,model,knowns,klabels)
#             return unknowns,unlabels,knowns,klabels
#         #     acc_known,_ = known_test(args,model,knowns,klabels)
#         #     fscore=calc(args,klabels,unlabels)
#         #     result['fscore']+=[fscore]
#         #     result['ak']+=[acc_known]
#         #     unknowns,unlabels,knowns,klabels = [],[],[],[]
#         # return result

def plot(args,model,test_fsl_loader):
        model.eval()
        result1,result2,thr1,thr2 = [],[],[],[]
        proto = model.fc.weight[:args.num_labeled_classes,:].detach()
        for idx,batch in enumerate(test_fsl_loader,1):
            data, label = [_.cuda() for _ in batch]
            data,label = data.squeeze(),label.squeeze()   
            feature = model.encode(data)
            feature,_ = LocalFeatureCluster(k_ratio=0.4)(feature) 
            all_prob= compute_feats(model, label[:args.n_ways*5],feature,proto)

            # 绘制热力图
            # query_probs = all_prob
            # similarity_matrix = query_probs.cpu().detach().numpy()
            # plt.figure(figsize=(12, 8))
            # sns.heatmap(similarity_matrix, cmap='viridis', cbar=True)
            # plt.title("Similarity Matrix Heatmap")
            # plt.xlabel("Categories")
            # plt.ylabel("Samples")
            # plt.show()
            # plt.savefig('/data/jessy/open world/new_save'+'/Heatmap.png')

            thr = all_prob[:,-1]
            query,_ = torch.max(all_prob[:,:args.num_labeled_classes],dim=1)
            result1.append(query[:args.n_ways*args.n_shots])
            result2.append(query[args.n_ways*args.n_shots:])
            thr1.append(thr[:args.n_ways*args.n_shots])
            thr2.append(thr[args.n_ways*args.n_shots:])
            
        result1 = torch.cat(result1)
        result2 = torch.cat(result2)
        thr1 = torch.cat(thr1)
        thr2 = torch.cat(thr2)
        count, bins, ignored = plt.hist(result1.tolist(), bins=30, alpha=0.75, color='darkorange')
        plt.hist(result2.tolist(), bins=30, alpha=0.75, color='lightgreen')
        # plt.hist(thr1.tolist(), bins=30, alpha=0.75, color='bisque')
        # plt.hist(thr2.tolist(), bins=30, alpha=0.75, color='yellow')
        data_sorted = sorted(zip(result1.tolist(), thr1.tolist()), key=lambda x: x[0])
        data_sorted, thresholds_sorted = zip(*data_sorted)
        # 绘制点图以展示阈值比较
        #plt.scatter(data_sorted, thresholds_sorted, color='red', label='Thresholds', alpha=0.5)
        #plt.plot(bin_centers, line_y_values, 'r--', linewidth=2)
        plt.xlabel('score')
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig('/data/lqq/openworld/save'+'/hist.png')

def compute_feats(model, label_id,features,proto):
    with torch.no_grad():
        test_cosine_scores= model.cls_classifier.incre_forward(features, proto,label_id)
        query_cls_probs = F.softmax(test_cosine_scores.detach(), dim=-1)
        #closed_cls_probs = F.softmax(supp_scores.detach(), dim=-1)
    return query_cls_probs.squeeze()


def known_test(args,model,data,label):
    feats=[]
    label = torch.tensor(label)
    model = model.eval()
    for i in range(len(data)):
        feat = model.encode(data[i])
        # feat,_ = LocalFeatureCluster(k_ratio=0.4)(feat)
        feats.append(feat)
    proto = model.fc.weight[:args.num_labeled_classes,:].detach().unsqueeze(0).unsqueeze(0)
    feats = torch.stack(feats)
    logits=F.cosine_similarity(feats, proto, dim=-1)
    logits=torch.squeeze(logits)
    acc = count_acc(logits, label.to('cuda'))
    preds = torch.argmax(logits, dim=1)
    score = f1_score(label.cpu().numpy(),preds.cpu().numpy(),average='macro')
    return acc,score


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h       
        
