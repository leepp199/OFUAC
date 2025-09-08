import scipy as sp
import scipy.spatial.distance as spd
import libmr
import numpy as np
from scipy.io import loadmat, savemat
#------------------------------------------------------------------------------------------
def compute_distance(args,model,train_loader):
    if args.dist_path is None:
        eu_channel, cos_channel, eu_cos_channel = [], [], []
        eucos_dist, eu_dist, cos_dist = [], [], []
        for i, batch in enumerate(train_loader, 1):
            data, label = [_.cuda() for _ in batch]
            model.mode="dist"
            features = model.encode(data)
            for n_class in range(args.num_base):
                for j in range(len(label)):
                    if (n_class==label[j]):
                        mean_feat = model.fc.weight.data[n_class, :]
                        eu_channel+=([spd.euclidean(mean_feat.cpu().detach().numpy(), features[j, :].cpu().detach().numpy())])
                        cos_channel+=([spd.cosine(mean_feat.cpu().detach().numpy(), features[j, :].cpu().detach().numpy())])
                        eu_cos_channel+=([spd.euclidean(mean_feat.cpu().detach().numpy(), features[j, :].cpu().detach().numpy())/200. +
                                    spd.cosine(mean_feat.cpu().detach().numpy(), features[j, :].cpu().detach().numpy())])

        eu_dist = [eu_channel[i:i+500] for i in range(0,len(eu_channel),500)]
        cos_dist = [cos_channel[i:i+500] for i in range(0,len(cos_channel),500)]
        eucos_dist = [eu_cos_channel[i:i+500] for i in range(0,len(eu_cos_channel),500)]
        distance_scores = {'eucos': eucos_dist, 'cosine': cos_dist, 'euclidean':eu_dist}
        args.dist_path = "/data/jessy/open world/save/dist.mat"
        savemat(args.dist_path,distance_scores)
    else:
        distance_scores = loadmat(args.dist_path)
    return distance_scores
def compute_test_dist(args,model,data,dist_type):
    eu_channel, cos_channel, eu_cos_channel = [], [], []
    for n_class in range(args.num_labeled_classes):
        feature = model.encode(data)
        mean_feat = model.fc.weight.data[n_class, :]
        eu_channel+=([spd.euclidean(mean_feat.cpu().detach().numpy(), feature[0,:].cpu().detach().numpy())])
        cos_channel+=([spd.cosine(mean_feat.cpu().detach().numpy(), feature[0,:].cpu().detach().numpy())])
        eu_cos_channel+=([spd.euclidean(mean_feat.cpu().detach().numpy(), feature[0,:].cpu().detach().numpy())/200. +
                                spd.cosine(mean_feat.cpu().detach().numpy(), feature[0,:].cpu().detach().numpy())])
    if dist_type=='eucos':
        return eu_cos_channel
    elif dist_type=='cosine':
        return cos_channel
    else:
        return eu_channel

def update_dist(feature,mean_feat,num,dist_type='eucos'):
    eu_channel, cos_channel, eu_cos_channel = [], [], []
    for i in range(num):
        eu_channel+=([spd.euclidean(mean_feat.cpu().detach().numpy(), feature[i].cpu().detach().numpy())])
        cos_channel+=([spd.cosine(mean_feat.cpu().detach().numpy(), feature[i].cpu().detach().numpy())])
        eu_cos_channel+=([spd.euclidean(mean_feat.cpu().detach().numpy(), feature[i].cpu().detach().numpy())/200. +
                                    spd.cosine(mean_feat.cpu().detach().numpy(), feature[i].cpu().detach().numpy())])
    if dist_type=='eucos':
        return eu_cos_channel
    elif dist_type=='cosine':
        return cos_channel
    else:
        return eu_channel
#------------------------------------------------------------------------------------------
def get_weibull_model(args,distance_scores,type='eucos',tailsize=25):
    weibull_model = {}
    for category in range(args.num_base):
        weibull_model[category] = {}
        weibull_model[category]['weibull_model'] = []
        mr = libmr.MR()
        tailtofit = sorted(distance_scores[type][category])[-tailsize:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[category]['weibull_model'] = [mr]
    return weibull_model

def update_weibull(weibull_model,args,dist,category,tailsize=20):
    if category>=args.num_labeled_classes:
        weibull_model[category] = {}
        weibull_model[category]['weibull_model'] = []
        mr = libmr.MR()
        tailtofit = sorted(dist)[-tailsize:]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[category]['weibull_model'] = [mr]
    return weibull_model   
#------------------------------------------------------------------------------------------
def openmax_scores(args,model,test_loader,weibull_model,distance_type = 'eucos'):
    # Now recalibrate each fc8 score for each channel and for each class
    # to include probability of unknown
    weibull = libmr.MR()
    scores = []
    unknowns,unlabels,knowns,klabels = [],[],[],[]
    for i, batch in enumerate(test_loader, 1):
        data, label= [_.cuda() for _ in batch]
        for j in range(len(label)):
            channel_distance = compute_test_dist(args,model,data[j,None,:],distance_type)
            for n_class in range(args.num_labeled_classes):
                weibull = weibull_model[n_class]['weibull_model'][0]
                scores.append(weibull.w_score(channel_distance[n_class]))
            min_ind = np.argmin(scores,axis=0)
            if scores[min_ind] < args.threshold: #判断为已知类
                knowns+=[data[j].view(1,-1)]
                klabels.append(label[j].item())
            else:
                unknowns+=[data[j].view(1,-1)]
                unlabels.append(label[j].item())
            scores=[]
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 #acc_know = sum(klabels[i]<args.num_base for i in range(len(klabels)))/float(len(klabels))
    #print("acc_know:{}".format(acc_know))
    return unknowns,unlabels,knowns,klabels
