
import torch
from sklearn.preprocessing import RobustScaler
import torch.nn as nn  
from utils.utils import *
from network import MYNET,get_optimizer,replace_base_fc
from data.dataloader import *
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from tqdm import tqdm
from openmax import *
from sklearn.cluster import KMeans
from CGA_att import ClusterGuidedAttention
class FeatureFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim//2),  # 压缩维度
            nn.ReLU(),
            nn.Linear(feat_dim//2, 1),          # 输出融合权重
            nn.Sigmoid()                         # 归一化到[0,1]
        )
    
    def forward(self, global_feat, local_feat):
        # 拼接特征 [B,1024]
        combined = torch.cat([global_feat, local_feat], dim=1)
        # 预测动态权重 [B,1]
        alpha = self.mlp(combined)
        return alpha * global_feat + (1-alpha) * local_feat
class EnhancedPositionEncoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 时间轴编码
        self.time_enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,1), padding=(1,0)),
            nn.GELU(),
            nn.Conv2d(16, feat_dim, kernel_size=(3,1), padding=(1,0))
        )
        
        # 频率轴编码
        self.freq_enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,3), padding=(0,1)),
            nn.GELU(),
            nn.Conv2d(16, feat_dim, kernel_size=(1,3), padding=(0,1))
        )
        
        # 动态融合门控（关键修改）
        self.gate = nn.Sequential(
            nn.Conv2d(feat_dim*2, feat_dim//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim//2, feat_dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, grid):
        # 输入验证
        assert grid.size(1) == 2, f"输入需要包含2个坐标通道，实际得到{grid.size(1)}"
        
        # 分别编码
        time_emb = self.time_enc(grid[:,:1])  # [B,512,H,W]
        freq_emb = self.freq_enc(grid[:,1:])  # [B,512,H,W]
        
        # 动态融合
        combined = torch.cat([time_emb, freq_emb], dim=1)  # [B,1024,H,W]
        gate = self.gate(combined)  # [B,512,H,W]
        
        return gate * time_emb + (1-gate) * freq_emb
class LocalFeatureCluster(nn.Module):
    def __init__(self, feat_dim=512, k_ratio=0.3, temporal_scale=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.k_ratio = k_ratio
        self.temporal_scale = temporal_scale
        self._current_device = None
        self.fusion_net = FeatureFusion(feat_dim)
        
        # 位置编码器（输入通道2对应x,y坐标）
        self.pos_encoder = EnhancedPositionEncoder(feat_dim)
        # self.pos_encoder = nn.Sequential(
        #     nn.Conv2d(2, 16, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, feat_dim, kernel_size=3, padding=1)
        # )
        
        # 空间权重网络
        self.spatial_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        B, C, H, W = features.shape
        device = features.device
        self._ensure_device_consistency(device)
        
        # 1. 位置编码
        grid = self._generate_grid(H, W, features.device)  # [1,2,H,W]
        pos_emb = self.pos_encoder(grid)  # [1,C,H,W]
        
        # 2. 特征增强
        enhanced_feat = features + pos_emb  # [B,C,H,W]
        flat_feat = enhanced_feat.flatten(2).permute(0,2,1)  # [B,HW,C]
        
        # 3. 动态聚类
        k = max(2, int(H*W*self.k_ratio))
        clustered_feat = []
        centers = []
        
        for i in range(B):
            # 3.1 构建时序相似度矩阵
            positions = torch.stack(torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            ), dim=-1).float().view(-1,2)  # [HW,2]
            temporal_sim = torch.exp(-torch.cdist(positions, positions)/self.temporal_scale)
            
            # 3.2 执行聚类
            with torch.no_grad():
                kmeans = KMeans(n_clusters=k, n_init=10).fit(flat_feat[i].cpu().numpy())
                center = torch.from_numpy(kmeans.cluster_centers_).float().to(device)
                label = torch.from_numpy(kmeans.labels_).to(device)
            
            # 3.3 时序加权原型
            for c in range(k):
                mask = (label == c)
                if mask.sum() > 0:
                    weights = temporal_sim[mask][:, mask].mean(1)
                    center[c] = (flat_feat[i][mask] * weights.unsqueeze(1)).sum(0) / (weights.sum() + 1e-6)
            
            clustered_feat.append(center[label])
            centers.append(center)
        
        # 4. 特征融合
        clustered_feat = torch.stack(clustered_feat)  # [B,HW,C]
        centers = torch.stack(centers)  # [B,k,C]
        
        # 5. 空间权重增强
        spatial_weight = self.spatial_net(flat_feat)  # [B,HW,1]
        enhanced = spatial_weight * clustered_feat + (1-spatial_weight) * flat_feat  # [B,HW,C]
        global_feat = features.mean(dim=[2,3])    # [B,512]
        local_feat = enhanced.mean(dim=[1])# [B,512]
        enhanced_feat = self.fusion_net(global_feat, local_feat)
        return enhanced_feat, centers
    def _generate_grid(self, H, W, device):
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1,2,H,W]
    def _ensure_device_consistency(self, device):
        if self._current_device != device:
            self.to(device)
            self._current_device = device
            # 验证关键参数设备
            assert next(self.pos_encoder.parameters()).device == device
            assert next(self.spatial_net.parameters()).device == device