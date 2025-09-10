from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class NMR_CNN_Encoder(nn.Module):
    def __init__(self, output_dim, dropout_rate=0.5):
        super(NMR_CNN_Encoder, self).__init__()
        
        # 第一个卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 第二个卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 第三个卷积块
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # 池化层，可以在卷积块之间使用
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 输出层，包含Dropout和全连接
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        # 假设输入 x 维度: (batch, 1, seq_len)
        x = self.conv_block1(x)
        x = self.max_pool(x) # 降采样
        
        x = self.conv_block2(x)
        x = self.max_pool(x) # 再次降采样
        
        x = self.conv_block3(x)
        
        # 全局池化，将序列信息聚合成一个向量
        x = self.global_avg_pool(x) # -> (batch, 512, 1)
        x = x.squeeze(-1) # -> (batch, 512)
        
        # 通过输出层得到最终结果
        output = self.output_layer(x) # -> (batch, output_dim)
        
        return output

class PeakPredictionHead(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 ppm_min, ppm_max):
        super(PeakPredictionHead, self).__init__()
        
        self.peak_pos_pred = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        

        self.num_atoms_pred = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.ppm_min = ppm_min
        self.ppm_max = ppm_max
        

    def proj_pos_pred(self, pos_logits):
        # 把预测从 [0,1] 映射回 ppm
        
        span = (self.ppm_max - self.ppm_min)
        ppm_pred = self.ppm_min + pos_logits * span      # (B, m)
        return ppm_pred
    
    def loss_pos(self, ppm_pred, ppm_true):
        return F.mse_loss(ppm_pred, ppm_true)
    
    def loss_num_atoms(self, num_atoms_pred, num_atoms_true):
        return F.binary_cross_entropy_with_logits(num_atoms_pred, num_atoms_true)
        
    def forward(self, x):
        peak_pos  = self.peak_pos_pred(x)
        ppm_pred = self.proj_pos_pred(peak_pos)
        
        num_atoms = self.num_atoms_pred(x)

        return ppm_pred, num_atoms

class NMR_CNN_Encoder_PeakPrediction(pl.LightningModule):
    def __init__(self, output_dim, ppm_min, ppm_max):
        super(NMR_CNN_Encoder_PeakPrediction, self).__init__()
        self.encoder = NMR_CNN_Encoder(output_dim)
        self.peak_prediction_head = PeakPredictionHead(output_dim, ppm_min, ppm_max)
    
    def compute_loss(self, ppm_pred, num_atoms, ppm_true, num_atoms_true):
        loss_pos = self.peak_prediction_head.loss_pos(ppm_pred, ppm_true)
        loss_num_atoms = self.peak_prediction_head.loss_num_atoms(num_atoms, num_atoms_true)
        loss = loss_pos + loss_num_atoms

    def forward(self, x, ppm_true, num_atoms_true):
        x = self.encoder(x)
        ppm_pred, num_atoms = self.peak_prediction_head(x)

        loss = self.compute_loss(ppm_pred, num_atoms, ppm_true, num_atoms_true)

        return x, ppm_pred, num_atoms, loss
    
    def step(self, batch, batch_idx):
        x, ppm_true, num_atoms_true = batch
        x, ppm_pred, num_atoms, loss = self(x, ppm_true, num_atoms_true)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
