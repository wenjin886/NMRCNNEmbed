from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import copy

from spec_utils import (
    Embeddings,
    PositionalEncoding,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
)

class NMR_CNN_Encoder(nn.Module):
    def __init__(self, output_dim, kernel_size, dropout_rate=0.1):
        super(NMR_CNN_Encoder, self).__init__()

        padding = (kernel_size - 1) // 2
        
        # 第一个卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 第二个卷积块
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 第三个卷积块
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=kernel_size, padding=padding),
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
        # 输入 x 维度: (batch, seq_len)
        x = self.conv_block1(x.unsqueeze(1))
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

class NMREmbedPatchAttention(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=256, h=8) -> None:
        super(NMREmbedPatchAttention, self).__init__()
        assert spec_len%patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.patch = nn.Linear(patch_len*d_model, d_model)
        self.attention = MultiHeadedAttention(h=h, d_model=d_model)
        # self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=h, batch_first=True)
        
    def forward(self, spec): #[batch_size, spec_len]
        batch_size = spec.shape[0]

        # print("spec", spec.unsqueeze(2).shape)
        spec = self.embed(spec.unsqueeze(2)) #[batch_size, spec_len, d_model]
        # print("spec", spec.shape)

        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.view(-1, self.patch_len, self.d_model) #[batch_size*(spec_len/patch_size), patch_size, d_model]
        spec = self.attention(spec, spec, spec)
 
        spec = spec.view(batch_size, -1, self.patch_len, self.d_model) #[batch_size, spec_len/patch_size, patch_size, d_model]
        spec = spec.reshape(batch_size, spec.shape[1], -1) #[batch_size, spec_len/patch_size, patch_size*d_model]
        
        return self.patch(spec) #[batch_size, spec_len/patch_size, d_model]

class NMR_Attn_Encoder(nn.Module):
    def __init__(self, 
                 spec_len=1600, patch_len=16,
                 d_model=512,
                 n_spec_attention_head=8,
                 n_spec_f_encoder_head=8,
                 n_spec_f_encoder_layer=4,
                 device=torch.device("cuda")
                 ):
        super().__init__()
        self.d_model = d_model
        self.spec_patch_num = int(spec_len/patch_len)
        
        c = copy.deepcopy
        pe = PositionalEncoding(d_model, dropout=0.1)
        spec_patch_att = NMREmbedPatchAttention(spec_len=spec_len, 
                                              patch_len=patch_len, 
                                              d_model=d_model,
                                              h=n_spec_attention_head)
        
        # Encoder part
        self.spec_embed = nn.Sequential(spec_patch_att, c(pe))
     
        encoder_attn = MultiHeadedAttention(h=n_spec_f_encoder_head,d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=2048, dropout=0.1)
        self.spec_atom_encoder = Encoder(EncoderLayer(d_model, c(encoder_attn),c(ff),0.1), 
                                         N=n_spec_f_encoder_layer)
        
        self.proj = nn.Sequential(
            nn.Linear(self.spec_patch_num * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.device_ = device
        self.__init_weights__()
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, spectra):
        B = spectra.shape[0]

        src = self.spec_embed(spectra)
        src_mask = torch.ones(spectra.shape[0], self.spec_patch_num, dtype=torch.bool).unsqueeze(-2).to(self.device_) # (B, 1, S)

        src = self.spec_atom_encoder(src, src_mask) # (B, S/P, D)
        # src = self.proj(src.view(B, -1)) # (B, S/P * D) --> # (B, D)
            
        return src, src_mask
    
class PeakPredictionHead(nn.Module):
    def __init__(self, input_dim, max_num_peaks, num_atom_types, 
                 ppm_min, ppm_max):
        super(PeakPredictionHead, self).__init__()
        
        self.peak_pos_pred = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, max_num_peaks)
        )
        

        self.num_atoms_pred = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, max_num_peaks*num_atom_types)
        )

        self.max_num_peaks = max_num_peaks
        self.num_atom_types = num_atom_types
        self.ppm_min = ppm_min
        self.ppm_max = ppm_max
        

    def proj_pos_pred(self, pos_logits):
        # 把预测从 [0,1] 映射回 ppm
        
        span = (self.ppm_max - self.ppm_min)
        ppm_pred = self.ppm_min + pos_logits * span      # (B, m)
        return ppm_pred
    
    
    def loss_pos(self, ppm_pred_val, ppm_true, ignore_idx=-100):
        # ppm_pred 和 ppm_true 的维度都是 (B, max_peaks)
        # mask 的维度是 (B, max_peaks)
        mask = torch.where(ppm_true == ignore_idx, 0, 1)
        # 逐元素计算损失
        loss = F.mse_loss(ppm_pred_val, ppm_true, reduction='none') # reduction='none' 保留每个元素的损失
        
        # 应用掩码，只保留真实峰的损失
        masked_loss = loss * mask
        
        # 计算平均损失时，分母应该是真实峰的数量，而不是总元素数量
        # 添加一个极小值防止 mask.sum() 为0
        final_loss = masked_loss.sum() / (mask.sum() + 1e-8) 
        
        return final_loss
    
    def loss_num_atoms(self, num_atoms_pred_logits, num_atoms_true_indices, ignore_idx=-100):
        """
        使用交叉熵计算原子数量的损失。
        
        Args:
            num_atoms_pred_logits (torch.Tensor): 模型的原始输出, 形状为 (B, max_peaks, num_classes)
            num_atoms_true_indices (torch.Tensor): 真实的类别索引, 形状为 (B, max_peaks), long 类型
            ignore_idx (int): 在真实标签中需要忽略的填充值
        """
        # 获取类别数量
        B, max_p, num_classes = num_atoms_pred_logits.shape
        
        # 为了匹配 F.cross_entropy 的输入要求，我们需要重塑张量
        # (B, max_peaks, num_classes) -> (B * max_peaks, num_classes)
        pred_logits_flat = num_atoms_pred_logits.view(-1, num_classes)
        
        # (B, max_peaks) -> (B * max_peaks,)
        true_indices_flat = num_atoms_true_indices.view(-1)
        
        # 计算损失，ignore_index 参数会自动处理我们之前用-100填充的假峰
        loss = F.cross_entropy(pred_logits_flat, true_indices_flat, ignore_index=ignore_idx)
        return loss
    
    def acc_num_atoms(self, num_atoms_pred_logits, num_atoms_true_indices, ignore_idx=-100):
        """
        在有padding的情况下，使用mask计算原子数量预测的准确率。

        Args:
            num_atoms_pred_logits (torch.Tensor): 模型输出, (B, max_peaks, num_classes)
            num_atoms_true_indices (torch.Tensor): 真实标签, (B, max_peaks), long类型
            mask (torch.Tensor): 掩码, (B, max_peaks), 真实峰为1, 填充为0

        Returns:
            float: 准确率 (0.0 to 1.0)
        """
        # 1. 获取预测的类别索引
        # torch.argmax在最后一个维度(-1)上操作，得到每个峰最可能的类别
        pred_indices = torch.argmax(num_atoms_pred_logits, dim=-1) # -> (B, max_peaks)
        
        # 确保mask是布尔类型以便索引
        mask = torch.where(num_atoms_true_indices == ignore_idx, 0, 1)
        bool_mask = mask.bool()
        
        # 2. 筛选出真实峰的预测和标签
        # 只选择 mask 为 True 的位置上的元素
        masked_preds = pred_indices[bool_mask]
        masked_true = num_atoms_true_indices[bool_mask]
        
        # 3. 计算这些真实峰中，有多少预测是正确的
        correct_predictions = (masked_preds == masked_true).sum().item()
        
        # 4. 真实峰的总数
        total_real_peaks = mask.sum().item()
        
        # 5. 计算准确率
        accuracy = correct_predictions / total_real_peaks if total_real_peaks > 0 else 0.0
        
        return accuracy

        
    def forward(self, x):
        peak_pos_logits  = self.peak_pos_pred(x).sigmoid()  # -> (B, max_num_peaks)
        ppm_pred_val = self.proj_pos_pred(peak_pos_logits)
        
        num_atoms_logits = self.num_atoms_pred(x) # -> (B, max_peaks * num_atom_types)
        num_atoms_logits = num_atoms_logits.view(-1, self.max_num_peaks, self.num_atom_types) # -> (B, max_peaks, num_atom_types)

        return ppm_pred_val, num_atoms_logits

class NMRPeakDecoder(nn.Module):
    def __init__(self, d_model, 
                 max_num_peaks, max_num_atoms_per_peak,
                 ppm_min, ppm_max,
                 n_decoder_head, decoder_dropout, n_decoder_layer,
                 device):
        super(NMRPeakDecoder, self).__init__()

        self.peak_queries = nn.Parameter(torch.randn(max_num_peaks, d_model))
        
        c = copy.deepcopy
        attn = MultiHeadedAttention(h=n_decoder_head, d_model=d_model, dropout=decoder_dropout)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=2048, dropout=0.1)
        self.decoder = Decoder(DecoderLayer(size=d_model, self_attn=c(attn), src_attn=c(attn), feed_forward=c(ff), dropout=0.1),
                                      N=n_decoder_layer)

        
        self.peak_pos_pred = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1)
        )

        # Classification part for num_atoms
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, max_num_atoms_per_peak)
        )    
       
        self.max_num_peaks = max_num_peaks
        self.max_num_atoms_per_peak = max_num_atoms_per_peak
        self.ppm_min = ppm_min
        self.ppm_max = ppm_max
        self.device_ = device

        self.__init_weights__()
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
    
    def proj_pos_pred(self, pos_logits):
        # 把预测从 [0,1] 映射回 ppm
        
        span = (self.ppm_max - self.ppm_min)
        ppm_pred = self.ppm_min + pos_logits * span      # (B, m)
        return ppm_pred
    
    def decode(self, enc_out, src_mask):
        bs = enc_out.shape[0]
        tgt_mask = torch.ones(bs, 1, self.peak_queries.shape[0], dtype=torch.bool).to(self.device_)
        peak_queries_ = self.peak_queries.unsqueeze(0).repeat(bs, 1, 1)
        peak_queries_ =  self.decoder(peak_queries_,
                            memory=enc_out,
                            src_mask=src_mask, 
                            tgt_mask=tgt_mask
                            )
        return peak_queries_
    
    def forward(self, enc_out, src_mask):
        
        peak_queries = self.decode(enc_out, src_mask)

        peak_pos_logits  = self.peak_pos_pred(peak_queries).squeeze().sigmoid()  # -> (B, max_num_peaks)
        ppm_pred_val = self.proj_pos_pred(peak_pos_logits)
        
        num_atoms_logits = self.cls_head(peak_queries) # -> (B, max_peaks, max_num_atoms_per_peak)

        return ppm_pred_val, num_atoms_logits


    def loss_pos(self, ppm_pred_val, ppm_true, ignore_idx=-100):
        # ppm_pred 和 ppm_true 的维度都是 (B, max_peaks)
        # mask 的维度是 (B, max_peaks)
        mask = torch.where(ppm_true == ignore_idx, 0, 1)
        # 逐元素计算损失
        loss = F.mse_loss(ppm_pred_val, ppm_true, reduction='none') # reduction='none' 保留每个元素的损失
        
        # 应用掩码，只保留真实峰的损失
        masked_loss = loss * mask
        
        # 计算平均损失时，分母应该是真实峰的数量，而不是总元素数量
        # 添加一个极小值防止 mask.sum() 为0
        final_loss = masked_loss.sum() / (mask.sum() + 1e-8) 
        
        return final_loss
    
    def loss_num_atoms(self, num_atoms_pred_logits, num_atoms_true_indices, ignore_idx=-100):
        """
        使用交叉熵计算原子数量的损失。
        
        Args:
            num_atoms_pred_logits (torch.Tensor): 模型的原始输出, 形状为 (B, max_peaks, num_classes)
            num_atoms_true_indices (torch.Tensor): 真实的类别索引, 形状为 (B, max_peaks), long 类型
            ignore_idx (int): 在真实标签中需要忽略的填充值
        """
        # 获取类别数量
        B, max_p, num_classes = num_atoms_pred_logits.shape
        
        # 为了匹配 F.cross_entropy 的输入要求，我们需要重塑张量
        # (B, max_peaks, num_classes) -> (B * max_peaks, num_classes)
        pred_logits_flat = num_atoms_pred_logits.view(-1, num_classes)
        
        # (B, max_peaks) -> (B * max_peaks,)
        true_indices_flat = num_atoms_true_indices.view(-1)
        
        # 计算损失，ignore_index 参数会自动处理我们之前用-100填充的假峰
        loss = F.cross_entropy(pred_logits_flat, true_indices_flat, ignore_index=ignore_idx)
        return loss
    
    def acc_num_atoms(self, num_atoms_pred_logits, num_atoms_true_indices, ignore_idx=-100):
        """
        在有padding的情况下，使用mask计算原子数量预测的准确率。

        Args:
            num_atoms_pred_logits (torch.Tensor): 模型输出, (B, max_peaks, num_classes)
            num_atoms_true_indices (torch.Tensor): 真实标签, (B, max_peaks), long类型
            mask (torch.Tensor): 掩码, (B, max_peaks), 真实峰为1, 填充为0

        Returns:
            float: 准确率 (0.0 to 1.0)
        """
        # 1. 获取预测的类别索引
        # torch.argmax在最后一个维度(-1)上操作，得到每个峰最可能的类别
        pred_indices = torch.argmax(num_atoms_pred_logits, dim=-1) # -> (B, max_peaks)
        
        # 确保mask是布尔类型以便索引
        mask = torch.where(num_atoms_true_indices == ignore_idx, 0, 1)
        bool_mask = mask.bool()
        
        # 2. 筛选出真实峰的预测和标签
        # 只选择 mask 为 True 的位置上的元素
        masked_preds = pred_indices[bool_mask]
        masked_true = num_atoms_true_indices[bool_mask]
        
        # 3. 计算这些真实峰中，有多少预测是正确的
        correct_predictions = (masked_preds == masked_true).sum().item()
        
        # 4. 真实峰的总数
        total_real_peaks = mask.sum().item()
        
        # 5. 计算准确率
        accuracy = correct_predictions / total_real_peaks if total_real_peaks > 0 else 0.0
        
        return accuracy

class NMRPeakPredictionModel(pl.LightningModule):
    def __init__(self, enc_output_dim, 
                 spec_len, patch_len, # attn encoder
                 max_num_peaks, one_hot_dim, ppm_min, ppm_max,
                 lr, warm_up_step,
                 device, dtype):
        super(NMRPeakPredictionModel, self).__init__()
        # self.encoder = NMR_CNN_Encoder(output_dim=enc_output_dim, kernel_size=7)
        self.encoder = NMR_Attn_Encoder(spec_len, patch_len, d_model=enc_output_dim, device=device)

        # self.peak_prediction_head = PeakPredictionHead(input_dim=enc_output_dim, 
        #                                                max_num_peaks=max_num_peaks,
        #                                                num_atom_types=one_hot_dim,
        #                                                ppm_min=ppm_min, ppm_max=ppm_max)

        self.peak_prediction_head = NMRPeakDecoder(d_model=enc_output_dim, max_num_peaks=max_num_peaks, 
                                                   max_num_atoms_per_peak=one_hot_dim,
                                                   ppm_min=ppm_min, ppm_max=ppm_max,
                                                   n_decoder_head=8, decoder_dropout=0.1, n_decoder_layer=4,
                                                   device=device)

        self.lr = lr
        self.warm_up_step = warm_up_step
        self.enc_output_dim = enc_output_dim
        self.device_ = device
        self.dtype_ = dtype

        self.save_hyperparameters()

    def compute_loss(self, ppm_pred_val, num_atoms_logits, ppm_true, num_atoms_true):

        loss_pos = self.peak_prediction_head.loss_pos(ppm_pred_val, ppm_true)
        loss_num_atoms = self.peak_prediction_head.loss_num_atoms(num_atoms_logits, num_atoms_true)
        loss = loss_pos + loss_num_atoms
        return loss, loss_pos, loss_num_atoms

    def forward(self, x, ppm_true, num_atoms_true):
        x, src_mask = self.encoder(x)
        # print("enc_out", x.shape)
        ppm_pred_val, num_atoms_logits = self.peak_prediction_head(x, src_mask)
        loss, loss_pos, loss_num_atoms = self.compute_loss(ppm_pred_val, num_atoms_logits, ppm_true, num_atoms_true)

        return x, ppm_pred_val, num_atoms_logits, loss, loss_pos, loss_num_atoms
    
    def step(self, batch, batch_idx):
        # nmr_h = batch['nmr_h'].unsqueeze(1).to(self.device_, dtype=self.dtype_) 
        nmr_h = batch['nmr_h'].to(self.device_, dtype=self.dtype_) 
        ppm_true = batch['x_grouped_H'].to(self.device_, dtype=self.dtype_) 
        # print("ppm_true")
        # print(ppm_true)
        num_atoms_true = batch['y_grouped_H'].to(self.device_) 
        # print("num_atoms_true")
        # print(num_atoms_true)
        x, ppm_pred_val, num_atoms_logits, loss, loss_pos, loss_num_atoms = self(nmr_h, ppm_true, num_atoms_true)
        return  x, ppm_pred_val, num_atoms_logits, loss, loss_pos, loss_num_atoms
    
    def training_step(self, batch, batch_idx):
        _, _, _, loss, loss_pos, loss_num_atoms = self.step(batch, batch_idx)
        batch_size = len(batch["id"])
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_loss_pos", loss_pos, batch_size=batch_size)
        self.log("train_loss_num_atoms", loss_num_atoms, batch_size=batch_size)

        return loss
    
    def validation_step(self, batch, batch_idx):
        _, _, num_atoms_logits, loss, loss_pos, loss_num_atoms = self.step(batch, batch_idx)
        batch_size = len(batch["id"])
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_loss_pos", loss_pos, batch_size=batch_size, prog_bar=True)
        self.log("val_loss_num_atoms", loss_num_atoms, batch_size=batch_size, prog_bar=True)

        num_atoms_true = batch['y_grouped_H'].to(self.device_) 
        acc_num_atoms = self.peak_prediction_head.acc_num_atoms(num_atoms_logits, num_atoms_true)
        self.log("val_acc_num_atoms", acc_num_atoms, batch_size=batch_size, prog_bar=True)


        return loss
    
    def test_step(self, batch, batch_idx):
        _, _, num_atoms_logits, loss, loss_pos, loss_num_atoms = self.step(batch, batch_idx)
        batch_size = len(batch["id"])
        self.log("test_loss", loss, batch_size=batch_size)
        self.log("test_loss_pos", loss_pos, batch_size=batch_size)
        self.log("test_loss_num_atoms", loss_num_atoms, batch_size=batch_size)

        num_atoms_true = batch['y_grouped_H'].to(self.device_) 
        acc_num_atoms = self.peak_prediction_head.acc_num_atoms(num_atoms_logits, num_atoms_true)
        self.log("test_acc_num_atoms", acc_num_atoms, batch_size=batch_size)

        return loss
    
    def configure_optimizers(self):
       
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.998),
            eps=1e-9,
            weight_decay=1e-5 
        )

        def rate(step):
           
            if step == 0:
                step = 1
            if self.warm_up_step > 0:
                value = self.enc_output_dim ** (-0.5) * min(step ** (-0.5), step * self.warm_up_step ** (-1.5))
            else:
                value = self.enc_output_dim ** (-0.5) * step ** (-0.5)
            lr_scale = 1 * value

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=rate
        )

        # 返回优化器和调度器
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
