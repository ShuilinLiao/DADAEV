import torch
import random
import numpy as np
import pandas as pd
from torch.optim import Adam,AdamW
import torch.nn.functional as F
import torch.nn as nn
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import copy
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from torch.distributions import MultivariateNormal
import torch.distributed as dist
from torch.autograd import Function
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

from utils import simdatset, realdatset, reproducibility

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # 前向传播不改变输入
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播时，将梯度乘以 -lambda_
        return grad_output.neg() * ctx.lambda_, None

class AdaptiveGradientReversal(nn.Module):
    """自适应梯度反转层"""
    def __init__(self, alpha=1.0, adapt_rate=0.01):
        super().__init__()
        self.alpha = alpha
        self.adapt_rate = adapt_rate
        self.cumulative_loss = 0
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def update_alpha(self, domain_loss):
        """根据域分类损失动态调整alpha"""
        if domain_loss > self.cumulative_loss:
            self.alpha = min(self.alpha + self.adapt_rate, 1.0)
        else:
            self.alpha = max(self.alpha - self.adapt_rate, 0.1)
        self.cumulative_loss = domain_loss

def compute_mmd(x, y, kernel='rbf', sigma_list=[1, 2, 4, 8, 16]):
    """
    计算两个样本集的 MMD 距离（多带宽 RBF kernel）
    x: [B, D]
    y: [B, D]
    返回: 标量 MMD
    """
    def rbf_kernel(a, b, sigma):
        dist = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    xx, yy, xy = 0, 0, 0
    for sigma in sigma_list:
        K_xx = rbf_kernel(x, x, sigma)
        K_yy = rbf_kernel(y, y, sigma)
        K_xy = rbf_kernel(x, y, sigma)
        xx += K_xx.mean()
        yy += K_yy.mean()
        xy += K_xy.mean()
    return xx + yy - 2 * xy

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def get_frac_from_sigmatrix(model, z, sigmatrix):
    model.eval()
    model.state = 'test'

    with torch.no_grad():
        x_recon = model.decode(z).detach().cpu()
        f = x_recon @ np.linalg.pinv(sigmatrix)

        f = f.numpy()
        f = np.maximum(f, 0) 
        row_sums = f.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-8  
        f = f / row_sums  

    return torch.tensor(f, dtype=torch.float32)

def evaluation(dat_loader, model, device='cuda'):
    model.eval()
    model.state = "test"
    
    x_recon_list, frac_list, z_list = [], [], []

    with torch.no_grad():
        for X, Y in dat_loader:
            X = X.to(device)
            x_recon, frac, z = model(X)
            x_recon_list.append(x_recon.cpu())
            frac_list.append(frac.cpu())
            z_list.append(z.cpu())

    x_recon_all = torch.cat(x_recon_list, dim=0).numpy()
    frac_all = torch.cat(frac_list, dim=0).numpy()
    z_all = torch.cat(z_list, dim=0).numpy()
    # sig_all = sig_list[0].numpy()  # sigmatrix 是固定的（每次一样），取第一个即可

    return x_recon_all, frac_all, z_all

class AdaptiveTAPEandDiffusion3(nn.Module):
    def __init__(self,input_dim, output_dim,Hidden_mul,T=2000):
        super().__init__()
        self.name = 'datape'
        self.state = 'train' # or 'test'
        self.inputdim = input_dim
        self.outputdim = output_dim

        latent_dim = 256 
        
        self.encoder = nn.Sequential(
            nn.Linear(self.inputdim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(), 

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(512, latent_dim),
            #nn.LayerNorm(256), 
        )

        self.predictor = nn.Sequential(
                                     nn.Linear(latent_dim, 128),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(64, output_dim),    # output_dim =23    
                                     )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.CELU(),
            nn.Linear(512, 1024),
            nn.CELU(),
            nn.Linear(1024, input_dim)
        )

        self.discriminator =  nn.Sequential(
                                     
                                     nn.Linear(latent_dim, 128),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(64, 2))
        self.grLaryer = AdaptiveGradientReversal(2)
        self.ref_creator =  MiniDiffuser2(latent_dim, T=T)
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def refraction(self, x, eps=1e-8):
        x = F.relu(x) # 确保非负
        x_nor = F.normalize(x, p=1, dim=1, eps=eps)
        return x_nor

    def adaptive_generater(self, x, z_fake):
        shape_batch, shape_feature = x.shape
        # z_fake = self.ref_creator.sample(shape_batch, shape_feature)
        domain_data = torch.cat([x, z_fake], dim=0)
        fake_label = torch.zeros(shape_batch, 1, device=x.device)
        real_label = torch.ones(shape_batch, 1, device=x.device)
        domain_label = torch.cat([real_label, fake_label], dim=0)
        perm = torch.randperm(domain_data.shape[0])
        domain_data = domain_data[perm]
        domain_label = domain_label[perm]
        return domain_data, domain_label
    
    def forward(self, x, z_fake = None):
        z = self.encode(x)
        if self.state == 'adapt':
            # adapt 模式下：使用梯度反转层和 discriminator 得到领域预测
            domain_data,ground_ture = self.adaptive_generater(z, z_fake)
            domain_out = self.discriminator(self.grLaryer(domain_data))
            domain_out2 = self.discriminator(domain_data)
            return domain_out,ground_ture, domain_out2
        elif self.state == 'diffusion':
            loss_diff, z_t  = self.ref_creator(z)
            return loss_diff, z_t  
        else:
            # 对于 train 和 test 模式，共同使用 predictor
            frac = self.predictor(z)
            frac = self.refraction(frac)

            # 重构过程：用 decoder 后的重建 
            x_recon = self.decode(z)
            return x_recon, frac, z

class TimeAwareResBlock(nn.Module):
    """时间感知残差块 (用于 Diffusion)"""
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim)
        )

    def forward(self, x, t_emb):
        t_feat = self.time_proj(t_emb)
        return x + self.mlp(x + t_feat)

class MiniDiffuser2(nn.Module):
    def __init__(self, latent_dim, T=2000):
        super().__init__()
        self.latent_dim = latent_dim
        self.T = T
        self.time_embed_dim = 128

        # 扩散超参数 Schedule
        self.betas = np.linspace(1e-4, 0.02, T)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # 注册缓冲区 (Buffers)
        self.register_buffer('sqrt_alphas_cumprod', torch.tensor(np.sqrt(self.alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(np.sqrt(1 - self.alphas_cumprod), dtype=torch.float32))
        self.register_buffer('betas_tensor', torch.tensor(self.betas, dtype=torch.float32))
        self.register_buffer('sqrt_recip_alphas', torch.tensor(np.sqrt(1.0 / self.alphas), dtype=torch.float32))
        self.register_buffer('posterior_variance', torch.tensor(self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), dtype=torch.float32))

        # 网络结构
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.input_proj = nn.Linear(latent_dim, 512)
        self.mid_blocks = nn.ModuleList([TimeAwareResBlock(512, 256) for _ in range(3)])
        self.output_proj = nn.Linear(512, latent_dim)

    def get_time_embedding(self, t):
        half_dim = self.time_embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if self.time_embed_dim % 2 == 1: emb = F.pad(emb, (0, 1, 0, 0))
        return self.time_mlp(emb)

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def forward(self, z0):
        """训练：预测噪声"""
        batch_size = z0.shape[0]
        device = z0.device
        t = torch.randint(0, self.T, (batch_size,), device=device).long()
        noise = torch.randn_like(z0)
        
        sqrt_alpha = self.extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_one_minus_alpha = self.extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        z_t = sqrt_alpha * z0 + sqrt_one_minus_alpha * noise
        
        t_emb = self.get_time_embedding(t)
        h = self.input_proj(z_t)
        for block in self.mid_blocks:
            h = block(h, t_emb)
        predicted_noise = self.output_proj(h)
        
        return F.mse_loss(predicted_noise, noise), z_t

    @torch.no_grad()
    def sample(self, batch_size, latent_dim_in_case_needed):
        """采样：生成潜在向量"""
        device = self.sqrt_alphas_cumprod.device
        shape = (batch_size, self.latent_dim)
        z = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_emb = self.get_time_embedding(t)
            
            h = self.input_proj(z)
            for block in self.mid_blocks:
                h = block(h, t_emb)
            predicted_noise = self.output_proj(h)
            
            betas_t = self.extract(self.betas_tensor, t, z.shape)
            sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, z.shape)
            sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, z.shape)
            
            model_mean = sqrt_recip_alphas_t * (z - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            
            if i > 0:
                noise = torch.randn_like(z)
                posterior_variance_t = self.extract(self.posterior_variance, t, z.shape)
                z = model_mean + torch.sqrt(posterior_variance_t) * noise 
            else:
                z = model_mean
        return z
    
class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        
        L2_distance = ((total0-total1)**2).sum(2)
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
            
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        if batch_size == 0 or target.size(0) == 0:
            return torch.tensor(0.0).to(source.device)
            
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX + YY - XY - YX)
        return loss
        
#======== train stage ==========
def alternate_training_earlyStop(model, dataloader, val_loader, optimizer_main, optimizer_diffusion, 
                      epochs_main, epochs_diffusion, device='cuda', patience=5, 
                      early_stop_start=400, early_stop_interval=50):
    criterion_reconstruct = nn.MSELoss()
    criterion_predict = nn.MSELoss()
    
    model.to(device)
    main_losses, diffusion_losses = [], []
    best_lccc = -1.0
    epochs_no_improve = 0
    best_model = model
    l1_lambda = 0.01 
    def compute_lccc(preds, targets):
        preds_mean = torch.mean(preds)
        targets_mean = torch.mean(targets)
        
        cov = torch.mean((preds - preds_mean) * (targets - targets_mean))
        preds_var = torch.var(preds)
        targets_var = torch.var(targets)
        
        numerator = 2 * cov
        denominator = preds_var + targets_var + (preds_mean - targets_mean)**2
        return numerator / denominator

    # 主网络训练阶段 Stage1
    model.state = 'train'
    model.ref_creator.requires_grad_(False)
    for epoch in range(epochs_main):
        model.train()
        total_loss = 0
        for data, label in dataloader:
            x = data.to(device)
            label = label.to(device)
            optimizer_main.zero_grad()
            x_recon, frac, z = model(x)
            loss_re = criterion_reconstruct(x_recon, x)
            loss_pr = criterion_predict(frac, label)
            
            loss_l1 = torch.mean(torch.abs(z))
            
            # 总 Loss
            loss = loss_re + 2 * loss_pr + l1_lambda * loss_l1
            # torch.isnan(frac).any().item()
            # torch.isnan(label).any().item()
            # print(f"loss_re {loss_re:.4f}, loss_pr: {loss_pr:.4f}, loss: {loss:.4f}")
            loss.backward()
            optimizer_main.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        main_losses.append(avg_loss)
        
        # 验证阶段（仅在满足条件时执行）
        if (epoch + 1) >= early_stop_start and (epoch + 1) % early_stop_interval == 0:
            model.eval()
            with torch.no_grad():
                
                val_preds, val_targets = [], []
                for val_data, val_label in val_loader:
                    val_x = val_data.to(device)
                    val_label = val_label.to(device)
                    _, frac, _ = model(val_x)
                    val_preds.append(frac)
                    val_targets.append(val_label)
                
                val_preds = torch.cat(val_preds)
                val_targets = torch.cat(val_targets)
                current_lccc = compute_lccc(val_preds, val_targets)
                
                print(f"Epoch [{epoch+1}/{epochs_main}], Main Loss: {avg_loss:.4f}, Val LCCC: {current_lccc:.4f}")
                
                # 早停逻辑
                if current_lccc > best_lccc:
                    best_lccc = current_lccc
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    # best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += early_stop_interval  # 因为我们每interval代才检查一次
                    if epochs_no_improve >= patience * early_stop_interval:
                        print(f"Early stopping at epoch {epoch+1} after {early_stop_start} epochs")
                        # best_model = model
                        # model.load_state_dict(best_model_state)
                        break
        else:
            print(f"Epoch [{epoch+1}/{epochs_main}], Main Loss: {avg_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 冻结主网络参数
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.predictor.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    # Diffusion模块训练阶段 Stage2
    model.state = 'diffusion'
    model.ref_creator.requires_grad_(True)
    diffusion_losses = []
    best_diff_loss = float('inf')
    epochs_no_improve_diff = 0
    # model = best_model
    # best_model = model

    for epoch in range(epochs_diffusion):
        model.train()
        total_loss_diff = 0
        for data, label in dataloader:
            x = data.to(device)
            optimizer_diffusion.zero_grad()
            with torch.no_grad():
                z = model.encode(x)
            loss_diff, z_t = model.ref_creator(z)
            #mmd_loss_func = MMDLoss(kernel_mul=2.0, kernel_num=3) 
            #loss_diff += mmd_loss_func(z_t, z)
            #loss_diff += torch.mean(torch.abs(z_t))
            loss_diff.backward()
            optimizer_diffusion.step()
            total_loss_diff += loss_diff.item()
        
        avg_loss_diff = total_loss_diff / len(dataloader)
        diffusion_losses.append(avg_loss_diff)
        
        # Diffusion阶段的验证（同样应用间隔条件）
        if (epoch + 1) >= early_stop_start and (epoch + 1) % early_stop_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss_diff = 0
                for val_data, _ in val_loader:
                    val_x = val_data.to(device)
                    z = model.encode(val_x)
                    loss_diff, _ = model.ref_creator(z)
                    val_loss_diff += loss_diff.item()
                
                avg_val_loss_diff = val_loss_diff / len(val_loader)
                print(f"Epoch [{epoch+1}/{epochs_diffusion}], Diffusion Loss: {avg_loss_diff:.5f}, Val Loss: {avg_val_loss_diff:.5f}")
                
                if avg_val_loss_diff < best_diff_loss:
                    best_diff_loss = avg_val_loss_diff
                    epochs_no_improve_diff = 0
                    best_diff_model_state = copy.deepcopy(model.state_dict())
                else:
                    epochs_no_improve_diff += early_stop_interval
                    if epochs_no_improve_diff >= patience * early_stop_interval:
                        # best_model = copy.deepcopy(model)
                        print(f"Early stopping diffusion at epoch {epoch+1} after {early_stop_start} epochs")
                        break
        else:
            print(f"Epoch [{epoch+1}/{epochs_diffusion}], Diffusion Loss: {avg_loss_diff:.5f}")

    # 解冻主网络参数
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.predictor.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True

    return model, main_losses, diffusion_losses

def predict2(test_x, trainX, model_name=None, adaptive=True, mode='overall', device='cuda'):

    if model_name is not None:
        model = torch.load(model_name + ".pth", weights_only=False, map_location=device)

    if adaptive is True:
        if mode == 'overall':
            #设置解码优化器
            decoder_optimizer  = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
            #设置编码优化器
            encoder_optimizer =  torch.optim.Adam(
                [
                    {'params':model.encoder.parameters()},
                    {'params':model.predictor.parameters()}
                ],lr=1e-4
            )
            #设置域对抗优化器
            optimizer_D = torch.optim.Adam(
                [{'params': model.encoder.parameters()},
                {'params': model.discriminator.parameters()}],
                lr=1e-5 )

            test_sigm, loss, test_pred,  x_recon_res = adaptive_stage_domain(model, test_x, trainX, optimizer_D, encoder_optimizer, decoder_optimizer, steps=500, max_iter=3, device=device)
            return test_sigm, test_pred, x_recon_res

    else:
        print('Predict cell fractions without adaptive training')
        model.eval()
        model.state = 'test'
        data = torch.from_numpy(test_x).float().to(device)
        _, pred, _ = model(data)
        pred = pred.cpu().detach().numpy()
        print('Prediction is done')
        return 
     
def adaptive_stage_domain_difSAwiwoS_noise(x, sour_x_train = None, model_name=None, 
                                           adaptive=True, mode='overall5', 
                           steps=20, max_iter=40, device='cuda', 
                           sigmatrix=None, ep_warmup=50, save_name=None, 
                           real_std=1, real_mean=0,
                           generator=None, modeS="wt",
                           teacher_noise_std=0.0  
                           ): 

    Domain_Loss = nn.CrossEntropyLoss()
    
    # 1. 加载模型
    if model_name is not None:
        model = torch.load(model_name + ".pth", weights_only=False, map_location=device)
        model = model.to(device)
    
    # 自动侦测隐层维度 (Latent Dimension)
    with torch.no_grad():
        # 取前2个样本跑一次 encode，看看输出是多少维 (比如 256)
        dummy_input = torch.tensor(x[0:2]).float().to(device)
        dummy_z = model.encode(dummy_input)
        latent_dim = dummy_z.shape[1] 
        print(f"Detected Latent Dimension: {latent_dim} (Original Gene Dim: {x.shape[1]})")

    realdata_loader = DataLoader(realdatset(x), batch_size=len(x), shuffle=False) 
    
    if sour_x_train is not None:
        print(sour_x_train.shape[0])
        print(x.shape[0])
        indices = np.random.choice(sour_x_train.shape[0], x.shape[0], replace=False)
        ori_sour = sour_x_train[indices]
        ori_sour = torch.tensor(ori_sour, device=device).to(torch.float32)
        
    elif sour_x_train is None:
        # 预生成数据 (Pre-generation)
        # 避免在循环中实时采样 Diffusion，解决训练慢的问题
        source_bank = None
        if adaptive and generator is None:
            print(">>> Pre-generating Source Latents via Diffusion...")
            n_generate = 50000 
            batch_gen = 2000
            bank_list = []
            
            model.eval()
            with torch.no_grad():
                for _ in range(0, n_generate, batch_gen):
                    z_gen = model.ref_creator.sample(batch_gen, latent_dim)
                    bank_list.append(z_gen.cpu())
            
            source_bank = torch.cat(bank_list, dim=0).to(device)
            print(f">>> Generated {source_bank.shape[0]} latent samples.")

        # 辅助函数：统一采样接口
        def get_source_samples(n_samples):
            if generator is not None:
                # 使用外部生成器 (GMM/Gaussian)
                z = generator.sample(n_samples)
            elif source_bank is not None:
                # 使用预生成池 (Diffusion 极速版)
                indices = torch.randint(0, source_bank.shape[0], (n_samples,), device=device)
                z = source_bank[indices]
                # 反标准化 (如果训练时做了标准化)
                # z = z * real_std + real_mean
            else:
                # 实时生成 (不推荐，很慢)
                z = model.ref_creator.sample(n_samples, latent_dim)
                z = z * real_std + real_mean
            
            return z.to(device).detach()

    best_pcc = -1
    best_model = None
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    model_copy.eval() 

    # === Main Adaptation Phase ===
    if adaptive is True:
        if mode == 'overall5':
            model.train()
            # 解冻需要训练的参数
            for param in model.encoder.parameters(): param.requires_grad = True
            for param in model.discriminator.parameters(): param.requires_grad = True
            for param in model.predictor.parameters(): param.requires_grad = True
            for param in model.decoder.parameters(): param.requires_grad = True 
            
            model.state = 'train'
            model.ref_creator.requires_grad_(False) 

            optimizer_da1 = torch.optim.AdamW([
                {'params':model.encoder.parameters()},
                {'params':model.predictor.parameters()},
                {'params':model.discriminator.parameters()}], lr=1e-5)
            optimizer_da2 = torch.optim.AdamW([
                {'params':model.encoder.parameters()},
                {'params':model.discriminator.parameters()}], lr=1e-5)
            optimizer_da3 = torch.optim.AdamW([
                {'params':model.encoder.parameters()},
                {'params':model.decoder.parameters()},
                {'params':model.predictor.parameters()}], lr=1e-5)
            
            for iter in range(max_iter):
                if (iter+1) % 10 == 0:
                    print(f"Iter [{iter+1}/{max_iter}]")
                
                # Domain adaptation Loop
                for _ in range(steps):
                    model.train()
                    model.state = 'adapt'
                    
                    for step, X in enumerate(realdata_loader):
                        X = X.to(device)
                        
                        if sour_x_train is not None:
                            z_batch = model_copy.encode(ori_sour).detach()    
                            
                        elif sour_x_train is None:
                            z_batch = get_source_samples(X.shape[0])    
                                        
                        z_target = model.encoder(X) 

                        # Step 1: Align Predictor & Discriminator
                        _, ground_true, preds2 = model(X, z_batch) 
                        ground_true = ground_true.squeeze().long().to(device)
                        disc_loss = Domain_Loss(preds2, ground_true)
                        # print(disc_loss)
                        
                        source_frac = get_frac_from_sigmatrix(model_copy, z_batch, sigmatrix) # z_batch z_target
                        source_frac = source_frac.to(device)
                        
                        # === 注入噪声测试鲁棒性 ===
                        if teacher_noise_std > 0:
                            noise = torch.randn_like(source_frac) * teacher_noise_std
                            source_frac = source_frac + noise
                            # source_frac = torch.clamp(source_frac, min=0.0)  # 保证非负
                            
                        frac = model.predictor(z_batch) # z_batch z_target 
                        frac = F.relu(frac)
                        frac_pred = model.refraction(frac)
                        pred_loss = compute_mmd(frac_pred, source_frac)
                            
                        if modeS=="wt":
                            loss =  disc_loss + pred_loss
                        else:
                            loss = disc_loss #  + 0.1 * pred_loss
                        
                        optimizer_da1.zero_grad()
                        loss.backward()
                        optimizer_da1.step()

                        # Step 2: Adversarial Training for Encoder
                        preds, ground_true, _ = model(X, z_batch) 
                        ground_true = ground_true.squeeze().long().to(device)
                        disc_loss_DA = Domain_Loss(preds, ground_true)
                        disc_loss_DA.backward()
                        optimizer_da2.step()
                        optimizer_da2.zero_grad()

                    # Step 3: Reconstruction
                    model.train()
                    model.state = 'adapt'
                    for step, X in enumerate(realdata_loader):
                        X = X.to(device)
                        z_targ = model.encode(X)
                        z_targ_recon = model.decoder(z_targ)
                        recon_loss = F.mse_loss(X, z_targ_recon)
                        
                        da3_loss = recon_loss
                        da3_loss.backward()
                        optimizer_da3.step()
                        optimizer_da3.zero_grad()
    else:
        model = model

    # 最终预测
    model.eval()
    model.state = 'test'
    with torch.no_grad():
        for _, X in enumerate(realdata_loader):
            X = X.to(device)
            x_recon_res, pred_res, z_res = model(X)
            break 
    
    return (
        x_recon_res.detach().cpu().numpy(),
        pred_res.detach().cpu().numpy(),
        z_res.detach().cpu().numpy(),
        model
    )
      
def adaptive_stage_domain_difSAwiwoS_latentMMD(x, model_name=None, adaptive=True, mode='overall5', 
                           steps=20, max_iter=200, device='cuda', 
                           sigmatrix=None, ep_warmup=50, save_name=None, 
                           real_std=1, real_mean=0,
                           generator=None, modeS="wt",
                           teacher_noise_std=0.0): 

    Domain_Loss = nn.CrossEntropyLoss()
    
    # 1. Load Model
    if model_name is not None:
        model = torch.load(model_name + ".pth", weights_only=False, map_location=device)
        model = model.to(device)
    
    # Detect Latent Dim
    with torch.no_grad():
        dummy_input = torch.tensor(x[0:2]).float().to(device)
        dummy_z = model.encode(dummy_input)
        latent_dim = dummy_z.shape[1] 
        print(f"Detected Latent Dimension: {latent_dim}")

    realdata_loader = DataLoader(realdatset(x), batch_size=len(x), shuffle=False) 
    
    # Pre-generate Source Bank
    source_bank = None
    if adaptive and generator is None:
        print(">>> Pre-generating Source Latents via Diffusion...")
        n_generate = 5000  # Adjusted size for evaluation
        batch_gen = 1000
        bank_list = []
        model.eval()
        with torch.no_grad():
            for _ in range(0, n_generate, batch_gen):
                z_gen = model.ref_creator.sample(batch_gen, latent_dim)
                bank_list.append(z_gen.cpu())
        source_bank = torch.cat(bank_list, dim=0).to(device)

    def get_source_samples(n_samples):
        if generator is not None:
            z = generator.sample(n_samples)
        elif source_bank is not None:
            indices = torch.randint(0, source_bank.shape[0], (n_samples,), device=device)
            z = source_bank[indices]
        else:
            z = model.ref_creator.sample(n_samples, latent_dim)
            z = z * real_std + real_mean
        return z.to(device).detach()

    def get_latents_subset(model, x_input, max_samples=2000):
        n_total = x_input.shape[0]
        
        if n_total > max_samples:
            indices = np.random.choice(n_total, max_samples, replace=False)
            x_subset = x_input[indices]
        else:
            x_subset = x_input
            
        x_subset = torch.tensor(x_subset).float().to(device)
        z_subset = model.encode(x_subset)
        return z_subset
    
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    model_copy.eval() 

    # === Main Adaptation Phase ===
    if adaptive is True:
        if mode == 'overall5':
            model.train()
            
            for param in model.encoder.parameters(): param.requires_grad = True
            for param in model.discriminator.parameters(): param.requires_grad = True
            for param in model.predictor.parameters(): param.requires_grad = True
            for param in model.decoder.parameters(): param.requires_grad = True 
            
            model.state = 'train'
            model.ref_creator.requires_grad_(False)
            
            # Define Optimizers
            optimizer_da1 = torch.optim.Adam([
                {'params':model.encoder.parameters()},
                {'params':model.predictor.parameters()},
                {'params':model.discriminator.parameters()}
                ], lr=1e-5)
            optimizer_da2 = torch.optim.Adam([
                {'params':model.encoder.parameters()},
                {'params':model.discriminator.parameters()}
                ], lr=1e-5)
            optimizer_da3 = torch.optim.Adam([
                {'params':model.encoder.parameters()},
                {'params':model.decoder.parameters()},
                {'params':model.predictor.parameters()}
                ], lr=1e-5)
            
            for iter in range(max_iter):
                if (iter+1) % 10 == 0:
                    print(f"Iter [{iter+1}/{max_iter}]")
                
                for _ in range(steps):
                    model.train()
                    model.state = 'adapt'
                    
                    for step, X in enumerate(realdata_loader):
                        X = X.to(device)
                        z_batch = get_source_samples(X.shape[0])                        
                        z_target = model.encoder(X) 
                        
                        # Step 1: Align Predictor & Discriminator
                        _, ground_true, preds2 = model(X, z_batch) 
                        ground_true = ground_true.squeeze().long().to(device)
                        disc_loss = Domain_Loss(preds2, ground_true)
                        
                        if modeS=="wt":
                            source_frac = get_frac_from_sigmatrix(model_copy, z_batch, sigmatrix)
                            source_frac = source_frac.to(device)
                            if teacher_noise_std > 0:
                                noise = torch.randn_like(source_frac) * teacher_noise_std
                                source_frac = source_frac + noise
                                
                            frac = model.predictor(z_batch) 
                            frac = F.relu(frac)
                            frac_pred = model.refraction(frac)
                        
                            pred_loss = compute_mmd(frac_pred, source_frac) # This is your existing Fraction MMD
                            loss = disc_loss + pred_loss
                        else:
                            loss = disc_loss
                        
                        optimizer_da1.zero_grad()
                        loss.backward()
                        optimizer_da1.step()

                        # Step 2: Adversarial Training for Encoder
                        preds, ground_true, _ = model(X, z_batch) 
                        ground_true = ground_true.squeeze().long().to(device)
                        disc_loss_DA = Domain_Loss(preds, ground_true)
                        
                        optimizer_da2.zero_grad()
                        disc_loss_DA.backward()
                        optimizer_da2.step()
                        
                    # Step 3: Reconstruction
                    model.train()
                    model.state = 'adapt'
                    for step, X in enumerate(realdata_loader):
                        X = X.to(device)
                        z_targ = model.encode(X)
                        z_targ_recon = model.decoder(z_targ)
                        recon_loss = F.mse_loss(X, z_targ_recon)
                        
                        optimizer_da3.zero_grad()
                        recon_loss.backward()
                        optimizer_da3.step()

    # Final Inference
    model.eval()
    model.state = 'test'
    with torch.no_grad():
        for _, X in enumerate(realdata_loader):
            X = X.to(device)
            x_recon_res, pred_res, z_res = model(X)
            break 
    
    return (
        x_recon_res.detach().cpu().numpy(),
        pred_res.detach().cpu().numpy(),
        z_res.detach().cpu().numpy(),
        model
    )

