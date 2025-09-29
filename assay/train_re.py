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
from torch.utils.data import DataLoader
from utils import simdatset, realdatset

import warnings
warnings.filterwarnings("ignore")
from simulation import generate_simulated_data
from utils import ProcessInputDataSplit,ProcessInputDataSplit_sim,ProcessInputData, simdatset, reproducibility

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None
class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
class AdaptiveGradientReversal(nn.Module):
    def __init__(self, alpha=1.0, adapt_rate=0.01):
        super().__init__()
        self.alpha = alpha
        self.adapt_rate = adapt_rate
        self.cumulative_loss = 0
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def update_alpha(self, domain_loss):

        if domain_loss > self.cumulative_loss:
            self.alpha = min(self.alpha + self.adapt_rate, 1.0)
        else:
            self.alpha = max(self.alpha - self.adapt_rate, 0.1)
        self.cumulative_loss = domain_loss

def plot_umap(real_z, fake_z, epoch, save_path=None):
    # real_z, fake_z: [N, D] tensor or np.array
    X = np.concatenate([real_z, fake_z], axis=0)
    labels = np.array(['Real']*len(real_z) + ['Fake']*len(fake_z))
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_umap[labels=='Real',0], X_umap[labels=='Real',1], c='blue', label='Real z', alpha=0.5, s=10)
    plt.scatter(X_umap[labels=='Fake',0], X_umap[labels=='Fake',1], c='orange', label='Fake z', alpha=0.5, s=10)
    plt.legend()
    plt.title(f'UMAP of Latent Vectors (Epoch {epoch})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_tsne(real_z, fake_z, epoch, save_path=None):

    X = np.concatenate([real_z, fake_z], axis=0)
    labels = np.array(['Real']*len(real_z) + ['Fake']*len(fake_z))
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_tsne[labels=='Real',0], X_tsne[labels=='Real',1], c='blue', label='Real z', alpha=0.5, s=10)
    plt.scatter(X_tsne[labels=='Fake',0], X_tsne[labels=='Fake',1], c='orange', label='Fake z', alpha=0.5, s=10)
    plt.legend()
    plt.title(f't-SNE of Latent Vectors (Epoch {epoch})')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def plot_loss_curves(losses_dict, epoch, save_path=None):
    plt.figure(figsize=(10,5))
    for name, losses in losses_dict.items():
        plt.plot(range(1, len(losses)+1), losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves (up to Epoch {epoch})")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def compute_mmd(x, y, kernel='rbf', sigma_list=[1, 2, 4, 8, 16]):
    """

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

class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()

        self.log_var_cell = nn.Parameter(torch.zeros(()))
        self.log_var_recon = nn.Parameter(torch.zeros(()))
    
    def forward(self, loss_cell, loss_recon):

        loss = torch.exp(-self.log_var_cell) * loss_cell + self.log_var_cell + \
               torch.exp(-self.log_var_recon) * loss_recon + self.log_var_recon
        return loss
    def get_weights(self):
        weight_cell = torch.exp(-self.log_var_cell)
        weight_recon = torch.exp(-self.log_var_recon)
        return weight_cell, weight_recon

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)

def generate_linear_schedule(T, low, high):
    return np.linspace(low, high, T)

class ResidualNoiseMLP(nn.Module):
    def __init__(self, latent_dim, time_dim=128):
        super().__init__()
        self.input_dim = latent_dim + time_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(self.input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim)

        if self.input_dim != latent_dim:
            self.shortcut = nn.Linear(self.input_dim, latent_dim)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        residual = self.shortcut(x)  
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        out = self.dropout(out)  
        return out + residual  

def time_embedding(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class MiniDiffuser2(nn.Module):
    def __init__(self, latent_dim, T=2000):
        super().__init__()
        self.latent_dim = latent_dim
        self.T = T

        # Generate the noise schedule
        self.betas = generate_cosine_schedule(T)

        self.num_timesteps = len(self.betas)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.time_embed_dim = 128  

        # Register buffers
        self.register_buffer('sqrt_alphas_cumprod', torch.tensor(np.sqrt(self.alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.tensor(np.sqrt(1 - self.alphas_cumprod), dtype=torch.float32))
        self.register_buffer("reciprocal_sqrt_alphas", torch.tensor(np.sqrt(1 / self.alphas), dtype=torch.float32))
        self.register_buffer("remove_noise_coeff",  torch.tensor(self.betas / np.sqrt(1 - self.alphas_cumprod), dtype=torch.float32))
        self.register_buffer('sigma', torch.tensor(np.sqrt(self.betas), dtype=torch.float32))
        
        # Simple denoising model for latent space
        self.noise_model = nn.Sequential(
            nn.Linear(latent_dim, 1024), 
            nn.SiLU(),
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim)
        )
    
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, 256),  
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, latent_dim * 2)  
        )

    def sinusoidal_embedding(self, t, embedding_dim):
  
        assert embedding_dim % 2 == 0, 
        half_dim = embedding_dim // 2
        
        exponents = torch.arange(half_dim, dtype=torch.float32, device=t.device)
        exponents = 2 * exponents / embedding_dim
        emb_factors = 1.0 / (10000.0 ** exponents) 
        
        t_flat = t.flatten().float()  
        t_emb = t_flat[:, None]       
        emb_factors = emb_factors[None, :] 
        
        angles = t_emb * emb_factors  
        
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        
        emb = torch.zeros((t_flat.size(0), embedding_dim), 
                        dtype=torch.float32, device=t.device)
        
        emb[:, 0::2] = sin_emb
        emb[:, 1::2] = cos_emb
        
        return emb
    def get_time_embedding(self, t):
        t_emb = self.sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        scale, shift = t_emb.chunk(2, dim=1)
        return scale, shift
    def apply_conditioning(self, x, scale, shift):
        
        return x * (scale + 1) + shift
    
    @torch.no_grad()
    def denoise(self, z_t, eta=0.1, clip_value=10.0, add_noise=True):
        x = z_t.clone().to(dtype=torch.float32)
        batch_size = x.shape[0]
        device = x.device

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            scale, shift = self.get_time_embedding(t_batch)
            x_conditioned = self.apply_conditioning(x, scale, shift)
            predicted_noise = self.noise_model(x_conditioned)
            coeff1 = extract(self.remove_noise_coeff, t_batch, x.shape)
            coeff2 = extract(self.reciprocal_sqrt_alphas, t_batch, x.shape)
            x_new = (x - coeff1 * predicted_noise) * coeff2
            if add_noise and t > 0:
                noise = torch.randn_like(x)
                sigma = extract(self.sigma, t_batch, x.shape)
                x_new += sigma * noise

            x = (1 - eta) * x + eta * x_new
            x = torch.clamp(x, -clip_value, clip_value)

        return x
    
    @torch.no_grad()
    def sample(self, batch_size, latent_dim=256, eta=0.1, clip_value=10.0, add_noise=True):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(batch_size, self.latent_dim, device=device) 

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            scale, shift = self.get_time_embedding(t_batch)
            x_conditioned = self.apply_conditioning(x, scale, shift)
            predicted_noise = self.noise_model(x_conditioned)
            coeff1 = extract(self.remove_noise_coeff, t_batch, x.shape)
            coeff2 = extract(self.reciprocal_sqrt_alphas, t_batch, x.shape)
            x_new = (x - coeff1 * predicted_noise) * coeff2

            if add_noise and t > 0:
                noise = torch.randn_like(x)
                sigma = extract(self.sigma, t_batch, x.shape)
                x_new += sigma * noise

            x = (1 - eta) * x + eta * x_new
            x = torch.clamp(x, -clip_value, clip_value)

        return x
    
    def forward(self, z0)
        batch_size = z0.shape[0]
        device = z0.device
        
        noise = torch.randn_like(z0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        scale, shift = self.get_time_embedding(t)

        sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)
        z_t = sqrt_alpha_cumprod_t * z0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        z_t_conditioned = self.apply_conditioning(z_t, scale, shift)
        
        predicted_noise = self.noise_model(z_t_conditioned)
        loss = F.mse_loss(predicted_noise, noise)
        return loss, z_t

def L1_loss(preds, gt):
    gt.to(preds.device)
    loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
    return loss

class AdaptiveTAPEandDiffusion2(nn.Module):
    def __init__(self,input_dim, output_dim,Hidden_mul,T=2000):
        super().__init__()
        self.name = 'datape'
        self.state = 'train' # or 'test'
        self.inputdim = input_dim
        self.outputdim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.inputdim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.predictor = nn.Sequential(
                                     nn.Linear(256, 128),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(64, output_dim),    # output_dim =23    
                                     )
        

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

        self.discriminator =  nn.Sequential(
                                     
                                     nn.Linear(256, 128),
                                     nn.CELU(),
                                     nn.Dropout(),
                                     nn.Linear(128, 64),
                                     nn.CELU(),

                                     nn.Dropout(),
                                     nn.Linear(64, 2))
        self.grLaryer = AdaptiveGradientReversal(2)
        self.ref_creator =  MiniDiffuser2(256, T=T)
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    def refraction(self, x, eps=1e-8):
        x_nor = F.normalize(x, p=1, dim=1, eps=eps)
        return x_nor
    
    def reconstruct_from_latent(self, z_fake):
        """
        接收外部给定的假潜在向量 z_fake，[B, 256]
        输出：
          x_recon   重构的输入空间数据 [B, inputdim]
          z_proc    经 ReLU + 归一化后的潜向量 [B, 256]
          sigmatrix 签名矩阵 [256, inputdim]
        """
        # 1）先拿签名矩阵
        sigmatrix = self.sigmatrix()
        z = self.predictor(z_fake)
        # 2）激活 + 归一化
        z = F.relu(z)
        z = self.refraction(z)

        # 3）用签名矩阵重构
        x_recon = torch.mm(z, sigmatrix)

        return x_recon, z, sigmatrix

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
    # 计算解码器的签名矩阵
        # sigmatrix = self.sigmatrix()
        z = self.encode(x)
        if self.state == 'adapt':
            # adapt 模式下：使用梯度反转层和 discriminator 得到领域预测
            domain_data,ground_ture = self.adaptive_generater(z, z_fake)
            domain_out = self.discriminator(self.grLaryer(domain_data))
            domain_out2 = self.discriminator(domain_data)
            return domain_out,ground_ture, domain_out2
        elif self.state == 'diffusion':
            # print(z.shape)
            loss_diff, z_t  = self.ref_creator(z)
            # z0_hat = self.ref_creator.denoise(z_t)
            return loss_diff, z_t  
        else:
            # 对于 train 和 test 模式，共同使用 predictor
            frac = self.predictor(z)
            frac = F.relu(frac)
            frac = self.refraction(frac)


            # 重构过程：用 decoder 后的重建 
            x_recon = self.decode(z)
            #print(x_recon.shape)
            return x_recon, frac, z

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

    def compute_lccc(preds, targets):
        preds_mean = torch.mean(preds)
        targets_mean = torch.mean(targets)
        
        cov = torch.mean((preds - preds_mean) * (targets - targets_mean))
        preds_var = torch.var(preds)
        targets_var = torch.var(targets)
        
        numerator = 2 * cov
        denominator = preds_var + targets_var + (preds_mean - targets_mean)**2
        return numerator / denominator

    # 主网络训练阶段
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
            loss = loss_re + 2 * loss_pr
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

    # Diffusion模块训练阶段
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

    return x_recon_all, frac_all, z_all

def soft_CE_loss(pred, soft_target):
    log_prob = F.log_softmax(pred, dim=1)
    loss = -(soft_target * log_prob).sum(dim=1).mean()
    return loss

def adaptive_stage_domain9(x, model_name=None, adaptive=True, mode='overall', steps=20, max_iter=200, device='cuda', sigmatrix = None,ep_warmup=50,save_name = None):

    Domain_Loss = nn.CrossEntropyLoss()
    
    if model_name is not None:
        model = torch.load(model_name + ".pth", weights_only=False, map_location=device)
        model = model.to(device)
    realdata_loader =DataLoader(realdatset(x), batch_size=len(x), shuffle=False) 
    shape_batch, shape_feature = x.shape
    #z_fake_all = model.ref_creator.sample(shape_batch, shape_feature)

    best_pcc = -1
    best_model = None
    ema_alpha = 0.9  # EMA decay
    ema_alpha = torch.tensor(ema_alpha, device=device)

    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    
    if adaptive and ep_warmup > 0:
        print(f"[Warmup] epochs = {ep_warmup}")
        model.train()
        # 只训练部分参数, 例如encoder和predictor
        for param in model.encoder.parameters():
            param.requires_grad = True
        for param in model.predictor.parameters():
            param.requires_grad = True
        for param in model.decoder.parameters():
            param.requires_grad = True
        model.ref_creator.requires_grad_(False)
        optimizer_warmup = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=1e-6
        )
        for ep in range(ep_warmup):
            warmup_loss1 = 0.0
            model.train()
            model.state = 'adapt'
            for _, X in enumerate(realdata_loader):
                X = X.to(device)
                z_batch = model.ref_creator.sample(X.shape[0], shape_feature).detach()
                z_targ = model.encode(X).detach()
                # z_batch = z_fake_all[step * X.size(0):(step + 1) * X.size(0)].to(device)
                z_targ_recon = model.decoder(z_targ)
                #recon_loss = F.mse_loss(X, z_targ_recon)
                            
                source_frac = get_frac_from_sigmatrix(model_copy, z_batch, sigmatrix) # 从固定sig中获取frac
                source_frac = source_frac.to(device)
                frac = model.predictor(z_batch) # 预测frac
                frac = F.relu(frac)
                frac_pred = model.refraction(frac)
                
                _, ground_true, preds2 = model(X, z_batch)
                ground_true = ground_true.squeeze().long().to(device)
                disc_loss = Domain_Loss(preds2, ground_true)
                
                pred_loss_DA = compute_mmd(frac_pred, source_frac)*0.1
                da3_loss = pred_loss_DA+disc_loss
                da3_loss.backward()
                optimizer_warmup.step()
                optimizer_warmup.zero_grad()
                warmup_loss1+=da3_loss.item()
            print(f"[Warmup phas 1 Epoch {ep+1}/{ep_warmup}] loss: {warmup_loss1/len(realdata_loader):.4f}")


    if adaptive is True:

        if mode == 'overall5':
            model.train()
            # 解冻主网络参数以备后续训练
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.discriminator.parameters():
                param.requires_grad = True
            for param in model.predictor.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = True 

            model.state = 'train'
            model.ref_creator.requires_grad_(False)

            #设置解码优化器
            optimizer_da1 =  torch.optim.Adam(
                [{'params':model.encoder.parameters()},
                {'params':model.predictor.parameters()},
                {'params':model.discriminator.parameters()}],lr=1e-5)
            #设置编码优化器
            optimizer_da2 =  torch.optim.Adam(
                [{'params':model.encoder.parameters()},
                {'params':model.discriminator.parameters()}],lr=1e-5)
            optimizer_da3 =  torch.optim.Adam(
                [{'params':model.encoder.parameters()},
                {'params':model.decoder.parameters()},
                {'params':model.predictor.parameters()}],lr=1e-5
            )
            
            for iter in range(max_iter):
                print(f"Iter [{iter+1}/{max_iter}]")
                # Domain adaptation
                for _ in range(steps):
                    model.train()
                    model.state = 'adapt'
                    for step, X in enumerate(realdata_loader):
                        X = X.to(device)
                        z_targ = model.encode(X).detach()
                        # z_batch = z_fake_all[step * X.size(0):(step + 1) * X.size(0)].to(device)
                        z_batch = model.ref_creator.sample(X.shape[0], shape_feature).detach()

                        # 反转前
                        _, ground_true, preds2 = model(X, z_batch) 
                        ground_true = ground_true.squeeze().long().to(device)
                        disc_loss = Domain_Loss(preds2, ground_true)

                        source_frac = get_frac_from_sigmatrix(model_copy, z_batch, sigmatrix) # 从固定sig中获取frac
                        source_frac = source_frac.to(device)
                        frac = model.predictor(z_batch) # 预测frac
                        frac = F.relu(frac)
                        frac_pred = model.refraction(frac)

                        pred_loss =compute_mmd(frac_pred, source_frac)
                        loss = pred_loss + disc_loss
                        loss.backward()
                        optimizer_da1.step()
                        optimizer_da1.zero_grad()

                        # 反转后
                        preds, ground_true, _ = model(X, z_batch) 
                        ground_true = ground_true.squeeze().long().to(device)
                        disc_loss_DA = Domain_Loss(preds, ground_true)
                        disc_loss_DA = disc_loss_DA
                        
                        disc_loss_DA.backward()
                        optimizer_da2.step()
                        optimizer_da2.zero_grad()

                    model.train()
                    model.state = 'adapt'
                    for step, X in enumerate(realdata_loader):
                            X = X.to(device)
                            z_targ = model.encode(X).detach()
                            # z_batch = z_fake_all[step * X.size(0):(step + 1) * X.size(0)].to(device)
                            z_targ_recon = model.decoder(z_targ)
                            recon_loss = F.mse_loss(X, z_targ_recon)
                            
                            source_frac = get_frac_from_sigmatrix(model_copy, z_batch, sigmatrix) # 从固定sig中获取frac
                            source_frac = source_frac.to(device)
                            frac = model.predictor(z_batch) # 预测frac
                            frac = F.relu(frac)
                            frac_pred = model.refraction(frac)
                            da3_loss = recon_loss
                            da3_loss.backward()
                            optimizer_da3.step()
                            optimizer_da3.zero_grad()

    else:
        model = model

    if best_model is not None:
        model.load_state_dict(best_model)

    # 最终预测输出
    model.eval()
    model.state = 'test'
    for _, X in enumerate(realdata_loader):
        X = X.to(device)
        x_recon_res, pred_res, z_res = model(X)
    
    return (
        x_recon_res.detach().cpu().numpy(),
        pred_res.detach().cpu().numpy(),
        z_res.detach().cpu().numpy(),
        model
    )

def adaptive_stage_domain9_noDF(x, sour_x_train, model_name=None, adaptive=True, mode='overall', steps=20, max_iter=200, device='cuda', sigmatrix = None,ep_warmup=50, save_name = None):

    Domain_Loss = nn.CrossEntropyLoss()
    
    if model_name is not None:
        model = torch.load(model_name + ".pth", weights_only=False, map_location=device)
        model = model.to(device)
    realdata_loader =DataLoader(realdatset(x), batch_size=len(x), shuffle=False) 
    
    indices = np.random.choice(sour_x_train.shape[0], x.shape[0], replace=False)
    ori_sour = sour_x_train[indices]
    ori_sour = torch.tensor(ori_sour, device=device).to(torch.float32)

    best_pcc = -1
    best_model = None
    ema_alpha = 0.9  # EMA decay
    ema_alpha = torch.tensor(ema_alpha, device=device)

    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    
    if adaptive and ep_warmup > 0:
        print(f"[Warmup] epochs = {ep_warmup}")
        model.train()
        # 只训练部分参数, 例如encoder和predictor
        for param in model.encoder.parameters():
            param.requires_grad = True
        for param in model.predictor.parameters():
            param.requires_grad = True
        for param in model.decoder.parameters():
            param.requires_grad = True
        model.ref_creator.requires_grad_(False)
        optimizer_warmup = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=1e-6
        )
        for ep in range(ep_warmup):
            warmup_loss1 = 0.0
            model.train()
            model.state = 'adapt'
            for _, X in enumerate(realdata_loader):
                X = X.to(device)
                # indices = np.random.choice(sour_x_train.shape[0], X.size(0), replace=False)
                # ori_sour = sour_x_train[indices]
                # ori_sour = torch.tensor(ori_sour, device=device).to(torch.float32)
                z_batch = model_copy.encode(ori_sour).detach()

                z_targ = model.encode(X).detach()
                z_targ_recon = model.decoder(z_targ)
                            
                source_frac = get_frac_from_sigmatrix(model_copy, z_batch, sigmatrix) # 从固定sig中获取frac
                source_frac = source_frac.to(device)
                frac = model.predictor(z_batch) # 预测frac
                frac = F.relu(frac)
                frac_pred = model.refraction(frac)
                
                _, ground_true, preds2 = model(X, z_batch)
                ground_true = ground_true.squeeze().long().to(device)
                disc_loss = Domain_Loss(preds2, ground_true)
                
                pred_loss_DA = compute_mmd(frac_pred, source_frac)*0.1
                da3_loss = pred_loss_DA+disc_loss
                da3_loss.backward()
                optimizer_warmup.step()
                optimizer_warmup.zero_grad()
                warmup_loss1+=da3_loss.item()
            print(f"[Warmup phas 1 Epoch {ep+1}/{ep_warmup}] loss: {warmup_loss1/len(realdata_loader):.4f}")


    if adaptive is True:

        if mode == 'overall5':
            model.train()
            # 解冻主网络参数以备后续训练
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.discriminator.parameters():
                param.requires_grad = True
            for param in model.predictor.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = True 

            model.state = 'train'
            model.ref_creator.requires_grad_(False)

            #设置解码优化器
            optimizer_da1 =  torch.optim.Adam(
                [{'params':model.encoder.parameters()},
                {'params':model.predictor.parameters()},
                {'params':model.discriminator.parameters()}],lr=1e-5)
            #设置编码优化器
            optimizer_da2 =  torch.optim.Adam(
                [{'params':model.encoder.parameters()},
                {'params':model.discriminator.parameters()}],lr=1e-5)
            optimizer_da3 =  torch.optim.Adam(
                [{'params':model.encoder.parameters()},
                {'params':model.decoder.parameters()},
                {'params':model.predictor.parameters()}],lr=1e-5
            )
            
            for iter in range(max_iter):
                print(f"Iter [{iter+1}/{max_iter}]")
                # Domain adaptation
                for _ in range(steps):
                    model.train()
                    model.state = 'adapt'
                    for step, X in enumerate(realdata_loader):
                        X = X.to(device)
                        z_targ = model.encode(X).detach()
                        z_batch = model_copy.encode(ori_sour).detach()

                        # 反转前
                        _, ground_true, preds2 = model(X, z_batch) 
                        ground_true = ground_true.squeeze().long().to(device)
                        disc_loss = Domain_Loss(preds2, ground_true)

                        source_frac = get_frac_from_sigmatrix(model_copy, z_batch, sigmatrix) # 从固定sig中获取frac
                        source_frac = source_frac.to(device)
                        frac = model.predictor(z_batch) # 预测frac
                        frac = F.relu(frac)
                        frac_pred = model.refraction(frac)

                        pred_loss =compute_mmd(frac_pred, source_frac)
                        loss = pred_loss + disc_loss
                        loss.backward()
                        optimizer_da1.step()
                        optimizer_da1.zero_grad()

                        # 反转后
                        preds, ground_true, _ = model(X, z_batch) 
                        ground_true = ground_true.squeeze().long().to(device)
                        disc_loss_DA = Domain_Loss(preds, ground_true)
                        disc_loss_DA = disc_loss_DA
                        
                        disc_loss_DA.backward()
                        optimizer_da2.step()
                        optimizer_da2.zero_grad()

                    model.train()
                    model.state = 'adapt'
                    for step, X in enumerate(realdata_loader):
                            X = X.to(device)
                            z_batch = model_copy.encode(ori_sour).detach()
                    
                            z_targ = model.encode(X).detach()
                            z_targ_recon = model.decoder(z_targ)
                            recon_loss = F.mse_loss(X, z_targ_recon)
                            
                            source_frac = get_frac_from_sigmatrix(model_copy, z_batch, sigmatrix) # 从固定sig中获取frac
                            source_frac = source_frac.to(device)
                            frac = model.predictor(z_batch) # 预测frac
                            frac = F.relu(frac)
                            frac_pred = model.refraction(frac)
                            da3_loss = recon_loss
                            da3_loss.backward()
                            optimizer_da3.step()
                            optimizer_da3.zero_grad()

    else:
        model = model

    if best_model is not None:
        model.load_state_dict(best_model)

    # 最终预测输出
    model.eval()
    model.state = 'test'
    for _, X in enumerate(realdata_loader):
        X = X.to(device)
        x_recon_res, pred_res, z_res = model(X)
    
    return (
        x_recon_res.detach().cpu().numpy(),
        pred_res.detach().cpu().numpy(),
        z_res.detach().cpu().numpy(),
        model
    )