import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class KinematicDynamics(nn.Module):
    def __init__(self, dt=0.1, v_max=15.0, k_max=0.2):
        super().__init__()
        self.dt = dt
        self.v_max = v_max
        self.k_max = k_max

    def forward(self, u):
        raw_v, raw_k = u.unbind(-1)

        v = self.v_max * torch.sigmoid(raw_v)
        k = self.k_max * torch.tanh(raw_k)

        B, T = v.shape

        x = torch.zeros(B, device=u.device)
        y = torch.zeros(B, device=u.device)
        yaw = torch.zeros(B, device=u.device)

        traj = []

        for t in range(T):
            x = x + v[:, t] * torch.cos(yaw) * self.dt
            y = y + v[:, t] * torch.sin(yaw) * self.dt

            yaw = yaw + v[:, t] * k[:, t] * self.dt

            traj.append(torch.stack([x, y], dim=-1))

        traj = torch.stack(traj, dim=1)  # [B, T, 2]
        dxy = torch.diff(torch.cat([torch.zeros(B, 1, 2, device=u.device), traj], dim=1), dim=1)

        return dxy
    

def kl_gaussian(mu_q, logvar_q, mu_p, logvar_p):
    kl = 0.5 * (
        logvar_p - logvar_q +
        (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p)
        - 1
    )

    # Sum over latent dimension (dim=1)
    kl = kl.sum(dim=1)
    
    return kl.mean()     


class ResNetTokenizer(nn.Module):
    def __init__(self, img_size=128, hidden_dim=64, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feat_dim = 512
        self.hidden_dim = hidden_dim
        self.tokens_h = img_size // 32
        self.tokens_w = img_size // 32
        self.num_tokens = self.tokens_h * self.tokens_w  # e.g., 4 x 4 = 16

        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim))
        self.proj = nn.Linear(self.feat_dim, hidden_dim)

    def forward(self, x):
        B = x.size(0)

        feats = self.backbone(x)  # (B, 512, H', W')
        B, C, Hf, Wf = feats.shape  
        tokens = feats.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)
        tokens = self.proj(tokens) + self.pos_embed

        return tokens


class VectorTokenizer(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, seq_len=8):
        super().__init__()
        self.seq_len=seq_len
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU()
        )
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, hidden_dim))
        
    def forward(self, x):
        x = self.mlp(x)                             # (B, L, hidden_dim)
        x = x + self.pos_embed
        return x                                    # (B, L, hidden_dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q, k, v):
        q_in = q
        q = self.norm_q(q)
        k = self.norm_kv(k)
        v = self.norm_kv(v)
        h, _ = self.attn(q, k, v)
        x = q_in + h
        h = self.mlp(self.norm2(x))
        x = x + h

        return x
    

class Head(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (B, seq_len, hidden_dim)
        x_pooled = x.mean(dim=1)  # (B, hidden_dim)
        mu = self.mu(x_pooled)          # (B, latent_dim)
        logvar = self.logvar(x_pooled)  # (B, latent_dim)
        logvar = torch.clamp(logvar, -10, 10)
        return mu, logvar


class LatentProjection(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, seq_len=8):
        super().__init__()
        self.proj = nn.Linear(input_dim, seq_len * hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, hidden_dim))

    def forward(self, x):
        x = self.proj(x)
        x = x.view(x.size(0), -1, self.pos_embed.size(-1))
        x = x + self.pos_embed
        return x


class EncoderLevel(nn.Module):
    def __init__(self, hidden_dim_in, hidden_dim_out):
        super().__init__()
        self.attn_n = TransformerBlock(hidden_dim_in)
        self.attn_c = TransformerBlock(hidden_dim_in) 
        self.proj = nn.Linear(hidden_dim_in, hidden_dim_out)

    def forward(self, x, c, n):
        h = self.attn_n(q=x, k=n, v=n)
        h = self.attn_c(q=h, k=c, v=c)
        h = self.proj(h)
        return h


class FinalHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, seq_len, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.01)
        self.attn_pool = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                               num_heads=num_heads, 
                                               batch_first=True)
        self.control_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, L, H = x.shape
        query = self.query.expand(B, -1, -1)                       
        attn_out, _ = self.attn_pool(query=query, key=x, value=x)  
        h = attn_out.squeeze(1)                               
        control = self.control_layer(h)                                  
        return control


class DecoderLevel(nn.Module):
    def __init__(self, hidden_dim_in, hidden_dim_out, seq_len, output_dim=3):
        super().__init__()
        self.attn_n = TransformerBlock(hidden_dim_in)
        self.attn_c = TransformerBlock(hidden_dim_in) 
        self.proj = nn.Linear(hidden_dim_in, hidden_dim_out)
        self.final_head = FinalHead(hidden_dim_out, output_dim, seq_len)

    def forward(self, x, c, n):
        h = self.attn_n(q=x, k=n, v=n)
        h = self.attn_c(q=h, k=c, v=c)
        h = self.proj(h)
        control = self.final_head(h)
        return control


class CVAE(nn.Module):
    def __init__(self, hidden_dim=32, input_dim=3, input_len=60, num_frames=3,  img_size=256, latent_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.patch_size = 16
        self.seq_len    = input_len
        
        # Conditioning image tokenizer
        self.cond_c = ResNetTokenizer(img_size=img_size, hidden_dim=hidden_dim)

        # Conditioning navigation tokenizer
        self.cond_n = VectorTokenizer(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=self.seq_len)

        # Input tokenizer
        self.input_tokenizer = VectorTokenizer(input_dim=input_dim, hidden_dim=hidden_dim, seq_len=self.seq_len)

        # Encoder
        self.enc = EncoderLevel(hidden_dim, hidden_dim)      

        # Posterior head
        self.pos = Head(hidden_dim, latent_dim)    

        # Latent Projection
        self.lat = LatentProjection(input_dim=latent_dim, hidden_dim=hidden_dim, seq_len=self.seq_len)

        # Decoder    
        self.dec =  DecoderLevel(hidden_dim, hidden_dim, output_dim=input_dim, seq_len=self.seq_len)    

        # Dynamics
        self.dynamics = KinematicDynamics(v_max=15.0, k_max=0.2)

    def reparameterize(self, mu, logvar, seed=None):
        std = torch.exp(0.5 * logvar)
        if seed is None: 
            eps = torch.randn_like(std) 
        else: 
            g = torch.Generator(device=mu.device) 
            g.manual_seed(seed) 
            eps = torch.randn(std.shape, generator=g, device=mu.device, dtype=std.dtype)
        return mu + eps * std

    def forward(self, x, c, n):
        c = self.cond_c(c)
        n = self.cond_n(n)
        x = self.input_tokenizer(x)

        h = self.enc(x, c, n)
        
        mu, logvar = self.pos(h)
        z = self.reparameterize(mu, logvar)
        q = self.lat(z)
        control = self.dec(q, c, n)
        out = self.dynamics(control)
        
        kl = kl_gaussian(mu, logvar, torch.zeros_like(mu), torch.zeros_like(logvar))
        
        return out, kl

    #@torch.no_grad()
    def generate(self, c, n, batch=1, device="cpu", seed=None):
        c = c.to(device).expand(batch, -1, -1, -1)
        n = n.to(device).expand(batch, -1, -1)
        
        c = self.cond_c(c)
        n = self.cond_n(n)
        mu = torch.zeros(batch, self.latent_dim).to(device)
        logvar = torch.zeros(batch, self.latent_dim).to(device)
        z = self.reparameterize(mu, logvar, seed=seed) 
        q = self.lat(z)
        control = self.dec(q, c, n)
        
        out = self.dynamics(control)

        return out


if __name__ == "__main__":
    import os
    import sys
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from dataset import Dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 4
    img_size = 256
    traj_len = 10
    history = 3
    model = CVAE(hidden_dim=32, input_dim=2, input_len=traj_len, img_size=img_size, latent_dim=64).to(device)

    base_dir = "../../../../mnt/nfs-share/AI_Datasets/_unzipped/world_model"
    val_data_root = os.path.join(base_dir, "validation")
    max_samples = 10
    
    image_transform = transforms.Compose([
        transforms.Resize(256),      # shortest side = 256
        transforms.ToTensor()
    ])

    dataset = Dataset(
        data_root=val_data_root,
        image_transform=image_transform,
        history=history,
        max_points=traj_len,
        max_samples=max_samples,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,  num_workers=4, pin_memory=True)
    
    sample = next(iter(dataloader))

    model.eval()
    with torch.no_grad():
        trajectory = sample["trajectories"][:,0].to(device)
        frames = sample["frames"][:,:3].to(device)
        reference_path = sample["reference_paths"][:,0].to(device)

        print("\ntrajectory shape:", trajectory.shape)
        print("reference_path shape:", reference_path.shape)
        print("frames shape:", frames.shape)

        output, kl = model(trajectory, frames, reference_path)
        output = model.generate(frames, reference_path, batch=batch_size, device=device)

        print("\nreconstructed output:", output.shape)
        print("generated output:", output.shape)