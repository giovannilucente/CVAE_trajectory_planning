import torch
import math
import torch.nn.functional as F
import math

class BetaAnnealer:
    def __init__(self, beta_start=0.0, beta_end=1.0, n_steps=10000, schedule='linear'):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.n_steps = n_steps
        self.schedule = schedule
        self.step_count = 0

    def step(self):
        """Compute beta value for current step and increment counter."""
        progress = min(self.step_count / self.n_steps, 1.0)
        
        if self.schedule == 'linear':
            beta = self.linear_schedule(progress)
        elif self.schedule == 'cosine':
            beta = self.cosine_schedule(progress)
        elif self.schedule == 'sigmoid':
            beta = self.sigmoid_schedule(progress)
        else:
            raise ValueError("Invalid schedule type. Choose from 'linear', 'cosine', 'sigmoid'.")
        
        self.step_count += 1
        return beta
    
    def linear_schedule(self, progress):
        return self.beta_start + progress * (self.beta_end - self.beta_start)
    
    def cosine_schedule(self, progress):
        cosine_progress = (1 - math.cos(progress * math.pi)) / 2
        return self.beta_start + cosine_progress * (self.beta_end - self.beta_start)
    
    def sigmoid_schedule(self, progress):
        x = (progress - 0.5) * 12
        sigmoid_progress = 1 / (1 + math.exp(-x))
        return self.beta_start + sigmoid_progress * (self.beta_end - self.beta_start)
        

def mse_kld(recon, target, kl_loss, beta=1.0):
    
    recon_loss = F.mse_loss(recon, target)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def l1_kld(recon, target, kl_loss, beta=1.0):
    
    recon_loss = F.l1_loss(recon, target)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def compute_ADE(pred, target, scale=10.0):
    pred = pred
    target = target

    pred = torch.cumsum(pred, dim=1) * scale
    target = torch.cumsum(target, dim=1) * scale

    error = torch.norm(pred - target, dim=-1)  # (B,T)
    return error.mean()


def compute_gray_iou(pred, target):
    pred_class = torch.zeros_like(pred, dtype=torch.long)
    target_class = torch.zeros_like(target, dtype=torch.long)

    pred_class[pred >= 0.33] = 1
    pred_class[pred >= 0.66] = 2

    target_class[target >= 0.33] = 1
    target_class[target >= 0.66] = 2

    ious = []

    for cls in range(3):
        pred_mask = pred_class == cls
        target_mask = target_class == cls

        intersection = (pred_mask & target_mask).sum(dim=(1, 2, 3))
        union = (pred_mask | target_mask).sum(dim=(1, 2, 3))

        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou)

    return torch.stack(ious, dim=1).mean().item()


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z):
        z = z.transpose(0, 1)
        A = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        A = A.div_(A.norm(p=2, dim=0, keepdim=True))
        x = (z @ A).unsqueeze(-1) * self.t
        err = (x.cos().mean(-3) - self.phi).square() + x.sin().mean(-3).square()
        return ((err @ self.weights) * z.size(-2)).mean()