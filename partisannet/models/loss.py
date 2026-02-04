import torch
import torch.nn as nn
import math


def disp_loss(x, tau = 0.5):
    z = torch.flatten(x, 1)
    dist = nn.functional.pdist(z, p=2).pow(2) / z.shape[1]
    dist = torch.concat([dist, dist, torch.zeros(z.shape[0]).to(dist.device)]) 
    loss = torch.logsumexp(-dist/tau, dim=0) - math.log(dist.numel())
    return loss

class ContrastiveCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(ContrastiveCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        # Learnable centers
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, x, labels):
        dist_mat = torch.cdist(x, self.centers, p=2).pow(2)
        pos_dist = dist_mat.gather(1, labels.unsqueeze(1))

        mask = torch.ones_like(dist_mat).scatter_(1, labels.unsqueeze(1), 0)
        neg_dist = dist_mat * mask
        sum_neg_dist = neg_dist.sum(dim=1, keepdim=True)
        
        loss = 0.5 * pos_dist / (sum_neg_dist + 1e-6)
        return loss.mean()


if __name__ == "__main__":
    # Example usage
    num_classes = 2
    feat_dim = 2
    batch_size = 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ccl_loss = ContrastiveCenterLoss(num_classes=num_classes, feat_dim=feat_dim, device=device)

    with torch.no_grad():
        # Center 0 at (-1, -1) | Center 1 at (1, 1)
        ccl_loss.centers.copy_(torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))

    x = torch.tensor([[-1.1, -1.1], 
                  [-0.9, -0.9]], dtype=torch.float32, device=device)
    
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    loss = ccl_loss(x, labels)

    print(f"Centers:\n{ccl_loss.centers.data}")
    print(f"Input Embeddings:\n{x}")
    print(f"Labels: {labels.tolist()}")
    print(f"\nCalculated Loss: {loss.item():.6f}")