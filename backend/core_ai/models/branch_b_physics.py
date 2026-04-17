import torch
import torch.nn as nn
import torchvision.models as models

class PhysicsExpert(nn.Module):
    def __init__(self, embed_dim=256):
        super(PhysicsExpert, self).__init__()
        # ResNet18 for processing High-Fidelity RAFT Optical Flow
        self.flow_cnn = models.resnet18(pretrained=True)
        # 2-channel optical flow (u, v) ke liye layer change ki hai
        self.flow_cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.flow_cnn.fc = nn.Linear(self.flow_cnn.fc.in_features, embed_dim)

    def forward(self, optical_flow):
        return self.flow_cnn(optical_flow)

# 👑 EXPLICIT PINN NOVELTY: Divergence & Acceleration Constraints
def calculate_physics_penalty(optical_flow, alpha=1.0, beta=1.0):
    """
    Mathematical Implementation of Physics Constraints on Optical Flow
    optical_flow shape: (batch, 2, H, W)
    """
    u = optical_flow[:, 0, :, :] 
    v = optical_flow[:, 1, :, :] 
    
    # ==========================================
    # 1. Continuity Constraint (Divergence-Free Flow ∇·F ≈ 0)
    # ==========================================
    du_dx = u[:, :, 1:] - u[:, :, :-1]  
    dv_dy = v[:, 1:, :] - v[:, :-1, :]  
    
    # Shape matching and calculating L2 Norm of Divergence
    divergence = torch.abs(du_dx[:, :-1, :] + dv_dy[:, :, :-1])
    loss_div = torch.mean(divergence ** 2) 
    
    # ==========================================
    # 2. Kinematic Smoothness (Acceleration Consistency d²x/dt²)
    # ==========================================
    # 2nd-order derivatives (Laplacian Approximation)
    d2u_dx2 = u[:, :, 2:] - 2*u[:, :, 1:-1] + u[:, :, :-2]
    d2u_dy2 = u[:, 2:, :] - 2*u[:, 1:-1, :] + u[:, :-2, :]
    
    d2v_dx2 = v[:, :, 2:] - 2*v[:, :, 1:-1] + v[:, :, :-2]
    d2v_dy2 = v[:, 2:, :] - 2*v[:, 1:-1, :] + v[:, :-2, :]
    
    # Matching dimensions
    laplacian_u = d2u_dx2[:, :-2, :] + d2u_dy2[:, :, :-2]
    laplacian_v = d2v_dx2[:, :-2, :] + d2v_dy2[:, :, :-2]
    
    loss_smooth = torch.mean(laplacian_u ** 2) + torch.mean(laplacian_v ** 2)
    
    # Final Explicit Physics Penalty
    total_pinn_loss = (alpha * loss_div) + (beta * loss_smooth)
    
    return total_pinn_loss