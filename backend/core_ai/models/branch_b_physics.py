import torch
import torch.nn as nn
import torchvision.models as models

class PhysicsExpert(nn.Module):
    def __init__(self, embed_dim=256):
        super(PhysicsExpert, self).__init__()
        # ResNet18 for processing Optical Flow
        self.flow_cnn = models.resnet18(pretrained=True)
        # 2-channel optical flow (u, v) ke liye layer change ki hai
        self.flow_cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.flow_cnn.fc = nn.Linear(self.flow_cnn.fc.in_features, embed_dim)

    def forward(self, optical_flow):
        return self.flow_cnn(optical_flow)

# 👑 PINN NOVELTY: Motion Hallucination Check
def calculate_physics_penalty(optical_flow):
    u = optical_flow[:, 0, :, :] 
    v = optical_flow[:, 1, :, :] 
    
    du_dx = u[:, :, 1:] - u[:, :, :-1]  
    dv_dy = v[:, 1:, :] - v[:, :-1, :]  
    
    divergence = torch.abs(du_dx[:, :-1, :] + dv_dy[:, :, :-1])
    return torch.mean(divergence)