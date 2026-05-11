import torch
import torch.nn as nn

class PhysicsInformedEngramLoss(nn.Module):
    def __init__(self, lambda_physics=0.1):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_weather, true_weather, engram_gates):
        data_loss = self.mse_loss(pred_weather, true_weather)
        P_pred = pred_weather[:, 0]
        rho_pred = pred_weather[:, 1]
        T_pred = pred_weather[:, 2]
        
        R_constant = 287.05
        physics_residual = P_pred - (rho_pred * R_constant * T_pred)
        physics_loss = torch.mean(physics_residual ** 2)


        conflict_loss = torch.mean(torch.sum(engram_gates, dim=-1) ** 2)
        total_loss = data_loss + (self.lambda_physics * physics_loss) + (0.01 * conflict_loss)
        
        return total_loss, data_loss, physics_loss
