import torch
import torch.nn as nn
from models.physical_augment.model_phy import MODEL_PHY
"""implementation of the Variational Auto Encoder Recurrent Neural Network (VAE-RNN) from 
https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf and partly from
https://arxiv.org/pdf/1710.05741.pdf using unimodal isotropic gaussian distributions for inference, prior, and 
generating models."""


class AE_RNN_U(nn.Module):
    def __init__(self, param, device,  sys_param={},  dataset="toy_lgssm", bias=False, ):
        super(AE_RNN_U, self).__init__()

        self.y_dim = param.y_dim
        self.u_dim = param.u_dim
        self.h_dim = param.h_dim
        self.z_dim = param.z_dim
        self.x_phy_w = param.x_phy_w
        self.x_nn_w = param.x_nn_w
        self.n_layers = param.n_layers
        self.device = device
        self.mpnt_wt = param.mpnt_wt
        self.param = sys_param
        self.dataset = dataset
        self.epoch_counter=0
        self.phypen_wt = 10
        self.nx = 1


        self.phy_aug = MODEL_PHY(self.dataset, self.param, self.device)


        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_u = nn.Sequential(
            nn.Linear(self.u_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        self.phi_x = nn.Sequential(
            nn.Linear(self.u_dim + self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        self.x_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
        )

        self.dynn = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        self.menn = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.y_dim),
        )
      
        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.h_dim, self.h_dim, self.n_layers, bias)

      
                  
    def dyphy(self, u, x, u_norm_dict, y_norm_dict):
        # with torch.no_grad():
        x_phy_t = self.phy_aug.dynamic_model(u,x, u_norm_dict, y_norm_dict)  
        return x_phy_t
    
    def mephy(self, u, x):
        # with torch.no_grad():
        y_phy_t = self.phy_aug.measurement_model(u,x)               
        return y_phy_t
    
    def phy_penalty(self, x, u, u_norm_dict, y_norm_dict):
        if self.phy_aug.model_type == "KUKA":
            loss = self.phy_aug.robo_phy_penalty_linear(x,u, u_norm_dict, y_norm_dict)
        else:
            loss = 1
        return loss
    def forward(self, u, y, u_norm_dict, y_norm_dict):
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        loss = 0
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, dtype=torch.float32, device=self.device)
        x = torch.zeros(batch_size, self.z_dim, seq_len, dtype=torch.float32, device=self.device)

        u_padded = torch.cat([
            torch.zeros(u.shape[0], u.shape[1], self.nx - 1, device=u.device),
            u
        ], dim=2)

        for t in range(seq_len):
            x_tm1 = x[:, :, t].clone() if t == 0 else x[:, :, t - 1].clone()

            u_lagged = u_padded[:, :, t: t + self.nx]
            u_lagged = u_lagged.transpose(1, 2).reshape(u.shape[0], -1)
            phi_u_t = self.phi_u(u_lagged)
            
            if self.mpnt_wt>100:
                # pure physical
                
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                x_t = x_mean_phy

            elif  self.mpnt_wt>=10:
                #physics augmented
                
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)                
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1, u_norm_dict, y_norm_dict)
                x_t = x_mean_nn + x_mean_phy
              
            elif self.mpnt_wt<=0:
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_t = x_mean_nn
            
            x[:, :, t] = x_t
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)

            if self.mpnt_wt > 100:
                y_hat_t = self.mephy(u[:, :, t], x_t)
                loss += torch.sum((y_hat_t - y[:, :, t]) ** 2)

            elif self.mpnt_wt >= 10:
                #physics augmented
                phi_x_t = self.phi_x(torch.cat([u[:,:,t], x_t], 1))
                y_hat_nn = self.menn(phi_x_t)
                
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_hat_t =  y_hat_nn +  y_hat_phy
                loss += (torch.sum((y_hat_t-y[:, :, t]) ** 2))
            elif self.mpnt_wt<=-10:
                #physics guided
                phi_x_t = self.phi_x(torch.cat([u[:,:,t], x_t], 1))
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                if y_hat_phy.dtype != self.y_phi_phy[0].weight.dtype:
                    y_hat_phy_f32 = y_hat_phy.to(self.y_phi_phy[0].weight.dtype)
                    y_phy_phi = self.y_phi_phy(y_hat_phy_f32)
                else:
                    y_phy_phi = self.y_phi_phy(y_hat_phy)
                    
                y_hat = self.menn_phy(torch.cat([phi_x_t,y_phy_phi],1))
                loss += torch.sum((y_hat-y[:, :, t]) ** 2)

            elif self.mpnt_wt<=0:
                #pure nn
                phi_x_t = self.phi_x(torch.cat([u[:,:,t], x_t], 1))
                # y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_hat_nn = self.menn(phi_x_t)
                y_hat_t = y_hat_nn
                loss += torch.sum((y_hat_t-y[:, :, t]) ** 2)
                
            else:       
                pass        
        
        self.epoch_counter += 1        
        return loss

    def generate(self, u, u_norm_dict, y_norm_dict):
        batch_size = u.shape[0]
        seq_len = u.shape[-1]
        y_hat = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        y_hat_sigma = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        x = torch.zeros(batch_size, self.z_dim, seq_len, device=self.device)
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        u_padded = torch.cat([
            torch.zeros(u.shape[0], u.shape[1], self.nx - 1, device=u.device),
            u
        ], dim=2)

        for t in range(seq_len):
            x_tm1 = torch.zeros(batch_size, self.z_dim, device=self.device) if t == 0 else x[:, :, t - 1]

            u_lagged = u_padded[:, :, t: t + self.nx]
            u_lagged = u_lagged.transpose(1, 2).reshape(u.shape[0], -1)
            phi_u_t = self.phi_u(u_lagged)

            if self.mpnt_wt > 100:
                x_t = self.dyphy(u[:, :, t], x_tm1, u_norm_dict, y_norm_dict)
            elif self.mpnt_wt >= 10:
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_mean_phy = self.dyphy(u[:, :, t], x_tm1, u_norm_dict, y_norm_dict)
                x_t = x_mean_nn + x_mean_phy
            elif self.mpnt_wt <= 0:
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_t = self.x_mean(dynn_phi)

            x[:, :, t] = x_t
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)

            if self.mpnt_wt > 100:
                y_hat[:, :, t] = self.mephy(u[:, :, t], x_t)
            elif self.mpnt_wt >= 10:
                phi_x_t = self.phi_x(torch.cat([u[:, :, t], x_t], 1))
                y_hat[:, :, t] = self.menn(phi_x_t) + self.mephy(u[:, :, t], x_t)
            elif self.mpnt_wt <= 0:
                phi_x = self.phi_x(torch.cat([u[:, :, t], x_t], 1))
                y_hat[:, :, t] = self.menn(phi_x)

        return y_hat, y_hat, y_hat_sigma, x

