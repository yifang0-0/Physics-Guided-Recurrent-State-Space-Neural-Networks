import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
from models.physical_augment.model_phy import MODEL_PHY

"""implementation of the Variational Auto Encoder Recurrent Neural Network (VAE-RNN) from 
https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf and partly from
https://arxiv.org/pdf/1710.05741.pdf using unimodal isotropic gaussian distributions for inference, prior, and 
generating models."""

def fun_RBF(x, label=2):
    if x.dim() == 1:
        x = x.unsqueeze(1)

    def rbf(x, c, sigma):
        return torch.exp(-((x - c) ** 2) / (sigma ** 2))

    # c1, sigma1 = -0.5, 0.1
    c2, sigma2 = -0.3, 0.2
    c3, sigma3 = 0.0, 0.2
    # c4, sigma4 = 0.1, 0.2
    # c5, sigma5 = 0.5, 0.1

    # y1 = rbf(x, c1, sigma1)
    y2 = -rbf(x, c2, sigma2)
    y3 = rbf(x, c3, sigma3)
    # y4 = -rbf(x, c4, sigma4)
    # y5 = rbf(x, c5, sigma5)


    if label == 2:
        return 0.2*(y2 + y3)
   
    else:
        raise ValueError("Invalid label for RBF.")

class MLP_U(nn.Module):
    def __init__(self, param, device,  sys_param={},  dataset="toy_lgssm", bias=False, ):
        super(MLP_U, self).__init__()


        self.y_dim = param.y_dim
        self.u_dim = param.u_dim
        self.h_dim = 20  # forced as requested
        self.z_dim = param.z_dim
        self.device = device
        self.x_phy_w = param.x_phy_w
        self.x_nn_w = param.x_nn_w
        self.n_layers = param.n_layers
        self.device = device
        self.mpnt_wt = param.mpnt_wt
        self.param = sys_param
        self.dataset = dataset
        self.epoch_counter=0
        self.phypen_wt = 10

        self.phy_aug = MODEL_PHY(self.dataset, self.param, self.device)

        
        # self.input_dyn_layer = nn.Linear(self.u_dim + self.z_dim, self.h_dim)
        # self.output_dyn_layer = nn.Linear(self.h_dim, self.z_dim)

        # self.input_meas_layer = nn.Linear(self.u_dim + self.z_dim, self.h_dim)
        # self.output_meas_layer = nn.Linear(self.h_dim, self.y_dim)
        
        self.input_dyn_layer = nn.Sequential(
            nn.Linear(self.u_dim + self.z_dim, self.h_dim),
            # CustomRBF(label=2),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2),
            # nn.ReLU()
        )
        self.output_dyn_layer = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2),
            # nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.z_dim),
            
        )

        # self.input_meas_layer = nn.Linear(self.u_dim + self.z_dim, self.h_dim)
        self.input_meas_layer = nn.Sequential(
            nn.Linear(self.u_dim + self.z_dim, self.h_dim),
            # CustomRBF(label=2)
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2)
            # nn.ReLU()
        )
        self.output_meas_layer = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            # CustomRBF(label=2)
            nn.ReLU(),
            nn.Linear(self.h_dim, self.y_dim)
        )
        
        
        
    def dyphy(self, u, x, u_norm_dict, y_norm_dict):

        x_phy_t = self.phy_aug.dynamic_model(u,x, u_norm_dict, y_norm_dict)  

        return x_phy_t

    
    def mephy(self, u, x):
        y_phy_t = self.phy_aug.measurement_model(u,x)               
        return y_phy_t
    


    def forward(self, u, y, u_norm_dict, y_norm_dict):
        #  batch size
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        # allocation
        loss = 0
        # initialization
        x = torch.zeros(batch_size, self.z_dim, seq_len, dtype=torch.float32, device=self.device)
        

        
        # for all time steps
        for t in range(seq_len):
            if t == 0:
                x_tm1 =  x[:,:,t].clone()
            else: 
                x_tm1 = x[:,:,t-1].clone()

