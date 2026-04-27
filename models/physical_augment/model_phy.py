from models.physical_augment.kuka300 import kuka300
from models.physical_augment.randomRobo import randomRobo
import roboticstoolbox as rtb


import torch
import torch.nn as nn
import numpy as np

class MODEL_PHY():
    def __init__(self, phy_type, sysparam, device,  normalizer_dict_input=None, normalizer_dict_output=None):
        self.phy_type = phy_type
        self.param = self._move_to_device(sysparam, device)
        self.device = device

        self.normalizer_dict_input=normalizer_dict_input
        self.normalizer_dict_output=normalizer_dict_output
        
        if self.phy_type == 'industrobo':
            self.if_clip = self.param['if_clip']
            self.if_level2 = self.param['if_level2']
            self.if_level0 = self.param['if_level0']
            self.if_bias = self.param['if_bias']
            self.if_G = self.param['if_G']
            print("self.param['roboname']: ",self.param['roboname'])
            print("if_clip: ",self.if_clip)
            print("if_level2: ",self.if_level2)
            print("if_bias: ",self.if_bias)
            print("if_level0: ",self.if_level0)
            print("if_G: ",self.if_G)
            if self.param['roboname'] == "KUKA300":
                self.model = kuka300(robot_type="correct")
                self.dof = self.model.dof
            # elif self.param['roboname'] == "KUKA300noffset":
            #     self.model = kuka300(robot_type="no_offset")
            #     self.dof = self.model.dof
                
            elif self.param['roboname'] == "Puma560":
                # self.if_clip = False
                # self.if_G = False
                self.model = rtb.models.DH.Puma560()
                self.dof = 6
            elif self.param['roboname'] == "randomRobo":
                # self.if_clip = False
                # self.if_G = False
                self.model = randomRobo()
                self.dof = 6
                
        elif self.phy_type == 'toy_lgssm':
            # decide how to change or where to add the congifuration that what parts of the models are available (do I need seperated model for that or maybe, no)
            # self.model == toy_lgssm()
            # can be initialed by adding A,B,C,D matrix here
            pass
            
    def _move_to_device(self, obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(v, device) for v in obj)
        else:
            return obj
              
    def qdd_func(self, torque, q, qd, robot):
        # Handle both single timestep and sequence inputs
        if len(q.shape) == 1:
            # Single timestep case
            qd = qd
            qdd = robot.accel(q, qd, torque, gravity=[0,0,9.81])
            return qd, qdd
        else:
            # Sequence case: q shape is (seq_len, q_dim)
            seq_len = q.shape[0]
            qdd_list = []
            for t in range(seq_len):
                qdd_t = robot.accel(q[t], qd[t], torque[t], gravity=[0,0,9.81])
                qdd_list.append(qdd_t)
            return qd, np.stack(qdd_list)

    def rk4_step(self, qdd_func, q, dq, dt, torque, robot):
        if len(q.shape) == 1:
            # Single timestep case
            k_dq1, k_ddq1 = qdd_func(torque, q, dq, robot)
            k_dq2, k_ddq2 = qdd_func(torque, q + 0.5*dt*k_dq1, dq + 0.5*dt*k_ddq1, robot)
            k_dq3, k_ddq3 = qdd_func(torque, q + 0.5*dt*k_dq2, dq + 0.5*dt*k_ddq2, robot)
            k_dq4, k_ddq4 = qdd_func(torque, q + dt*k_dq3, dq + dt*k_ddq3, robot)

            new_ddq = (k_ddq1 + 2.0*k_ddq2 + 2.0*k_ddq3 + k_ddq4)/6.0*dt
            new_dq = (k_dq1 + 2.0*k_dq2 + 2.0*k_dq3 + k_dq4)/6.0*dt
            
            q_next = q + new_dq
            dq_next = dq + new_ddq
            return q_next, dq_next, k_ddq1
        else:
            # Sequence case: process all timesteps at once
            seq_len = q.shape[0]
            q_next_list = []
            dq_next_list = []
            new_ddq_list = []
            
            for t in range(seq_len):
                k_dq1, k_ddq1 = qdd_func(torque[t], q[t], dq[t], robot)
                k_dq2, k_ddq2 = qdd_func(torque[t], q[t] + 0.5*dt*k_dq1, dq[t] + 0.5*dt*k_ddq1, robot)
                k_dq3, k_ddq3 = qdd_func(torque[t], q[t] + 0.5*dt*k_dq2, dq[t] + 0.5*dt*k_ddq2, robot)
                k_dq4, k_ddq4 = qdd_func(torque[t], q[t] + dt*k_dq3, dq[t] + dt*k_ddq3, robot)

                new_ddq = (k_ddq1 + 2.0*k_ddq2 + 2.0*k_ddq3 + k_ddq4)/6.0*dt
                new_dq = (k_dq1 + 2.0*k_dq2 + 2.0*k_dq3 + k_dq4)/6.0*dt
                
                q_next = q[t] + new_dq
                dq_next = dq[t] + new_ddq
                
                q_next_list.append(q_next)
                dq_next_list.append(dq_next)
                new_ddq_list.append(new_ddq)
            
            return np.stack(q_next_list), np.stack(dq_next_list), np.stack(new_ddq_list)

    def dynamic_model(self, u, x_pre, normalizer_dict_input=None,normalizer_dict_output=None):
        is_3d = len(u.shape) == 3
        if is_3d:
            batch_size, seq_len, u_dim = u.shape
            # Reshape to 2D for processing
            u_reshaped = u.reshape(-1, u_dim)  # [batch_size * seq_len, u_dim]
            x_pre_reshaped = x_pre.reshape(-1, x_pre.shape[-1])  # [batch_size * seq_len, x_dim]
        else:
            batch_size = u.shape[0]
            u_reshaped = u
            x_pre_reshaped = x_pre

        
        # print("x_pre_reshaped.shape: ",x_pre_reshaped.shape)
        
        
        
        
        if self.phy_type == 'industrobo':
            # forward dynamics (start from 6 dim)
            x_dim = x_pre_reshaped.shape[1]
            dof = int(x_dim/2)

            q = x_pre_reshaped[:,0:dof].clone()  # Initial joint positions
            qd = x_pre_reshaped[:,dof:].clone() # Initial joint velocities

            q = q * normalizer_dict_output.scale + normalizer_dict_output.offset
            qd = qd * normalizer_dict_output.scale + normalizer_dict_output.offset

            u_p = u_reshaped.clone() * normalizer_dict_input.scale + normalizer_dict_input.offset

                
                
            
            torque = torch.zeros(u_p.size())
            
            dof_num = self.dof

            q_lim_max =[]
            q_lim_min = []

            for i in range(dof_num):

                # u_p +=  torch.tensor([-2,-2,-2,-2,-2,-2]).to(device='cuda')
                if self.if_G == True:
                    G = [212.76, 203.52, 192.75, 156, 156, 102.17]
                    torque[:, i] = u_p[:, i]*G[i]
                if self.if_bias == True:
                    # bias  =  torch.tensor([-2,-2,-2,-2,-2,-2]).to(device='cuda')[i]
                    device = torque.device  
                    bias = torch.tensor([4, 5, 5, 0.5, 2, 0.5], device=device, dtype=u_p.dtype)
                    # bias = torch.tensor([-2,-2,-2,-2,-2,-2], device=device, dtype=u_p.dtype)


                    torque[:, i] +=  bias[i]
                    
                
                    
            if self.if_clip == True: 
                q_lim_min = [-90*np.pi/180, -30*np.pi/180, -110*np.pi/180, -180*np.pi/180, -90*np.pi/180, -180*np.pi/180]      
                q_lim_max = [90*np.pi/180,  40*np.pi/180,  40*np.pi/180,  180*np.pi/180, 90*np.pi/180,  180*np.pi/180]
                qd_lim = [63.4, 61.7, 59.5 , 91.5,85.8,131.3]
            else:
                q_lim_min = [-500*np.pi/180, -500*np.pi/180, -500*np.pi/180, -500*np.pi/180, -500*np.pi/180, -500*np.pi/180]      
                q_lim_max = [500*np.pi/180,  500*np.pi/180,  500*np.pi/180,  500*np.pi/180, 500*np.pi/180,  500*np.pi/180]
                qd_lim = [360, 360, 360, 360,360,360]
                
            q_lim_min = torch.tensor(q_lim_min).to(device='cuda')
            q_lim_max = torch.tensor(q_lim_max).to(device='cuda')
            
            q_list = []
            qd_list = []
            qdd_list = []
            dt = self.param['dt']
            
            qd_lim_min = []
            qd_lim_max = []

            for i in qd_lim:
                qd_lim_min.append(-i/180*np.pi)
                qd_lim_max.append(i/180*np.pi)
            
            qd_lim_min = torch.tensor(qd_lim_min).to(device='cuda')
            qd_lim_max = torch.tensor(qd_lim_max).to(device='cuda')

            for i_batch in range(batch_size):
                if is_3d:
                    # Process entire sequence at once
                    q_np = q[i_batch*seq_len:(i_batch+1)*seq_len].clone().detach().cpu().numpy()
                    qd_np = qd[i_batch*seq_len:(i_batch+1)*seq_len].clone().detach().cpu().numpy()
                    torque_np = torque[i_batch*seq_len:(i_batch+1)*seq_len].clone().detach().cpu().numpy()
                    
                    q_i, qd_i, qdd_i = self.rk4_step(self.qdd_func, q_np, qd_np, dt, torque_np, self.model)
                    
                    # Convert back to tensors and apply limits
                    qd_ib = torch.tensor(qd_i).to(device='cuda')
                    q_ib = torch.tensor(q_i).to(device='cuda')
                    
                    qd_ib = torch.max(torch.min(qd_ib, qd_lim_max), qd_lim_min)
                    q_ib = torch.max(torch.min(q_ib, q_lim_max), q_lim_min)
                    
                    if self.if_level2 == True:
                        # qd_ib += qd[i_batch*seq_len:(i_batch+1)*seq_len] * torch.tensor([-0.05,0.1,-0.2,0.05,0.1,-0.1]).to(device='cuda')
                        qd_ib += qd[i_batch*seq_len:(i_batch+1)*seq_len] * torch.tensor([-0.02,0.06,-0.1,0.02,0.05,-0.05]).to(device='cuda')

                        # qd_ib += qd[i_batch*seq_len:(i_batch+1)*seq_len] * torch.tensor([-0.1,0.14,-0.4,0.1,0.2,-0.2]).to(device='cuda')

                    
                    q_list.append(q_ib)
                    qd_list.append(qd_ib)
                    qdd_list.append(torch.tensor(qdd_i).to(device='cuda'))
                else:
                    # Process single timestep
                    q_np = q[i_batch].clone().detach().cpu().numpy()
                    qd_np = qd[i_batch].clone().detach().cpu().numpy()
                    torque_np = torque[i_batch].clone().detach().cpu().numpy()

                    q_i, qd_i, qdd_i = self.rk4_step(self.qdd_func, q_np, qd_np, dt, torque_np, self.model)
                    
                    qd_ib = torch.tensor(qd_i).to(device='cuda')
                    q_ib = torch.tensor(q_i).to(device='cuda')
                    
                    qd_ib = torch.max(torch.min(qd_ib, qd_lim_max), qd_lim_min)
                    q_ib = torch.max(torch.min(q_ib, q_lim_max), q_lim_min)
                    
                    if self.if_level2 == True:
                        qd_ib += qd[i_batch] * torch.tensor([-0.05,0.1,-0.2,0.05,0.1,-0.1]).to(device='cuda')
                    
                    q_list.append(q_ib)
                    qd_list.append(qd_ib)
                    qdd_list.append(torch.tensor(qdd_i).to(device='cuda'))

            q_array = torch.stack(q_list)
            qd_array = torch.stack(qd_list)
            qdd_array = torch.stack(qdd_list)

            q_array = (q_array - normalizer_dict_output.offset) / normalizer_dict_output.scale 
            qd_array = (qd_array - normalizer_dict_output.offset) / normalizer_dict_output.scale 

            xt = torch.cat((q_array,qd_array), dim=1).to(dtype=torch.float32,device='cuda')
            # qdd = torch.cat((qdd_array), dim=1).to(dtype=torch.float32,device='cuda')
            
            self.qdd = torch.asarray(qdd_array).to(device='cuda')
            # Reshape back to 3D if input was 3D
            if is_3d:
                xt = xt.reshape(batch_size, seq_len, -1)
        # 0.0464    0.0003    0.0412    0.0586    0.0039    0.0146  
        elif self.phy_type == 'cascaded_tank':
            batch_size = u.shape[0]
            # u = u_reshaped.clone() * normalizer_dict_input.scale + normalizer_dict_input.offset
            u_reshaped = u.reshape(u.shape[0],-1)

            x_dim = x_pre_reshaped.shape[1]
            x1 = x_pre_reshaped[:,0:1].clone()  # Initial joint positions
            
            x2 = x_pre_reshaped[:,1:2].clone() # Initial joint velocities
            if normalizer_dict_output is not None:
                u_reshaped = u_reshaped.clone() * normalizer_dict_input.scale + normalizer_dict_input.offset
                x1 = x1.clone() * normalizer_dict_output.scale + normalizer_dict_output.offset
                x2 = x2.clone() * normalizer_dict_output.scale + normalizer_dict_output.offset
            # print("x1.shape",x1.shape, "x2.shape",x2.shape)
            eps = 1e-6  # to avoid sqrt of negatives
            Ts = self.param['dt']
            # print("!!!!!!!!!!!!Ts ", Ts)
            # Compute next x1
            # print("k1: ",self.param['k1'])
            # print("k2: ",self.param['k2'])
            # print("k3: ",self.param['k3'])
            # print("k4: ",self.param['k4'])
            # print("k5: ",self.param['k5'])
            x1_sqrt = torch.sqrt(torch.clamp(x1, min=0.0) + eps)
            x2_sqrt = torch.sqrt(torch.clamp(x2, min=0.0) + eps)
            # x1_sqrt = torch.sqrt(x1 + eps)
            
            x1_next = x1 + Ts * (
                -self.param['k1'] * x1_sqrt + self.param['k2'] * x1 + self.param['k3'] * u_reshaped
            )
            # x1_next = torch.minimum(x1_next, torch.tensor(self.param['x1Max'], device=x1.device))
            x1_next = torch.clamp(x1_next, max=self.param['x1Max'])
            x2_next  = x2.clone()
            # x2_sqrt = torch.sqrt(x2 + eps)
            # print("x2_sqrt", x2_sqrt)
            # for i in range(batch_size):
                # print("u[i]", u.reshape(x2.shape[0],-1)[i])
            addition =  (self.param['k1'] * x1_sqrt
                    - self.param['k2'] * x1
                    - self.param['k4'] * x2_sqrt
                    + self.param['k5'] * x2)
            x2_next = x2 + Ts * addition
            '''
            # for i in range(batch_size):

            if x1[i] >= self.param['x1Max']:
                addition =(
                    self.param['k1'] * x1_sqrt[i]
                    - self.param['k2'] * x1[i]
                    - self.param['k4'] * x2_sqrt[i]
                    + self.param['k5'] * x2[i]
                    + self.param['k5'] * u_reshaped[i]
                )
                # print("x2[i].shape:", x2[i].shape)
                # print("addition.shape:", addition.shape)
                x2_next[i] = x2[i] + Ts * addition
                # print("addition 1: ",addition)
                
            else:
                addition =  (self.param['k1'] * x1_sqrt[i]
                    - self.param['k2'] * x1[i]
                    - self.param['k4'] * x2_sqrt[i]
                    + self.param['k5'] * x2[i])
                x2_next[i] = x2[i] + Ts * addition
            '''
            x2_next_update = torch.clamp(x2_next, min=self.param['xMin'], max=self.param['x2Max'])
            if normalizer_dict_output is not None:
                x1_next = (x1_next - normalizer_dict_output.offset) / normalizer_dict_output.scale 
                x2_next_update = (x2_next_update - normalizer_dict_output.offset) / normalizer_dict_output.scale 
            xt = torch.cat((x1_next,x2_next_update), dim=1).to(dtype=torch.float32,device='cuda')
            
        elif self.phy_type == 'toy_lgssm' or self.phy_type == 'toy_lgssm_5_pre':
            batch_size = u.shape[0]
            # print("x_pre",x_pre)
            A_prt =torch.tensor(self.param['A_prt'], dtype=torch.float32,device=self.device)
            # print("A_prt: ",A_prt)
            
            A_prt = A_prt.expand(batch_size, -1, -1)
            

            B_prt =torch.tensor(self.param['B_prt'], dtype=torch.float32,device=self.device)
            B_prt = B_prt.expand(batch_size, -1, -1)
            xt =  torch.matmul(A_prt,x_pre.unsqueeze(-1)).squeeze(-1)+torch.matmul(B_prt,u.unsqueeze(-1)).squeeze(-1)
            # print("A_prt,x_pre.unsqueeze(-1)).squeeze(-1)", torch.matmul(A_prt,x_pre.unsqueeze(-1)).squeeze(-1))
            # print("xt: ",xt)
        return xt
    
    def measurement_model(self, ut, xt ):
        # Check if input is 3D tensor (batch_size, seq_len, x_dim)
        is_3d = len(xt.shape) == 3
        if is_3d:
            batch_size, seq_len, x_dim = xt.shape
            xt_reshaped = xt.reshape(-1, x_dim)  # [batch_size * seq_len, x_dim]
        else:
            xt_reshaped = xt
            batch_size = xt.shape[0]
            x_dim = xt.shape[1]
            
        if self.phy_type == 'industrobo':
            if self.if_level0==True:
                x_dim = xt.shape[1]
                y = xt[:,0:int(x_dim/2)].clone()
                return y
            else:
                x_dim = xt.shape[1]
                yt = xt[:,0:int(x_dim/2)].clone()
                return yt-yt
        
        elif self.phy_type == 'cascaded_tank':
            # print("xt_reshaped.shape: ",xt_reshaped.shape)
            # print("xt_reshaped: ",xt_reshaped)
            
            # offset = torch.tensor([[self.param['offset']]], dtype=torch.float32,device=self.device)
            # print("offset: ",offset)
            y = xt_reshaped[:,1].clone()
            # y = xt_reshaped[:,1].clone()+offset
            
            # print("y: ",y)
            # print("y.shape: ",y.shape)
            
            return y.reshape(batch_size, seq_len, -1) if is_3d else y.reshape(batch_size, -1)
        
        elif self.phy_type == 'toy_lgssm' or self.phy_type == 'toy_lgssm_5_pre':
            
            C_prt = torch.tensor(self.param['C_prt'], dtype=torch.float32,device=self.device)
            
            # Reshape back to 3D if input was 3D
            if is_3d:
                C_prt = C_prt.expand(batch_size, -1, -1)
                y = torch.matmul(C_prt, xt_reshaped.T).T  # [batch_size, y_dim]
                y = y.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, y_dim]
            else:
                batch_size = ut.shape[0]
                C_prt = torch.tensor( self.param['C_prt'], dtype=torch.float32,device=self.device)
                C_prt = C_prt.expand(batch_size, -1, -1)
                # print(C.shape,y.shape,ut.shape)
                y = torch.matmul(C_prt,xt_reshaped.unsqueeze(-1)).squeeze(-1)


            return y

    def simulation(self, ut, xt, normalizer_dict_input=None,normalizer_dict_output=None):
        u = u.clone() * normalizer_dict_input.scale + normalizer_dict_input.offset
        if self.phy_type == 'cascaded_tank':
            batch_size = u.shape[0]
            x_dim = xt.shape[1]
            x1 = xt[:,0:1].clone()  # Initial joint positions
            x2 = xt[:,1:2].clone() # Initial joint velocities

            eps = 1e-6  # to avoid sqrt of negatives
            Ts = self.param['dt']
            x1_sqrt = torch.sqrt(torch.clamp(x1, min=eps))
            x1_next = x1 + Ts * (
                -self.param['k1'] * x1_sqrt + self.param['k2'] * x1 + self.param['k3'] * u.reshape(x1.shape[0],-1)
            )
            x1_next = torch.minimum(x1_next, torch.tensor(self.param['x1Max'], device=x1.device))
            x2_next  = x2.clone()
            x2_sqrt = torch.sqrt(torch.clamp(x2.clone(), min=eps))
            for i in range(batch_size):
                if x1[i] >= self.param['x1Max']:
                    addition =(
                        self.param['k1'] * x1_sqrt[i]
                        - self.param['k2'] * x1[i]
                        - self.param['k4'] * x2_sqrt[i]
                        + self.param['k5'] * x2[i]
                        + self.param['k5'] * u.reshape(x2.shape[0],-1)[i]
                    )
                    x2_next[i] = x2[i] + Ts * addition
                else:
                    addition =  (self.param['k1'] * x1_sqrt[i]
                        - self.param['k2'] * x1[i]
                        - self.param['k4'] * x2_sqrt[i]
                        + self.param['k5'] * x2[i])
                    x2_next[i] = x2[i] + Ts * addition
            x2_next_update = torch.minimum(x2_next, torch.tensor(self.param['x2Max'], device=x2.device))
            x2_next_update = torch.maximum(x2_next_update, torch.tensor(self.param['xMin'], device=x2.device))
            xt = torch.cat((x1_next,x2_next_update), dim=1).to(dtype=torch.float32,device='cuda')
            offset = torch.tensor([[self.param['offset']]], dtype=torch.float32,device=self.device)
            # print("offset: ",offset)
            y = xt[:,1].clone()+offset
            # print("y: ",y)
            # print("y.shape: ",y.shape)
            
            return y.reshape(batch_size, seq_len, -1) if is_3d else y.reshape(batch_size, -1)
        
    def robo_phy_penalty(self, xt, u, normalizer_dict_input=None,normalizer_dict_output=None):
        # Check if input is 3D tensor (batch_size, seq_len, x_dim)
        # forward dynamics (start from 6 dim)
        batch_size = u.shape[0]
        x_dim = xt.shape[1]
        dof = int(x_dim/2)
        
        q = xt[:,0:dof].clone()  # Initial joint positions
        qd = xt[:,dof:].clone() # Initial joint velocities
        # print("q.shape: ",q.shape)
        qdd = self.qdd

        q = q * normalizer_dict_output.scale + normalizer_dict_output.offset
        qd = qd * normalizer_dict_output.scale + normalizer_dict_output.offset

        u_p = u.clone() * normalizer_dict_input.scale + normalizer_dict_input.offset

        torque = torch.zeros(u_p.size())
        
        dof_num = self.dof


        for i in range(dof_num):
            torque[:, i] = u_p[:, i]
        torque = torque.to(device='cuda')
        rne_tau_list = []
        # print("batch_size: ",batch_size)
        for i_batch in range(batch_size):
            
            q_np = q[i_batch].clone().detach().cpu().numpy()
            qd_np = qd[i_batch].clone().detach().cpu().numpy()
            qdd_np = qdd[i_batch].clone().detach().cpu().numpy()
            rne_tau = self.model.rne(q_np, qd_np, qdd_np, gravity=[0,0,9.81])
            rne_tau_list.append(torch.tensor(rne_tau).to(device='cuda'))
        rne_tau_list = torch.stack(rne_tau_list).to(device='cuda')
        # print("rne_tau_list.shape: ",rne_tau_list.shape)
        # print("torque.shape: ",torque.shape)
        for i in range(dof_num):
            if self.if_G == True:
                G = [212.76, 203.52, 192.75, 156, 156, 102.17]
                rne_tau_list[:, i] = rne_tau_list[:, i]/G[i]
        penalty = torch.sum((rne_tau_list-torque)**2)
        
        # print("torque[0]", torque[0]," rne_tau_list[0]",rne_tau_list[0])
        # print("penalty.shape: ",penalty.shape)
        # print("penalty: ",penalty)
        return penalty

        