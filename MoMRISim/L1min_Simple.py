import torch
from torch.autograd import Variable
import ptwt, pywt
import numpy as np

from MoMRISim.util.helpers_math import complex_mul, fft2c_ndim
from MoMRISim.motion_simulation.motion_forward_backward_models import motion_corruption_NUFFT


class L1minModuleBase():

    def __init__(
            self,
            smaps3D,
            binary_background_mask,
            masked_corrupted_kspace3D,
            mask3D,
            gpu,
            L1min_lr,
            L1min_lambda,
            L1min_num_steps,
            traj=None,
            pred_motion_params=None,
            L1min_optimizer_type="SGD",
            Ns = 52,
            args=None,
            ) -> None:
        
        self.smaps3D = smaps3D
        self.binary_background_mask = binary_background_mask
        self.masked_corrupted_kspace3D = masked_corrupted_kspace3D
        self.mask3D = mask3D
        self.traj = traj
        self.pred_motion_params = pred_motion_params
        self.gpu = gpu
        self.L1min_lr = L1min_lr
        self.L1min_optimizer_type = L1min_optimizer_type
        self.L1min_lambda = L1min_lambda
        self.L1min_num_steps = L1min_num_steps
        self.args = args
        self.Ns = Ns
        self.final_result_dict = {}

    def run_L1min(self):
        ###############
        # Init Reconstruction Volume
        mse = torch.nn.MSELoss()
        recon = Variable(torch.zeros(self.masked_corrupted_kspace3D.shape[1:])).cuda(self.gpu)
        recon.data.uniform_(0,1)
        recon.requires_grad = True
        # Define the optimizer:
        optimizer = self.init_optimizer(recon, self.L1min_optimizer_type, self.L1min_lr)
        if self.pred_motion_params is None:
            self.pred_motion_params = torch.zeros(self.Ns, 6).cuda(self.gpu)
    
        ###############
        # Run L1-min
        print(f"Starting L1-minimization with {self.L1min_num_steps} steps, lr {self.L1min_lr:.1e}, lambda {self.L1min_lambda:.1e} and optimizer {self.L1min_optimizer_type}.")
        
        for iteration in range(self.L1min_num_steps):            
            optimizer.zero_grad()
            
            # Step 1: Apply forward model
            recon_coil = complex_mul(recon.unsqueeze(0), self.smaps3D)
            recon_kspace3d_coil = fft2c_ndim(recon_coil, 3)
            recon_kspace3d_coil = motion_corruption_NUFFT(recon_kspace3d_coil, recon_coil, self.pred_motion_params, self.traj, 
                                                        weight_rot=True, args=self.args, max_coil_size=2)
            
            # Step 2: Calculating the loss and backward
            # a. take wavelet of reconstruction
            coefficient = ptwt.wavedec3(recon, pywt.Wavelet("haar"),level=1)[0]
            # b. Calculating the loss and backward
            loss_dc = mse( recon_kspace3d_coil , self.masked_corrupted_kspace3D )
            loss_reg = self.L1min_lambda*torch.norm(coefficient,p=1)
            loss_recon = loss_dc + loss_reg

            loss_recon.backward()
            optimizer.step()
            print(f"Step {iteration+1}/{self.L1min_num_steps}, Loss: {loss_recon.item():.2e}, Loss_DC: {loss_dc.item():.2e}, Loss_Reg: {loss_reg.item():.2e}")
            
            recon_for_eval = recon.detach()*self.binary_background_mask

        return recon_for_eval

    def init_optimizer(self, param, optimizer_type, lr):
        '''
        Run this to define the optimizer for recon or motion estimation.
        '''
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam([param], lr=lr)
        elif optimizer_type == "SGD":
            optimizer = torch.optim.SGD([param], lr=lr)
        else:
            raise ValueError("Unknown optimizer")
        return optimizer
