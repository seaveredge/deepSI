import deepSI
import numpy as np
import torch
from torch import nn
from deepSI import model_augmentation
from deepSI.model_augmentation.utils import verifySystemType, verifyNetType, RK4_step, assign_param


###################################################################################
####################         DEFAULT/GENERIC FUNCTIONS         ####################
###################################################################################
class default_encoder_net(nn.Module):  # a simple FC net with a residual (default approach)
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_encoder_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), n_out=nx,
                                  n_nodes_per_layer=n_nodes_per_layer,  n_hidden_layers=n_hidden_layers,
                                  activation=activation)

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)

class default_state_net(nn.Module):
    def __init__(self, nu, nx, augmentation_params):
        super(default_state_net, self).__init__()
        self.MApar = augmentation_params

    def forward(self, x, u):
        # in:                | out:
        #  - x (Nd, Nx)      |  - x+ (Nd, Nx)
        #  - u (Nd, Nu + np) |
        return self.MApar.f_h(x,u)[0] # Select f(x,u) function

class default_output_net(nn.Module):
    def __init__(self, nu, nx, ny, augmentation_params):
        super(default_output_net, self).__init__()
        self.MApar = augmentation_params

    def forward(self, x, u):
        # in:                | out:
        #  - x (Nd, Nx)      |  - y (Nd, Ny)
        #  - u (Nd, Nu + np) |
        return self.MApar.f_h(x,u)[1] # Select h(x,u) function

def get_augmented_fitsys(augmentation_structure, known_system, wnet, aug_kwargs={}, e_net=default_encoder_net,
                         y_lag_encoder=None, u_lag_encoder=None, enet_kwargs={}, na_right=0, nb_right=0):
    if augmentation_structure is model_augmentation.augmentationstructures.SSE_DynamicAugmentation:
        # Learn the state of the augmented model as well
        nx_system = known_system.Nx
        nx_hidden = wnet.n_state
        nx_encoder = nx_system + nx_hidden
    elif augmentation_structure is model_augmentation.augmentationstructures.SSE_StaticAugmentation:
        nx_encoder = known_system.Nx
    else: raise ValueError("'augmentation_structure' should be either " +
                           "'SSE_DynamicAugmentation' or 'SSE_StaticAugmentation'")
    if y_lag_encoder is None: y_lag_encoder = nx_encoder + 1
    if u_lag_encoder is None: u_lag_encoder = nx_encoder + 1
    if e_net is None: raise ValueError("Encoder net (e_net) must be defined")
    return deepSI.fit_systems.SS_encoder_general_hf(feedthrough=True, nx=nx_encoder, na=y_lag_encoder, nb=u_lag_encoder,
                                                    e_net=e_net, e_net_kwargs=dict(**enet_kwargs), hf_net=augmentation_structure,
                                                    hf_net_kwargs=dict(known_system=known_system, wnet=wnet, **aug_kwargs),
                                                    na_right=na_right, nb_right=nb_right)

def get_CT_augmented_fitsys(augmentation_params, y_lag_encoder, u_lag_encoder, e_net=None, f_net=None, h_net=None):
    raise NotImplementedError('Not verified or tested or whatevered yet...')
    # Input checks:
    if type(augmentation_params) is model_augmentation.augmentationstructures.SSE_DynamicAugmentation:
        # Learn the state of the augmented model as well
        nx_system = augmentation_params.Nx
        nx_hidden = augmentation_params.Nxh
        nx_encoder = nx_system + nx_hidden
    elif type(augmentation_params) is model_augmentation.augmentationstructures.SSE_StaticAugmentation:
        nx_encoder = augmentation_params.Nx
    else: raise ValueError("'augmentation_params' should be of type " +
                           "'SSE_DynamicAugmentation' or 'SSE_StaticAugmentation'")
    if augmentation_params.sys.Ts is None: raise ValueError("Sampling time associated with the system not defined...")  # ToDo: IS this the correct way of having a samping time here?
    if e_net is None: e_net = default_encoder_net
    if f_net is None: f_net = default_CT_state_net
    else: print("Make sure that your custom net is of the form as given in 'augmentationstructures.default_CT_state_net'")
    if h_net is None: h_net = default_output_net
    else: print("Make sure that your custom net is of the form as given in 'augmentationstructures.default_output_net'")
    ### TODO: For now have my own RK4 step integrator, Later... have it integrated with deepSI (also include normalization
    ### TODO: factor and stuff)
    # calculate the normalization factor for f  TODO: For now set it to zero...
    # integratornet = integrator_RK4

    return deepSI.fit_systems.SS_encoder_general(feedthrough=True, nx=nx_encoder,
            na=y_lag_encoder, nb=u_lag_encoder, e_net=e_net, e_net_kwargs=dict(),
            f_net=f_net, f_net_kwargs=dict(augmentation_params=augmentation_params),
            h_net=h_net, h_net_kwargs=dict(augmentation_params=augmentation_params))


###################################################################################
##############         SUB SPACE ENCODER BASED AUGMENTATIONS         ##############
##############                  STATIC AUGMENTATION                  ##############
###################################################################################
class SSE_StaticAugmentation(nn.Module):
    def __init__(self, nx, nu, ny, known_system, wnet, initial_scaling_factor=1e-3, Dzw_is_zero=True, feedthrough=True):
        super(SSE_StaticAugmentation, self).__init__()
        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(wnet, 'static')
        # Save parameters
        self.sys = known_system
        self.net = wnet
        self.Nu = self.sys.Nu
        self.Nx = self.sys.Nx
        self.Ny = self.sys.Ny
        self.Nz = self.net.n_in
        self.Nw = self.net.n_out
        self.Bw = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nx, self.Nw))
        self.Cz = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nx))
        self.Dzw = None if Dzw_is_zero else nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nw))
        self.Dzu = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nu))
        self.Dyw = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Ny, self.Nw))

    def initialize_parameters(self, Bw = None, Cz = None, Dzw = None, Dzu = None, Dyw = None):
        self.Bw.data = assign_param(self.Bw, Bw, 'Bw')
        self.Cz.data = assign_param(self.Cz, Cz, 'Cz')
        self.Dzw.data = assign_param(self.Dzw, Dzw, 'Dzw')
        self.Dzu.data = assign_param(self.Dzu, Dzu, 'Dzu')
        self.Dyw.data = assign_param(self.Dyw, Dyw, 'Dyw')

    def compute_z(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - z (Nd, Nz)
        #  - u (Nd, Nu) |
        zx = torch.einsum('ij, bj -> bi', self.Cz, x)   # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        zu = torch.einsum('ij, bj -> bi', self.Dzu, u)  # (Nz, Nu)*(Nd, Nu)->(Nd, Nz)
        zw = torch.zeros(zu.shape) if self.Dzw is None else torch.zeros(zu.shape)  # Not sure how to implement this yet... Should be something like :torch.einsum('ij, bj -> bi', self.Dzw, w)  # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        return zx + zu + zw

    def compute_ynet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Dyw*w (Nd, Ny)
        return torch.einsum('ij, bj -> bi', self.Dyw, w) # (Ny, Nw)*(Nd, Nw)->(Nd, Ny)

    def compute_xnet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Bw*w (Nd, Nx)
        return torch.einsum('ij, bj -> bi', self.Bw, w)  # (Nx, Nw)*(Nd, Nw)->(Nd, Nx)

    def forward(self,x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - y  (Nd, Ny)
        #  - u (Nd, Nu) |  - x+ (Nd, Nx)
        # compute network contribution
        z = self.compute_z(x, u)
        w = self.net(z)
        x_plus = self.sys.f(x,u) + self.compute_xnet_contribution(w)
        y_k    = self.sys.h(x,u) + self.compute_ynet_contribution(w)
        return y_k, x_plus


###################################################################################
##############                  DYNAMIC AUGMENTATION                 ##############
###################################################################################
class SSE_DynamicAugmentation(nn.Module):
    def __init__(self, nx, nu, ny, known_system, wnet, initial_scaling_factor=1e-3, Dzw_is_zero=True, feedthrough=True):
        super(SSE_DynamicAugmentation, self).__init__()
        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(wnet, 'dynamic')
        # Save parameters
        self.sys = known_system
        self.net = wnet
        self.Nu  = self.sys.Nu
        self.Nx  = self.sys.Nx
        self.Ny  = self.sys.Ny
        self.Nz  = self.net.n_in
        self.Nw  = self.net.n_out
        self.Nxh = self.net.n_state
        self.Bw  = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nx, self.Nw))
        self.Cz  = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nx))
        self.Dzw = None if Dzw_is_zero else nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nw))
        self.Dzu = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nu))
        self.Dyw = nn.Parameter(data=initial_scaling_factor * torch.rand(self.Ny, self.Nw))
        assert self.Dzw is None, 'No implementation yet for non-zero Dzw'

    def initialize_parameters(self, Bw=None, Cz=None, Dzw=None, Dzu=None, Dyw=None):
        self.Bw.data = assign_param(self.Bw, Bw, 'Bw')
        self.Cz.data = assign_param(self.Cz, Cz, 'Cz')
        #self.Dzw.data = assign_param(self.Dzw, Dzw, 'Dzw')
        self.Dzu.data = assign_param(self.Dzu, Dzu, 'Dzu')
        self.Dyw.data = assign_param(self.Dyw, Dyw, 'Dyw')

    def compute_z(self, x, w, u):
        # in:            | out:
        #  - x (Nd, Nx)  |  - z (Nd, Nz)
        #  - u (Nd, Nu)  |
        zx = torch.einsum('ij, bj -> bi', self.Cz, x)   # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        zu = torch.einsum('ij, bj -> bi', self.Dzu, u)  # (Nz, Nu)*(Nd, Nu)->(Nd, Nz)
        zw = torch.zeros(zu.shape) if self.Dzw is None else torch.zeros(zu.shape)  # Not sure how to implement this yet... Should be something like :torch.einsum('ij, bj -> bi', self.Dzw, w)  # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        return zx + zu + zw

    def compute_ynet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Dyw*w (Nd, Ny)
        return torch.einsum('ij, bj -> bi', self.Dyw, w) # (Ny, Nw)*(Nd, Nw)->(Nd, Ny)

    def compute_xnet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Bw*w (Nd, Nx)
        return torch.einsum('ij, bj -> bi', self.Bw, w)  # (Nx, Nw)*(Nd, Nw)->(Nd, Nx)

    def forward(self,x, u):
        # in:                 | out:
        #  - x (Nd, Nx + Nxh) |  - x+ (Nd, Nx + Nxh)
        #  - u (Nd, Nu)       |  - y  (Nd, Ny)
        # split up the state from the encoder in the state of the known part
        # and the state of the unknown (to be learned) part
        if x.ndim == 1: # shape of x is (Nx+Nxh)
            assert x.shape != self.Nx+self.Nxh, 'dimension of state is not nx+nx_hidden... nx = ' +  str(self.Nx) + \
                                                ', nx_h = ' + str(self.Nxh) + ', while x.shape = ' + str(x.shape)
            x_known = x[:self.Nx]
            x_learn = x[-self.Nxh:]
        else:
            x_known = x[:, :self.Nx]
            x_learn = x[:, -self.Nxh:]
        # compute the input for the network
        z = self.compute_z(x=x_known, w=None, u=u)  # z = Cz x + Dzw w + Dzu u  --> Dzw = 0
        # calculate w from NN and update hidden state
        x_learn_plus, w = self.net(hidden_state=x_learn, u=z)  # u_net = z_model
        x_known_plus = self.sys.f(x_known, u) + self.compute_xnet_contribution(w)
        y_k          = self.sys.h(x_known, u) + self.compute_ynet_contribution(w)
        x_plus = torch.cat((x_known_plus,x_learn_plus), dim=x.ndim-1)
        return y_k, x_plus






