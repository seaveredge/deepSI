import deepSI
import numpy as np
import torch
from torch import nn
from deepSI import model_augmentation


def simple_res_net_2_LFR(net):
    seq = net.net_non_lin.net
    Nu = seq[0].in_features
    Nh = seq[0].out_features
    Ny = seq[-1].out_features
    nr_layers = int(0.5*(len(seq)-1))
    Dzu = torch.cat((seq[0].weight.data ,torch.zeros(((nr_layers-1)*Nh, Nu))),dim=0)
    Dyw = torch.cat((torch.zeros((Ny, (nr_layers - 1) * Nh)), seq[-1].weight.data), dim=1)
    Dyu = net.net_lin.weight.data
    if nr_layers == 1: Dzw = torch.zeros((Nh,Nh))
    else:
        Dzw = torch.zeros((Nh, nr_layers * Nh))
        for i in range(1,nr_layers):
            Dzw_row = torch.cat((torch.zeros((Nh, (i - 1) * Nh)), seq[2 * i].weight.data, torch.zeros((Nh, Nh*(nr_layers - i)))), dim=1)
            Dzw = torch.cat((Dzw, Dzw_row))
    return Dzw, Dzu, Dyw, Dyu

# Allowed systems:
def verifySystemType(sys):
    if   type(sys).__base__ is model_augmentation.lpvsystem.lpv_model_grid: return
    elif type(sys).__base__ is model_augmentation.lpvsystem.lpv_model_aff:  return
    elif type(sys)          is model_augmentation.lpvsystem.lti_system:     return
    elif type(sys)          is model_augmentation.lpvsystem.lti_affine_system: return
    elif type(sys).__base__ is model_augmentation.lpvsystem.lti_affine_system: return
    else: raise ValueError("Systems must be of the types defined in 'model_augmentation.lpvsystem'")

def verifyNetType(net,nettype):
    if nettype in 'static':
        if type(net) is not deepSI.utils.contracting_REN: return
        elif type(net) is not deepSI.utils.LFR_ANN: return
        else: raise ValueError("Static network required...")
    elif nettype in 'dynamic':
        if type(net) is deepSI.utils.contracting_REN: return
        elif type(net) is deepSI.utils.LFR_ANN: return
        else: raise ValueError("Dynamic network required...")
    else: raise ValueError('Unknown net type, only dynamic or static supported')

# some generic functions
def to_torch_tensor(A): # Obsolete?
    if torch.is_tensor(A):
        return A
    else:
        return torch.tensor(A, dtype=torch.float)

def RK4_step(f, x, u, h): # Functions of the form f(x,u). See other scripts for time-varying cases
    # one step of runge-kutta integration. u is zero-order-hold
    k1 = h * f(x, u)
    k2 = h * f(x + k1 / 2, u)
    k3 = h * f(x + k2 / 2, u)
    k4 = h * f(x + k3, u)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Function used for parameter initialization
def assign_param(A_old, A_new, nm):
    if A_new is not None:
        assert torch.is_tensor(A_new), nm + ' must be of the Tensor type'
        assert A_new.size() == A_old.size(), nm + ' must be of size' + str(A_old.size())
        return A_new.data
    else:
        return A_old.data