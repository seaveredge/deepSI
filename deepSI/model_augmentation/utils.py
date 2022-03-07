import deepSI
import numpy as np
import torch
from torch import nn
from deepSI import model_augmentation


# Allowed systems:
def verifySystemType(sys):
    if   type(sys).__base__ is model_augmentation.lpvsystem.lpv_model_grid: return
    elif type(sys).__base__ is model_augmentation.lpvsystem.lpv_model_aff:  return
    elif type(sys)          is model_augmentation.lpvsystem.lti_system:     return
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