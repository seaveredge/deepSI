import deepSI
import numpy as np
import torch
from torch import nn
from scipy.io import loadmat
from scipy.interpolate import interpn
from deepSI.model_augmentation.utils import RK4_step


# Apply experiment wrapper
def hidden_apply_experiment(sys, data, x0):
    # Todo: What if we have data with non-equidistant time-steps
    if sys.Ts is None and data.dt is None: raise ValueError('Sample time of data not specified')
    if type(data) is deepSI.system_data.System_data_list:
        lis = []
        for i in range(len(data)):
            y, x = hidden_apply_one_experiment(sys,data[i].u,x0, data[i].N_samples, data.dt)
            lis.append(deepSI.System_data(u=data[i].u, y=y, x=x))
        retrn = deepSI.System_data_list(lis)
    else:
        y, x = hidden_apply_one_experiment(sys, data.u, x0, data.N_samples, data.dt)
        retrn = deepSI.System_data(u=data.u, y=y, x=x)
    return retrn


def hidden_apply_one_experiment(sys, u, x0, T, dt):
    u = torch.tensor(u,dtype=torch.float) if not torch.is_tensor(u) else u
    x0 = torch.tensor(x0, dtype=torch.float) if not torch.is_tensor(x0) else x0
    y = torch.zeros(T, sys.Ny)
    x = torch.zeros(T+1, sys.Nx)
    x[0,:] = x0
    for k in range(T):
        y[k, :] = sys.h(x[k, :], u[k, :])
        # Discrete simulation
        if sys.Ts is not None:
            x[k + 1, :] = sys.f(x[k, :], u[k, :])
        else:
            x[k + 1, :] = RK4_step(sys.f, x[k, :], u[k, :], dt)
    return y, x[:-1,:]


class lpv_model_grid:
    def __init__(self, A, B, C, D, grid, Ts=-1):
        # This system must be able to process the inputs and outputs for the subspace encoder,
        # hence: shape({x,u,y,p}) = (Nd,{nx,nu,ny,np})
        # The matrices A,...,D are NPARRAYS!! (NOT tensors)
        self.A = A # shape: (nx, nx, ng_p1, ..., ng_pnp)
        self.B = B # shape: (nx, nu, ng_p1, ..., ng_pnp)
        self.C = C # shape: (ny, nx, ng_p1, ..., ng_pnp)
        self.D = D # shape: (nx, nx, ng_p1, ..., ng_pnp)
        self.grid = grid  # Tuple of np array-like elements, with (gridp1, gridp2, ..., gridp_np)
        self.Nx = self.A.shape[0]
        self.Nu = self.B.shape[1]
        self.Ny = self.C.shape[0]
        self.Np = len(grid)
        self.interp_A = self.getcorrect_interpmat(self.A)
        self.interp_B = self.getcorrect_interpmat(self.B)
        self.interp_C = self.getcorrect_interpmat(self.C)
        self.interp_D = self.getcorrect_interpmat(self.D)
        self.Ts = Ts

    def getcorrect_interpmat(self,matr):
        l = list(range(len(matr.shape)))
        return np.transpose(matr, axes=l[2:] + l[:2])  # (pdims, n, m)

    def f(self, x, u):
        # in:                   | out:
        #     x (Nd,Nx)         |      x+ (Nd,Nx)
        #     u (Nd,Nu+Np_ext)  |
        einsumequation = 'bik, bk->bi' if x.ndim > 1 else 'bik, k->i'
        p, unow = self.schedulingFunction(x, u) # (Nd, Np), (Nd, Nu) (both tensors)
        A_now = interpn(self.grid,self.interp_A,p.numpy(), method='linear', bounds_error=False, fill_value=None) # (Nd, nx, nx)
        B_now = interpn(self.grid,self.interp_B,p.numpy(), method='linear', bounds_error=False, fill_value=None) # (Nd, nx, nu)
        A = torch.tensor(A_now,dtype=torch.float)  # (Nd,Nx,Nx)  (when x.shape = (nx,), Nd = 1)
        B = torch.tensor(B_now,dtype=torch.float)  # (Nd,Nx,Nu)
        # Batch wise matrix-vector multiplication
        Ax = torch.einsum(einsumequation, A, x) # (Nd,Nx,Nx)*(Nd,Nx)->(Nd,Nx)
        Bu = torch.einsum(einsumequation, B, unow) # (Nd,Nx,Nu)*(Nd,Nu)->(Nd,Nx)
        return Ax + Bu

    def h(self,x, u):
        # in:                   | out:
        #     x (Nd,Nx)         |      y (Nd,Ny)
        #     u (Nd,Nu+Np_ext)  |
        einsumequation = 'bik, bk->bi' if x.ndim > 1 else 'bik, k->i'
        p, unow = self.schedulingFunction(x, u)  # (Nd, Np), (Nd, Nu) (both tensors)
        C_now = interpn(self.grid,self.interp_C,p.numpy(), method='linear', bounds_error=False, fill_value=None) # (Nd, ny, nx)
        D_now = interpn(self.grid,self.interp_D,p.numpy(), method='linear', bounds_error=False, fill_value=None) # (Nd, ny, nu)
        C = torch.tensor(C_now,dtype=torch.float)  # (Nd,Ny,Nx)  (when x.shape = (nx,), Nd = 1)
        D = torch.tensor(D_now,dtype=torch.float)  # (Nd,Ny,Nu)
        # Batch wise matrix-vector multiplication
        Cx = torch.einsum(einsumequation, C, x) # (Nd,ny,nx)*(Nd,nx)->(Nd,ny)
        Du = torch.einsum(einsumequation, D, unow) # (Nd,ny,nu)*(Nd,nu)->(Nd,ny)
        return Cx + Du

    def apply_experiment(self, data, x0=None):
        if x0 is None:
            x0 = torch.zeros(self.Nx)
        return hidden_apply_experiment(self, data, x0)

    def schedulingFunction(self, x, u):
        # USER DEFINED FUNCTION
        # in:                 | out:
        #     x (Nd, Nx)      |      p (Nd, Np)
        #     u (Nd, Nu + Np) |      ubar (Nd, Nu)
        #
        # EXAMPLE:
        #     ToDo: Example
        #
        raise NotImplementedError('Scheduling function should be implemented in child')


class lpv_model_aff:
    def __init__(self, A, B, C, D, Ts=-1):
        # This system must be able to process the inputs and outputs for the subspace encoder,
        # hence: shape({x,u,y,p}) = (Nd,{Nx, Nu+np, Ny, Np}),
        # Note: np=/=Np!!
        #       np: # of external scheduling variables
        #       Np: dimension of scheduling vector
        # A...D are tensors!
        self.A = A  # shape: (Nx, Nx, Np+1)
        self.B = B  # shape: (Nx, Nu, Np+1)
        self.C = C  # shape: (Ny, Nx, Np+1)
        self.D = D  # shape: (Ny, Nu, Np+1)
        self.Nx = self.A.shape[0]
        self.Nu = self.B.shape[1]
        self.Ny = self.C.shape[0]
        self.Np = self.A.shape[2]-1
        self.Ts = Ts

    def getAp(self, A, p):
        # In:               | Out:
        #  - A (n, m, Np+1) |   - A(p) (Nd, n, m)
        #  - p (Nd, Np)     |
        Nd = p.shape[0]
        pbar = torch.cat((torch.ones((Nd,1)), p), dim=1) # (Nd,np+1)
        return torch.einsum('ijk, bk -> bij', A, pbar) # (n, m, (Nd,np+1))*(Nd,np+1) -> (Nd, n, m)

    def f(self, x, u):
        # in:                | out:
        #  - x (Nd, Nx)      |  - x+ (Nd, Nx)
        #  - u (Nd, Nu + np) |
        einsumequation =  'bik, bk->bi' if x.ndim > 1 else 'bik, k->i'
        p, unow = self.schedulingFunction(x,u)      # (Nd, Np), (Nd, Nu)
        A = self.getAp(self.A, p)                   # (Nd, Nx, Nx)
        B = self.getAp(self.B, p)                   # (Nd, Nx, Nu)
        Ax = torch.einsum(einsumequation, A, x)      # (Nd, Nx, Nx)*(Nd, Nx)->(Nd, Nx)
        Bu = torch.einsum(einsumequation, B, unow)   # (Nd, Nx, Nu)*(Nd, Nu)->(Nd, Nx)
        return Ax + Bu

    def h(self,x, u):
        # in:                | out:
        #  - x (Nd, Nx)      |  - y (Nd, Ny)
        #  - u (Nd, Nu + np) |
        einsumequation = 'bik, bk->bi' if x.ndim > 1 else 'bik, k->i'
        p, unow = self.schedulingFunction(x, u)  # (Nd, Np), (Nd, Nu)
        C = self.getAp(self.C, p)  # (Nd, Ny, Nx)
        D = self.getAp(self.D, p)  # (Nd, Ny, Nu)
        Cx = torch.einsum(einsumequation, C, x)  # (Nd, Nx, Nx)*(Nd, Nx)->(Nd, Nx)
        Du = torch.einsum(einsumequation, D, unow)  # (Nd, Nx, Nu)*(Nd, Nu)->(Nd, Nx)
        return Cx + Du

    def apply_experiment(self, data, x0=None):
        if x0 is None:
            x0 = torch.zeros(self.Nx)
        return hidden_apply_experiment(self, data, x0)

    # From now on, user defined!
    def schedulingFunction(self, x, u):
        # in:                 | out:
        #  - x (Nd, Nx)       |  - p (Nd, Np)
        #  - u (Nd, Nu + np) |  - ubar (Nd, Nu) (True input of the system, i.e., no scheduling variables)
        # note that np =/= Np, np is the number of external scheduling variables (only used here)
        #
        # EXAMPLE:
        #     ToDo: Example
        raise NotImplementedError('Scheduling function should be implemented in child')


class lti_system:
    def __init__(self, A, B, C, D, Ts=-1):
        self.A = A  # shape: (Nx, Nx)
        self.B = B  # shape: (Nx, Nu)
        self.C = C  # shape: (Ny, Nx)
        self.D = D  # shape: (Ny, Nu)
        self.Nx = self.A.shape[0]
        self.Nu = self.B.shape[1]
        self.Ny = self.C.shape[0]
        self.Ts = Ts

    def f(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - x+ (Nd, Nx)
        #  - u (Nd, Nu) |
        Ax = torch.einsum('ik, bk->bi', self.A, x) # (Nx, Nx)*(Nd, Nx)->(Nd, Nx)
        Bu = torch.einsum('ik, bk->bi', self.B, u)   # (Nd, Nx, Nu)*(Nd, Nu)->(Nd, Nx)
        return Ax + Bu

    def h(self,x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - y (Nd, Ny)
        #  - u (Nd, Nu) |
        Cx = torch.einsum('ik, bk->bi', self.C, x)  # (Nd, Nx, Nx)*(Nd, Nx)->(Nd, Nx)
        Du = torch.einsum('ik, bk->bi', self.D, u)  # (Nd, Nx, Nu)*(Nd, Nu)->(Nd, Nx)
        return Cx + Du

    def apply_experiment(self, data, x0=None):
        if x0 is None:
            x0 = torch.zeros(self.Nx)
        return hidden_apply_experiment(self, data, x0)
