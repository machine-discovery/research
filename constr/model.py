from abc import abstractmethod
import FrEIA.framework as FF
import FrEIA.modules as FM
from tqdm import tqdm
import math
import torch
from typing import List, Union, Tuple
import functorch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np


class MLP(torch.nn.Module):
    def __init__(self, ninp: int, nout: int, ndepths: int, nhidden: int, activation: List[torch.nn.Module]):
        super().__init__()

        assert len(activation) == ndepths - 1
        layers: List[torch.nn.Module] = []
        linear_layer = torch.nn.Linear(ninp, nhidden)
        layers.append(linear_layer)
        layers.append(activation[0])
        for i in range(1, ndepths - 1):
            linear_layer = torch.nn.Linear(nhidden, nhidden)
            layers.append(linear_layer)
            layers.append(activation[i])
        linear_layer = torch.nn.Linear(nhidden, nout)
        layers.append(linear_layer)

        self._module = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._module.forward(x)


class NeuralODE(torch.nn.Module):
    def __init__(self, func: torch.nn.Module, hmax: float = 0.001, solver: str = 'rk4'):
        super().__init__()
        self.func = func
        self.hmax = hmax
        self.solver = solver
        self.extra_in_dim = self.func.get_extra_in_dim()

    def forward(self, z0: torch.Tensor, t: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        bs, *z_shape = z0.size()
        time_len = t.size(0)
        z = torch.zeros(time_len, bs, *z_shape)
        z = z.type_as(z0)
        z[0] = z0
        z0 = z0.requires_grad_(True)

        for i_t in tqdm(range(time_len - 1)):
            z0 = self.rk4_solve(z0, t[i_t], t[i_t + 1])
            z[i_t + 1] = z0

        return z

    def rk4_solve(self, z0: torch.Tensor, t0: torch.Tensor,
                  t1: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        n_steps = math.ceil((abs(t1 - t0) / self.hmax).max().item())

        h = (t1 - t0) / n_steps
        t = t0
        z = z0

        # tmp solution
        h = h[0]
        for _ in range(n_steps):
            k1 = self.func(z, compute_jacobian=False)
            k1 = self.pad(z, k1)

            z2 = z + 0.5 * k1 * h
            k2 = self.func(z2, compute_jacobian=False)
            k2 = self.pad(z2, k2)

            z3 = z + 0.5 * k2 * h
            k3 = self.func(z3, compute_jacobian=False)
            k3 = self.pad(z3, k3)

            z4 = z + k3 * h
            k4 = self.func(z4, compute_jacobian=False)
            k4 = self.pad(z4, k4)

            z = z + h * (k1 + 2 * (k2 + k3) + k4) / 6
            t = t + h

        return z

    def pad(self, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if self.extra_in_dim > 0:
            *dim, _ = z.shape
            pad = torch.zeros((*dim, self.extra_in_dim)).type_as(z)
            k = torch.cat((k, pad), dim=-1)
        return k


class MLPAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

        self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, input_dim)

        self.nonlinearity = torch.nn.ReLU()

    def encode(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = h + self.nonlinearity(self.linear2(h))
        h = h + self.nonlinearity(self.linear3(h))
        return self.linear4(h)

    def decode(self, z):
        h = self.nonlinearity(self.linear5(z))
        h = h + self.nonlinearity(self.linear6(h))
        h = h + self.nonlinearity(self.linear7(h))
        return self.linear8(h)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class BaseModel(torch.nn.Module):
    def __init__(self, ninp: int = 2, nhid: int = 200, ndepth: int = 3,
                 extra_ninp: int = 0, norm: bool = False):
        super().__init__()
        self._ninp = ninp
        self._nhid = nhid
        self._ndepth = ndepth
        self._extra_ninp = extra_ninp  # number of input constants
        self._norm = norm  # whether or not normalisation is applied (at both inps and outs)
        self.register_buffer('jacobian', torch.Tensor([]))

        if self._norm:
            # special case only for double pendulum
            A = np.loadtxt('/home/ubuntu/research/node/version_2/data/norm.txt')
            self.A = torch.Tensor(A)

    @abstractmethod
    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        pass

    def compute_jacobian(self, dx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        jacobian_rows = []
        for i in range(dx.shape[-1]):
            grad = torch.autograd.grad(dx[..., i], x, torch.ones_like(dx[..., i]), True, True)
            jacobian_rows.append(grad[0][..., :dx.shape[-1]])
        jacobian = torch.stack(jacobian_rows, dim=-2).squeeze()
        return jacobian

    def get_extra_in_dim(self):
        return self._extra_ninp


class Baseline(BaseModel):
    """
    Our model uses baseline + regularisation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._ninp + self._extra_ninp, nout=self._ninp, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )


    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        x = x.requires_grad_(True)
        if self._norm and not compute_jacobian:
            # always set compute_jacobian to True when training and validating
            # when testing this is always set to False
            self.A = self.A.type_as(self.jacobian)
            x = x @ torch.linalg.inv(self.A)

        dx = self.linear(x)
        if compute_jacobian:
            self.jacobian = self.compute_jacobian(dx, x)
        if self._norm and not compute_jacobian:
            dx = dx @ self.A
        return dx


class NSF(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._ninp + self._extra_ninp, nout=self._ninp, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )
        self.hnn = MLP(
            ninp=self._ninp + self._extra_ninp, nout=1, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )

    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        x = x.requires_grad_(True)
        if self._norm and not compute_jacobian:
            self.A = self.A.type_as(self.jacobian)
            x = x @ torch.linalg.inv(self.A)

        hnn_x = self.hnn(x)
        hnn_x = torch.autograd.grad(hnn_x, x, torch.ones_like(hnn_x), retain_graph=True, create_graph=True)[0]

        linear_x = self.linear(x)
        jac = self.compute_jacobian(linear_x, x)
        if len(jac.shape) == 2:
            jac = jac[None, ...]
        M = jac - torch.transpose(jac, 1, 2)
        inv_M = torch.inverse(M)

        dx = torch.matmul(inv_M, hnn_x[..., None])

        if compute_jacobian:
            self.jacobian = jac
        if self._norm and not compute_jacobian:
            return dx[..., 0] @ self.A
        return dx[..., 0]


class AutoEncoderReg(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._ninp + self._extra_ninp, nout=self._ninp, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )
        self.autoencoder = MLPAutoencoder(self._ninp, self._nhid, self._ninp)

    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        x = x.requires_grad_(True)
        if self._norm and not compute_jacobian:
            self.A = self.A.type_as(self.jacobian)
            x = x @ torch.linalg.inv(self.A)
        x = self.autoencoder.encode(x)
        dx = self.linear(x)
        if compute_jacobian:
            self.jacobian = self.compute_jacobian(dx, x)
        transformed_grad = functorch.jvp(self.autoencoder.decode, (x,), (dx,))
        if self._norm and not compute_jacobian:
            return transformed_grad[1] @ self.A
        return transformed_grad[1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(x)

class InvertibleNNReg(BaseModel):
    def __init__(self, nblocks: int = 8, inn_nhid: int = 512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._ninp + self._extra_ninp, nout=self._ninp, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )

        def subnet_fc(c_in, c_out):
            return torch.nn.Sequential(
                torch.nn.Linear(c_in, inn_nhid),
                torch.nn.Softplus(),
                torch.nn.Linear(inn_nhid, c_out)
            )
        inn = FF.SequenceINN(self._ninp)
        for _ in range(nblocks):
            inn.append(FM.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
        self.inn_net = inn
        self.inn_net_inv = lambda x: self.inn_net(x, rev=True)[0]

    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        x = x.requires_grad_(True)
        if self._norm and not compute_jacobian:
            self.A = self.A.type_as(self.jacobian)
            x = x @ torch.linalg.inv(self.A)
        x = self.inn_net(x)[0]
        dx = self.linear(x)
        if compute_jacobian:
            self.jacobian = self.compute_jacobian(dx, x)
        transformed_grad = functorch.jvp(self.inn_net_inv, (x,), (dx,))
        if self._norm and not compute_jacobian:
            return transformed_grad[1] @ self.A
        return transformed_grad[1]


class HNN(BaseModel):
    def __init__(self, field_type: str = 'solenoidal', *args, **kwargs):
        super().__init__(*args, **kwargs)
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._ninp + self._extra_ninp, nout=self._ninp, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )
        self.field_type = field_type.lower()
        assert self.field_type in ['solenoidal', 'conservative']
        self.M = self.permutation_tensor(self._ninp)

    def permutation_tensor(self, n: int) -> torch.Tensor:
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])
        return M

    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        # not implemented for double pendulum so norm is not needed
        x = x.requires_grad_(True)
        self.M = self.M.type_as(x)

        F1, F2 = self.linear(x).split(self._ninp // 2, -1)  # traditional forward pass

        conservative_field = torch.zeros_like(x)  # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            # this will probably break if batch size is larger than 1
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]  # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape).type_as(x)
        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]  # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()
        dx = conservative_field + solenoidal_field
        if compute_jacobian:
            self.jacobian = self.compute_jacobian(dx, x)
        return dx


class LNN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._ninp + self._extra_ninp, nout=1, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )

    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        if len(x.shape) > 2:
            assert x.shape[0] == 1
            x = x[0]

        x = x.requires_grad_(True)
        if self._norm and not compute_jacobian:
            self.A = self.A.type_as(self.jacobian)
            x = x @ torch.linalg.inv(self.A)
        n = self._ninp // 2
        _, qt = torch.split(x, n, dim=-1)

        H = self.functorch_hessian(x)
        J = self.functorch_jacobian(x)
        A = J[..., :n]  # (..., nout=1, nstates // 2)
        B = H[..., n:, n:]  # (..., nout=1, nstates // 2, nstates // 2)
        C = H[..., n:, :n]  # (..., nout=1, nstates // 2, nstates // 2)
        A = A.unsqueeze(-1)  # (..., nout=1, nstates // 2, 1)

        qtt = torch.linalg.pinv(B) @ (A - C @ qt[..., None, :, None])
        qtt = qtt.squeeze(-1).squeeze(-2)

        res = torch.cat((qt, qtt), dim=-1)
        if self._norm and not compute_jacobian:
            res = res @ self.A
        return res

    def functorch_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        return functorch.vmap(functorch.jacrev(self.linear))(x)

    def functorch_hessian(self, x: torch.Tensor) -> torch.Tensor:
        # return functorch.vmap(functorch.hessian(self.linear))(x)
        return functorch.vmap(functorch.jacrev(functorch.jacrev((self.linear))))(x)


class PixelHNNAutoEncoder(BaseModel):
    def __init__(self, field_type: str = 'solenoidal', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nlatent = 2
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._nlatent, nout=self._nlatent, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )
        self.autoencoder = MLPAutoencoder(self._ninp, self._nhid, self._nlatent)
        self.field_type = field_type.lower()
        assert self.field_type in ['solenoidal', 'conservative']
        self.M = self.permutation_tensor(self._nlatent)

    def permutation_tensor(self, n: int) -> torch.Tensor:
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[:n // 2]])
        return M

    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        # not implemented for double pendulum so norm is not needed
        x = x.requires_grad_(True)
        self.M = self.M.type_as(x)

        F1, F2 = self.linear(x).split(self._nlatent // 2, -1)  # traditional forward pass

        conservative_field = torch.zeros_like(x)  # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            # this will probably break if batch size is larger than 1
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0]  # gradients for conservative field
            conservative_field = dF1 @ torch.eye(*self.M.shape).type_as(x)
        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True)[0]  # gradients for solenoidal field
            solenoidal_field = dF2 @ self.M.t()
        dx = conservative_field + solenoidal_field
        if compute_jacobian:
            self.jacobian = self.compute_jacobian(dx, x)
        return dx

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(x)


class PixelRegAutoEncoder(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nlatent = 2
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._nlatent, nout=self._nlatent, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )
        self.autoencoder = MLPAutoencoder(self._ninp, self._nhid, self._nlatent)

    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        # x is the latent embedding
        x = x.requires_grad_(True)
        if self._norm and not compute_jacobian:
            self.A = self.A.type_as(self.jacobian)
            x = x @ torch.linalg.inv(self.A)
        dx = self.linear(x)
        if compute_jacobian:
            self.jacobian = self.compute_jacobian(dx, x)
        return dx

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(x)


class PixelRegAutoEncoderRedundant(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nlatent = 20
        activations = [torch.nn.Softplus() for _ in range(self._ndepth - 1)]
        self.linear = MLP(
            ninp=self._nlatent, nout=self._nlatent, ndepths=self._ndepth,
            nhidden=self._nhid, activation=activations
        )
        self.autoencoder = MLPAutoencoder(self._ninp, self._nhid, self._nlatent)

    def forward(self, x: torch.Tensor, compute_jacobian: bool = True) -> torch.Tensor:
        # x is the latent embedding
        x = x.requires_grad_(True)
        if self._norm and not compute_jacobian:
            self.A = self.A.type_as(self.jacobian)
            x = x @ torch.linalg.inv(self.A)
        dx = self.linear(x)
        if compute_jacobian:
            self.jacobian = self.compute_jacobian(dx, x)
        return dx

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(x)


def get_model(model: str):
    model = model.lower()
    opt = {
        'baseline': Baseline,
        'baselinereg': Baseline,
        'autoencoderreg': AutoEncoderReg,
        'invertiblennreg': InvertibleNNReg,
        'hnn': HNN,
        'nsf': NSF,
        'lnn': LNN,
        'damped-baseline': Baseline,
        'damped-nsf': NSF,
        'dampedreg': InvertibleNNReg,
        'dampedregbaseline': Baseline,
        'pixelhnn': PixelHNNAutoEncoder,
        'pixelreg': PixelRegAutoEncoder,
        'dampedpixelreg': PixelRegAutoEncoderRedundant
    }
    return opt[model]
