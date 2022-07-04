import torch
import pytorch_lightning as pl
import numpy as np
from utils import CONSOLE_ARGUMENTS as hparams
from model import NeuralODE
from misc import calc_double_pdl_energy, calc_mass_spring_energy, calc_single_pdl_energy
from abc import abstractmethod
from typing import Union, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class BaseLitModel(pl.LightningModule):
    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self._backbone = backbone
        self._ode_trained = NeuralODE(self._backbone, hparams.hmax, hparams.solver)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hparams.lr, amsgrad=True)
        return optimizer

    @abstractmethod
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor,
             x: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        pass

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass


class BaselineLitModel(BaseLitModel):
    def __init__(self, backbone: torch.nn.Module):
        super().__init__(backbone)
        self._backbone = backbone
        self._ode_trained = NeuralODE(self._backbone, hparams.hmax, hparams.solver)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        loss_fn = torch.nn.MSELoss()
        return loss_fn(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, dx = batch  # (batch_size, nb_data_per_batch, dim)
        dx_hat = self._backbone(x, compute_jacobian=True)
        loss = self.loss(dx_hat, dx)
        self.log(
            'train_loss/original_loss', loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        return {'loss': loss, 'y': dx, 'y_hat': dx_hat}

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            x, dx = batch  # (batch_size, nb_data_per_batch, dim)
            dx_hat = self._backbone(x, compute_jacobian=True)
        loss = self.loss(dx_hat, dx)
        self.log(
            'val_loss/original_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )


class BaselineRegLitModel(BaseLitModel):
    def __init__(self, backbone: torch.nn.Module, reg_coef: float):
        super().__init__(backbone)
        self._backbone = backbone
        self._ode_trained = NeuralODE(self._backbone, hparams.hmax, hparams.solver)
        self._reg_coef = reg_coef

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        loss_fn = torch.nn.MSELoss()
        for buffer in self._backbone.named_buffers():
            if buffer[0] == 'jacobian':
                jacobian = buffer[1]

        J = torch.eye(int(y_hat.shape[-1]))
        J = torch.cat([J[int(y_hat.shape[-1]) // 2:], -J[:int(y_hat.shape[-1]) // 2]])
        J = J.type_as(y_hat)

        mse = loss_fn(y_hat, y)
        reg = loss_fn(J.T @ jacobian, torch.transpose(jacobian, 1, 2) @ J)
        loss = mse + self._reg_coef * reg
        return loss, mse, reg

    def training_step(self, batch, batch_idx):
        x, dx = batch  # (batch_size, nb_data_per_batch, dim)
        dx_hat = self._backbone(x, compute_jacobian=True)
        loss, mse, reg = self.loss(dx_hat, dx)
        self.log(
            'train_loss/total_loss', loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/original_loss', mse, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/reg', reg, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        return {'loss': loss, 'y': dx, 'y_hat': dx_hat}

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            x, dx = batch  # (batch_size, nb_data_per_batch, dim)
            dx_hat = self._backbone(x, compute_jacobian=True)
        loss, mse, reg = self.loss(dx_hat, dx)
        self.log(
            'val_loss/total_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/original_loss', mse, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/reg', reg, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )


class DampedRegLitModel(BaselineRegLitModel):
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        loss_fn = torch.nn.MSELoss()
        for buffer in self._backbone.named_buffers():
            if buffer[0] == 'jacobian':
                jacobian = buffer[1]
        eigvals = torch.linalg.eig(jacobian)[0]
        eigvals = torch.view_as_real(eigvals)
        real_pt = eigvals[:, :, 0]

        mse = loss_fn(y_hat, y)
        reg = loss_fn(torch.nn.functional.relu(real_pt), torch.zeros_like(real_pt))
        loss = mse + self._reg_coef * reg
        return loss, mse, reg


class PixelHNNLitModel(BaselineLitModel):
    def __init__(self, backbone: torch.nn.Module):
        super().__init__(backbone)
        self._backbone = backbone
        self._ode_trained = NeuralODE(self._backbone, hparams.hmax, hparams.solver)

    def loss(self, x: torch.Tensor, xnext: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        loss_fn = torch.nn.MSELoss()

        z = self._backbone.encode(x)
        x_hat = self._backbone.decode(z)        

        znext = self._backbone.encode(xnext)
        noise = torch.randn_like(z) * 0.05
        znext_hat = z + self._backbone(z + noise)

        w, dw = torch.split(z, z.shape[-1] // 2, dim=-1)
        wnext, _ = torch.split(znext, znext.shape[-1] // 2, dim=-1)

        mse = loss_fn(znext, znext_hat)
        ae_loss = loss_fn(x, x_hat)
        cc_loss = loss_fn(dw, wnext - w)
        loss = 1e-1 * mse + ae_loss + cc_loss
        return loss, mse, cc_loss, ae_loss

    def training_step(self, batch, batch_idx):
        x, xnext = batch  # (batch_size, nb_data_per_batch, dim)
        loss, mse, cc_loss, ae_loss = self.loss(x, xnext)
        self.log(
            'train_loss/total_loss', loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/original_loss', mse, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/cc_loss', cc_loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/ae_loss', ae_loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            x, xnext = batch  # (batch_size, nb_data_per_batch, dim)
            loss, mse, cc_loss, ae_loss = self.loss(x, xnext)
        self.log(
            'val_loss/total_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/original_loss', mse, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/cc_loss', cc_loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/ae_loss', ae_loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hparams.lr, amsgrad=True, weight_decay=1e-5)
        return optimizer


class PixelRegLitModel(BaselineRegLitModel):
    def __init__(self, backbone: torch.nn.Module, reg_coef: float):
        super().__init__(backbone, reg_coef)
        self._backbone = backbone
        self._ode_trained = NeuralODE(self._backbone, hparams.hmax, hparams.solver)
        self._reg_coef = reg_coef

    def loss(self, x: torch.Tensor, xnext: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        loss_fn = torch.nn.MSELoss()

        z = self._backbone.encode(x)
        x_hat = self._backbone.decode(z)

        znext = self._backbone.encode(xnext)
        noise = torch.randn_like(znext) * 0.05
        znext_hat = z + self._backbone(z + noise)

        w, dw = torch.split(z, z.shape[-1] // 2, dim=-1)
        wnext, _ = torch.split(znext, znext.shape[-1] // 2, dim=-1)

        for buffer in self._backbone.named_buffers():
            if buffer[0] == 'jacobian':
                jacobian = buffer[1]

        J0 = torch.zeros((int(z.shape[-1] / 2), int(z.shape[-1] / 2))).type_as(z)
        J1 = torch.eye(int(z.shape[-1] / 2)).type_as(z)
        J2 = -torch.eye(int(z.shape[-1] / 2)).type_as(z)
        J3 = torch.zeros((int(z.shape[-1] / 2), int(z.shape[-1] / 2))).type_as(z)
        J = torch.cat((torch.cat((J0, J1), dim=1), torch.cat((J2, J3), dim=1)), dim=0).type_as(z)

        mse = loss_fn(znext, znext_hat)
        ae_loss = loss_fn(x, x_hat)
        cc_loss = loss_fn(dw, wnext - w)
        reg = loss_fn(J.T @ jacobian, torch.transpose(jacobian, 1, 2) @ J)
        loss = 1e-1 * mse + ae_loss + cc_loss + self._reg_coef * reg
        return loss, mse, cc_loss, ae_loss, reg

    def training_step(self, batch, batch_idx):
        x, xnext = batch  # (batch_size, nb_data_per_batch, dim)
        loss, mse, cc_loss, ae_loss, reg = self.loss(x, xnext)
        self.log(
            'train_loss/total_loss', loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/original_loss', mse, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/reg', reg, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/cc_loss', cc_loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/ae_loss', ae_loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            x, xnext = batch  # (batch_size, nb_data_per_batch, dim)
            loss, mse, cc_loss, ae_loss, reg = self.loss(x, xnext)
        self.log(
            'val_loss/total_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/original_loss', mse, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/reg', reg, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/cc_loss', cc_loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/ae_loss', ae_loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hparams.lr, amsgrad=True, weight_decay=1e-5)
        return optimizer


class DampedPixelRegLitModel(BaselineRegLitModel):
    def __init__(self, backbone: torch.nn.Module, reg_coef: float):
        super().__init__(backbone, reg_coef)
        self._backbone = backbone
        self._ode_trained = NeuralODE(self._backbone, hparams.hmax, hparams.solver)
        self._reg_coef = reg_coef

    def loss(self, x: torch.Tensor, xnext: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        loss_fn = torch.nn.MSELoss()

        z = self._backbone.encode(x)
        x_hat = self._backbone.decode(z)

        znext = self._backbone.encode(xnext)
        noise = torch.randn_like(znext) * 0.05
        znext_hat = z + self._backbone(z + noise)


        for buffer in self._backbone.named_buffers():
            if buffer[0] == 'jacobian':
                jacobian = buffer[1]

        eigvals = torch.linalg.eig(jacobian)[0]
        eigvals = torch.view_as_real(eigvals)
        real_pt = eigvals[:, :, 0]
    
        mse = loss_fn(znext, znext_hat)
        ae_loss = loss_fn(x, x_hat)
        reg = loss_fn(torch.nn.functional.relu(real_pt), torch.zeros_like(real_pt))
        loss = 1e-1 * mse + ae_loss + self._reg_coef * reg
        return loss, mse, ae_loss, reg

    def training_step(self, batch, batch_idx):
        x, xnext = batch  # (batch_size, nb_data_per_batch, dim)
        loss, mse, ae_loss, reg = self.loss(x, xnext)
        self.log(
            'train_loss/total_loss', loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/original_loss', mse, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/reg', reg, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'train_loss/ae_loss', ae_loss, on_step=True, on_epoch=False,
            prog_bar=True, logger=True, sync_dist=True
        )
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            x, xnext = batch  # (batch_size, nb_data_per_batch, dim)
            loss, mse, ae_loss, reg = self.loss(x, xnext)
        self.log(
            'val_loss/total_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/original_loss', mse, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/reg', reg, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            'val_loss/ae_loss', ae_loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hparams.lr, amsgrad=True, weight_decay=1e-5)
        return optimizer


class InvertibleNNRegLitModel(BaselineRegLitModel):
    pass


class HNNLitModel(BaselineLitModel):
    pass


class NSFLitModel(BaselineLitModel):
    pass


class LNNLitModel(BaselineLitModel):
    pass


def get_litmodel(model: str):
    model = model.lower()
    opt = {
        'baseline': BaselineLitModel,
        'baselinereg': BaselineRegLitModel,
        'invertiblennreg': InvertibleNNRegLitModel,
        'hnn': HNNLitModel,
        'nsf': NSFLitModel,
        'lnn': LNNLitModel,
        'damped-baseline': BaselineLitModel,
        'damped-nsf': NSFLitModel,
        'dampedreg': DampedRegLitModel,
        'dampedregbaseline': DampedRegLitModel,
        'pixelhnn': PixelHNNLitModel,
        'pixelreg': PixelRegLitModel,
        'dampedpixelreg': DampedPixelRegLitModel,
    }
    return opt[model]
