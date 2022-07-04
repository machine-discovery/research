import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
import torch
from utils import CONSOLE_ARGUMENTS as hparams
from model import get_model
from litmodel import get_litmodel
from data.double_pendulum import general_double_pendulum
from data.single_pendulum import general_pendulum
from data.mass_spring import general_spring
from data.damped_pendulum import general_damped_pendulum
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pickle5 as pickle


if hparams.experiment == 'mass-spring':
    train_x, train_dx = general_spring(
        (0, 2 * np.pi), nb_teval=30, nb_samples=250, seed=hparams.train_seed, noiseless=hparams.noiseless,
    )
    val_x, val_dx = general_spring(
        (0, 2 * np.pi), nb_teval=30, nb_samples=25, seed=hparams.val_seed, noiseless=hparams.noiseless,
    )
    model = get_model(hparams.model)()
    pl_model = get_litmodel(hparams.model)
    if hparams.model == 'baselinereg':
        pl_model = pl_model(backbone=model, reg_coef=hparams.reg_real)
    else:
        pl_model = pl_model(backbone=model)
elif hparams.experiment == 'single-pdl':
    train_x, train_dx = general_pendulum(
        (0, 2 * np.pi), nb_teval=30, nb_samples=250, seed=hparams.train_seed, noiseless=hparams.noiseless,
    )
    val_x, val_dx = general_pendulum(
        (0, 2 * np.pi), nb_teval=30, nb_samples=25, seed=hparams.val_seed, noiseless=hparams.noiseless,
    )
    model = get_model(hparams.model)()
    pl_model = get_litmodel(hparams.model)
    if hparams.model == 'baselinereg':
        pl_model = pl_model(backbone=model, reg_coef=hparams.reg_real)
    else:
        pl_model = pl_model(backbone=model)
elif hparams.experiment == 'damped-single-pdl':
    train_x, train_dx = general_damped_pendulum(
        (0, 2 * np.pi), nb_teval=30, nb_samples=250, seed=hparams.train_seed, noiseless=hparams.noiseless,
    )
    val_x, val_dx = general_damped_pendulum(
        (0, 2 * np.pi), nb_teval=30, nb_samples=25, seed=hparams.val_seed, noiseless=hparams.noiseless,
    )
    if hparams.model == 'dampedreg':
        model = get_model(hparams.model)(nblocks=8, inn_nhid=100, ninp=4)
        pl_model = get_litmodel(hparams.model)
        pl_model = pl_model(backbone=model, reg_coef=hparams.reg_real)
    elif hparams.model == 'dampedregbaseline':
        model = get_model(hparams.model)(ninp=4)
        pl_model = get_litmodel(hparams.model)
        pl_model = pl_model(backbone=model, reg_coef=hparams.reg_real)
    else:
        model = get_model(hparams.model)(ninp=4)
        pl_model = get_litmodel(hparams.model)
        pl_model = pl_model(backbone=model)
elif hparams.experiment == 'double-pdl':
    train_x = np.loadtxt('data/train_x_v2.txt')
    train_dx = np.loadtxt('data/train_dx_v2.txt')
    dxs_max = np.abs(train_dx).max(axis=0)
    norm = np.diag(dxs_max)

    train_x = train_x @ np.linalg.inv(norm)
    train_dx = train_dx @ np.linalg.inv(norm)

    val_x, val_dx = general_double_pendulum(
        (0, 2 * np.pi), nb_teval=30, nb_samples=25, seed=hparams.val_seed
    )
    val_x = val_x @ np.linalg.inv(norm)
    val_dx = val_dx @ np.linalg.inv(norm)
    if hparams.model == 'invertiblennreg':
        model = get_model(hparams.model)(nblocks=8, inn_nhid=100, ninp=4, norm=True)
        pl_model = get_litmodel(hparams.model)(model, hparams.reg_real)
    else:
        model = get_model(hparams.model)(ninp=4, norm=True)
        pl_model = get_litmodel(hparams.model)(model)
elif hparams.experiment == 'pixel-pdl':
    def from_pickle(path):  # load something
        thing = None
        with open(path, 'rb') as handle:
            thing = pickle.load(handle)
        return thing
    data = from_pickle('data/pendulum-pixels-dataset.pkl')
    train_x = data['pixels']
    val_x = data['test_pixels']
    train_nextx = data['next_pixels']
    val_nextx = data['test_next_pixels']
    train_dx = train_nextx
    val_dx = val_nextx
    if hparams.model == 'pixelhnn':
        model = get_model(hparams.model)(ninp=train_x.shape[-1])
        pl_model = get_litmodel(hparams.model)(model)
    elif hparams.model == 'pixelreg':
        model = get_model(hparams.model)(ninp=train_x.shape[-1])
        pl_model = get_litmodel(hparams.model)(model, hparams.reg_real)
    else:
        raise NotImplementedError
elif hparams.experiment == 'damped-pixel-pdl':
    def from_pickle(path):  # load something
        thing = None
        with open(path, 'rb') as handle:
            thing = pickle.load(handle)
        return thing
    data = from_pickle('data/modified-pdl-pixels-dataset-test2.pkl')
    train_x = data['pixels']
    val_x = data['test_pixels']
    train_nextx = data['next_pixels']
    val_nextx = data['test_next_pixels']
    train_dx = train_nextx
    val_dx = val_nextx

    if hparams.model == 'dampedpixelreg':
        model = get_model(hparams.model)(ninp=train_x.shape[-1])
        pl_model = get_litmodel(hparams.model)(model, hparams.reg_real)
    else:
        raise NotImplementedError
else:
    raise NotImplementedError

train_x = torch.Tensor(train_x)
train_dx = torch.Tensor(train_dx)
val_x = torch.Tensor(val_x)
val_dx = torch.Tensor(val_dx)

train_dataset = TensorDataset(train_x, train_dx)
val_dataset = TensorDataset(val_x, val_dx)

train_dataloader = DataLoader(
    train_dataset, batch_size=hparams.batch, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=hparams.batch, num_workers=8, persistent_workers=True, pin_memory=True
)

logger = TensorBoardLogger(save_dir=hparams.logdir,
                           name=hparams.logname,
                           version=hparams.version,
                           default_hp_metric=False)
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(max_epochs=hparams.nb_epoch,
                     gpus=hparams.nb_gpus,
                     log_every_n_steps=10,
                     logger=logger,
                     gradient_clip_val=hparams.gradient_clip_val,
                     callbacks=[
                        ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss/original_loss'),
                        lr_monitor,
                    ])
trainer.fit(pl_model, train_dataloader, val_dataloader)
