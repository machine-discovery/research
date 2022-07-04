import numpy as np
import torch
from data.double_pendulum import general_double_pendulum
from data.single_pendulum import general_pendulum
from data.mass_spring import general_spring
from typing import Tuple

def calc_mass_spring_energy(x: np.ndarray) -> np.ndarray:
    k = m = 1
    q, qt = np.split(x, 2, axis=-1)
    T = 0.5 * m * qt ** 2
    V = 0.5 * k * q ** 2
    return T + V

def calc_single_pdl_energy(x: np.ndarray) -> np.ndarray:
    g = 3
    m = r = 1
    q, qt = np.split(x, 2, axis=-1)
    T = 0.5 * m * (r * qt) ** 2
    V = m * g * r * (1 - np.cos(q))
    return T + V

def calc_damped_pdl_energy(x: np.ndarray) -> np.ndarray:
    g = m = r = 1
    x, y, dx, dy = np.split(x, 4, axis=-1)
    T = 0.5 * m * (r ** 2 ) * (dx ** 2 + dy ** 2)
    V = m * g * r * (1 + y)
    return T + V

def calc_double_pdl_energy(x: np.ndarray) -> np.ndarray:
    g = 9.8  # fix g to be 9.8
    r1 = r2 = m1 = m2 = 1
    q, p, qt, pt = np.split(x, 4, axis=-1)
    T = 0.5 * m1 * (r1 ** 2) * (qt ** 2)
    T = T + 0.5 * m2 * (r1 ** 2) * (qt ** 2)
    T = T + 0.5 * m2 * (r2 ** 2) * (pt ** 2)
    T = T + 0.5 * m2 * 2 * r1 * r2 * qt * pt * np.cos(q - p)
    V = -(m1 + m2) * g * r1 * np.cos(q) - m2 * g * r2 * np.cos(p)
    return T + V

def generate_init_state(self, seed: int, experiment: str) -> Tuple[torch.Tensor]:
    if experiment == 'mass-spring':
        x, _ = general_spring(
            t_span=(0, 100), nb_teval=1000, seed=seed, nb_samples=1, noiseless=True
        )
        x0 = torch.Tensor(x[0]).reshape(1, 1, -1)
        t = torch.linspace(0, 100, 1000)[None, :, None, None].permute(1, 0, 2, 3)
    elif experiment == 'single-pdl':
        x, _ = general_pendulum(
            t_span=(0, 100), nb_teval=1000, seed=seed, nb_samples=1, noiseless=True
        )
        x0 = torch.Tensor(x[0]).reshape(1, 1, -1)
        t = torch.linspace(0, 100, 1000)[None, :, None, None].permute(1, 0, 2, 3)
    elif experiment == 'double-pdl':
        x, _ = general_double_pendulum(
            t_span=(0, 50), nb_teval=2000, seed=seed, nb_samples=1
        )
        x0 = torch.Tensor(x[0]).reshape(1, 1, -1)
        t = torch.linspace(0, 50, 2000)[None, :, None, None].permute(1, 0, 2, 3)
    return x0, t, x
