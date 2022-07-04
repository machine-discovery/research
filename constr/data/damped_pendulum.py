import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
from tqdm import tqdm

def damped_pendulum_dynamics(t: Optional[np.ndarray], y0: np.ndarray) -> np.ndarray:
    g = 1
    q, qt = np.split(y0, 2, axis=-1)

    qtt = -g * np.sin(q) - 0.05 * qt
    return np.concatenate((qt, qtt), axis=-1)

def general_damped_pendulum(t_span: Tuple[float] = (0, 15), y0: Optional[np.ndarray] = None,
                            nb_teval: int = 30, seed: int = 1, nb_samples: int = 25,
                            noiseless: bool = False) -> Tuple[np.ndarray]:
    np.random.seed(seed)

    t_eval = np.linspace(t_span[0], t_span[1], num=nb_teval)

    xs = []
    dxs = []
    for _ in tqdm(range(nb_samples)):
        y0 = np.random.randn(2)

        # get the points along the curve using numerical solver
        res = solve_ivp(damped_pendulum_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-10)

        theta, omega = np.split(res['y'].T, 2, axis=-1)
        x = np.sin(theta)
        y = -np.cos(theta)
        dx = omega * np.cos(theta)
        dy = omega * np.sin(theta)
        ddx = -omega ** 2 * np.sin(theta) - np.sin(theta) * np.cos(theta) - dx * 0.05
        ddy = omega ** 2 * np.cos(theta) - np.sin(theta) ** 2 - dy * 0.05

        inp = np.concatenate([x, y, dx, dy], axis=-1)
        grad = np.concatenate([dx, dy, ddx, ddy], axis=-1)

        xs.append(inp)
        dxs.append(grad)

    xs = np.concatenate(xs)
    if not noiseless:
        xs = xs + np.random.randn(*xs.shape) * 0.1
    dxs = np.concatenate(dxs)
    return xs, dxs
