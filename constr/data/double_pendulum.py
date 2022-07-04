import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
from tqdm import tqdm


def double_pendulum_dynamics(t: Optional[np.ndarray], y0: np.ndarray) -> np.ndarray:
    g = 9.8
    r1 = r2 = m1 = 1
    m2 = 1

    q, p, qt, pt = np.split(y0, 4, axis=-1)

    coef = r2 / r1 * (m2 / (m1 + m2))
    a1 = coef * np.cos(q - p)
    a2 = r1 / r2 * np.cos(q - p)
    f1 = -coef * (pt ** 2) * np.sin(q - p) - g / r1 * np.sin(q)
    f2 = r1 / r2 * (qt ** 2) * np.sin(q - p) - g / r2 * np.sin(p)

    qtt = (f1 - a1 * f2) / (1 - a1 * a2)
    ptt = (f2 - a2 * f1) / (1 - a1 * a2)

    return np.concatenate((qt, pt, qtt, ptt), axis=-1)

def general_double_pendulum(t_span: Tuple[float] = (0, 15), y0: Optional[np.ndarray] = None,
                            nb_teval: int = 30, seed: int = 1, nb_samples: int = 25) -> Tuple[np.ndarray]:
    np.random.seed(seed)

    t_eval = np.linspace(t_span[0], t_span[1], num=nb_teval)

    xs = []
    dxs = []
    for _ in tqdm(range(nb_samples)):
        y0 = np.random.randn(4)

        # get the points along the curve using numerical solver
        res = solve_ivp(double_pendulum_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-10, atol=1e-10)

        # get the gradient of each point obtained
        grad = [double_pendulum_dynamics(None, y) for y in res['y'].T]
        grad = np.stack(grad)

        xs.append(res['y'].T)
        dxs.append(grad)
        assert np.allclose(grad[:, :2], res['y'].T[:, 2:])

    xs = np.concatenate(xs)
    dxs = np.concatenate(dxs)

    return xs, dxs
