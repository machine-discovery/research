import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Optional
from tqdm import tqdm

def spring_dynamics(t: Optional[np.ndarray], y0: np.ndarray) -> np.ndarray:
    k = m = 1
    q, qt = np.split(y0, 2, axis=-1)
    qtt = - k / m * q
    return np.concatenate((qt, qtt), axis=-1)

def general_spring(t_span: Tuple[float] = (0, 15), y0: Optional[np.ndarray] = None,
                   nb_teval: int = 30, seed: int = 1, nb_samples: int = 25,
                   noiseless: bool = False) -> Tuple[np.ndarray]:
    np.random.seed(seed)

    t_eval = np.linspace(t_span[0], t_span[1], num=nb_teval)

    xs = []
    dxs = []
    for _ in tqdm(range(nb_samples)):
        y0 = np.random.randn(2)

        # get the points along the curve using numerical solver
        res = solve_ivp(spring_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-10)

        # get the gradient of each point obtained
        grad = [spring_dynamics(None, y) for y in res['y'].T]
        grad = np.stack(grad)

        xs.append(res['y'].T)
        dxs.append(grad)
        assert np.allclose(grad[:, 0], res['y'].T[:, 1])

    xs = np.concatenate(xs)
    if not noiseless:
        xs = xs + np.random.randn(*xs.shape) * 0.1
    dxs = np.concatenate(dxs)
    return xs, dxs
