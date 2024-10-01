import numpy as np
import jax.numpy as jnp
from jax import jit

class Centreline:
    """
    members:
    int N
    float p[N][2]
    float n[N][2]
    (x_i -> p[i, 0], y_i -> p[i, 1])
    """
    def __init__(self, N, p, half_track_width, vehicle_width):
        self.N = np.uint32(N)
        self.p = jnp.array(np.array(p, dtype=np.float32))
        self.half_w_tr = jnp.array(np.array(half_track_width, dtype=np.float32))
        self.w_veh = jnp.float32(vehicle_width)
        self.n = None

    def calc_n(self, x_derivatives, y_derivatives):
        self.n = jnp.stack([y_derivatives, -x_derivatives], axis=-1)
        self.n /= jnp.linalg.norm(self.n, axis=1, keepdims=True)