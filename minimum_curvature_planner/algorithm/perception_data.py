import numpy as np

class Centreline:
    """
    members:
    int N
    float p[N][2]
    float n[N][2]
    (x_i -> p[i, 0], y_i -> p[i, 1])
    """
    def __init__(self, N, p, n, half_track_width, vehicle_width):
        self.N = np.uint32(N)
        self.p = np.array(p, dtype=np.float64)
        self.n = np.array(n, dtype=np.float64)
        self.half_w_tr = np.array(half_track_width, dtype=np.float64)
        self.w_veh = np.float64(vehicle_width)
