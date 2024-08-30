# Import packages.
import cvxpy as cp
import numpy as np
from matrices import matAInv, A_ex_comp, q_comp, M_comp, first_derivatives, matPxx, matPxy, matPyy, matrices_H_f

t = 0

x1, y1 = 0, 0

x2, y2 = 1, 1

x3, y3 = -3,-4


a = [(1,2)]
b = [(3,4)]
c = [(5,6)]
d = [(6,7)]

x, y = [], []
x_bar, y_bar = [], []
x_bar_bar, y_bar_bar = [], []


for i in range(3):
    x.append(a[0][0] + b[0][0] * t + c[0][0] * (t**2) + d[0][0] * (t**3))
    y.append(a[0][1] + b[0][1] * t + c[0][1] * (t**2) + d[0][1] * (t**3))

    x_bar.append(b[0][0] + 2 * c[0][0] * t + 3 * d[0][0] * (t**2))
    y_bar.append(b[0][1] + 2 * c[0][1] * t + 3 * d[0][1] * (t**2))

    x_bar_bar.append(2 * c[0][0] + 6 * d[0][0] * t)
    y_bar_bar.append(2 * c[0][1] + 6 * d[0][1] * t)

x_bar_bar_bar = 6 * d[0][0]
y_bar_bar_bar = 6 * d[0][1]
# def PxCoeff(x_bar, y_bar):
#     return (y_bar)**2/(x_bar**2 + y_bar**2)**3

# def PxyCoeff(x_bar, y_bar):
#     return (y_bar)*(x_bar)*(-2)/(x_bar**2 + y_bar**2)**3

# def PyCoeff(x_bar, y_bar):
#     return (x_bar)**2/(x_bar**2 + y_bar**2)**3

# xx, yy, xy = [], [], []
# for i in range(3):
#     xx.append(PxCoeff(x_bar[i], y_bar[i]))
#     yy.append(PyCoeff(x_bar[i], y_bar[i]))
#     xy.append(PxyCoeff(x_bar[i], y_bar[i]))

def kappa(x_bar,y_bar,x_bar_bar, y_bar_bar):
    return (x_bar*y_bar_bar-y_bar*x_bar_bar)/(x_bar**2+y_bar**2)**1.5

def kappa_sq(x_bar,y_bar,x_bar_bar, y_bar_bar):
    return ((x_bar*y_bar_bar)**2-2*x_bar*x_bar_bar*y_bar*y_bar_bar+(y_bar*x_bar_bar)**2)/(x_bar**2+y_bar**2)**3

Pxx = matPxx(N: np.int32, x_dashed: x_bar, y_dashed: y_bar):
Pyy = matPyy(N: np.int32, x_dashed:x_bar , y_dashed: y_bar)
Pxy = matPxy(N: np.int32, x_dashed: x_bar, y_dashed: y_bar):

qx=q_comp(centreline= None, component=0)
qy=q_comp(centreline= None, component=1)

#qx = np.array([x1, x2, 0, 0, x2, x3, 0, 0, x3, x1, 0, 0])
#qy = np.array([y1, y2, 0, 0, y2, y3, 0, 0, y3, y1, 0, 0])

Aexc = A_ex_comp(2)
A_inv = matAInv(3)
Mx =M_comp(centreline=None, component=0)
My=M_comp(centreline=None, component=1)

Tc=2 * Aexc@A_inv
Tnx=2 * Aexc @ A_inv @ Mx
Tny=2 * Aexc @ A_inv @ My

fx = 2* Tnx.T @ Pxx.T @ Tc @ qx
fxy = Tny.T @ Pxy.T @ Tc @ qx + Tnx.T @ Pxy.T @ Tc @ qy
fy = 2*Tny.T @ Pyy.T @ Tc @ qy

Hx = Tnx.T @ Pxx @ Tnx
Hxy = Tny.T @ Pxy @ Tnx
Hy = Tny.T @ Pyy @ Tny



P = Hx + Hxy + Hy
q = fx + fxy + fy


alpha = cp.variable(3)

alpha_max = 5
alpha_min = -5

# Define and solve the CVXPY problem.

prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(alpha, P) + q.T @ alpha),
                 [alpha <= alpha_max,
                  alpha >= alpha_max])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)
