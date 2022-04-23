import casadi as c
import matplotlib.pyplot as plt
import numpy as np



opti = c.Opti()

n = 40
m = 20

x = opti.variable(n, 1)
u = opti.variable(m, 1)

Xref = np.zeros((n, 1))
Uref = np.zeros((m, 1))

Q = c.MX.eye(n)
Qf = c.MX.eye(n)
R = c.MX.eye(m)

a = c.DM([[1,0.1],[0,1]])
A = c.diagcat(a, a)

for _ in range(2, 20):
    A = c.diagcat(A, a)

b = c.DM([[0],[0.1]])
B = c.diagcat(b, b)

for _ in range(2, 20):
    B = c.diagcat(B, b)

print (A@x + B @ u)


# xdot = B @ u
# print(xdot)


