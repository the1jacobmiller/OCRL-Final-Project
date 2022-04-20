import casadi as c
import matplotlib.pyplot as plt
import numpy as np


# Q = c.diag(c.MX([1.2, 10, 2, 4]))
# Qf = c.MX.eye(4) * 100
# R = c.diag(c.SX([3, 3]))

R = c.MX.eye(20)
Q = c.MX.eye(20)

# a = c.DM([[1,0.1],[0,1]])
# Q = c.diagcat(a, a, a, a, a, a, a, a, a, a)

opti = c.Opti()

x = opti.variable(40, 1)
u = opti.variable(40, 1)

Xref = np.zeros((40, 1))
Uref = np.zeros((40, 1))


# Q = c.MX(2, 2)

# c.kron(Q,Q)
# Q = c.diag(c.MX([[1.0, 0.1], [0.0, 1.0]]))

#print((x - Xref).T @ Q)