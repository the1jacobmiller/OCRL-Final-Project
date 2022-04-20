import casadi as c
import matplotlib.pyplot as plt
import numpy as np


# Q = c.diag(c.MX([1.2, 10, 2, 4]))
# Qf = c.MX.eye(4) * 100
# R = c.diag(c.SX([3, 3]))

# R = c.MX.eye(20)
# Q = c.MX.eye(20)

# a = c.DM([[1,0.1],[0,1]])
# Q = c.diagcat(a, a, a, a, a, a, a, a, a, a)

opti = c.Opti()

x = opti.variable(40, 1)
u = opti.variable(40, 1)

Xref = np.zeros((40, 1))
Uref = np.zeros((40, 1))

n = 40
m = 40

# n = len(Xref_flattened)
# m = len(Uref_flattened)

x = opti.variable(m,1)
u = opti.variable(n,1)

R = c.MX.eye(n)
Q = c.MX.eye(m)
Qf = c.MX.eye(m)

# q = c.DM([[1,0.1],[0,1]])
# A = c.diagcat(q, q, q, q, q, q, q, q, q, q)

stage_cost = (x - Xref).T @ Q @ (x - Xref) + u.T @ R @ u
term_cost = (x - Xref).T @ Qf @ (x - Xref)

print(x[:,-1])


# Q = c.MX(2, 2)

# c.kron(Q,Q)
# Q = c.diag(c.MX([[1.0, 0.1], [0.0, 1.0]]))

#print((x - Xref).T @ Q)