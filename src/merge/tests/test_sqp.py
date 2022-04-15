from unittest import result
import casadi
import numpy as np

def test_casadi_interface():
	opti = casadi.Opti()
	x = opti.variable()
	y = opti.variable()

	opti.minimize((y-x**2)**2)
	opti.subject_to(x**2 + y**2==1)
	opti.subject_to(x + y>=1)

	opti.solver('ipopt')
	sol = opti.solve()
	result_x, result_y = sol.value(x), sol.value(y)
	assert result_x == 0.78615
	assert result_y == 0.61803

if __name__ == '__main__':
	test_casadi_interface()