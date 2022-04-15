import casadi as c
import numpy as np

def test_casadi_interface():
	opti = c.Opti()
	x = opti.variable()
	y = opti.variable()

	opti.minimize((y-x**2)**2)
	opti.subject_to(x**2 + y**2==1)
	opti.subject_to(x + y>=1)

	opti.solver('ipopt')
	sol = opti.solve()
	result_x, result_y = sol.value(x), sol.value(y)

	assert np.allclose(result_x , 0.78615)
	assert np.allclose(result_y , 0.61803)

def test_sqp():
	pass

if __name__ == '__main__':
	test_casadi_interface()