import osqp
import numpy as np
from scipy import sparse

def test_osqp_solve():
	P = sparse.csc_matrix([[4, 1], [1, 2]])
	q = np.array([1, 1])
	A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
	l = np.array([1, 0, 0])
	u = np.array([1, 0.7, 0.7])
	prob = osqp.OSQP()
	prob.setup(P, q, A, l, u, alpha=1.0)
	# x is the primal, y is the dual 
	res = prob.solve()
	assert res.info.status == "solved"
	assert res.info.run_time < 1

def test_ipopt_solve():
	assert True


if __name__ == '__main__':
	test_osqp_solve()