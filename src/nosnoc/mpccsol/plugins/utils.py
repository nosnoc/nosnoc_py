import casadi as ca
import numpy as np
 
def ind2sub(array_shape, ind):
    rows = (ind.astype('int') // array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

# TODO(@anton) This may be slower but it is actually correct.
#              We should move this directly into CasADi.

# TODO(@anton) This may or may not work for ca.MX :/

def find_nonscalar(g,w,p=None):
    if p is None:
        p = type(w)([])
    #     g_sym
    # Get indices of all g which are linear in x.
    b_lin = np.array([ca.is_linear(gi,w) for gi in ca.vertsplit(g,1)])
    ind_linear, = np.nonzero(b_lin)
    g_linear = g[ind_linear]
    A, b = ca.linear_coeff(g_linear, w)
    # Find exactly scalar
    I,J = A.sparsity().get_triplet()
    v,i,c = np.unique(I, return_counts=True, return_index=True)
    ind_monomial = v[c==1]
    ind_mult1, = np.where(np.array(ca.DM(A).nonzeros()) == 1.0)
    ind_scalar_monomial = np.intersect1d(ind_monomial, ind_mult1, assume_unique=True)
    ind_nonoffset,_ = np.where(ca.DM(b).full() == 0.0)
    ind_scalar_ = np.intersect1d(ind_scalar_monomial, ind_nonoffset, assume_unique=True)
    ind_map = np.array(J)[i[ind_scalar_]]
    ind_scalar = ind_linear[ind_scalar_]
    ind_nonscalar = np.setdiff1d(np.arange(0,g.size(1)),ind_scalar, assume_unique=True)

    return ind_scalar, ind_nonscalar, ind_map

