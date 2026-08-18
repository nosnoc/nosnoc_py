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
    """
    Returns the indicies in g which are scalar functions of w, the complement of that, and the selection map from w which corresponds to the scalar indices.
    """
    Sym = type(w)
    if p is None:
        p = Sym([])
    #     g_sym
    # Get indices of all g which are linear in x.
    b_lin = np.array([ca.is_linear(gi,w) for gi in ca.vertsplit(g,1)])
    ind_linear, = np.nonzero(b_lin)
    g_linear = g[ind_linear]
    A, b = ca.linear_coeff(g_linear, w)
    # Find linear parts of g
    I,J = A.sparsity().get_triplet()
    v,i,c = np.unique(I, return_counts=True, return_index=True)
    # Find monomial parts of g
    ind_monomial = v[c==1]
    # Find monomial and scalar parts of g
    ind_mult1, = np.where(np.array([ca.is_equal(Ai,Sym(1.0)) for Ai in A.nonzeros()]))
    ind_scalar_monomial = np.intersect1d(ind_monomial, ind_mult1, assume_unique=True)
    # Find monomial and scalar parts of g with no offset
    # TODO(@anton) we can actually handle offsets by offsetting the bounds but for now we do not.
    ind_nonoffset = np.where(np.array([ca.is_equal(bi,Sym(0.0)) for bi in b.nonzeros()]))
    # Get scalar indices, map, and nonscalar via the A matrix and ind_linear
    ind_scalar_ = np.intersect1d(ind_scalar_monomial, ind_nonoffset, assume_unique=True)
    ind_map = np.array(J)[i[ind_scalar_]]
    ind_scalar = ind_linear[ind_scalar_]
    ind_nonscalar = np.setdiff1d(np.arange(0,g.size(1)),ind_scalar, assume_unique=True)

    return ind_scalar, ind_nonscalar, ind_map
