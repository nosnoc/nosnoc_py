import casadi as ca
import numpy as np
import json

from nosnoc import RKScheme, RKRepresentation
from nosnoc.rk import IntegralRKRepresentation, DifferentialRKRepresentation, LiftedDifferentialRKRepresentation

from .model import Pss
from .model import Cls
from .model import Dae as DaeModel
from .dcs import Stewart as StewartDCS
from .dcs import Heaviside as HeavisideDCS
from .dcs import Cls as ClsDCS
from .dcs import Dae as Dae
from .discrete_time_problem import Stewart as StewartDTP
from .discrete_time_problem import Heaviside as HeavisideDTP
from .discrete_time_problem import Cls as ClsDTP
from .discrete_time_problem import Ocp as OcpDTP

def _bound_kernel(x):
    idx, = np.where(np.isfinite(x))
    bound = x[idx]
    return idx,bound

    

def dump_hp_functions(model: DaeModel, opts, name):
    if isinstance(model, DaeModel):
        dcs = Dae(model)
    else:
        raise NotImplementedError("Only smooth DAEs can be dumped to the required structure exploiting functions.")

    if opts.rk_representation == RKRepresentation.INTEGRAL:
        rk = IntegralRKRepresentation(opts.n_s, opts.rk_scheme)
    elif opts.rk_representation == RKRepresentation.DIFFERENTIAL:
        rk = DifferentialRKRepresentation(opts.n_s, opts.rk_scheme)
        raise NotImplementedError("Only Integral rk representation supported")
    elif opts.rk_representation == RKRepresentation.DIFFERENTIAL_LIFT_X:
        rk = LiftedDifferentialRKRepresentation(opts.n_s, opts.rk_scheme)
        raise NotImplementedError("Only Integral rk representation supported")

    n_s = opts.n_s
    n_x = model.dims.n_x
    n_z = model.dims.n_z
    n_u = model.dims.n_u

    # TODO(@anton) params, global vars etc handling

    x_0 = ca.SX.sym("x_0", n_x)
    x_next = ca.SX.sym("x_0", n_x)
    # Build a single stage:
    # TODO(@anton) handle non-integral correctly
    x_rk = [ca.SX.sym(f"x_{jj+1}", n_x) for jj in range(n_s)]
    z_rk = [ca.SX.sym(f"z_{jj+1}", n_z) for jj in range(n_s)]
    k_rk = [ca.vertcat(x_ii_jj,z_ii_jj) for x_ii_jj,z_ii_jj in zip(x_rk,z_rk)]
    u_rk = ca.SX.sym(f"u", n_u)
    p_global = ca.SX.sym(f"p", model.dims.n_p_global)
    p_rk = ca.vertcat(u_rk,p_global)

    x_end, q_end, dynamic, algebraic = rk.collocation_constraints(
        x_0,
        k_rk,
        p_rk,
        opts.h,
        dcs.f_x_rk,
        dcs.f_q_rk,
        dcs.g_rk,
    )
    G = ca.vertcat(*dynamic,*algebraic)
    # TODO(@anton) fix this for non-integral
    C = ca.vertcat(*[dcs.g_path_rk(k_ii,p_rk) for k_ii in k_rk])

    idxlx, lx = _bound_kernel(model.lbx)
    idxux, ux = _bound_kernel(model.ubx)

    idxlz, lz = _bound_kernel(model.lbz)
    idxuz, uz = _bound_kernel(model.ubz)

    idxlu, lu = _bound_kernel(model.lbu)
    idxuu, uu = _bound_kernel(model.ubu)

    idxlt, lt = _bound_kernel(model.lbg_path)
    idxut, ut = _bound_kernel(model.ubg_path)

    idxlk, lk, idxuk, uk = create_k_bound(model, opts, idxlx, lx, idxlz, lz, idxux, ux, idxuz, uz)


    # TODO(include params etc)
    F_N = model.f_q_T
    
    mult_G = ca.SX.sym(f"mu", G.size1())
    mult_C = ca.SX.sym(f"zeta", C.size1(),C.size2())
    mult_next = ca.SX.sym(f"lambda", n_x)
    mult_prev = ca.SX.sym(f"lambda_prev", n_x)
    k_rk = ca.vertcat(*k_rk)
    F = q_end
    Fu = ca.jacobian(q_end,u_rk)
    Fk = ca.jacobian(q_end,k_rk)
    Fx_N = ca.jacobian(q_end,k_rk)
    L = q_end + ca.dot(G, mult_G) + ca.dot(C, mult_C)
    H,_ = ca.hessian(L, ca.vertcat(u_rk,k_rk))
    H_N,_ = ca.hessian(F_N, model.x)

    B = ca.jacobian(x_end, k_rk)
    
    B_fun = ca.Function("B", [], [B])
    G_fun = ca.Function("G", [x_0, u_rk, k_rk, p_global], [G])
    nablaG_fun = ca.Function("nablaG", [x_0, u_rk, k_rk, p_global], [ca.jacobian(G,x_0), ca.jacobian(G,u_rk), ca.jacobian(G,k_rk)])
    # TODO(@anton) implement generic path constraints
    C_fun = ca.Function("C", [u_rk, k_rk, p_global], [C])
    nablaC_fun = ca.Function("nablaC", [u_rk, k_rk, p_global], [ca.jacobian(C, ca.vertcat(u_rk, k_rk))])
    F_fun = ca.Function("F", [u_rk, k_rk, p_global], [F])
    F_N_fun = ca.Function("F_N", [model.x, model.p_global], [F_N])
    nablaF_fun = ca.Function("nablaF", [u_rk, k_rk, p_global], [Fu.T, Fk.T])
    nablaF_N_fun = ca.Function("nablaF_N", [model.x, model.p_global], [ca.jacobian(F_N,model.x).T])
    H_fun = ca.Function("H", [u_rk, k_rk, mult_G, mult_C, p_global], [H])
    H_N_fun = ca.Function("H_N", [model.x, model.p_global], [H_N])

    cg = ca.CodeGenerator(name, {})
    cg.add(B_fun)
    cg.add(G_fun)
    cg.add(C_fun)
    cg.add(F_fun)
    cg.add(F_N_fun)
    cg.add(nablaG_fun)
    cg.add(nablaC_fun)
    cg.add(nablaF_fun)
    cg.add(nablaF_N_fun)
    cg.add(H_fun)
    cg.add(H_N_fun)
    cg.generate()

    bound_data = {
        "idxlx":idxlx.astype(int).tolist(), "lx": lx.tolist(),
        "idxux":idxux.astype(int).tolist(), "ux": ux.tolist(),
        "idxlu":idxlu.astype(int).tolist(), "lu": lu.tolist(),
        "idxuu":idxuu.astype(int).tolist(), "uu": uu.tolist(),
        "idxlk":idxlk.astype(int).tolist(), "lk": lk.tolist(),
        "idxuk":idxuk.astype(int).tolist(), "uk": uk.tolist(),
        "idxlt":idxlt.astype(int).tolist(), "lt": lt.tolist(),
        "idxut":idxut.astype(int).tolist(), "ut": ut.tolist(),
    }
    
    with open(name+"_bounds.json", "w") as f_json:
        json.dump(bound_data, f_json)
        
    

def create_k_bound(model, opts, idxlx, lx, idxlz, lz, idxux, ux, idxuz, uz):
    n_s = opts.n_s
    n_x = model.dims.n_x
    n_z = model.dims.n_z
    # TODO(@anton) handle non
    if opts.x_box_at_stg:
        idxlk = np.concatenate([np.concatenate([np.atleast_1d(idxlx), np.atleast_1d(idxlz)]) + (ii*(n_x+n_z)) for ii in range(n_s)])
        lk = np.concatenate([np.concatenate([np.atleast_1d(lx), np.atleast_1d(lz)]) for ii in range(n_s)])
        idxuk = np.concatenate([np.concatenate([np.atleast_1d(idxux), np.atleast_1d(idxuz)]) + (ii*(n_x+n_z)) for ii in range(n_s)])
        uk = np.concatenate([np.concatenate([np.atleast_1d(ux), np.atleast_1d(uz)]) for ii in range(n_s)])
    else:
        idxlk = np.array([])
        lk = np.array([])
        idxuk = np.array([])
        uk = np.array([])
    return idxlk, lk, idxuk, uk
    
