# URDF robot model

`UrdfRobotModel` loads a URDF with Pinocchio and exposes its coordinates,
kinematic map `N(q)`, joint metadata, limits, unit-norm helpers, and symbolic
mass matrix for use with CasADi and nosnoc CLS models.

## Creating a model

```python
from pathlib import Path

from urdf_robot_model import UrdfRobotModel

urdf_path = Path("two_link.urdf")
robot = UrdfRobotModel(urdf_path)

# Body-only URDFs (e.g. Crazyflie cf2x): prepend a free-flying root.
drone = UrdfRobotModel(Path("cf2x.urdf"), free_flyer=True)
```

The path is resolved to an absolute path and must point to an existing URDF
file.

Floating bases can be created in two ways:

1. Declare a `floating` (or `planar`) joint in the URDF itself.
2. Pass `free_flyer=True` to prepend Pinocchio's `JointModelFreeFlyer` from
   the universe to the URDF root link. Fixed joints in the file (props, frames)
   are **not** rewritten. This cannot be combined with a URDF that already
   declares a `floating` joint.

A URDF with no movable joints and `free_flyer=False` raises a clear error
(`nq = 0`). This layer still does not parse custom tags such as Crazyflie
`<properties>` or build thrust/`f_v`; those remain application-side.

## Supported URDF joint types

| URDF type | Pinocchio `(nq, nv)` (typical) | Notes |
| --- | --- | --- |
| `fixed` | `(0, 0)` | Collapsed; recorded in `joint_types` only. |
| `revolute` | `(1, 1)` | Standard Euclidean joint. |
| `continuous` | rewritten to `(1, 1)` | Converted to revolute with huge artificial limits so Pinocchio does not use a sin/cos pair. Original type is kept in metadata. |
| `prismatic` | `(1, 1)` | Standard Euclidean joint. |
| `floating` | `(7, 6)` | Free-flyer: `xyz` + quaternion. Needs `N(q)`. |
| `planar` | `(4, 3)` | `x, y, cos, sin` in modern Pinocchio. Needs `N(q)`. |

Joints with a `<mimic>` tag are rejected. Unknown joint types are rejected.

## Public attributes

### Model and dimensions

| Attribute | Type | Description |
| --- | --- | --- |
| `urdf_path` | `pathlib.Path` | Absolute path to the loaded URDF. |
| `free_flyer` | `bool` | Whether a free-flyer root was prepended in code. |
| `pinocchio_model` | `pinocchio.Model` | Numeric Pinocchio model. |
| `casadi_model` | `pinocchio.casadi.Model` | Pinocchio model using CasADi scalars. |
| `casadi_data` | Pinocchio data | Workspace associated with `casadi_model`. |
| `n_q` | `int` | Number of generalized position coordinates. |
| `n_v` | `int` | Number of generalized velocity coordinates. |
| `q_neutral` | `numpy.ndarray` | Neutral configuration from `pin.neutral` (use this, not zeros, for freeflyers). |

### Symbolic coordinates and maps

| Attribute | Type | Description |
| --- | --- | --- |
| `q` | `casadi.SX` | Symbolic generalized-position vector of length `nq`. |
| `v` | `casadi.SX` | Symbolic generalized-velocity vector of length `nv`. |
| `x` | `casadi.SX` | `vertcat(q, v)`, the CLS state ordering. |
| `N` | `casadi.SX` | Kinematic map with shape `(nq, nv)` such that `q_dot = N(q) v`. |
| `N_fun` | `casadi.Function` | Numeric evaluator `q -> N(q)`. |
| `q_names` | `list[str]` | Joint name corresponding to every entry of `q`. |
| `v_names` | `list[str]` | Joint name corresponding to every entry of `v`. |

`N(q)` is built as the CasADi Jacobian of `pinocchio.casadi.integrate(model, q, v)`
with respect to `v`, evaluated at `v = 0`. For Euclidean robots this is the
identity. For freeflyers it is the usual 7×6 map (translation + quaternion
kinematics).

### Original joint metadata

| Attribute | Type | Description |
| --- | --- | --- |
| `joint_types` | `dict[str, str]` | Maps every URDF joint name to its original type, including fixed joints. |
| `continuous_joint_names` | `list[str]` | Names of all joints declared as `continuous` in the original URDF. |

### Joint limits

| Attribute | Length | Description |
| --- | ---: | --- |
| `lower_position_limits` | `nq` | Lower bound for every position coordinate. |
| `upper_position_limits` | `nq` | Upper bound for every position coordinate. |
| `velocity_limits` | `nv` | Maximum absolute velocity for every velocity coordinate. |
| `effort_limits` | `nv` | Maximum absolute effort for every velocity coordinate. |

All four attributes are NumPy arrays. An absent finite bound is `±np.inf`.

Scalar URDF position limits are applied only to single-DoF Euclidean joints
(`revolute` / `prismatic`). Quaternion and planar `(cos, sin)` coordinates are
left unbounded. Continuous joints keep unbounded position limits even though
they are rewritten internally as revolute.

### Ready-to-use nosnoc CLS state bounds

Nosnoc's CLS state is ordered as `x = [q, v]`:

| Attribute | Length | Contents |
| --- | ---: | --- |
| `lbx` | `nq + nv` | `[lower_position_limits, -velocity_limits]` |
| `ubx` | `nq + nv` | `[upper_position_limits, velocity_limits]` |

```python
import casadi as ca
import nosnoc

cls_model = nosnoc.model.Cls(
    x=robot.x,
    q=robot.q,
    v=robot.v,
    N=robot.N,  # required when nq != nv; fine to always pass
    lbx=robot.lbx,
    ubx=robot.ubx,
    M=robot.M,
    f_v=f_v,  # Generalized-force expression supplied by the application.
    f_c=f_c,  # Contact-gap expression supplied by the application.
    e=0.0,
)
```

The layer intentionally does not expose `lbu` or `ubu`. A URDF effort limit
cannot by itself determine the control bounds without knowing which joints
are actuated and how the control vector maps to generalized forces.

This layer also does not build `f_v` (gravity, Coriolis, actuation) or contact
gaps `f_c`. Those remain application-supplied.

### Mass matrix

| Attribute | Type | Description |
| --- | --- | --- |
| `M` | `casadi.SX` | Symbolic joint-space mass matrix evaluated at `q`, shape `(nv, nv)`. |
| `M_fun` | `casadi.Function` | Function mapping a numeric position vector to the numeric mass matrix. |

```python
mass_matrix = robot.M_fun(robot.q_neutral)
```

### Unit-norm helpers (quaternions / planar embeddings)

Floating and planar joints embed orientations with unit-norm coordinates.
Correct `N(q)` keeps those norms instantaneously invariant, but long NLP solves
may still drift. Helpers:

| Attribute | Type | Description |
| --- | --- | --- |
| `unit_constraint_slices` | `list[tuple[int, int]]` | Index ranges in `q` that should satisfy `||q_s|| = 1`. |
| `g_unit_norm` | `casadi.SX` | Column of residuals `||q_s||^2 - 1` (empty if none). |
| `g_unit_norm_fun` | `casadi.Function` | Numeric evaluator of `g_unit_norm`. |

Initialize freeflyer orientations from `robot.q_neutral` (identity quaternion),
not from zeros. Optionally pass `g_unit_norm` into a nosnoc path/equality
constraint later if drift becomes an issue.

## Complete inspection example

Run `main.py` from this directory to exercise fixed-base, URDF-floating, and
`cf2x` with `free_flyer=True`:

```powershell
python main.py
```

The harness prints dimensions, maps, limits, `N`, `M`, unit-norm helpers, and
runs shape / symmetry / finite-difference sanity checks.
