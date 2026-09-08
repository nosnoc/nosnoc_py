# URDF robot model

`UrdfRobotModel` loads a fixed-base URDF with Pinocchio and exposes its
coordinates, joint metadata, limits, and symbolic mass matrix for use with
CasADi and nosnoc.

## Creating a model

```python
from pathlib import Path

from urdf_robot_model import UrdfRobotModel

urdf_path = Path("two_link.urdf")
robot = UrdfRobotModel(urdf_path)
```

The path is resolved to an absolute path and must point to an existing URDF
file. The current implementation supports models for which `nq == nv`.

## Public attributes

### Model and dimensions

| Attribute | Type | Description |
| --- | --- | --- |
| `urdf_path` | `pathlib.Path` | Absolute path to the loaded URDF. |
| `pinocchio_model` | `pinocchio.Model` | Numeric Pinocchio model. |
| `casadi_model` | `pinocchio.casadi.Model` | Pinocchio model using CasADi scalars. |
| `casadi_data` | Pinocchio data | Workspace associated with `casadi_model`. |
| `nq` | `int` | Number of generalized position coordinates. |
| `nv` | `int` | Number of generalized velocity coordinates. |

### Symbolic coordinates

| Attribute | Type | Description |
| --- | --- | --- |
| `q` | `casadi.SX` | Symbolic generalized-position vector of length `nq`. |
| `v` | `casadi.SX` | Symbolic generalized-velocity vector of length `nv`. |
| `q_names` | `list[str]` | Joint name corresponding to every entry of `q`. |
| `v_names` | `list[str]` | Joint name corresponding to every entry of `v`. |

<!-- For a joint with several coordinates, the names include an index such as
`joint_name[0]`. -->

### Original joint metadata

| Attribute | Type | Description |
| --- | --- | --- |
| `joint_types` | `dict[str, str]` | Maps every URDF joint name to its original type, including fixed joints. |
| `continuous_joint_names` | `list[str]` | Names of all joints declared as `continuous` in the original URDF. |

Continuous joints are converted to revolute joints internally so that
Pinocchio uses one position coordinate instead of a sine/cosine pair. The
attributes above preserve the original URDF meaning after that conversion.

Example:

```python
for joint_name, joint_type in robot.joint_types.items():
    print(f"{joint_name}: {joint_type}")
```

### Joint limits

| Attribute | Length | Description |
| --- | ---: | --- |
| `lower_position_limits` | `nq` | Lower bound for every position coordinate. |
| `upper_position_limits` | `nq` | Upper bound for every position coordinate. |
| `velocity_limits` | `nv` | Maximum absolute velocity for every velocity coordinate. |
| `effort_limits` | `nv` | Maximum absolute effort for every velocity coordinate. |

All four attributes are NumPy arrays. Effort means torque for a revolute or
continuous joint and force for a prismatic joint.

An absent finite bound is represented numerically by infinity. Continuous
joints therefore have position bounds `[-np.inf, np.inf]`. Missing velocity
or effort limits are represented by `np.inf`.

```python
print(robot.lower_position_limits)
print(robot.upper_position_limits)
print(robot.velocity_limits)
print(robot.effort_limits)
```

### Ready-to-use nosnoc CLS state bounds

Nosnoc's CLS state is ordered as `x = [q, v]`. The robot exposes bounds in
that order:

| Attribute | Length | Contents |
| --- | ---: | --- |
| `lbx` | `nq + nv` | `[lower_position_limits, -velocity_limits]` |
| `ubx` | `nq + nv` | `[upper_position_limits, velocity_limits]` |

They can be passed directly to `nosnoc.model.Cls`:

```python
import casadi as ca
import nosnoc

x = ca.vertcat(robot.q, robot.v)

cls_model = nosnoc.model.Cls(
    x=x,
    q=robot.q,
    v=robot.v,
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

<!-- Whether nosnoc enforces `lbx` and `ubx` at finite-element boundaries, RK
stage points, or both is controlled by its `x_box_at_fe` and `x_box_at_stg`
options. -->

### Mass matrix

| Attribute | Type | Description |
| --- | --- | --- |
| `M` | `casadi.SX` | Symbolic joint-space mass matrix evaluated at `q`. |
| `M_fun` | `casadi.Function` | Function mapping a numeric position vector to the numeric mass matrix. |

```python
import casadi as ca

q_value = ca.DM.zeros(robot.nq)
mass_matrix = robot.M_fun(q_value)
```

## Complete inspection example

Run `main.py` from this directory to print the currently exposed model data:

```powershell
python main.py
```

The example prints the dimensions, symbolic coordinates, coordinate-to-joint
mapping, original joint types, URDF limits, CLS state bounds, and mass matrix.
