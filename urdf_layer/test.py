"""Minimal URDF-to-CasADi robot model built with Pinocchio."""

from pathlib import Path
from typing import List, Tuple, Union
import xml.etree.ElementTree as ET


import casadi as ca

try:
    import pinocchio as pin
    import pinocchio.casadi as cpin
except ImportError as exc:
    raise ImportError(
        "UrdfRobotModel requires Pinocchio with its CasADi bindings."
    ) from exc



def _make_coordinate_names(nq, nv, pinocchio_model) -> Tuple[List[str], List[str]]:
    """Map every entry of q and v to the URDF joint that owns it."""
    q_names = [""] * nq
    v_names = [""] * nv

    # Joint zero is Pinocchio's synthetic "universe" joint.
    for joint_id in range(1, pinocchio_model.njoints):
        joint = pinocchio_model.joints[joint_id]
        joint_name = pinocchio_model.names[joint_id]

        for local_index in range(joint.nq):
            suffix = "" if joint.nq == 1 else f"[{local_index}]"
            q_names[joint.idx_q + local_index] = joint_name + suffix

        for local_index in range(joint.nv):
            suffix = "" if joint.nv == 1 else f"[{local_index}]"
            v_names[joint.idx_v + local_index] = joint_name + suffix

    return q_names, v_names
def _build_model(urdf_path):
        """Build a model using one unwrapped angle for continuous joints."""
        urdf_tree = ET.parse(urdf_path)
        urdf_root = urdf_tree.getroot()
        continuous_joint_names = []

        # Turn continuous joints into revolut joints because pinnochio 
        # Associates 2 states q for continuous joints instead of just 1
        # This causes nq != nv, undesirable behavior
        for joint_element in urdf_root.findall("joint"):
            if joint_element.get("type") != "continuous":
                continue

            joint_name = joint_element.get("name")
            if joint_name is not None:
                continuous_joint_names.append(joint_name)

            # Revolut joints require at least upper and lower bounds to be valid 
            # Very large artificial position bounds preserve the unbounded meaning 
            joint_element.set("type", "revolute")
            limit_element = joint_element.find("limit")
            if limit_element is None:
                limit_element = ET.SubElement(joint_element, "limit")
            limit_element.set("lower", "-1e100")
            limit_element.set("upper", "1e100")
            # Some older versions also require effort and velocity
            if limit_element.get("effort") is None:
                limit_element.set("effort", "1e100")
            if limit_element.get("velocity") is None:
                limit_element.set("velocity", "1e100")

        urdf_xml = ET.tostring(urdf_root, encoding="unicode")
        return pin.buildModelFromXML(urdf_xml), continuous_joint_names

urdf_path = Path(__file__).with_name("spinner_friction.urdf")
urdf_path = Path(urdf_path).resolve()
if not urdf_path.is_file():
    raise FileNotFoundError(f"URDF file not found: {urdf_path}")

# The ordinary Pinocchio model provides the robot structure and names.

pinocchio_model, continuous_joint_names = _build_model(urdf_path)
nq = pinocchio_model.nq
nv = pinocchio_model.nv

print(f"nq: {nq}")
print(f"nv: {nv}")

# This is also the representation currently expected by nosnoc Cls.
# if nq != nv:
    # raise NotImplementedError(
    #     "This basic layer only supports robots for which nq == nv."
    # )

q = ca.SX.sym("q", nq)
v = ca.SX.sym("v", nv)

print(f"q:  {q}")
print(f"v:  {v}")

q_names, v_names = _make_coordinate_names(nq, nv, pinocchio_model)

print("\nCoordinate mapping:")
for index, name in enumerate(q_names):
    print(f"  q[{index}] -> {name}")
for index, name in enumerate(v_names):
    print(f"  v[{index}] -> {name}")


print("\nJoint types:")
urdf_root = ET.parse(urdf_path).getroot()
for joint_element in urdf_root.findall("joint"):
    joint_name = joint_element.get("name", "<unnamed>")
    joint_type = joint_element.get("type", "<unknown>")
    print(f"  {joint_name}: {joint_type}")

# Convert the numeric model to a model whose scalar type is CasADi SX.
casadi_model = cpin.Model(pinocchio_model)
casadi_data = casadi_model.createData()

# CRBA computes the upper triangular part of the joint-space inertia
# matrix, so mirror it to obtain the complete symmetric matrix M(q).
M_upper = cpin.crba(casadi_model, casadi_data, q)
M = M_upper + M_upper.T - ca.diag(ca.diag(M_upper))
M_fun = ca.Function("mass_matrix", [q], [M])


print("\nSymbolic mass matrix M(q):")
print(M)

q_zero = ca.DM.zeros(nq)
print("\nMass matrix at q = 0:")
print(M_fun(q_zero))
