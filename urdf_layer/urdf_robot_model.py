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


class UrdfRobotModel:
    """Load a fixed-base URDF and expose its basic symbolic dynamics."""

    def __init__(self, urdf_path: Union[str, Path]):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        self.pinocchio_model = self._build_model()
        self.nq = self.pinocchio_model.nq
        self.nv = self.pinocchio_model.nv

        # This is the representation currently expected by nosnoc Cls.
        if self.nq != self.nv:
            raise NotImplementedError(
                "This basic layer only supports robots for which nq == nv."
            )

        self.q = ca.SX.sym("q", self.nq)
        self.v = ca.SX.sym("v", self.nv)

        self.q_names, self.v_names = self._make_coordinate_names()

        # Convert the numeric model to a model whose scalar type is CasADi SX.
        self.casadi_model = cpin.Model(self.pinocchio_model)
        self.casadi_data = self.casadi_model.createData()

        # Pinocchio guarantees the upper triangular part of the CRBA result,
        # while some bindings also populate its lower triangular part.  Keep
        # only the guaranteed half before mirroring it; otherwise an already
        # populated off-diagonal entry would be added twice.
        M_raw = cpin.crba(self.casadi_model, self.casadi_data, self.q)
        M_upper = ca.triu(M_raw)
        self.M = M_upper + M_upper.T - ca.diag(ca.diag(M_upper))
        self.M_fun = ca.Function("mass_matrix", [self.q], [self.M])

    def _build_model(self):
        """Build a model using one unwrapped angle for continuous joints."""
        urdf_tree = ET.parse(self.urdf_path)
        urdf_root = urdf_tree.getroot()
        self.continuous_joint_names = []

        # Turn continuous joints into revolut joints because pinnochio 
        # Associates 2 states q for continuous joints instead of just 1
        # This causes nq != nv, undesirable behavior
        for joint_element in urdf_root.findall("joint"):
            if joint_element.get("type") != "continuous":
                continue

            joint_name = joint_element.get("name")
            if joint_name is not None:
                self.continuous_joint_names.append(joint_name)

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
        return pin.buildModelFromXML(urdf_xml)

    def _make_coordinate_names(self) -> Tuple[List[str], List[str]]:
        """Map every entry of q and v to the URDF joint that owns it."""
        q_names = [""] * self.nq
        v_names = [""] * self.nv

        # Joint zero is Pinocchio's synthetic "universe" joint.
        for joint_id in range(1, self.pinocchio_model.njoints):
            joint = self.pinocchio_model.joints[joint_id]
            joint_name = self.pinocchio_model.names[joint_id]

            for local_index in range(joint.nq):
                suffix = "" if joint.nq == 1 else f"[{local_index}]"
                q_names[joint.idx_q + local_index] = joint_name + suffix

            for local_index in range(joint.nv):
                suffix = "" if joint.nv == 1 else f"[{local_index}]"
                v_names[joint.idx_v + local_index] = joint_name + suffix

        return q_names, v_names

    def __repr__(self) -> str:
        return (
            f"UrdfRobotModel(name={self.pinocchio_model.name!r}, "
            f"nq={self.nq}, nv={self.nv})"
        )
