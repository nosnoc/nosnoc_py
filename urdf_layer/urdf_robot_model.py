"""Minimal URDF-to-CasADi robot model built with Pinocchio."""

from pathlib import Path
from typing import List, Tuple, Union
import xml.etree.ElementTree as ET

import casadi as ca
import numpy as np

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
        (
            self.lower_position_limits,
            self.upper_position_limits,
            self.velocity_limits,
            self.effort_limits,
        ) = self._make_limit_vectors()
        # nosnoc CLS orders its differential state as x = [q, v].
        self.lbx = np.concatenate(
            (self.lower_position_limits, -self.velocity_limits)
        )
        self.ubx = np.concatenate(
            (self.upper_position_limits, self.velocity_limits)
        )

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
        self.joint_types = {}
        self._original_joint_limits = {}

        # Turn continuous joints into revolut joints because pinnochio 
        # Associates 2 states q for continuous joints instead of just 1
        # This causes nq != nv, undesirable behavior
        for joint_element in urdf_root.findall("joint"):
            joint_name = joint_element.get("name")
            joint_type = joint_element.get("type")
            if joint_name is None or joint_type is None:
                raise ValueError("Every URDF joint must have a name and type.")

            # Store the original URDF type before continuous joints are
            # converted for Pinocchio's internal representation.
            self.joint_types[joint_name] = joint_type
            limit_element = joint_element.find("limit")
            self._original_joint_limits[joint_name] = {
                attribute: (
                    float(limit_element.get(attribute))
                    if limit_element is not None
                    and limit_element.get(attribute) is not None
                    else None
                )
                for attribute in ("lower", "upper", "velocity", "effort")
            }

            if joint_type != "continuous":
                continue

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

    def _make_limit_vectors(self):
        """Return limits from the original URDF, aligned with q and v."""
        lower_position_limits = np.full(self.nq, -np.inf)
        upper_position_limits = np.full(self.nq, np.inf)
        velocity_limits = np.full(self.nv, np.inf)
        effort_limits = np.full(self.nv, np.inf)

        # Joint zero is Pinocchio's synthetic "universe" joint.
        for joint_id in range(1, self.pinocchio_model.njoints):
            joint = self.pinocchio_model.joints[joint_id]
            joint_name = self.pinocchio_model.names[joint_id]
            joint_type = self.joint_types[joint_name]
            limits = self._original_joint_limits[joint_name]

            if joint_type != "continuous":
                if limits["lower"] is not None:
                    lower_position_limits[joint.idx_q : joint.idx_q + joint.nq] = limits["lower"]
                if limits["upper"] is not None:
                    upper_position_limits[joint.idx_q : joint.idx_q + joint.nq] = limits["upper"]

            if limits["velocity"] is not None:
                velocity_limits[joint.idx_v : joint.idx_v + joint.nv] = abs(limits["velocity"])
            if limits["effort"] is not None:
                effort_limits[joint.idx_v : joint.idx_v + joint.nv] = abs(limits["effort"])

        return (
            lower_position_limits,
            upper_position_limits,
            velocity_limits,
            effort_limits,
        )

    def __repr__(self) -> str:
        return (
            f"UrdfRobotModel(name={self.pinocchio_model.name!r}, "
            f"nq={self.nq}, nv={self.nv})"
        )
