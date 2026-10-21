"""Minimal URDF-to-CasADi robot model built with Pinocchio."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
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

# frozen set is an immutable set, so it's faster to iterate over than a list
SUPPORTED_URDF_JOINT_TYPES = frozenset(
    {
        "fixed",
        "revolute",
        "continuous",
        "prismatic",
        "floating",
        "planar",
    }
)

# Joints whose configuration includes a unit-norm embedding (quaternion or
# planar cos/sin). 
# Scalar URDF position limits must not be applied to those coordinates.
_MANIFOLD_URDF_TYPES = frozenset({"floating", "planar"})
# Planar has n_q = 4 [x, y, cos(θ), sin(θ)], I might want to make it 
# n_q = 3 [x, y, θ] instead, but that would require a custom Pinocchio joint type.

class UrdfRobotModel:
    """Load a URDF and expose its basic symbolic dynamics."""

    def __init__(self, urdf_path: Union[str, Path], *, free_flyer: bool = False):
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        self.free_flyer = free_flyer
        self.pinocchio_model = self._build_model()
        self.n_q = self.pinocchio_model.nq
        self.n_v = self.pinocchio_model.nv

        if self.n_q == 0:
            raise ValueError(
                f"URDF {self.urdf_path.name!r} has no degrees of freedom "
                "(nq = 0). Add a URDF joint with type='floating', or pass "
                "free_flyer=True to prepend a free-flying root."
            )

        # Stores a safe default configuration for the robot as a flat NumPy vector.
        # It's the configuration where the robot is at rest, with all joints at their neutral position.
        # This is useful for initializating the robot's state, might need it later
        # Can't just initialize to all zeros because that's not a valid configuration 
        # for some joints (e.g. floating joints)
        self.q_neutral = np.asarray(
            pin.neutral(self.pinocchio_model), dtype=float
        ).flatten()

        self.q = ca.SX.sym("q", self.n_q)
        self.v = ca.SX.sym("v", self.n_v)
        self.x = ca.vertcat(self.q, self.v)

        # Map every entry of q and v to the URDF joint that owns it
        # Basically if I want to know which joint a particular state belongs to,
        # I do q_names[index] it will return the joint name
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

        self.unit_constraint_slices = self._make_unit_constraint_slices()
        self.g_unit_norm = self._make_g_unit_norm()
        # Passes the unit norm constraints as a function of q as a CasADi function
        self.g_unit_norm_fun = ca.Function(
            "g_unit_norm", [self.q], [self.g_unit_norm]
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

        # Ambient kinematic map q_dot = N(q) v for nosnoc CLS.
        self.N = self._make_kinematic_map()
        self.N_fun = ca.Function("N", [self.q], [self.N])

    def _build_model(self):
        """Build a model using one unwrapped angle for continuous joints."""
        # Might want to add unwrapped angle for planar joints later
        
        urdf_tree = ET.parse(self.urdf_path)
        urdf_root = urdf_tree.getroot()
        self.continuous_joint_names = []
        self.joint_types = {}
        self._original_joint_limits = {}

        # Turn continuous joints into revolut joints because pinnochio 
        # Associates 2 states q for continuous joints instead of just 1
        for joint_element in urdf_root.findall("joint"):
            joint_name = joint_element.get("name")
            joint_type = joint_element.get("type")
            if joint_name is None or joint_type is None:
                raise ValueError("Every URDF joint must have a name and type.")

            if joint_element.find("mimic") is not None:
                raise ValueError(
                    f"Joint {joint_name!r} uses a <mimic> tag, which is not "
                    "supported by UrdfRobotModel."
                )

            if joint_type not in SUPPORTED_URDF_JOINT_TYPES:
                supported = ", ".join(sorted(SUPPORTED_URDF_JOINT_TYPES))
                raise ValueError(
                    f"Unsupported URDF joint type {joint_type!r} on joint "
                    f"{joint_name!r}. Supported types: {supported}."
                )

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

            # Revolute joints require at least upper and lower bounds.
            # Very large artificial position bounds preserve the unbounded meaning.
            joint_element.set("type", "revolute")
            limit_element = joint_element.find("limit")
            if limit_element is None:
                limit_element = ET.SubElement(joint_element, "limit")
            limit_element.set("lower", "-1e100")
            limit_element.set("upper", "1e100")
            # Some older versions also require effort and velocity.
            if limit_element.get("effort") is None:
                limit_element.set("effort", "1e100")
            if limit_element.get("velocity") is None:
                limit_element.set("velocity", "1e100")

        has_floating_urdf = any(
            joint_type == "floating" for joint_type in self.joint_types.values()
        )
        if self.free_flyer and has_floating_urdf:
            raise ValueError(
                "free_flyer=True cannot be combined with a URDF that already "
                "declares a floating joint."
            )

        urdf_xml = ET.tostring(urdf_root, encoding="unicode")
        if self.free_flyer:
            # Prepend universe --FreeFlyer--> URDF root link. Does not rewrite
            # any fixed joints in the file.
            model = pin.buildModelFromXML(urdf_xml, pin.JointModelFreeFlyer())
            self.joint_types["root_joint"] = "floating"
            self._original_joint_limits["root_joint"] = {
                "lower": None,
                "upper": None,
                "velocity": None,
                "effort": None,
            }
            return model

        return pin.buildModelFromXML(urdf_xml)

    def _make_coordinate_names(self) -> Tuple[List[str], List[str]]:
        """Map every entry of q and v to the URDF joint that owns it."""
        q_names = [""] * self.n_q
        v_names = [""] * self.n_v

        # Joint zero is Pinocchio's synthetic "universe" joint.
        for joint_id in range(1, self.pinocchio_model.njoints):
            joint = self.pinocchio_model.joints[joint_id]
            joint_name = self.pinocchio_model.names[joint_id]
            if not joint_name:
                joint_name = f"joint_{joint_id}"

            for local_index in range(joint.nq):
                suffix = "" if joint.nq == 1 else f"[{local_index}]"
                q_names[joint.idx_q + local_index] = joint_name + suffix

            for local_index in range(joint.nv):
                suffix = "" if joint.nv == 1 else f"[{local_index}]"
                v_names[joint.idx_v + local_index] = joint_name + suffix

        return q_names, v_names

    def _unit_slice_for_joint(
        self, joint_name: str, joint
    ) -> Optional[Tuple[int, int]]:
        """Return the (start, stop) q-slice that must stay unit-norm, if any."""
        joint_type = self.joint_types[joint_name]
        # Floating: q = [xyz(3), quat(4)].
        if joint_type == "floating" and joint.nq == 7 and joint.nv == 6:
            start = joint.idx_q + 3
            return (start, start + 4)
        # Planar: q = [x, y, cos, sin] in modern Pinocchio.
        if joint_type == "planar" and joint.nq == 4 and joint.nv == 3:
            start = joint.idx_q + 2
            return (start, start + 2)
        return None

    def _make_unit_constraint_slices(self) -> List[Tuple[int, int]]:
        slices: List[Tuple[int, int]] = []
        for joint_id in range(1, self.pinocchio_model.njoints):
            joint = self.pinocchio_model.joints[joint_id]
            joint_name = self.pinocchio_model.names[joint_id]
            unit_slice = self._unit_slice_for_joint(joint_name, joint)
            if unit_slice is not None:
                slices.append(unit_slice)
        return slices

    def _make_g_unit_norm(self) -> ca.SX:
        residuals = []
        for start, stop in self.unit_constraint_slices:
            q_s = self.q[start:stop]
            residuals.append(ca.dot(q_s, q_s) - 1.0)
        if not residuals:
            return ca.SX(0, 1)
        return ca.vertcat(*residuals)

    def _make_limit_vectors(self):
        """Return limits from the original URDF, aligned with q and v."""
        lower_position_limits = np.full(self.n_q, -np.inf)
        upper_position_limits = np.full(self.n_q, np.inf)
        velocity_limits = np.full(self.n_v, np.inf)
        effort_limits = np.full(self.n_v, np.inf)

        # Joint zero is Pinocchio's synthetic "universe" joint.
        for joint_id in range(1, self.pinocchio_model.njoints):
            joint = self.pinocchio_model.joints[joint_id]
            joint_name = self.pinocchio_model.names[joint_id]
            joint_type = self.joint_types[joint_name]
            limits = self._original_joint_limits.get(
                joint_name,
                {
                    "lower": None,
                    "upper": None,
                    "velocity": None,
                    "effort": None,
                },
            )

            # Scalar URDF limits only apply to single-DoF Euclidean joints.
            # Manifold embeddings (quaternion, planar cos/sin) stay unbounded.
            if (
                joint_type not in _MANIFOLD_URDF_TYPES
                and joint_type != "continuous"
                and joint_type != "spherical"
                and joint.nq == 1
            ):
                if limits["lower"] is not None:
                    lower_position_limits[joint.idx_q] = limits["lower"]
                if limits["upper"] is not None:
                    upper_position_limits[joint.idx_q] = limits["upper"]

            # Velocity / effort: only broadcast a scalar when n_v == 1.
            if limits["velocity"] is not None and joint.nv == 1:
                velocity_limits[joint.idx_v] = abs(limits["velocity"])
            if limits["effort"] is not None and joint.nv == 1:
                effort_limits[joint.idx_v] = abs(limits["effort"])

        return (
            lower_position_limits,
            upper_position_limits,
            velocity_limits,
            effort_limits,
        )

    def _make_kinematic_map(self) -> ca.SX:
        """Build N(q) = d(integrate(q, v))/dv at v = 0."""
        q_next = cpin.integrate(self.casadi_model, self.q, self.v)
        N_full = ca.jacobian(q_next, self.v)
        return ca.substitute(N_full, self.v, ca.SX.zeros(self.n_v))

    def __repr__(self) -> str:
        return (
            f"UrdfRobotModel(name={self.pinocchio_model.name!r}, "
            f"n_q={self.n_q}, n_v={self.n_v}, free_flyer={self.free_flyer})"
        )
