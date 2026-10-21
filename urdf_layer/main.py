"""Small manual test for the primitive URDF layer."""

from pathlib import Path

import numpy as np
import pinocchio as pin

from urdf_robot_model import UrdfRobotModel


HERE = Path(__file__).resolve().parent


def _as_numpy(value) -> np.ndarray:
    return np.array(value, dtype=float).reshape(
        value.shape if hasattr(value, "shape") else -1
    )


def _check(condition: bool, message: str) -> None:
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise AssertionError(message)


def inspect_robot(label: str, robot: UrdfRobotModel) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)

    print(robot)
    print(f"n_q: {robot.n_q}")
    print(f"n_v: {robot.n_v}")
    print(f"q:  {robot.q}")
    print(f"v:  {robot.v}")
    print(f"x:  {robot.x}")
    print(f"N shape: {robot.N.shape}")
    # print(f"N: {robot.N}")
    print(f"M shape: {robot.M.shape}")
    # print(f"M: {robot.M}")
    print(f"unit_constraint_slices: {robot.unit_constraint_slices}")
    print(f"g_unit_norm: {robot.g_unit_norm}")

    print("\nCoordinate mapping:")
    for index, name in enumerate(robot.q_names):
        print(f"  q[{index}] -> {name}")
    for index, name in enumerate(robot.v_names):
        print(f"  v[{index}] -> {name}")

    print("\nJoint types:")
    for joint_name, joint_type in robot.joint_types.items():
        print(f"  {joint_name}: {joint_type}")

    print("\nJoint limits:")
    print(f"  lower position: {robot.lower_position_limits}")
    print(f"  upper position: {robot.upper_position_limits}")
    print(f"  velocity:       {robot.velocity_limits}")
    print(f"  effort:         {robot.effort_limits}")

    print("\nCLS state bounds (x = [q, v]):")
    print(f"  lbx: {robot.lbx}")
    print(f"  ubx: {robot.ubx}")

    q_neutral = robot.q_neutral
    print("\nq_neutral:")
    print(q_neutral)

    N_neutral = _as_numpy(robot.N_fun(q_neutral))
    M_neutral = _as_numpy(robot.M_fun(q_neutral))
    print("\nN(q_neutral):")
    print(N_neutral)
    print("\nM(q_neutral):")
    print(M_neutral)

    if robot.g_unit_norm.numel() > 0:
        g_neutral = _as_numpy(robot.g_unit_norm_fun(q_neutral)).reshape(-1)
        print("\ng_unit_norm(q_neutral):")
        print(g_neutral)
    else:
        g_neutral = np.zeros(0)

    print("\nSanity checks:")
    _check(
        robot.N.shape == (robot.n_q, robot.n_v),
        f"N.shape == ({robot.n_q}, {robot.n_v})",
    )
    _check(
        robot.M.shape == (robot.n_v, robot.n_v),
        f"M.shape == ({robot.n_v}, {robot.n_v})",
    )
    _check(
        robot.x.shape == (robot.n_q + robot.n_v, 1),
        f"x.shape == ({robot.n_q + robot.n_v}, 1)",
    )
    _check(
        np.allclose(M_neutral, M_neutral.T, atol=1e-10),
        "M(q_neutral) is symmetric",
    )
    _check(
        len(robot.lbx) == robot.n_q + robot.n_v
        and len(robot.ubx) == robot.n_q + robot.n_v,
        "lbx/ubx have length n_q + n_v",
    )

    for start, stop in robot.unit_constraint_slices:
        norm = np.linalg.norm(q_neutral[start:stop])
        _check(
            np.isclose(norm, 1.0, atol=1e-10),
            f"||q[{start}:{stop}]|| == 1 at neutral (got {norm})",
        )
        N_block = N_neutral[start:stop, :]
        tangency = N_block.T @ q_neutral[start:stop]
        _check(
            np.allclose(tangency, 0.0, atol=1e-8),
            f"q_s^T N_block ~= 0 for slice [{start}:{stop}]",
        )

    if robot.g_unit_norm.numel() > 0:
        _check(
            np.allclose(g_neutral, 0.0, atol=1e-10),
            "g_unit_norm(q_neutral) ~= 0",
        )

    if robot.n_q == robot.n_v and not robot.unit_constraint_slices:
        _check(
            np.allclose(N_neutral, np.eye(robot.n_q), atol=1e-10),
            "N(q_neutral) ~= I for Euclidean n_q == n_v model",
        )

    # Finite-difference check: d/de integrate(q, e*v) at e=0 ~= N(q) v.
    rng = np.random.default_rng(0)
    v_sample = rng.normal(size=robot.n_v)
    eps = 1e-7
    q_plus = np.asarray(
        pin.integrate(robot.pinocchio_model, q_neutral, eps * v_sample),
        dtype=float,
    ).reshape(-1)
    fd = (q_plus - q_neutral) / eps
    Nv = N_neutral @ v_sample
    _check(
        np.allclose(fd, Nv, atol=1e-5, rtol=1e-4),
        "finite-difference integrate matches N(q) v",
    )


def main() -> None:
    # inspect_robot(
    #     "Fixed-base two_link.urdf",
    #     UrdfRobotModel(HERE / "two_link.urdf"),
    # )
    # inspect_robot(
    #     "URDF floating: free_body.urdf",
    #     UrdfRobotModel(HERE / "free_body.urdf"),
    # )
    inspect_robot(
        "cf2x.urdf with free_flyer=True",
        UrdfRobotModel(HERE / "cf2x.urdf", free_flyer=True),
    )

    # Zero-DOF URDFs without free_flyer should fail clearly.
    try:
        UrdfRobotModel(HERE / "cf2x.urdf")
        raise AssertionError("expected ValueError for cf2x without free_flyer")
    except ValueError as exc:
        print(f"\n[OK] cf2x without free_flyer raises: {exc}")


if __name__ == "__main__":
    main()
