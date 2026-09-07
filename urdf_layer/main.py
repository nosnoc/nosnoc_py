"""Small manual test for the primitive URDF layer."""

from pathlib import Path

import casadi as ca

from urdf_robot_model import UrdfRobotModel


def main() -> None:
    urdf_path = Path(__file__).with_name("two_link.urdf")
    # urdf_path = Path(__file__).with_name("spinner_friction.urdf")
    robot = UrdfRobotModel(urdf_path)

    print(robot)
    print(f"nq: {robot.nq}")
    print(f"nv: {robot.nv}")
    print(f"q:  {robot.q}")
    print(f"v:  {robot.v}")

    print("\nCoordinate mapping:")
    for index, name in enumerate(robot.q_names):
        print(f"  q[{index}] -> {name}")
    for index, name in enumerate(robot.v_names):
        print(f"  v[{index}] -> {name}")

    print("\nSymbolic mass matrix M(q):")
    print(robot.M)

    q_zero = ca.DM.zeros(robot.nq)
    print("\nMass matrix at q = 0:")
    print(robot.M_fun(q_zero))


if __name__ == "__main__":
    main()
