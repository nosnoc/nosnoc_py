"""Display URDF configurations and trajectories in Meshcat.

Trajectory columns are Pinocchio configurations: shape (model.nq, frames).
Import URDFViewer to reuse a loaded model from another Python module.
"""

from pathlib import Path
import time

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


class URDFViewer:
    """Load a fixed-base URDF once and update its pose or play trajectories."""

    def __init__(self, urdf_path: str | Path) -> None:
        urdf_path = Path(urdf_path).resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.visual_model = pin.buildGeomFromUrdf(
            self.model, str(urdf_path), pin.GeometryType.VISUAL,
            package_dirs=str(urdf_path.parent),
        )
        collision_model = pin.buildGeomFromUrdf(
            self.model, str(urdf_path), pin.GeometryType.COLLISION,
            package_dirs=str(urdf_path.parent),
        )
        self.visualizer = MeshcatVisualizer(
            self.model, collision_model, self.visual_model,
        )
        # Automatic browser launching interrupts loading on this Windows setup.
        self.visualizer.initViewer(open=False)
        self.visualizer.viewerRootNodeName = "pinocchio"
        self.visualizer.viewerVisualGroupName = "pinocchio/visuals"

        # Avoid loadViewerModel()/display() mesh-scaling issues in this
        # Pinocchio environment by sending ordinary NumPy transforms ourselves.
        for geometry in self.visual_model.geometryObjects:
            self.visualizer.loadViewerGeometryObject(geometry, pin.GeometryType.VISUAL)

        self.data = self.model.createData()
        self.visual_data = self.visual_model.createData()
        self.display(pin.neutral(self.model))
        print(f"Displaying: {urdf_path}")
        print(f"Open this URL: {self.visualizer.viewer.url()}")

    def display(self, q: np.ndarray | list[float]) -> None:
        """Show one configuration, without sleeping (suitable for an MPC loop)."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            raise ValueError(f"Expected {self.model.nq} finite configuration values.")

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(
            self.model, self.data, self.visual_model, self.visual_data,
        )
        for geometry_id, geometry in enumerate(self.visual_model.geometryObjects):
            node_name = self.visualizer.getViewerNodeName(geometry, pin.GeometryType.VISUAL)
            world_transform = np.array(
                self.visual_data.oMg[geometry_id].homogeneous, dtype=float, copy=True,
            )
            world_transform[:3, :3] *= np.asarray(geometry.meshScale, dtype=float)
            self.visualizer.viewer[node_name].set_transform(world_transform)

    def play(self, q_traj: np.ndarray, dt: float) -> None:
        """Play once, then return with the last pose visible.

        q_traj has shape (model.nq, frames), matching the drone example.
        dt is the positive sample interval in seconds. Playback blocks the
        caller; use display(q) for individual updates inside an MPC loop.
        Configurations must use Pinocchio's joint ordering/representation
        (radians for the two-link robot's revolute joints).
        """
        q_traj = np.asarray(q_traj, dtype=float)
        if (q_traj.ndim != 2 or q_traj.shape[0] != self.model.nq
                or q_traj.shape[1] == 0 or not np.all(np.isfinite(q_traj))):
            raise ValueError(
                f"Expected a finite trajectory of shape ({self.model.nq}, frames) "
                "with at least one frame."
            )
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and greater than zero.")

        start = time.perf_counter()
        for k in range(q_traj.shape[1]):
            # Account for rendering time instead of adding it to every dt.
            delay = start + k * dt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self.display(q_traj[:, k])

    def close(self) -> None:
        """Close the Meshcat viewer connection when finished."""
        window = self.visualizer.viewer.window
        if callable(getattr(window, "close", None)):
            window.close()
        else:
            # This Meshcat version exposes Visualizer.close(), but its
            # ViewerWindow has no close method.
            window.zmq_socket.close(linger=0)
            if window.server_proc is not None:
                window.server_proc.terminate()
                window.server_proc.wait()


def main() -> None:
    """Repeat a smooth two-link test trajectory until Ctrl+C."""
    dt = 0.02
    duration = 6.0
    t = np.arange(0.0, duration, dt)
    q_traj = np.vstack((
        0.8 * np.sin(2.0 * np.pi * t / duration),
        1.2 * np.sin(2.0 * np.pi * t / duration + np.pi / 2.0),
    ))

    viewer = URDFViewer(Path(__file__).with_name("two_link.urdf"))
    print("Repeating the test trajectory. Open the URL above; press Ctrl+C to stop.")
    try:
        while True:
            viewer.play(q_traj, dt)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
