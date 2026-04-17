"""🔭 Visualize reconstruction pointclouds with viser.

Usage:
    python scripts/vis_pcd.py ../out [--port 8080]
"""
import argparse
from pathlib import Path

import numpy as np
import viser

_DUMMY = np.zeros((1, 3), dtype=np.float32)


def _load(task_dir: Path) -> np.ndarray | None:
    """Load scene pointcloud from task output dir."""
    p = task_dir / "data" / "scene_pcd.npz"
    return np.load(p)["points"] if p.exists() else None


def _update(server: viser.ViserServer, scene: np.ndarray | None, point_size: float) -> None:
    """Overwrite scene point cloud (same name = in-place update)."""
    pts = scene if scene is not None else _DUMMY
    server.scene.add_point_cloud(
        "/scene_pcd",
        points=pts,
        colors=np.tile([100, 150, 255], (pts.shape[0], 1)).astype(np.uint8),
        point_size=point_size,
        visible=scene is not None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize task pointclouds")
    parser.add_argument("out_dir", type=Path, help="Path to out/ directory")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.001)
    args = parser.parse_args()

    tasks = sorted([d for d in args.out_dir.iterdir() if d.is_dir()])
    if not tasks:
        raise FileNotFoundError(f"❌ No task directories found in {args.out_dir}")

    names = [t.name for t in tasks]
    print(f"  📂 Found {len(tasks)} tasks: {names}")

    server = viser.ViserServer(host="localhost", port=args.port)
    dropdown = server.gui.add_dropdown("Task", options=names, initial_value=names[0])

    def show(task_name: str) -> None:
        scene = _load(args.out_dir / task_name)
        _update(server, scene, args.point_size)
        s = scene.shape[0] if scene is not None else 0
        print(f"  [{task_name}]  🔵 scene {s} pts")

    @dropdown.on_update
    def _(_) -> None:
        show(dropdown.value)

    show(names[0])
    print(f"\n🔭 viser [{len(tasks)} tasks] → open http://localhost:{args.port}")
    server.sleep_forever()


if __name__ == "__main__":
    main()
