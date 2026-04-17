"""🎨 Visualize H2R retargeting results with viser.

Usage:
    cd omnidexgrasp
    conda activate omnidexgrasp
    python scripts/vis_dexgrasp.py
    python scripts/vis_dexgrasp.py --output ../out --assets ../assets/robo --port 8080
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))          # for _viser_utils
sys.path.append(str(Path(__file__).parent.parent))   # for human2robo

import argparse
from datetime import datetime

import trimesh
import viser
from PIL import Image

from _viser_utils import _ROT90X, load_tasks, get_hand_mesh, parse_entry

# Hand color presets (RGBA, alpha=180)
HAND_COLORS = {
    "Teal":   (100, 210, 230, 180),
    "Pink":   (230, 130, 140, 180),
    "Sage":   (160, 210, 185, 180),
    "Silver": (190, 190, 195, 180),
    "Blue":   (0,   100, 255, 180),
    "Orange": (255, 140,   0, 180),
}


def main():
    parser = argparse.ArgumentParser(description="Visualize H2R dexterous grasp results")
    parser.add_argument("--output",    default="../out",          help="Output directory")
    parser.add_argument("--assets",    default="../assets/robo",  help="Robot hand assets")
    parser.add_argument("--mano-root", default="../assets/mano",  help="MANO assets root")
    parser.add_argument("--port",      type=int, default=8080)
    args = parser.parse_args()

    output_dir  = Path(args.output)
    assets_root = Path(args.assets)
    tasks = load_tasks(output_dir)

    if not tasks:
        print("❌ No tasks with robo.json found!"); return

    try:
        from human2robo.dataloader import RetargetDataLoader
        dataloader = RetargetDataLoader(
            output_dir=output_dir, n_obj_pts=1, device="cpu",
            mano_assets_root=args.mano_root,
        )
    except Exception:
        dataloader = None

    server = viser.ViserServer(port=args.port)
    print(f"🌐 Viser: http://localhost:{args.port} | {len(tasks)} tasks")

    # Scene root frame rotated 90° CCW around X so objects face forward
    server.scene.add_frame("/scene", wxyz=_ROT90X, show_axes=False)

    task_names = list(tasks.keys())
    hand_types = [ht for ht in ["inspire", "wuji", "shadow"]
                  if any(ht in r for r in tasks.values())]

    state: dict = {
        "task": task_names[0],
        "hand": hand_types[0] if hand_types else "",
        "show_init": False,
        "show_mano": False,
        "hand_color": "Teal",
    }
    handles: dict = {}

    task_slider = server.gui.add_slider(
        "Task Index", min=0, max=len(task_names) - 1, step=1, initial_value=0
    )
    task_dd    = server.gui.add_dropdown("Task",       options=task_names,          initial_value=task_names[0])
    hand_dd    = server.gui.add_dropdown("Hand Type",  options=hand_types,          initial_value=hand_types[0])
    color_dd   = server.gui.add_dropdown("🎨 Hand Color", options=list(HAND_COLORS), initial_value="Teal")
    show_init  = server.gui.add_checkbox("Show Init Hand",  initial_value=False)
    show_mano  = server.gui.add_checkbox("Show MANO Hand",  initial_value=False, disabled=dataloader is None)
    shot_btn   = server.gui.add_button("📸 Screenshot")
    info_md    = server.gui.add_markdown("*Select task and hand type*")

    def refresh():
        task_name = state["task"]
        hand_type = state["hand"]
        robo      = tasks[task_name]

        # Remove previous mesh handles (keep /scene frame)
        for h in handles.values():
            h.remove()
        handles.clear()

        # Object mesh under /scene so it inherits the X rotation
        mesh_path = output_dir / task_name / "scaled_mesh.obj"
        if mesh_path.exists():
            handles["object"] = server.scene.add_mesh_trimesh("/scene/object", trimesh.load(str(mesh_path)))

        if hand_type in robo:
            entry = robo[hand_type]
            init_pose, final_pose = parse_entry(entry)

            # Init robot hand (orange) — before optimization
            if state["show_init"]:
                init_mesh = get_hand_mesh(hand_type, init_pose, assets_root)
                if init_mesh is not None:
                    init_mesh.visual.vertex_colors = [255, 140, 0, 80]  # orange, semi-transparent
                    handles["hand_init"] = server.scene.add_mesh_trimesh("/scene/hand_init", init_mesh)

            # Final robot hand — after optimization, color from state
            final_mesh = get_hand_mesh(hand_type, final_pose, assets_root)
            if final_mesh is not None:
                final_mesh.visual.vertex_colors = list(HAND_COLORS[state["hand_color"]])
                handles["hand_final"] = server.scene.add_mesh_trimesh("/scene/hand_final", final_mesh)

            # MANO hand mesh in obj_cam frame via dataloader
            if state["show_mano"] and dataloader is not None:
                task_data = dataloader.load(task_name)
                if task_data is not None:
                    mano_verts = task_data.mano_verts_obj[0].cpu().numpy()
                    mano_faces = dataloader.mano_faces.cpu().numpy()
                    mano_mesh  = trimesh.Trimesh(vertices=mano_verts, faces=mano_faces)
                    mano_mesh.visual.vertex_colors = [0, 200, 0, 80]   # green, semi-transparent
                    handles["hand_mano"] = server.scene.add_mesh_trimesh("/scene/hand_mano", mano_mesh)

            mano_part = " &nbsp; 🟢 MANO" if state["show_mano"] else ""
            legend = ("🟠 init &nbsp; 🔵 final" if state["show_init"] else "🔵 final") + mano_part
            pose_str = ", ".join(f"{v:.3f}" for v in final_pose)
            info_md.content = (
                f"**{task_name}** | `{hand_type}`\n\n"
                f"{legend}\n\n"
                f"**dex_pose final** ({len(final_pose)} dims):\n\n"
                f"```\n{pose_str}\n```"
            )
        else:
            info_md.content = f"⚠️ No {hand_type} result"

    def on_state(key: str):
        def handler(e): state[key] = e.target.value; refresh()
        return handler

    @task_slider.on_update
    def _(e):
        idx = int(e.target.value)
        if task_names[idx] == state["task"]:
            return   # already in sync, skip
        state["task"] = task_names[idx]
        task_dd.value = task_names[idx]   # sync dropdown
        refresh()

    @task_dd.on_update
    def _(e):
        if e.target.value == state["task"]:
            return   # already in sync, skip
        state["task"] = e.target.value
        task_slider.value = task_names.index(e.target.value)  # sync slider
        refresh()

    hand_dd.on_update(on_state("hand"))
    show_init.on_update(on_state("show_init"))
    show_mano.on_update(on_state("show_mano"))
    color_dd.on_update(on_state("hand_color"))

    @shot_btn.on_click
    def _(_):
        clients = server.get_clients()
        if not clients:
            print("⚠️  No client connected"); return
        client = next(iter(clients.values()))
        img_arr = client.get_render(height=1080, width=1920, transport_format="png")
        ts = datetime.now().strftime("%H%M%S")
        save_dir = output_dir / "screenshots"
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / f"{state['task']}_{ts}.png"
        Image.fromarray(img_arr).save(save_path)
        print(f"📸 Saved: {save_path}")

    refresh()
    print("🎨 Ready. Press Ctrl+C to exit.")
    server.sleep_forever()


if __name__ == "__main__":
    main()
