"""Base class for robot hand forward kinematics and retargeting."""

from pathlib import Path
import torch
import trimesh
import pytorch3d.transforms
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
from csdf import compute_sdf, index_vertices_by_faces


def _get_convex_path(mesh_dir: Path, filename: str) -> Path:
    """Return convex hull mesh path if exists, else original."""
    stem = Path(filename).stem
    for suffix in [".convex.STL", ".convex.stl"]:
        candidate = mesh_dir / (stem + suffix)
        if candidate.exists():
            return candidate
    return mesh_dir / filename


class RobotHandModel:
    """Base class for robot hand models — FK, mesh generation, penetration."""

    def __init__(
        self,
        urdf_path: Path,
        mesh_dir: Path,
        fingertip_links: list[str],
        device: str = "cuda",
        use_convex: bool = True,
    ):
        self.device = device
        self.use_convex = use_convex
        self.fingertip_links = fingertip_links
        self.mesh: dict = {}

        self.chain = self._load_urdf(urdf_path)
        self._build_mesh(urdf_path, mesh_dir)
        self._parse_joint_limits(urdf_path)

    def _load_urdf(self, urdf_path: Path) -> pk.Chain:
        with open(urdf_path, "rb") as f:
            chain = pk.build_chain_from_urdf(f.read())
        return chain.to(device=self.device)

    def _build_mesh(self, urdf_path: Path, mesh_dir: Path) -> None:
        device = self.device

        def build_recurse(body):
            link_name = body.link.name
            link_verts, link_faces = [], []
            n_verts = 0

            for visual in body.link.visuals:
                if visual.geom_type is None:
                    continue
                scale = torch.tensor([1.0, 1.0, 1.0], device=device)

                if visual.geom_type == "sphere":
                    link_mesh = trimesh.primitives.Sphere(radius=float(visual.geom_param))
                elif visual.geom_type == "mesh":
                    param = visual.geom_param
                    if isinstance(param, tuple):
                        mesh_path_str, scale_param = param
                        if scale_param is not None:
                            scale = torch.tensor(scale_param, device=device)
                    else:
                        mesh_path_str = param

                    filename = Path(mesh_path_str).name
                    if self.use_convex:
                        path = _get_convex_path(mesh_dir, filename)
                    else:
                        path = mesh_dir / filename
                    link_mesh = trimesh.load_mesh(str(path), process=False)
                else:
                    raise ValueError(f"Unsupported geometry type for link {link_name}")

                verts = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                pos = visual.offset.to(device)
                verts = pos.transform_points(verts * scale)
                link_verts.append(verts)
                link_faces.append(faces + n_verts)
                n_verts += len(verts)

            if link_verts:
                lv = torch.cat(link_verts, dim=0)
                lf = torch.cat(link_faces, dim=0)
                self.mesh[link_name] = dict(
                    vertices=lv,
                    faces=lf,
                    face_verts=index_vertices_by_faces(lv, lf),
                )

            for child in body.children:
                build_recurse(child)

        build_recurse(self.chain._root)

    def _parse_joint_limits(self, urdf_path: Path) -> None:
        root_xml = ET.parse(str(urdf_path)).getroot()
        limits = {}
        for joint in root_xml.findall("joint"):
            name = joint.get("name")
            tag = joint.find("limit")
            if tag is not None:
                limits[name] = (float(tag.get("lower", -1e9)), float(tag.get("upper", 1e9)))
            else:
                limits[name] = (float("-inf"), float("inf"))

        self.joints_names, self.joints_lower, self.joints_upper = [], [], []

        def recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                lo, hi = limits.get(body.joint.name, (float("-inf"), float("inf")))
                self.joints_lower.append(lo)
                self.joints_upper.append(hi)
            for child in body.children:
                recurse(child)

        recurse(self.chain._root)
        self.joints_lower = torch.tensor(self.joints_lower, device=self.device)
        self.joints_upper = torch.tensor(self.joints_upper, device=self.device)

    def _compute_penetration(self, obj_pc, global_t, global_R, current_status):
        """Compute max SDF penetration of object points vs each hand link."""
        # Transform object points to robot wrist frame
        x = (obj_pc - global_t.unsqueeze(1)) @ global_R  # (B, N, 3)

        pen_list = []
        for link_name in self.mesh:
            if link_name in self.fingertip_links:
                continue  # Skip fingertips
            mat = current_status[link_name].get_matrix()  # (B, 4, 4)
            x_local = (x - mat[:, :3, 3].unsqueeze(1)) @ mat[:, :3, :3]
            x_local = x_local.reshape(-1, 3)
            dis, _, signs, _, _ = compute_sdf(x_local, self.mesh[link_name]["face_verts"])
            dis = dis * (-signs)  # positive = inside mesh (penetrating), negative = outside
            pen_list.append(dis.reshape(x.shape[0], x.shape[1]))

        return torch.max(torch.stack(pen_list), dim=0)[0]  # (B, N)

    def forward(self, hand_pose, object_pc=None, with_penetration=False, include_fingertip_mesh=True):
        """FK: (B, 3+3+N_joints) → vertices, faces, fingertip_keypoints [, penetration].

        Returns dict with keys: vertices, faces, fingertip_keypoints, penetration (optional).
        """
        B = hand_pose.shape[0]
        global_t = hand_pose[:, :3]
        global_R = pytorch3d.transforms.axis_angle_to_matrix(hand_pose[:, 3:6])  # (B, 3, 3)

        joint_pose = self._get_joint_pose(hand_pose[:, 6:])
        current_status = self.chain.forward_kinematics(joint_pose)

        hand: dict = {}

        if object_pc is not None and with_penetration:
            hand["penetration"] = self._compute_penetration(obj_pc=object_pc,
                                                             global_t=global_t,
                                                             global_R=global_R,
                                                             current_status=current_status)

        # Aggregate link vertices into world frame (optionally skip fingertip link spheres)
        link_names = [ln for ln in self.mesh
                      if include_fingertip_mesh or ln not in self.fingertip_links]
        verts_list = [
            current_status[ln].transform_points(self.mesh[ln]["vertices"]).expand(B, -1, -1)
            for ln in link_names
        ]
        hand["vertices"] = (torch.cat(verts_list, dim=1) @ global_R.transpose(1, 2)
                            + global_t.unsqueeze(1))

        # Build global face index
        offset, faces_list = 0, []
        for ln in link_names:
            faces_list.append(self.mesh[ln]["faces"] + offset)
            offset += self.mesh[ln]["vertices"].shape[0]
        hand["faces"] = torch.cat(faces_list, dim=0)

        # Fingertip keypoints
        tips = []
        for ln in self.fingertip_links:
            tip = (current_status[ln].transform_points(torch.zeros(1, 3, device=self.device))
                   @ global_R.transpose(1, 2) + global_t.unsqueeze(1))
            tips.append(tip)
        hand["fingertip_keypoints"] = torch.cat(tips, dim=1)  # (B, 5, 3)

        return hand

    def _get_joint_pose(self, joint_pose):
        """Hook for subclasses (e.g., Inspire mimic joints)."""
        return joint_pose

    def mano2robot_batch(
        self, mano_trans: torch.Tensor, mano_axis_angle: torch.Tensor, mano_pose: torch.Tensor
    ) -> torch.Tensor:
        """Map MANO parameters to robot-specific dex_pose. Must be overridden by subclasses."""
        raise NotImplementedError

    def __call__(self, hand_pose, object_pc=None, with_penetration=False, include_fingertip_mesh=True):
        return self.forward(hand_pose, object_pc, with_penetration, include_fingertip_mesh)
