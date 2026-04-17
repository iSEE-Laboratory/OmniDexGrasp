"""📐 Point cloud utilities for object scale computation."""
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import pycocotools.mask as mask_util
import trimesh

from utils.camera import CameraIntrinsics, dynamic_intrinsics


def decode_mask_rle(mask_rle: dict) -> np.ndarray:
    """🎭 Decode COCO RLE mask to binary array.

    Args:
        mask_rle: COCO RLE dict {"size": [h, w], "counts": str}.

    Returns:
        Binary mask (H, W) uint8.
    """
    rle = mask_rle.copy()
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_util.decode(rle)


def preprocess_depth(
    depth_uint16: np.ndarray,
    mask: np.ndarray,
    edge_erode_px: int = 3,
    bilateral_d: int = 5,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """🧹 Preprocess D435 depth: spatial smoothing + edge artifact removal.

    Equivalent to librealsense spatial_filter + mask erosion.
    Removes "flying pixels" at object boundaries before point cloud projection.

    Args:
        depth_uint16: Raw depth (H, W) uint16 in mm.
        mask: Binary object mask (H, W) uint8.
        edge_erode_px: Erode mask by N pixels to exclude boundary flying pixels.
        bilateral_d: bilateralFilter kernel diameter.
        bilateral_sigma_color: Depth value sigma in mm; controls edge preservation
            (pixels differing more than this are not blended). Typical: 50.0.
            ⚠️  Large values (>200) cause zero-depth holes to bleed into valid regions.
        bilateral_sigma_space: Space sigma (spatial smoothing radius, pixels).

    Returns:
        (filtered_depth_uint16, eroded_mask): Both (H, W).
    """
    # 🌊 Edge-preserving spatial smoothing (≈ librealsense spatial_filter)
    depth_f = depth_uint16.astype(np.float32)
    depth_f = cv2.bilateralFilter(depth_f, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    filtered = depth_f.astype(np.uint16)

    # ✂️ Erode mask to exclude boundary flying pixels
    if edge_erode_px > 0:
        kernel = np.ones((edge_erode_px * 2 + 1, edge_erode_px * 2 + 1), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
    else:
        eroded_mask = mask

    return filtered, eroded_mask


def depth_to_pointcloud(
    depth_path: Path,
    mask: np.ndarray,
    camera: CameraIntrinsics,
    depth_scale: float = 0.001,
    max_depth_m: float = 3.0,
    edge_erode_px: int = 3,
) -> np.ndarray:
    """🌊 Unproject masked depth to 3D point cloud via pinhole model.

    Applies 2D preprocessing (spatial filter + mask erosion) before projection.

    Args:
        depth_path: Path to depth.png (uint16, mm).
        mask: Binary object mask (H, W).
        camera: Camera intrinsic parameters.
        depth_scale: Depth unit conversion factor (default 0.001 = mm→meters).
        max_depth_m: Maximum valid depth in meters.
        edge_erode_px: Pixels to erode from mask edge (removes flying pixels).

    Returns:
        Point cloud (N, 3) float32 in meters.
    """
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    H, W = depth_raw.shape[:2]

    # 🔄 Resize mask if resolution mismatch
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # 🧹 2D preprocessing: spatial smoothing + mask erosion
    depth_filt, mask = preprocess_depth(depth_raw, mask, edge_erode_px=edge_erode_px)

    depth_m = depth_filt.astype(np.float32) * depth_scale
    cam = dynamic_intrinsics(camera, W, H)

    valid = (depth_m > 0) & (depth_m < max_depth_m) & np.isfinite(depth_m) & (mask > 0)

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_m[valid]
    x = (u[valid] - cam.ppx) * z / cam.fx
    y = (v[valid] - cam.ppy) * z / cam.fy

    return np.column_stack([x, y, z]).astype(np.float32)


def denoise_pointcloud(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """🧹 Remove statistical outliers from point cloud (open3d).

    Args:
        points: Input point cloud (N, 3).
        nb_neighbors: Neighbors used to compute mean distance.
        std_ratio: Points beyond mean + std_ratio*std are removed.

    Returns:
        Cleaned point cloud (M, 3).
    """
    if points.shape[0] == 0:
        return points
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, ind = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    return points[ind]


def compute_obj_scale(
    points: np.ndarray, mesh_path: Path
) -> tuple[float, float, float]:
    """📏 Compute object scale factor: pcd_max_extent / mesh_max_extent.

    Uses Minimal Oriented Bounding Box (OBB) for both point cloud and mesh.

    Args:
        points: Cleaned point cloud (M, 3).
        mesh_path: Path to base.obj mesh file.

    Returns:
        (scale_factor, pcd_max_extent, mesh_max_extent).
    """
    # 📐 Point cloud OBB
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd_obb = pcd.get_minimal_oriented_bounding_box()
    pcd_max_extent = float(np.max(pcd_obb.extent))

    # 📐 Mesh OBB
    mesh = trimesh.load(str(mesh_path), process=False)
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
        o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
    )
    mesh_obb = o3d_mesh.get_minimal_oriented_bounding_box()
    mesh_max_extent = float(np.max(mesh_obb.extent))

    scale_factor = 1.0 if mesh_max_extent == 0 else pcd_max_extent / mesh_max_extent
    return scale_factor, pcd_max_extent, mesh_max_extent
