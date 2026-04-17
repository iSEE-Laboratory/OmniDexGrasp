# =============================================================================
# Wrapper: Panda3dBatchRenderer (Sequential In-Process Mode)
# =============================================================================
# Source:   thirdparty/megapose6d/src/megapose/panda3d_renderer/panda3d_batch_renderer.py
# Issue:    https://github.com/megapose6d/megapose6d/issues/66
# Problem:  torch.multiprocessing worker processes deadlock at _out_queue.get()
#           when running alongside other CUDA code. Confirmed deadlock (not slow rendering).
# Fix:      Monkey-patch Panda3dBatchRenderer to render sequentially in main process,
#           eliminating cross-process IPC queue entirely.
# Usage:    import recons.panda3d_batch_renderer_wrapper  # before load_named_model
# =============================================================================
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from megapose.panda3d_renderer.panda3d_batch_renderer import (
    BatchRenderOutput,
    CameraRenderingData,
    Panda3dBatchRenderer,
)
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer


def _init_renderers(self, preload_cache: bool) -> None:
    """Create Panda3dSceneRenderer in main process instead of spawning subprocesses."""
    import panda3d.core as p3d
    # Panda3D resolves textures via model-path (not relative to .obj), so add each mesh dir
    for obj in self._object_dataset.list_objects:
        p3d.get_model_path().prepend_directory(str(obj.mesh_path.parent))

    object_labels = [obj.label for obj in self._object_dataset.list_objects]
    preload_labels = set(object_labels) if preload_cache else set()
    self._renderer = Panda3dSceneRenderer(
        asset_dataset=self._object_dataset,
        preload_labels=preload_labels,
    )


def _render(
    self,
    labels: List[str],
    TCO: torch.Tensor,
    K: torch.Tensor,
    light_datas,
    resolution,
    render_depth: bool = False,
    render_mask: bool = False,
    render_normals: bool = False,
) -> BatchRenderOutput:
    """Render sequentially in main process, bypassing IPC queue entirely."""
    if render_mask:
        raise NotImplementedError

    scene_datas = self.make_scene_data(labels, TCO, K, light_datas, resolution)
    bsz = len(scene_datas)
    list_rgbs: List[Optional[torch.Tensor]] = [None] * bsz
    list_depths: List[Optional[torch.Tensor]] = [None] * bsz
    list_normals: List[Optional[torch.Tensor]] = [None] * bsz

    for n, sd in enumerate(scene_datas):
        is_valid = (
            np.isfinite(sd.object_datas[0].TWO.toHomogeneousMatrix()).all()
            and np.isfinite(sd.camera_data.TWC.toHomogeneousMatrix()).all()
            and np.isfinite(sd.camera_data.K).all()
        )
        if is_valid:
            r = self._renderer.render_scene(
                object_datas=sd.object_datas,
                camera_datas=[sd.camera_data],
                light_datas=sd.light_datas,
                render_normals=render_normals,
                render_depth=render_depth,
                copy_arrays=True,
            )[0]
        else:
            h, w = sd.camera_data.resolution
            r = CameraRenderingData(
                rgb=np.zeros((h, w, 3), dtype=np.uint8),
                normals=np.zeros((h, w, 1), dtype=np.uint8),
                depth=np.zeros((h, w, 1), dtype=np.float32),
            )
        list_rgbs[n] = torch.tensor(r.rgb).share_memory_()
        if render_depth:
            list_depths[n] = torch.tensor(r.depth).share_memory_()
        if render_normals:
            list_normals[n] = torch.tensor(r.normals).share_memory_()

    rgbs = torch.stack(list_rgbs).pin_memory().cuda(non_blocking=True)
    rgbs = rgbs.float().permute(0, 3, 1, 2) / 255

    depths = None
    if render_depth:
        depths = torch.stack(list_depths).pin_memory().cuda(non_blocking=True)
        depths = depths.float().permute(0, 3, 1, 2)

    normals = None
    if render_normals:
        normals = torch.stack(list_normals).pin_memory().cuda(non_blocking=True)
        normals = normals.float().permute(0, 3, 1, 2) / 255

    return BatchRenderOutput(rgbs=rgbs, depths=depths, normals=normals)


def _stop(self) -> None:
    """No subprocess workers to clean up in sequential mode."""
    self._is_closed = True


def _new_init(
    self,
    object_dataset,
    n_workers: int = 8,
    preload_cache: bool = True,
    split_objects: bool = False,
) -> None:
    """Drop the `assert n_workers >= 1` guard; n_workers is unused in sequential mode."""
    self._object_dataset = object_dataset
    self._n_workers = n_workers
    self._split_objects = split_objects
    self._init_renderers(preload_cache)
    self._is_closed = False


# Apply patch at import time
Panda3dBatchRenderer.__init__ = _new_init
Panda3dBatchRenderer._init_renderers = _init_renderers
Panda3dBatchRenderer.render = _render
Panda3dBatchRenderer.stop = _stop
