"""🤖 Human-to-Robot hand retargeting pipeline.

Usage:
    cd omnidexgrasp
    conda activate omnidexgrasp
    python -m human2robo.main
    python -m human2robo.main hand_types=[inspire]
    python -m human2robo.main hand_types=[inspire,wuji] device=cpu
"""

from pathlib import Path
import json
import logging
import hydra
from omegaconf import DictConfig

from .dataloader import RetargetDataLoader
from .models import HAND_MODELS
from .retarget import retarget_pose


@hydra.main(config_path="../cfg", config_name="h2r", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Run H2R retargeting for all tasks in the output directory."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir   = Path(cfg.output)
    assets_root  = Path(cfg.assets_root)
    mano_root    = cfg.mano_assets_root

    # Load all requested hand models once (reuse across tasks)
    models = {
        ht: HAND_MODELS[ht](assets_root=assets_root, device=cfg.device)
        for ht in cfg.hand_types
    }
    logging.info(f"✅ Loaded models: {list(models.keys())}")

    dataloader = RetargetDataLoader(
        output_dir=output_dir,
        n_obj_pts=cfg.n_obj_pts,
        device=cfg.device,
        mano_assets_root=mano_root,
    )

    task_dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir()]
    logging.info(f"🤖 H2R: {len(task_dirs)} tasks × {list(cfg.hand_types)}")

    for task_dir in task_dirs:
        data = dataloader.load(task_dir.name)
        if data is None:
            logging.warning(f"⚠️  skip {task_dir.name}")
            continue

        logging.info(f"🔄 {task_dir.name}")
        result: dict = {"task_name": data.task_name}

        for hand_type, model in models.items():
            ret = retarget_pose(
                model=model,
                gt_fingertip=data.gt_fingertip,
                obj_pc=data.obj_pc,
                mano_trans=data.mano_trans,
                mano_axis_angle=data.mano_axis_angle,
                mano_pose=data.mano_pose,
                cfg=cfg,
                hand_type=hand_type,
            )
            result[hand_type] = {
                "init":  ret.init_dex_pose_obj.squeeze(0).tolist(),
                "final": ret.dex_pose_obj.squeeze(0).tolist(),
            }

        out_path = task_dir / "robo.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logging.info(f"  ✅ → {out_path}")

    logging.info("🎉 H2R retargeting complete!")


if __name__ == "__main__":
    main()
