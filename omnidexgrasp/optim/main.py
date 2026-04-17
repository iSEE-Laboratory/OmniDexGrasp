"""🤚 MANO hand optimization with EasyHOI."""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gc
import logging
import shutil
import time
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig

from models.hoi_optim_module import HOI_Sync
from optim.dataloader import OptimDataLoader


def run_optimization_stages(hoi_sync: HOI_Sync) -> None:
    """🔄 Execute three-stage optimization pipeline."""
    stages = [
        ("step1: Contact Preparation", hoi_sync.prepare_contact_data, "before_contact_align"),
        ("step2: Contact Alignment", hoi_sync.run_handpose_global, "contact_align"),
        ("step3: Hand Refinement", hoi_sync.run_handpose_refine, "optim_final"),
    ]
    for name, func, prefix in stages:
        logging.info(f"{name} Optimization")
        start = time.time()
        func()
        logging.info(f"{name} duration: {time.time() - start:.2f}s")
        hoi_sync.export(prefix=prefix)
        hoi_sync.export_mano(prefix=prefix)


def _extract_final_results(optim_dir: Path, out_dir: Path, task_name: str) -> None:
    """📤 Extract final optimized outputs flat into out_dir."""
    # HOI_Sync internally appends "easyhoi/" to its output_dir
    hoi_dir = optim_dir / "easyhoi"

    # optim.ply: merged hand+obj mesh from final stage
    ply_src = hoi_dir / f"optim_final_{task_name}.ply"
    if ply_src.exists():
        shutil.copy2(ply_src, out_dir / "optim_res.ply")
        logging.info(f"  💾 optim_res.ply → {out_dir / 'optim_res.ply'}")
    else:
        logging.warning(f"  ⚠️ optim_final ply not found: {ply_src}")

    # manopose.json: final stage MANO parameters
    json_src = hoi_dir / "export_optim_final" / "res.json"
    if json_src.exists():
        shutil.copy2(json_src, out_dir / "optim_res.json")
        logging.info(f"  💾 optim_res.json → {out_dir / 'optim_res.json'}")
    else:
        logging.warning(f"  ⚠️ export_optim_final/res.json not found: {json_src}")


def process_single_task(
    task_name: str, datasets_dir: Path, output_dir: Path,
    project_root: str, inter_out: bool,
) -> None:
    """🎯 Run optimization for one task."""
    data_dir = datasets_dir / task_name
    out_dir = output_dir / task_name
    optim_dir = out_dir / "data" / "optim"

    logging.info(f"🎯 Processing: {task_name}")

    # 🧹 Clean old directories
    old_easyhoi = out_dir / "easyhoi"
    if old_easyhoi.exists():
        shutil.rmtree(old_easyhoi)
    if optim_dir.exists():
        shutil.rmtree(optim_dir)

    # 📦 Load data
    loader = OptimDataLoader(data_dir=data_dir, output_dir=out_dir)
    data_item = loader.load_data()
    if data_item is None:
        logging.error(f"❌ Failed to load data for {task_name}")
        return

    # 🚀 Run optimization
    (optim_dir / "debug").mkdir(parents=True, exist_ok=True)
    hoi_sync = HOI_Sync(str(optim_dir), project_root=project_root)
    logging.info(f"📦 Data loaded: {data_item['name']}")

    try:
        hoi_sync.get_data(data_item)
        hoi_sync.get_hamer_hand_mask()
        run_optimization_stages(hoi_sync)
    finally:
        del hoi_sync
        torch.cuda.empty_cache()

    # 📤 Extract final results
    _extract_final_results(optim_dir, out_dir, task_name)

    # 🧹 Remove intermediate optim dir if not needed
    if not inter_out:
        shutil.rmtree(optim_dir)
        logging.info(f"🧹 Removed intermediate optim dir (inter_out={inter_out})")


@hydra.main(config_path="../cfg", config_name="optim", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Run MANO hand optimization for all tasks."""
    logging.basicConfig(level=logging.INFO)
    datasets_dir = Path(cfg.datasets)
    output_dir = Path(cfg.output)
    project_root = str(Path(cfg.easyhoi_root).resolve())
    inter_out = cfg.out.inter_out

    for task_dir in sorted(datasets_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue
        out_dir = output_dir / task_dir.name
        if not (out_dir / "hand_params.pt").exists():
            logging.warning(f"⏭️ Skipping {task_dir.name}: no recons output")
            continue
        try:
            process_single_task(task_dir.name, datasets_dir, output_dir, project_root, inter_out)
        except Exception:
            logging.exception(f"❌ {task_dir.name} failed")

    # 🧹 Cleanup
    torch.cuda.empty_cache()
    gc.collect()
    # WORKAROUND: nvdiffrast holds a CUDA context that hangs on normal Python exit.
    # os._exit() bypasses cleanup to avoid the hang.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
