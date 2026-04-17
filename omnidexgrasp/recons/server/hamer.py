"""🤚 HaMeR hand reconstruction server.

POST /predict - Reconstruct hand mesh from image.

Usage: python -m recons.server.hamer
"""
import base64
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from omegaconf import DictConfig
from PIL import Image
from pydantic import BaseModel

# ViTPose COCO-WholeBody keypoint slice indices
_LEFT_HAND = slice(-42, -21)
_RIGHT_HAND = slice(-21, None)

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


# ══════════════════════════════════════════════════════════════════════════════
# 🧠 Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HaMeRModel:
    """🤚 HaMeR model wrapper for hand reconstruction."""
    hamer_model: "HAMER"  # type: ignore
    model_cfg: DictConfig
    body_detector: "DefaultPredictor_Lazy"  # type: ignore
    keypoint_detector: "ViTPoseModel"  # type: ignore
    renderer: "Renderer"  # type: ignore
    device: str
    cfg: DictConfig

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "HaMeRModel":
        """🚀 Load HaMeR models from config."""
        import hamer
        import hamer.configs
        from hamer.models import load_hamer
        from hamer.utils.renderer import Renderer
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from vitpose_model import ViTPoseModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Loading HaMeR models on {device}...")

        hamer.configs.CACHE_DIR_HAMER = str(Path(cfg.model.checkpoint).parent)
        ViTPoseModel.MODEL_DICT = {
            "ViTPose+-G (multi-task train, COCO)": {
                "config": cfg.model.vitpose_config,
                "model": cfg.model.vitpose_ckpt,
            },
        }

        print("  🤚 Loading HaMeR...")
        model, model_cfg = load_hamer(str(cfg.model.checkpoint))
        model = model.to(device).eval()

        print("  🔍 Loading body detector...")
        body_detector = cls._build_body_detector(cfg, hamer, DefaultPredictor_Lazy)

        print("  🦴 Loading keypoint detector...")
        keypoint_detector = ViTPoseModel(device)

        renderer = Renderer(model_cfg, faces=model.mano.faces)

        print("✅ All HaMeR models loaded!")
        return cls(model, model_cfg, body_detector, keypoint_detector, renderer, device, cfg)

    @staticmethod
    def _build_body_detector(
        cfg: DictConfig, hamer_pkg: "module", DefaultPredictor_Lazy: type  # type: ignore
    ) -> "DefaultPredictor_Lazy":
        """Build detectron2 body detector from config."""
        if cfg.model.body_detector == "vitdet":
            from detectron2.config import LazyConfig
            det_cfg_path = Path(hamer_pkg.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
            det_cfg = LazyConfig.load(str(det_cfg_path))
            det_cfg.train.init_checkpoint = str(cfg.model.vitdet_checkpoint)
            for i in range(3):
                det_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        else:
            from detectron2 import model_zoo
            det_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
            )
            det_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            det_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(det_cfg)

    def detect_best_hand(self, img_cv2: np.ndarray) -> dict | None:
        """🔍 Detect the best hand in image via body detection + ViTPose keypoints."""
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        det_instances = self.body_detector(img_cv2)["instances"]
        valid = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        bboxes = det_instances.pred_boxes.tensor[valid].cpu().numpy()
        scores = det_instances.scores[valid].cpu().numpy()

        if len(bboxes) == 0:
            return None

        vitposes_out = self.keypoint_detector.predict_pose(
            img_rgb,
            [np.concatenate([bboxes, scores[:, None]], axis=1)],
        )

        # 🏆 Select hand with highest keypoint confidence
        best_conf, best_hand = 0.0, None
        for vitposes in vitposes_out:
            hands = [(0, vitposes["keypoints"][_LEFT_HAND]),
                     (1, vitposes["keypoints"][_RIGHT_HAND])]
            for is_right, keyp in hands:
                valid_mask = keyp[:, 2] > 0.5
                if valid_mask.sum() <= 3:
                    continue
                conf = keyp[valid_mask, 2].mean()
                if conf > best_conf:
                    best_conf = conf
                    kp_valid = keyp[valid_mask]
                    best_hand = {
                        "bboxes": np.array([[kp_valid[:, 0].min(), kp_valid[:, 1].min(),
                                             kp_valid[:, 0].max(), kp_valid[:, 1].max()]]),
                        "is_right": np.array([is_right]),
                        "keypts": np.array([keyp[:, :2]]),
                    }
        return best_hand

    def reconstruct(
        self, img_cv2: np.ndarray, hand_data: dict, focal_length: float
    ) -> dict:
        """🤚 Run HaMeR reconstruction on detected hand."""
        from hamer.datasets.vitdet_dataset import ViTDetDataset
        from hamer.utils import recursive_to
        from hamer.utils.renderer import cam_crop_to_full

        dataset = ViTDetDataset(
            self.model_cfg, img_cv2, hand_data["bboxes"], hand_data["is_right"],
            rescale_factor=self.cfg.inference.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.cfg.inference.batch_size, shuffle=False, num_workers=0
        )

        batch = recursive_to(next(iter(dataloader)), self.device)
        with torch.no_grad():
            out = self.hamer_model(batch)

        # Flip x-axis of predicted camera for left/right hand convention
        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        img_size = batch["img_size"].float()
        pred_cam_t = cam_crop_to_full(
            pred_cam, batch["box_center"].float(), batch["box_size"].float(),
            img_size, focal_length,
        ).detach().cpu().numpy()

        # Extract first (only) result
        verts = out["pred_vertices"][0].detach().cpu().numpy()
        is_right = batch["right"][0].cpu().numpy()
        verts[:, 0] = (2 * is_right - 1) * verts[:, 0]

        return {
            "vertices": verts,
            "cam_transl": pred_cam_t[0],
            "is_right": bool(is_right),
            "mano_params": {k: v.detach().cpu().numpy().tolist() for k, v in out["pred_mano_params"].items()},
            "img_size": img_size[0].cpu().numpy().tolist(),
        }

    def render_mask(self, recon_data: dict, focal_length: float) -> np.ndarray:
        """🎨 Render binary hand mask (0=hand, 255=background)."""
        img_size = recon_data["img_size"]
        cam_view = self.renderer.render_rgba_multiple(
            [recon_data["vertices"]],
            cam_t=[recon_data["cam_transl"]],
            render_res=img_size,
            is_right=[recon_data["is_right"]],
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=focal_length,
        )
        mask = np.full((int(img_size[1]), int(img_size[0]), 3), 255, dtype=np.uint8)
        mask[cam_view[:, :, 3] > 0] = 0
        return mask


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 API
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    image_b64: str
    focal_length: float


class PredictResponse(BaseModel):
    status: str              # success | warning | error
    message: str
    mano_params: dict = {}
    vertices_b64: str = ""   # base64 numpy (778, 3)
    cam_transl: list[float] = []
    is_right: bool = False
    mask_b64: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# 🔧 Helpers
# ══════════════════════════════════════════════════════════════════════════════

def encode_array_b64(arr: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def encode_image_b64(img: np.ndarray) -> str:
    """Convert image to base64 PNG string."""
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 Server
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="🤚 HaMeR Hand Reconstruction Server")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """🤚 Reconstruct hand mesh from image."""
    model: HaMeRModel = request.app.state.model
    print(f"\n{'='*60}")
    print(f"📨 New request: image_b64 ({len(req.image_b64)} chars)")

    img_bytes = base64.b64decode(req.image_b64)
    img_cv2 = cv2.cvtColor(
        np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )

    hand_data = model.detect_best_hand(img_cv2)
    if hand_data is None:
        return PredictResponse(status="warning", message="No hands detected")

    print(f"🤚 Reconstructing with focal_length={req.focal_length:.2f}")
    recon_data = model.reconstruct(img_cv2, hand_data, req.focal_length)

    print(f"🎉 Done! is_right={recon_data['is_right']}")
    return PredictResponse(
        status="success",
        message="Hand reconstructed successfully",
        mano_params=recon_data["mano_params"],
        vertices_b64=encode_array_b64(recon_data["vertices"]),
        cam_transl=recon_data["cam_transl"].tolist(),
        is_right=recon_data["is_right"],
        mask_b64=encode_image_b64(model.render_mask(recon_data, req.focal_length)),
    )


@hydra.main(config_path="../../cfg/model", config_name="hamer", version_base=None)
def main(cfg: DictConfig) -> None:
    """🚀 Start HaMeR server with Hydra config."""
    app.state.model = HaMeRModel.from_config(cfg)
    print(f"🌐 Server starting at http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    Path("log").mkdir(exist_ok=True)
    main()
