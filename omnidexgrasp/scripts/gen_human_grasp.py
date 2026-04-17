"""Generate human grasp images for the OmniDexGrasp dataset.

Usage:
    python scripts/gen_human_grasp.py [--data /path/to/data] [--workers 8] [--test]
"""

import os
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from google import genai
from google.genai import types
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Gemini API config — read from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "")
GEMINI_MODEL = "gemini-3-pro-image-preview"


BASE_PROMPT = """
# Generate Realistic Image Based on Input Image

Generate a realistic image from the input image, adding a human right hand naturally grasping the central object. The generated image must meet the following requirements:

## Grasping Style Consistent with Object Use

{intention}

## Object Placement and Scene Consistency

- The object must remain in its original position, stably placed on the table, exactly as in the input image.
- Do not alter the object's characteristics, including material, shape, color, or size. The object in the output must be identical to that in the input image.
- Do not change the object's relative or absolute position.
- Do not change the inclination angle of the object relative to the reference plane.
- Do not change the number, position, or layout of other surrounding objects.

## First-Person Perspective and using right hand

- The image should simulate a first-person human viewpoint, as if the viewer is seeing their own hand interacting with the object.
- Ensure correct hand features and occlusion relationships; objects must not intersect or penetrate unnaturally.
- You must using right hand to grasp the object, do not show the left hand in the frame.
- Optionally, part of the wrist or forearm may be shown to enhance the spatial relationship between the hand and object. The framing should focus on the interaction between hand and object.

## Visual Style and Integration

- The added hand must match the original image in style, tone, and level of detail for seamless integration.
- The overall image must remain realistic, avoiding visual artifacts, structural inconsistencies, or unnatural composition.

## Image Size

Output image size: 1536 × 1024, landscape orientation.

## Number of Generated Images

{gen_n} images
"""

# Lazy-initialized client (thread-safe for concurrent requests)
_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("Set GEMINI_API_KEY environment variable")
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1beta", "base_url": GEMINI_BASE_URL},
        )
    return _client


def load_intention(camera_yaml: Path) -> str | None:
    """Load grasp intention from camera.yaml. Returns None if not found."""
    with open(camera_yaml) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("grasp_intention")


def build_tasks(data_dir: Path, test_mode: bool) -> dict[Path, str]:
    """Scan data dir, return {task_dir: intention} for unprocessed tasks."""
    tasks = {}
    for task_dir in sorted(data_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        scene = task_dir / "scene_image.png"
        camera = task_dir / "camera.yaml"
        if not scene.exists() or not camera.exists():
            continue
        if (task_dir / "generated_human_grasp_0.png").exists():
            continue  # Skip already processed
        intention = load_intention(camera)
        if intention is None:
            print(f"⏭️ SKIP {task_dir.name}: no grasp_intention in camera.yaml")
            continue
        tasks[task_dir] = intention
        if test_mode and len(tasks) >= 1:
            break
    return tasks


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def generate_image(image_path: Path, prompt: str) -> bytes:
    """Call Gemini generateContent API to edit image with hand."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    image_part = types.Part.from_bytes(data=image_data, mime_type="image/png")

    response = _get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="3:2"),
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data

    raise ValueError("No image data in Gemini response")


def process_task(task_dir: Path, intention: str, gen_n: int) -> tuple[str, str]:
    """Process one task. Returns (status, message)."""
    output_0 = task_dir / "generated_human_grasp_0.png"
    if output_0.exists():
        return "SKIP", "already exists"
    prompt = BASE_PROMPT.format(intention=intention, gen_n=gen_n)
    try:
        saved = []
        for i in range(gen_n):
            image_bytes = generate_image(task_dir / "scene_image.png", prompt)
            out_path = task_dir / f"generated_human_grasp_{i}.png"
            out_path.write_bytes(image_bytes)
            saved.append(out_path.name)
        return "OK", ", ".join(saved)
    except Exception as e:
        return "ERROR", str(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate human grasp images")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gen_n", type=int, default=1, help="Number of images to generate per task")
    parser.add_argument("--test", action="store_true", help="Process 1 task only")
    args = parser.parse_args()

    tasks = build_tasks(args.data, args.test)
    total = len(tasks)
    print(f"🚀 Found {total} tasks, gen_n={args.gen_n}, starting {args.workers} workers...")

    ok, skip, err = 0, 0, 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_task, d, i, args.gen_n): d for d, i in tasks.items()}
        for n, fut in enumerate(as_completed(futs), 1):
            d = futs[fut]
            status, msg = fut.result()
            if status == "OK":
                ok += 1
            elif status == "SKIP":
                skip += 1
            else:
                err += 1
            print(f"[{n}/{total}] [{status}] {d.name}: {msg}")

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s — OK:{ok} SKIP:{skip} ERR:{err}")


if __name__ == "__main__":
    main()
