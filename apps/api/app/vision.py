import asyncio
import json
import subprocess
from pathlib import Path


async def run_yolo_detection(
    video_path: Path, model_path: Path, output_path: Path, annotated_output_path: Path,
    ball_model_path: Path | None = None,
) -> dict:
    """Run detection in the isolated vision environment."""
    from .config import settings

    if not model_path.exists():
        raise RuntimeError(f"YOLO checkpoint is missing: {model_path}")
    if not settings.vision_python.exists() or not settings.vision_worker.exists():
        raise RuntimeError("The isolated vision worker is not installed")
    command = [str(settings.vision_python), str(settings.vision_worker),
               "--video", str(video_path), "--model", str(model_path), "--output", str(output_path),
               "--annotated-output", str(annotated_output_path)]
    if ball_model_path and ball_model_path.exists():
        command.extend(["--ball-model", str(ball_model_path)])
    process = await asyncio.to_thread(
        subprocess.run,
        command,
        capture_output=True, check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.decode(errors="replace").strip() or process.stdout.decode(errors="replace").strip()
        raise RuntimeError(detail.splitlines()[-1] if detail else f"YOLO worker exited with code {process.returncode}")
    try:
        return json.loads(process.stdout.decode().strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise RuntimeError("YOLO worker returned an invalid result") from error
