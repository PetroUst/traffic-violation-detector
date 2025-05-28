from typing import Union, Iterable
from tqdm.auto import tqdm
import json
import cv2
from pathlib import Path
from typing import List, Tuple
import numpy as np
from Scene_config import Rect


def make_video_from_dir(
    img_dir: Union[str, Path],
    out_path: Union[str, Path],
    fps: int = 30,
    codec: str = "mp4v",
    glob: str = "*",
    recursive: bool = False,
    bgr_input: bool | None = None,
) -> None:
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(img_dir)

    files: Iterable[Path] = (
        img_dir.rglob(glob) if recursive else img_dir.glob(glob)
    )
    files = sorted(f for f in files if f.is_file())

    if not files:
        raise FileNotFoundError(f"No matching images in {img_dir} (pattern '{glob}').")

    first = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise IOError(f"Could not read image {files[0]}")
    if first.dtype != np.uint8:
        raise TypeError(f"Unsupported dtype {first.dtype}; expected uint8.")
    h, w = first.shape[:2]

    channels = 1 if first.ndim == 2 else first.shape[2]
    if channels not in (1, 3):
        raise ValueError(f"Unsupported channel count: {channels}")

    if bgr_input is None:                       # auto mode
        bgr_input = channels == 3 and bool(np.mean(first[..., 0]) < np.mean(first[..., 2]))
    rgb2bgr_needed = (channels == 3) and (bgr_input is False)

    out_path = Path(out_path).with_suffix(".mp4" if codec.lower() in {"mp4v", "avc1"} else ".avi")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=(channels == 3))
    if not writer.isOpened():
        raise IOError("VideoWriter failed to open. Check codec/container support.")

    frame_count = 0
    try:
        for fp in tqdm(files, desc="Creating Video", unit="frame"):
            img = cv2.imread(str(fp), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Failed to read {fp}")
            if img.shape[:2] != (h, w):
                raise ValueError(f"Geometry mismatch on {fp}: {img.shape[:2]} vs {(h, w)}")
            if img.dtype != np.uint8:
                raise TypeError(f"{fp} has dtype {img.dtype}, expected uint8.")

            if rgb2bgr_needed:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif channels == 1:                 # grayscale → replicate channels
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            writer.write(img)
            frame_count += 1
    finally:
        writer.release()

    print(f"[INFO] Video saved → {out_path.resolve()}  ({frame_count} frames @ {fps} fps)")


def crop_by_rect(img: np.ndarray, rect: Rect) -> np.ndarray:
    H, W = img.shape[:2]

    x1, y1 = rect.x, rect.y
    x2, y2 = x1 + rect.width, y1 + rect.height

    if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
        raise ValueError(f"Rect {rect} lies partially outside image of size {W}×{H}")

    return img[y1:y2, x1:x2]

def get_fps(video_path: str | Path) -> float:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps else 0.0

def save_json_and_clip(
    *,
    obj_id: int,
    violation_name: str,
    slice_: List[Tuple[int, np.ndarray]],
    video_path: str | Path,
    out_dir: str | Path,
    fps: float,
    save_clip: bool = False,
) -> None:
    if not slice_:
        raise ValueError("slice_ is empty – nothing to save")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_fr, end_fr = slice_[0][0], slice_[-1][0]
    stem = f"{violation_name.lower()}_{obj_id}_{start_fr:06d}_{end_fr:06d}"
    json_path = out_dir / f"{stem}.json"
    clip_path = out_dir / f"{stem}.mp4"

    # ── 1. write JSON ───────────────────────────────────────────────
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(
            {
                "violation": violation_name,
                "object_id": obj_id,
                "video_source": str(video_path),
                "frames": [
                    {"frame": int(fr), "bbox": [float(x) for x in box]}
                    for fr, box in slice_
                ],
            },
            jf,
            indent=2,
        )

    if save_clip:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_fr)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise IOError(f"Cannot read frame {start_fr} from {video_path}")

        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (w, h))

        current_fr = start_fr
        slice_iter = iter(slice_)
        next_fr, next_bbox = next(slice_iter)
        while ok and current_fr <= end_fr:
            if current_fr == next_fr:
                x1, y1, x2, y2 = map(int, next_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                try:
                    next_fr, next_bbox = next(slice_iter)
                except StopIteration:
                    next_fr = None
            writer.write(frame)
            ok, frame = cap.read()
            current_fr += 1

        writer.release()
        cap.release()
        print(f"[✓] Saved {violation_name} clip → {clip_path.name}")

    print(f"[✓] Saved JSON → {json_path.name}")



def slice_track(
        track: List[Tuple[int, int, np.ndarray]],
        start_fr: int,
        end_fr: int,
        pad_sec: float,
        fps: float,
) -> List[Tuple[int, np.ndarray]]:
    pad = int(round(pad_sec * fps))
    low = max(start_fr - pad, track[0][0])
    high = min(end_fr + pad, track[-1][0])

    fr2box = {fr: box for fr, _c, box in track}
    return [(fr, fr2box[fr]) for fr in range(low, high + 1) if fr in fr2box]