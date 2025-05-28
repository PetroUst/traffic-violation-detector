import threading
import queue
import time
import pathlib
import cv2
from Processor import TrackProcessor
from helpers import crop_by_rect, get_fps
from Scene_config import load_scene
from TL_classifier import TLClassifier
from ultralytics import YOLO


class ViolationsDetector:
    def __init__(
        self,
        model_path: str,
        video_path: str,
        scene_config_path: str,
        tl_model_path: str,
        device: str = "cuda:0",
        frame_stride: int = 1,
        save_frames: bool = False,
        frames_dir: str = "out_frames",
        violation_dir: str = "violations",
        save_clips: bool = False,
        tracker_cfg: str = "botsort_reid.yaml",
        yolo_verbose: bool = False,
    ):
        self.video_path = video_path
        self.save_frames = save_frames
        self.frames_dir = pathlib.Path(frames_dir)
        if save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        cfg = load_scene(scene_config_path)
        self.scene = cfg.subscenes[list(cfg.subscenes)[0]]
        self.clf = TLClassifier(tl_model_path, device=device)

        framerate = get_fps(video_path)
        self.processor = TrackProcessor(
            frame_rate=framerate,
            frame_stride=frame_stride,
            vid_path=video_path,
            violation_dir=violation_dir,
            scene_config=self.scene,
            save_clips=save_clips,
            track_window=36000,
        )

        self.tracking_results = YOLO(model_path).track(
            video_path,
            stream=True,
            verbose=yolo_verbose,
            stream_buffer=True,
            vid_stride=frame_stride,
            tracker=tracker_cfg,
        )

        self.queue: queue.Queue = queue.Queue(maxsize=2)

        self.tl_rect = (
            self.scene.traffic_light_area.traffic_light_rect
            if self.scene.traffic_light_area is not None
            else None
        )

    def _producer(self):
        for result in self.tracking_results:
            self.queue.put(result)
        self.queue.put(None)

    def run(self):
        threading.Thread(target=self._producer, daemon=True).start()

        start_time = time.time()
        idx = 0

        while True:
            result = self.queue.get()
            if result is None:
                break

            go = True
            if self.tl_rect is not None:
                patch = crop_by_rect(result.orig_img, self.tl_rect)
                go = self.clf(patch)

            self.processor.add_frame(idx, result, go)

            if idx % 30 == 0:
                print(self.processor.process(idx, False))

            if self.save_frames:
                frame_path = self.frames_dir / f"frame_{idx:07d}.jpg"
                cv2.imwrite(str(frame_path), result.plot(line_width=2, font_size=5))

            idx += 1

        print(self.processor.process(idx, True))
        print("Total time taken:", time.time() - start_time)
