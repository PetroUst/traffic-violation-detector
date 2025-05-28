from collections import defaultdict, deque
from enum import Enum, unique
from typing import Dict, List, Tuple, Iterable, Deque, Set, Callable

import cv2
import numpy as np

from Scene_config import SubScene
from helpers import save_json_and_clip, slice_track


@unique
class Violations(Enum):
    RED_LIGHT = 1
    STOP_SIGN = 2
    RESTRICTED_AREA = 3
    RESTRICTED_PARKING = 4
    DIVIDING_LINE = 5


class TrackProcessor:
    BBox = np.ndarray
    Track = Tuple[int, int, BBox]

    TRACK_EXPIRY = 90
    PX_THR = 0.5
    W_STOP = 5
    GAP_TOL = 2
    STILL_RATIO = 0.9
    PX_THR_PARK = 2.0
    CAR_CLASS_ID = 1

    def __init__(self, frame_rate: float, frame_stride: int, vid_path: str, violation_dir: str, save_clips: bool, scene_config: SubScene, track_window: int | None = None) -> None:
        self.frame_rate = frame_rate
        self.frame_stride = frame_stride
        self.scene_config = scene_config
        self.vid_path = vid_path
        self.violation_dir = violation_dir
        self.save_clips = save_clips
        factory: Callable[[], Iterable[TrackProcessor.Track]]
        factory = (lambda: deque(maxlen=track_window)) if track_window else list

        self.track_history: Dict[int, Iterable[TrackProcessor.Track]] = defaultdict(factory)
        self.frame_flags: List[Tuple[int, bool]] = []

        self.red_light_ids: Set[int] = set()
        self.stop_sign_ids: Set[int] = set()
        self.restricted_parking_ids: Set[int] = set()
        self.restricted_area_ids: Set[int] = set()
        self.dividing_line_ids: Set[int] = set()

    def add_frame(self, frame_idx: int, results, go_flag: bool) -> None:
        self.frame_flags.append((frame_idx, go_flag))

        if results.boxes is None or results.boxes.id is None:
            return

        ids = results.boxes.id.cpu().numpy().tolist()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
        for obj_id, cls_id, box in zip(ids, clses, boxes):
            self.track_history[obj_id].append((frame_idx, cls_id, box.copy()))

    def process(
            self,
            current_frame: int,
            final_flush: bool = False
    ) -> Dict[Violations, List[int]]:
        new_red: List[int] = []
        new_stop: List[int] = []
        new_rpark: List[int] = []
        new_rarea: List[int] = []
        new_dline: List[int] = []

        done_ids = [
            oid for oid, trk in self.track_history.items()
            if final_flush or trk[-1][0] <= current_frame - self.TRACK_EXPIRY
        ]

        for oid in done_ids:
            track = list(self.track_history[oid])

            if (self.scene_config.traffic_light_area is not None
                    and oid not in self.red_light_ids):
                slice_ = self._red_light_violation(track)
                if slice_:
                    save_json_and_clip(
                        obj_id=oid,
                        violation_name=Violations.RED_LIGHT.name,
                        slice_=slice_,
                        video_path=self.vid_path,
                        out_dir=self.violation_dir,
                        fps=self.frame_rate,
                        save_clip=self.save_clips,
                    )
                    self.red_light_ids.add(oid)
                    new_red.append(oid)

            if (self.scene_config.stop_area is not None
                    and oid not in self.stop_sign_ids):
                slice_ = self._stop_sign_violation(track)
                if slice_:
                    save_json_and_clip(
                        obj_id=oid,
                        violation_name=Violations.STOP_SIGN.name,
                        slice_=slice_,
                        video_path=self.vid_path,
                        out_dir=self.violation_dir,
                        fps=self.frame_rate,
                        save_clip=self.save_clips,
                    )
                    self.stop_sign_ids.add(oid)
                    new_stop.append(oid)

            if (self.scene_config.restricted_parking is not None
                    and oid not in self.restricted_parking_ids):
                slice_ = self._restricted_parking_violation(track)
                if slice_:
                    save_json_and_clip(
                        obj_id=oid,
                        violation_name=Violations.RESTRICTED_PARKING.name,
                        slice_=slice_,
                        video_path=self.vid_path,
                        out_dir=self.violation_dir,
                        fps=self.frame_rate,
                        save_clip=self.save_clips,
                    )
                    self.restricted_parking_ids.add(oid)
                    new_rpark.append(oid)

            if (self.scene_config.restricted_area is not None
                    and oid not in self.restricted_area_ids):
                slice_ = self._restricted_area_violation(track)
                if slice_:
                    save_json_and_clip(
                        obj_id=oid,
                        violation_name=Violations.RESTRICTED_AREA.name,
                        slice_=slice_,
                        video_path=self.vid_path,
                        out_dir=self.violation_dir,
                        fps=self.frame_rate,
                        save_clip=self.save_clips,
                    )
                    self.restricted_area_ids.add(oid)
                    new_rarea.append(oid)

            if (self.scene_config.dividing_line is not None
                    and oid not in self.dividing_line_ids):
                slice_ = self._dividing_line_violation(track)
                if slice_:
                    save_json_and_clip(
                        obj_id=oid,
                        violation_name=Violations.DIVIDING_LINE.name,
                        slice_=slice_,
                        video_path=self.vid_path,
                        out_dir=self.violation_dir,
                        fps=self.frame_rate,
                        save_clip=self.save_clips,
                    )
                    self.dividing_line_ids.add(oid)
                    new_dline.append(oid)

            del self.track_history[oid]

        return {
            Violations.RED_LIGHT: new_red,
            Violations.STOP_SIGN: new_stop,
            Violations.RESTRICTED_PARKING: new_rpark,
            Violations.RESTRICTED_AREA: new_rarea,
            Violations.DIVIDING_LINE: new_dline,
        }

    def _red_light_violation(
            self,
            track: List[Track],
            pad_sec: float = 5.0,
    ) -> List[Tuple[int, np.ndarray]]:
        tl_poly = self.scene_config.traffic_light_area.polygon
        contour = np.asarray(tl_poly.as_tuples, np.int32)
        frame_map = dict(self.frame_flags)  # frame_idx → go flag
        fps = self.frame_rate

        last_in, first_out = None, None
        for fr, _cls, box in track:
            cx, cy = (box[0] + box[2]) / 2.0, box[3]
            if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                last_in = fr
            elif last_in is not None:
                first_out = fr
                break

        if last_in is None or first_out is None:
            return []

        if frame_map.get(last_in, True) or frame_map.get(first_out, True):
            return []

        return slice_track(track, last_in, first_out, pad_sec, fps)

    def _stop_sign_violation(
            self,
            track: List[Track],
            pad_sec: float = 2.0,
    ) -> List[Tuple[int, np.ndarray]]:
        ss_poly = self.scene_config.stop_area
        contour = np.asarray(ss_poly.as_tuples, np.int32)
        fps = self.frame_rate

        inside = False
        stopped_ok = False
        win: Deque[float] = deque(maxlen=self.W_STOP)
        prev_pt, prev_fr = None, None
        entry_fr, exit_fr = None, None

        for fr, _cls, box in track:
            cx, cy = (box[0] + box[2]) / 2.0, box[3]
            inside_now = cv2.pointPolygonTest(contour, (cx, cy), False) >= 0

            if inside_now:
                if not inside:
                    inside, entry_fr = True, fr
                    win.clear()
                    prev_pt, prev_fr = (cx, cy), fr
                    win.append(0.0)
                    continue

                gap = fr - prev_fr
                dist = np.hypot(cx - prev_pt[0], cy - prev_pt[1])
                disp = dist / max(gap, 1)

                for _ in range(min(gap, self.GAP_TOL + 1)):
                    win.append(disp)

                if len(win) == self.W_STOP and (sum(win) / self.W_STOP) <= self.PX_THR:
                    stopped_ok = True

                prev_pt, prev_fr = (cx, cy), fr

            elif inside:
                exit_fr = fr
                break

        if inside and exit_fr is None:
            exit_fr = track[-1][0]

        if inside and not stopped_ok:
            return slice_track(track, entry_fr, exit_fr, pad_sec, fps)
        return []

    def _restricted_parking_violation(
            self,
            track: List[Track],
            pad_sec: float = 2.0
    ) -> List[Tuple[int, np.ndarray]]:

        rp = self.scene_config.restricted_parking
        contour = np.asarray(rp.polygon.as_tuples, np.int32)
        fps = self.frame_rate
        limit_s = rp.seconds_timelimit
        pad_frames = int(round(pad_sec * fps))

        inside = []
        for frame_idx, _cls, box in track:
            cx = (box[0] + box[2]) / 2.0
            cy = box[3]
            if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                inside.append((frame_idx, cx, cy, box))

        if not inside:
            return []

        first_fr, last_fr = inside[0][0], inside[-1][0]
        if (last_fr - first_fr) / fps < limit_s:
            return []

        centres = np.array([[cx, cy] for _, cx, cy, _ in inside])
        cluster_cx, cluster_cy = np.median(centres, axis=0)

        dists = np.hypot(centres[:, 0] - cluster_cx,
                         centres[:, 1] - cluster_cy)
        ratio = np.mean(dists <= self.PX_THR_PARK)

        if ratio < 0.90:
            return []

        idx_to_box = {fr: box for fr, _cx, _cy, box in inside}
        low = max(first_fr - pad_frames, track[0][0])
        high = min(last_fr + pad_frames, track[-1][0])

        out = []
        for fr in range(low, high + 1):
            if fr in idx_to_box:
                out.append((fr, idx_to_box[fr]))

        return out

    def _restricted_area_violation(
            self,
            track: List[Track],
            pad_sec: float = 2.0,
    ) -> List[Tuple[int, np.ndarray]]:
        ra_poly = self.scene_config.restricted_area
        contour = np.asarray(ra_poly.as_tuples, np.int32)
        fps = self.frame_rate

        inside = [
            (fr, box)
            for fr, _, box in track
            if cv2.pointPolygonTest(
                contour,
                ((box[0] + box[2]) / 2.0, box[3]),
                False
            ) >= 0
        ]
        if not inside:
            return []

        cls_totals = {}
        for _, cls, _ in track:
            cls_totals[cls] = cls_totals.get(cls, 0) + 1

        top_cls = max(cls_totals.items(), key=lambda kv: kv[1])[0]
        if top_cls != self.CAR_CLASS_ID:
            return []

        first_fr = inside[0][0]
        last_fr = inside[-1][0]
        return slice_track(track, first_fr, last_fr, pad_sec, fps)

    def _dividing_line_violation(
            self,
            track: List[Track],
            pad_sec: float = 2.0,
    ) -> List[Tuple[int, np.ndarray]]:
        dl = self.scene_config.dividing_line
        if dl is None or len(dl.x) < 2:
            return []

        segments = dl.segments
        fps = self.frame_rate

        prev_pt = None
        cross_fr = None
        for fr, _cls, box in track:
            curr_pt = ((box[0] + box[2]) / 2.0, box[3])
            if prev_pt is not None:
                if any(intersect(prev_pt, curr_pt, q1, q2) for q1, q2 in segments):
                    cross_fr = fr
                    break
            prev_pt = curr_pt

        if cross_fr is None:
            return []

        return slice_track(track, cross_fr, cross_fr, pad_sec, fps)


def intersect(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0:
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1:
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1:
        return None
    return True