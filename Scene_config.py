from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

@dataclass(frozen=True)
class Polygon:
    x: List[int]
    y: List[int]

    @property
    def as_tuples(self):
        return list(zip(self.x, self.y))


@dataclass(frozen=True)
class PolyLine:
    x: List[int]
    y: List[int]

    @property
    def as_tuples(self):
        return list(zip(self.x, self.y))

    @property
    def segments(self):
        pts = self.as_tuples
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class TrafficLightArea:
    polygon: Polygon
    traffic_light_rect: Rect


@dataclass(frozen=True)
class RestrictedParkingArea:
    polygon: Polygon
    seconds_timelimit: int


@dataclass(frozen=True)
class SubScene:
    name: str
    traffic_light_area: Optional[TrafficLightArea] = None
    restricted_area: Optional[Polygon] = None
    stop_area: Optional[Polygon] = None
    dividing_line: Optional[PolyLine] = None
    restricted_parking: Optional[RestrictedParkingArea] = None


@dataclass(frozen=True)
class SceneConfig:
    scene_id: str
    subscenes: Dict[str, SubScene]

    def list_subscenes(self) -> List[str]:
        return list(self.subscenes.keys())

    def tl_polygon(self, subscene: str) -> Polygon:
        return self._require(subscene).traffic_light_area.polygon

    def tl_rect(self, subscene: str) -> Rect:
        return self._require(subscene).traffic_light_area.traffic_light_rect

    def restricted(self, subscene: str) -> Polygon:
        return self._require(subscene).restricted_area

    def stop(self, subscene: str) -> Polygon:
        return self._require(subscene).stop_area

    def dividing(self, subscene: str) -> PolyLine:
        return self._require(subscene).dividing_line

    def _require(self, subscene: str) -> SubScene:
        try:
            return self.subscenes[subscene]
        except KeyError:
            raise KeyError(f"Sub-scene '{subscene}' not found in '{self.scene_id}'.") from None


def load_scene(path: str | Path) -> SceneConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def build_polygon(d: Any) -> Polygon | None:
        if d is None:
            return None
        return Polygon(d["all_points_x"], d["all_points_y"])

    def build_polyline(d: Any) -> Optional[PolyLine]:
        if d is None:
            return None
        return PolyLine(d["all_points_x"], d["all_points_y"])

    def build_subscene(name: str, d: dict) -> SubScene:
        tla = None
        if "traffic_light_area" in d:
            tld = d["traffic_light_area"]
            tla = TrafficLightArea(
                polygon=build_polygon(tld["polygon"]),
                traffic_light_rect=Rect(**tld["traffic_light_rect"]),
            )

        rpa = None
        if "restricted_parking" in d:
            rpd = d["restricted_parking"]
            rpa = RestrictedParkingArea(
                polygon=build_polygon(rpd["polygon"]),
                seconds_timelimit=int(rpd["seconds_timelimit"]),
            )

        return SubScene(
            name=name,
            traffic_light_area=tla,
            restricted_area=build_polygon(d.get("restricted_area", {}).get("polygon")),
            stop_area=build_polygon(d.get("stop_area", {}).get("polygon")),
            dividing_line=build_polyline(d.get("dividing_line")),
            restricted_parking=rpa,
        )

    cfg = SceneConfig(
        scene_id=raw["scene_id"],
        subscenes={k: build_subscene(k, v) for k, v in raw["subscenes"].items()},
    )
    return cfg
