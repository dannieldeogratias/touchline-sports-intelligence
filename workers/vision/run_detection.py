import argparse
import json
import subprocess
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

CLASS_ALIASES = {"person": "player", "player": "player", "sports ball": "ball", "ball": "ball", "referee": "referee"}


def cluster_teams(samples: dict[int, list[list[float]]]) -> dict[int, int]:
    """Cluster stable player tracks into two shirt-colour groups without another dependency."""
    track_colours = {track_id: np.median(values, axis=0) for track_id, values in samples.items() if len(values) >= 2}
    if len(track_colours) < 2:
        return {}
    ids, values = list(track_colours), np.array(list(track_colours.values()), dtype=np.float32)
    centres = np.array([values[0], values[np.argmax(np.linalg.norm(values - values[0], axis=1))]])
    labels = np.zeros(len(values), dtype=np.int32)
    for _ in range(12):
        labels = np.argmin(np.linalg.norm(values[:, None] - centres[None, :], axis=2), axis=1)
        next_centres = np.array([values[labels == index].mean(axis=0) if np.any(labels == index) else centres[index]
                                 for index in range(2)])
        if np.allclose(next_centres, centres):
            break
        centres = next_centres
    return {track_id: int(label) for track_id, label in zip(ids, labels)}


def infer_events(frames: list[dict], track_teams: dict[int, int]) -> list[dict]:
    """Produce review-only candidates from ball motion and possession transitions."""
    candidates, confirmed_owner = [], None
    owner_run, owner_run_length = None, 0
    possession_start, release_frame = None, None
    last_emitted = {"pass": -10.0}
    for frame in frames:
        balls = [item for item in frame["d"] if item["c"] == "ball"]
        players = [item for item in frame["d"] if item["c"] == "player" and "id" in item]
        if not balls:
            continue
        ball = max(balls, key=lambda item: item["p"])
        ball_centre = np.array([(ball["b"][0] + ball["b"][2]) / 2, (ball["b"][1] + ball["b"][3]) / 2])
        nearest = None
        nearest_distance = 1.0
        for player in players:
            foot = np.array([(player["b"][0] + player["b"][2]) / 2, player["b"][3]])
            distance = float(np.linalg.norm(ball_centre - foot))
            if distance < nearest_distance:
                nearest, nearest_distance = player, distance
        owner = nearest.get("id") if nearest is not None and nearest_distance < 0.055 else None
        if owner == owner_run:
            owner_run_length += 1
        else:
            owner_run, owner_run_length = owner, 1
        if owner is None and confirmed_owner is not None and release_frame is None:
            release_frame = (frame["t"], ball_centre.copy(), confirmed_owner, possession_start)
        if owner is not None and owner_run_length >= 2:
            if confirmed_owner is None:
                confirmed_owner, possession_start = owner, frame["t"]
            elif owner != confirmed_owner:
                old_team, new_team = track_teams.get(confirmed_owner), track_teams.get(owner)
                released_distance = (float(np.linalg.norm(ball_centre - release_frame[1]))
                                     if release_frame is not None else 0.0)
                held_seconds = ((release_frame[0] if release_frame else frame["t"]) -
                                (possession_start if possession_start is not None else frame["t"]))
                if (old_team is not None and old_team == new_team and release_frame is not None and
                        released_distance > .035 and held_seconds >= .4 and
                        frame["t"] - last_emitted["pass"] > 2.5):
                    candidates.append({"type": "PASS_CANDIDATE", "t": frame["t"],
                                       "from_track": confirmed_owner, "to_track": owner, "team": old_team,
                                       "confidence": round(min(.85, .45 + released_distance), 2),
                                       "status": "REVIEW"})
                    last_emitted["pass"] = frame["t"]
                confirmed_owner, possession_start = owner, frame["t"]
            release_frame = None
    return candidates


def plausible_player_box(box: list[float]) -> bool:
    width, height = box[2] - box[0], box[3] - box[1]
    return .004 <= width <= .16 and .025 <= height <= .48 and height >= width * 1.05


def box_iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def tiled_ball_candidates(model: YOLO, frame: np.ndarray, device) -> list[dict]:
    height, width = frame.shape[:2]
    regions = [(0, 0, .6, .6), (.4, 0, 1, .6), (0, .4, .6, 1), (.4, .4, 1, 1)]
    crops, offsets = [], []
    for x1, y1, x2, y2 in regions:
        left, top, right, bottom = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        crops.append(frame[top:bottom, left:right])
        offsets.append((left, top))
    predictions = model.predict(crops, imgsz=960, conf=.06, iou=.45, device=device, verbose=False)
    candidates = []
    for prediction, (left, top) in zip(predictions, offsets):
        if prediction.boxes is None:
            continue
        for box, confidence in zip(prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.conf.cpu().tolist()):
            normalized = [
                (box[0] + left) / width, (box[1] + top) / height,
                (box[2] + left) / width, (box[3] + top) / height,
            ]
            ball_width, ball_height = normalized[2] - normalized[0], normalized[3] - normalized[1]
            if not (.001 <= ball_width <= .04 and .0015 <= ball_height <= .065):
                continue
            candidates.append({"p": float(confidence), "b": normalized})
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["p"], reverse=True):
        centre = ((candidate["b"][0] + candidate["b"][2]) / 2, (candidate["b"][1] + candidate["b"][3]) / 2)
        if any(box_iou(candidate["b"], item["b"]) > .35 or
               np.linalg.norm(np.array(centre) - np.array(((item["b"][0] + item["b"][2]) / 2,
                                                           (item["b"][1] + item["b"][3]) / 2))) < .012
               for item in kept):
            continue
        kept.append(candidate)
    return kept[:8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotated-output", required=True)
    parser.add_argument("--ball-model")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--tracker", default=str(Path(__file__).with_name("football_botsort.yaml")))
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    model = YOLO(args.model)
    ball_model = YOLO(args.ball_model) if args.ball_model else None
    BYTETracker.reset_id()
    device = 0 if torch.cuda.is_available() else "cpu"
    frames, detections_by_class = [], Counter()
    unique_tracks: dict[str, set[int]] = defaultdict(set)
    track_observations: dict[str, Counter] = defaultdict(Counter)
    track_distance: dict[int, float] = defaultdict(float)
    previous_centres: dict[int, tuple[float, float]] = {}
    confidence_sum, peak_vram = Counter(), 0
    colour_samples: dict[int, list[list[float]]] = defaultdict(list)
    trail_points: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=45))
    video_writer = None
    annotated_path = Path(args.annotated_output)
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    last_ball_centre = None
    ball_velocity = np.zeros(2)
    missing_ball_frames = 0
    recovered_ball_frames = 0
    previous_scene_histogram = None
    camera_cuts: list[float] = []

    results = model.track(source=args.video, stream=True, persist=True, tracker=args.tracker,
                          vid_stride=args.stride, imgsz=args.image_size, conf=args.confidence,
                          device=device, verbose=False)
    for sample_index, result in enumerate(results):
        frame_time = round(sample_index * args.stride / fps, 3)
        hsv = cv2.cvtColor(cv2.resize(result.orig_img, (160, 90)), cv2.COLOR_BGR2HSV)
        scene_histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(scene_histogram, scene_histogram)
        if previous_scene_histogram is not None:
            similarity = cv2.compareHist(previous_scene_histogram, scene_histogram, cv2.HISTCMP_CORREL)
            if similarity < .42 and (not camera_cuts or frame_time - camera_cuts[-1] > 1.0):
                camera_cuts.append(frame_time)
        previous_scene_histogram = scene_histogram
        frame_detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(boxes)
            for box, class_id, confidence, track_id in zip(boxes, classes, confidences, ids):
                label = CLASS_ALIASES.get(result.names[class_id])
                if label is None:
                    continue
                if label == "ball" and ball_model is not None:
                    continue
                x1, y1, x2, y2 = box
                normalized = [round(max(0.0, min(1.0, value)), 4) for value in
                              (x1 / width, y1 / height, x2 / width, y2 / height)]
                if label == "player" and not plausible_player_box(normalized):
                    continue
                detection = {"c": label, "p": round(confidence, 3), "b": normalized}
                if track_id is not None:
                    detection["id"] = track_id
                    unique_tracks[label].add(track_id)
                    track_observations[label][track_id] += 1
                    centre = ((normalized[0] + normalized[2]) / 2, (normalized[1] + normalized[3]) / 2)
                    if track_id in previous_centres:
                        old = previous_centres[track_id]
                        track_distance[track_id] += ((centre[0] - old[0]) ** 2 + (centre[1] - old[1]) ** 2) ** 0.5
                    previous_centres[track_id] = centre
                    trail_points[track_id].append((int(centre[0] * width), int(centre[1] * height)))
                    if label == "player" and len(colour_samples[track_id]) < 30:
                        px1, py1, px2, py2 = map(int, box)
                        box_width, box_height = px2 - px1, py2 - py1
                        shirt = result.orig_img[
                            max(0, py1 + int(box_height * .12)):max(py1 + 1, py1 + int(box_height * .55)),
                            max(0, px1 + int(box_width * .2)):max(px1 + 1, px2 - int(box_width * .2)),
                        ]
                        if shirt.size:
                            lab = cv2.cvtColor(shirt, cv2.COLOR_BGR2LAB).reshape(-1, 3)
                            colour_samples[track_id].append(np.median(lab, axis=0).tolist())
                frame_detections.append(detection)
                detections_by_class[label] += 1
                confidence_sum[label] += confidence
        if ball_model is not None:
            frame_detections = [item for item in frame_detections if item["c"] != "ball"]
            candidates = tiled_ball_candidates(ball_model, result.orig_img, device)
            predicted = last_ball_centre + ball_velocity if last_ball_centre is not None else None
            chosen = None
            if candidates:
                if predicted is None:
                    chosen = max(candidates, key=lambda item: item["p"])
                else:
                    chosen = max(candidates, key=lambda item: item["p"] - 2.8 * float(np.linalg.norm(
                        np.array(((item["b"][0] + item["b"][2]) / 2, (item["b"][1] + item["b"][3]) / 2)) - predicted)))
                    chosen_centre = np.array(((chosen["b"][0] + chosen["b"][2]) / 2,
                                              (chosen["b"][1] + chosen["b"][3]) / 2))
                    if np.linalg.norm(chosen_centre - predicted) > .14 and chosen["p"] < .55:
                        chosen = None
            if chosen is not None:
                centre = np.array(((chosen["b"][0] + chosen["b"][2]) / 2,
                                   (chosen["b"][1] + chosen["b"][3]) / 2))
                if last_ball_centre is not None:
                    ball_velocity = .65 * ball_velocity + .35 * (centre - last_ball_centre)
                last_ball_centre = centre
                missing_ball_frames = 0
                ball_detection = {"c": "ball", "p": round(chosen["p"], 3),
                                  "b": [round(value, 4) for value in chosen["b"]], "id": 0}
                frame_detections.append(ball_detection)
                detections_by_class["ball"] += 1
                confidence_sum["ball"] += chosen["p"]
            elif last_ball_centre is not None and missing_ball_frames < 4:
                missing_ball_frames += 1
                last_ball_centre = np.clip(last_ball_centre + ball_velocity, 0, 1)
                radius = .009
                frame_detections.append({"c": "ball", "p": round(.18 * (.78 ** missing_ball_frames), 3),
                                         "b": [round(last_ball_centre[0] - radius, 4), round(last_ball_centre[1] - radius, 4),
                                               round(last_ball_centre[0] + radius, 4), round(last_ball_centre[1] + radius, 4)],
                                         "id": 0, "predicted": True})
                recovered_ball_frames += 1
        frames.append({"t": frame_time, "d": frame_detections})
        plotted = result.orig_img.copy()
        for item in (detection for detection in frame_detections if detection["c"] != "ball"):
            x1, y1, x2, y2 = (int(item["b"][0] * width), int(item["b"][1] * height),
                              int(item["b"][2] * width), int(item["b"][3] * height))
            colour = (255, 210, 65) if item["c"] == "referee" else (83, 220, 145)
            label = f"{item['c']} {item.get('id', '-')} · {item['p']:.0%}"
            cv2.rectangle(plotted, (x1, y1), (x2, y2), colour, max(1, width // 900))
            cv2.putText(plotted, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        .42, colour, 1, cv2.LINE_AA)
        if ball_model is not None:
            for item in (detection for detection in frame_detections if detection["c"] == "ball"):
                x1, y1, x2, y2 = (int(item["b"][0] * width), int(item["b"][1] * height),
                                  int(item["b"][2] * width), int(item["b"][3] * height))
                colour = (0, 170, 255) if item.get("predicted") else (0, 230, 255)
                cv2.rectangle(plotted, (x1, y1), (x2, y2), colour, 3)
                cv2.putText(plotted, "ball predicted" if item.get("predicted") else f"ball {item['p']:.2f}",
                            (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55, colour, 2, cv2.LINE_AA)
        for points in trail_points.values():
            if len(points) > 1:
                cv2.polylines(plotted, [np.array(points)], False, (83, 220, 145), 2, cv2.LINE_AA)
        if video_writer is None:
            video_writer = subprocess.Popen([
                imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps / args.stride),
                "-i", "-", "-an", "-vcodec", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(annotated_path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if video_writer.stdin:
            video_writer.stdin.write(plotted.tobytes())
        if torch.cuda.is_available():
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())

    if video_writer is not None:
        if video_writer.stdin:
            video_writer.stdin.close()
        if video_writer.wait() != 0:
            raise RuntimeError("FFmpeg could not encode the annotated video")

    track_teams = cluster_teams(colour_samples)
    for frame in frames:
        for detection in frame["d"]:
            if detection.get("id") in track_teams:
                detection["team"] = track_teams[detection["id"]]
    event_candidates = infer_events(frames, track_teams)

    duration_seconds = total_source_frames / fps if total_source_frames else (frames[-1]["t"] if frames else 0)
    minute_count = max(1, int(duration_seconds // 60) + 1)
    activity = [{"minute": minute, "players": 0, "balls": 0, "referees": 0} for minute in range(minute_count)]
    for frame in frames:
        bucket = activity[min(int(frame["t"] // 60), minute_count - 1)]
        for detection in frame["d"]:
            bucket[f"{detection['c']}s"] += 1

    summary = {
        "frames_sampled": len(frames), "sample_stride": args.stride, "source_fps": round(fps, 3),
        "duration_seconds": round(duration_seconds, 2), "detections": dict(detections_by_class),
        "unique_tracks": {key: len(value) for key, value in unique_tracks.items()},
        "stable_track_segments": {key: sum(count >= 5 for count in values.values())
                                  for key, values in track_observations.items()},
        "average_confidence": {key: round(confidence_sum[key] / count, 3)
                               for key, count in detections_by_class.items() if count},
        "ball_visible_percent": round(100 * sum(any(item["c"] == "ball" for item in frame["d"])
                                                  for frame in frames) / len(frames), 1) if frames else 0,
        "most_active_track": max(track_distance, key=track_distance.get, default=None),
        "peak_vram_mb": round(peak_vram / 1024**2, 1),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model": Path(args.model).name, "image_size": args.image_size,
        "annotated_video": annotated_path.name,
        "team_tracks": {"home": sum(team == 0 for team in track_teams.values()),
                        "away": sum(team == 1 for team in track_teams.values())},
        "event_candidates": dict(Counter(item["type"] for item in event_candidates)),
        "ball_pipeline": "tiled_dedicated_temporal" if ball_model is not None else "multiclass",
        "ball_predicted_frames": recovered_ball_frames,
        "tracking_scope": "scene_local",
        "tracker": Path(args.tracker).stem,
        "shot_detection": "requires_pitch_calibration",
        "camera_cuts": camera_cuts,
    }
    payload = {"version": 1,
               "video": {"width": width, "height": height, "fps": round(fps, 3),
                         "duration_seconds": round(duration_seconds, 2)},
               "summary": summary, "activity": activity, "events": event_candidates, "frames": frames}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
