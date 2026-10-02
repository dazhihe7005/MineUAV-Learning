"""Correct undefined V2 near-target-speed scalars without retraining.

The JSON report already records these means as null when no qualifying
policy steps exist. This utility preserves the raw TensorBoard event file
under the V2 log directory and replaces it with a verified filtered copy.
"""

import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.summary.writer.event_file_writer import EventFileWriter


TAG = "eval/mean_speed_within_0_1_m_s"
RL_DIR = Path(__file__).resolve().parent
EVENT_DIR = RL_DIR / "tensorboard" / "ppo_waypoint_reward_v2" / "PPO_1"
BACKUP_DIR = RL_DIR / "logs" / "ppo_waypoint_reward_v2" / "original_tensorboard_events"
REPORT_PATH = RL_DIR.parent / "reports" / "ppo_waypoint_reward_v2.json"


def copy_without_undefined_near_speed(source: Path, output_dir: Path,
                                      invalid_positions: set[int]) -> Path:
    """Copy all events, dropping selected occurrences of one undefined scalar."""
    if not source.is_file() or invalid_positions and min(invalid_positions) < 0:
        raise ValueError("Invalid source or scalar position")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = set(output_dir.glob("events.out.tfevents.*"))
    writer = EventFileWriter(str(output_dir))
    count = 0
    try:
        # EventFileLoader normalizes simple_value summaries into tensor
        # summaries; copying that output would silently alter scalar tags.
        for raw_event in RawEventFileLoader(str(source)).Load():
            event = Event.FromString(raw_event)
            values = list(event.summary.value) if event.HasField("summary") else []
            if not any(value.tag == TAG for value in values):
                writer.add_event(event)
                continue
            kept = []
            for value in values:
                if value.tag == TAG:
                    position = count
                    count += 1
                    if position in invalid_positions:
                        continue
                kept.append(value)
            if kept:
                copied = type(event)()
                copied.CopyFrom(event)
                del copied.summary.value[:]
                copied.summary.value.extend(kept)
                writer.add_event(copied)
        if invalid_positions and max(invalid_positions) >= count:
            raise ValueError("An invalid scalar position exceeds event count")
    finally:
        writer.close()
    generated = list(set(output_dir.glob("events.out.tfevents.*")) - existing)
    if len(generated) != 1:
        raise RuntimeError("Expected exactly one corrected TensorBoard event file")
    return generated[0]


def repair_completed_run() -> tuple[Path, Path]:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    milestones = sorted(report["milestones"].values(),
                        key=lambda item: item["actual_timesteps"])
    invalid_positions = {index for index, item in enumerate(milestones)
                         if item["evaluation"]["mean_speed_within_0_1_m_s"] is None}
    sources = list(EVENT_DIR.glob("events.out.tfevents.*"))
    if len(sources) != 1 or BACKUP_DIR.exists():
        raise RuntimeError("Expected one original event file and no prior correction")
    original = sources[0]
    original_events = EventAccumulator(str(original), size_guidance={"scalars": 0})
    original_events.Reload()
    if len(original_events.Scalars(TAG)) != len(milestones):
        raise RuntimeError("TensorBoard evaluation count disagrees with report")
    corrected = copy_without_undefined_near_speed(original, EVENT_DIR,
                                                  invalid_positions)
    clean_events = EventAccumulator(str(corrected), size_guidance={"scalars": 0})
    clean_events.Reload()
    expected = len(milestones) - len(invalid_positions)
    if len(clean_events.Scalars(TAG)) != expected:
        raise RuntimeError("Corrected near-target speed count is wrong")
    for tag in original_events.Tags()["scalars"]:
        if tag != TAG and len(clean_events.Scalars(tag)) != len(original_events.Scalars(tag)):
            raise RuntimeError(f"Corrected run lost scalar events: {tag}")
    BACKUP_DIR.mkdir(parents=True, exist_ok=False)
    archived = BACKUP_DIR / original.name
    original.rename(archived)
    return corrected, archived


if __name__ == "__main__":
    corrected_path, archived_path = repair_completed_run()
    print(f"Verified corrected event file: {corrected_path}")
    print(f"Original preserved at: {archived_path}")
