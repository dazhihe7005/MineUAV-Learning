"""The completed run's TensorBoard correction preserves real scalar history."""

import sys
import tempfile
import unittest
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_reward_v2_tensorboard import copy_without_undefined_near_speed  # noqa: E402


class TensorBoardRepairTests(unittest.TestCase):
    def test_copy_can_write_beside_original_event_file(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            writer = EventFileWriter(str(directory))
            writer.add_event(Event(step=20, summary=Summary(value=[
                Summary.Value(tag="eval/mean_speed_within_0_1_m_s", simple_value=0.0)])))
            writer.close()
            source = next(directory.glob("events.out.tfevents.*"))
            corrected = copy_without_undefined_near_speed(source, directory,
                                                          invalid_positions={0})
            self.assertNotEqual(corrected, source)
            self.assertTrue(source.is_file())
            self.assertTrue(corrected.is_file())

    def test_undefined_scalar_is_removed_without_touching_other_training_metrics(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source_dir = root / "source"
            source_dir.mkdir()
            writer = EventFileWriter(str(source_dir))
            for step, tag, value in ((20, "train/value_loss", 1.2),
                                     (20, "eval/mean_speed_within_0_1_m_s", 0.0),
                                     (50, "train/value_loss", 0.8),
                                     (50, "eval/mean_speed_within_0_1_m_s", 0.79)):
                writer.add_event(Event(step=step,
                                       summary=Summary(value=[Summary.Value(tag=tag,
                                                                           simple_value=value)])))
            writer.close()
            source = next(source_dir.glob("events.out.tfevents.*"))
            corrected = copy_without_undefined_near_speed(
                source, root / "corrected", invalid_positions={0})
            events = EventAccumulator(str(corrected), size_guidance={"scalars": 0})
            events.Reload()
            self.assertEqual([point.step for point in events.Scalars("train/value_loss")],
                             [20, 50])
            self.assertEqual([point.step for point in events.Scalars(
                "eval/mean_speed_within_0_1_m_s")], [50])
            self.assertTrue(source.is_file(), "raw event file must remain recoverable")


if __name__ == "__main__":
    unittest.main()
