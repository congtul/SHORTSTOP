import numpy as np

from shortstop.calvin_obstacle_viz import _overlay_obstacle, _to_world_frame, save_sequence_video
from shortstop.env import Obstacle


class _FakeCamera:
    """Minimal stand-in for calvin_env's StaticCamera -- only the
    attributes save_sequence_video/camera_projection actually read."""

    def __init__(self, width=32, height=32, fov=60.0):
        self.width = width
        self.height = height
        self.fov = fov
        # a camera sitting on +y looking at the origin, +z up -- any
        # valid OpenGL view/projection matrix works, the exact pose
        # doesn't matter for these tests (see test_camera_projection.py
        # for projection-math correctness itself).
        eye, target, up = np.array([0.0, 1.0, 0.5]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = true_up
        view[2, :3] = -forward
        view[:3, 3] = -view[:3, :3] @ eye
        self.viewMatrix = view.flatten(order="F")

        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        proj = np.zeros((4, 4))
        proj[0, 0] = f / (width / height)
        proj[1, 1] = f
        proj[2, 2] = (5.0 + 0.01) / (0.01 - 5.0)
        proj[2, 3] = (2 * 5.0 * 0.01) / (0.01 - 5.0)
        proj[3, 2] = -1.0
        self.projectionMatrix = proj.flatten(order="F")


def _fake_frames(n_steps, height=32, width=32):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8) for _ in range(n_steps)]


def test_to_world_frame_applies_translation_and_rotation():
    # identity orientation: pure translation.
    world = _to_world_frame([1.0, 2.0, 3.0], base_position=[-0.34, -0.46, 0.24], base_orientation=[0.0, 0.0, 0.0, 1.0])
    assert np.allclose(world, [0.66, 1.54, 3.24])

    # 90-degree yaw (matches basic_playtable.yaml's robot_base_orientation
    # convention): local +x should end up along world +y.
    quarter_turn_z = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    world = _to_world_frame([1.0, 0.0, 0.0], base_position=[0.0, 0.0, 0.0], base_orientation=quarter_turn_z)
    assert np.allclose(world, [0.0, 1.0, 0.0], atol=1e-9)


def test_overlay_obstacle_moves_when_base_offset_is_supplied():
    """Root-cause regression test: sample_obstacle_from_reference_chunk's
    obstacle.center is in the robot's local base frame (see
    shortstop.calvin_obstacle's docstring), not world coordinates -- a
    real CALVIN scene's robot base sits away from the world origin (e.g.
    calvin_scene_D.yaml's [-0.34, -0.46, 0.24]), so ignoring the offset
    projects the obstacle to the wrong pixel entirely. Drawing with vs.
    without the offset from the same local-frame obstacle must land on
    different pixels."""
    camera = _FakeCamera()
    frame = _fake_frames(1)[0]
    obstacle = Obstacle(center=[0.0, 0.0, 0.0], radius=0.05)

    without_offset = np.asarray(_overlay_obstacle(frame, obstacle, camera))
    with_offset = np.asarray(_overlay_obstacle(
        frame, obstacle, camera, base_position=[0.3, 0.15, 0.6], base_orientation=[0.0, 0.0, 0.0, 1.0],
    ))
    assert not np.array_equal(without_offset, with_offset)


def test_save_sequence_video_writes_a_nonempty_file_with_and_without_obstacle(tmp_path):
    camera = _FakeCamera()
    subtask_records = [
        {
            "subtask": "fake_subtask_a",
            "frames": _fake_frames(4),
            "obstacle": Obstacle(center=[0.0, 0.0, 0.0], radius=0.05),
            "outcome": "violated",
        },
        {
            "subtask": "fake_subtask_b",
            "frames": _fake_frames(3),
            "obstacle": None,
            "outcome": "reached",
        },
    ]
    out_path = tmp_path / "sequence.mp4"

    save_sequence_video(subtask_records, camera, str(out_path), fps=5, freeze_frames=2)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_sequence_video_duration_includes_freeze_frames(tmp_path):
    """Checks the written MP4's duration matches the total number of
    logical frames requested -- (2 + 2 freeze) + (3 + 2 freeze) = 9
    frames at fps=5 -> 1.8s -- not just that some file got written."""
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip

    camera = _FakeCamera()
    subtask_records = [
        {"subtask": "a", "frames": _fake_frames(2), "obstacle": None, "outcome": "reached"},
        {"subtask": "b", "frames": _fake_frames(3), "obstacle": None, "outcome": "failed"},
    ]
    out_path = tmp_path / "sequence.mp4"
    save_sequence_video(subtask_records, camera, str(out_path), fps=5, freeze_frames=2)

    with VideoFileClip(str(out_path)) as clip:
        duration = clip.duration
    assert np.isclose(duration, 9 / 5, atol=0.15)
