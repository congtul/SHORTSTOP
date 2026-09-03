"""Debug-only visualization for CALVIN's privileged obstacle (Stage 7b).

Composites the (otherwise invisible) obstacle onto the SAME rgb_static
camera frames the policy itself sees (see camera_projection.py for the
pure-math 3D->2D projection), producing an MP4 that looks like the real
CALVIN eval pipeline's own rollout recording
(mdt.rollout.rollout_video.RolloutVideo, which is literally what
results/calvin_eval.gif is -- unnormalized obs["rgb_obs"]["rgb_static"]
frames, one per step), plus a red circle marking where X_u actually is.
Earlier revisions of this module instead plotted an abstract matplotlib
scatter of the arm's own sphere/frame chain -- dropped entirely, since a
schematic plot in its own coordinate frame is much harder to sanity-check
by eye than the real rendered scene the policy conditions on. A later
revision wrote a GIF instead of MP4 -- also dropped, since GIF's 256-
color-per-frame palette and lack of inter-frame motion compression make
it *larger*, not smaller, than a real video codec for photorealistic
camera footage like this (GIF was only ever chosen for the earlier
schematic-plot version's convenience, not for any size/quality reason).

This is a SEPARATE artifact, built strictly AFTER the real rollout's
actions have already been decided and the env has already stepped --
drawing this overlay never adds anything to the live PyBullet scene and
never changes what obs the policy receives; it only paints on top of an
already-captured image, using the camera's own (real, not approximated)
viewMatrix/projectionMatrix purely to know where to paint. See
docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md's design decision for why the
obstacle must stay invisible to the policy's own vision input.
"""
import numpy as np
from PIL import Image, ImageDraw

from .camera_projection import project_point, project_radius


def _import_image_sequence_clip():
    """moviepy's import path changed across major versions: <2.0 exposes
    ImageSequenceClip under moviepy.editor (the same import mdt_policy's
    own mdt/rollout/rollout_video.py uses to write calvin_eval.gif, and
    what mdt_policy/requirements.txt's unpinned `moviepy` line could
    resolve to depending on when a given env was set up), >=2.0 removed
    that submodule and exposes it directly from the top-level package.
    Try both rather than assuming one, since either could be what's
    actually installed on a given machine."""
    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        from moviepy import ImageSequenceClip
    return ImageSequenceClip


def _quaternion_to_rotation_matrix(quaternion):
    """pybullet's xyzw quaternion convention -> 3x3 rotation matrix, pure
    numpy (no pybullet import needed -- just the 4 components as plain
    floats, e.g. straight from robot.base_orientation)."""
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _to_world_frame(local_point, base_position, base_orientation):
    """calvin_obstacle.sample_obstacle_from_reference_chunk's obstacle.
    center is deliberately expressed in the robot's own *local* base
    frame (see that module's docstring: the safety math never needs the
    base-to-world transform, since both the obstacle and the real
    per-step joint angles live in that same local frame). But CALVIN's
    robot base does NOT sit at the world origin (e.g.
    calvin_env/conf/scene/calvin_scene_D.yaml's robot_base_position =
    [-0.34, -0.46, 0.24]), and the camera's viewMatrix is built in true
    world coordinates -- so only this visualization needs to leave the
    local frame, via the same base_position/base_orientation
    (robot.base_position/robot.base_orientation, the latter already a
    quaternion) PyBullet placed the robot's URDF at."""
    rotation = _quaternion_to_rotation_matrix(base_orientation)
    return np.asarray(base_position) + rotation @ np.asarray(local_point)


def _overlay_obstacle(frame, obstacle, camera, base_position=(0.0, 0.0, 0.0), base_orientation=(0.0, 0.0, 0.0, 1.0)):
    """`frame`: HxWx3 uint8 array (one rgb_static frame). `camera`: the
    calvin_env StaticCamera object that rendered it (read-only here --
    its .viewMatrix/.projectionMatrix/.fov/.height, never mutated).
    `base_position`/`base_orientation`: the robot's base pose in world
    coordinates (see _to_world_frame) -- default is the identity
    transform (local frame == world frame), for callers/tests that don't
    care about the offset. Returns a new PIL Image with the obstacle
    drawn as a red circle, or an unmodified copy if `obstacle` is None or
    projects behind the camera (nothing sane to draw)."""
    image = Image.fromarray(frame).convert("RGB")
    if obstacle is None:
        return image

    world_center = _to_world_frame(obstacle.center, base_position, base_orientation)
    projected = project_point(world_center, camera.viewMatrix, camera.projectionMatrix, camera.width, camera.height)
    if projected is None:
        return image

    pixel_x, pixel_y, depth = projected
    pixel_r = max(2.0, project_radius(obstacle.radius, depth, camera.fov, camera.height))
    ImageDraw.Draw(image).ellipse(
        [pixel_x - pixel_r, pixel_y - pixel_r, pixel_x + pixel_r, pixel_y + pixel_r],
        outline=(255, 0, 0), width=2,
    )
    return image


def save_sequence_video(
    subtask_records, camera, out_path, fps=10, freeze_frames=5,
    base_position=(0.0, 0.0, 0.0), base_orientation=(0.0, 0.0, 0.0, 1.0),
):
    """Merge every subtask attempt of one sequence into a single MP4,
    each frame the real rgb_static camera image with that subtask's
    obstacle (if any) composited on top -- matches how the real CALVIN
    eval pipeline records one continuous video per sequence, not a
    separate file per subtask attempt.

    `subtask_records`: list of dicts, one per subtask attempt in order,
    each {"subtask": str, "frames": [HxWx3 uint8 rgb_static array, ...]
    (one per step of that attempt, see
    shortstop.calvin_experiment.run_calvin_unshielded_subtask's
    `record_camera_frames=True`), "obstacle": Obstacle or None,
    "outcome": "reached"|"violated"|"failed"}. `camera`: the StaticCamera
    that rendered every one of these frames (the same viewMatrix/
    projectionMatrix must have produced all of them, or the overlay
    would be projected against the wrong camera pose).

    `base_position`/`base_orientation`: the robot's base pose in world
    coordinates (env.env.robot.base_position/.base_orientation) -- every
    `obstacle.center` is expressed in that base's own local frame (see
    _to_world_frame), so this must match the same env instance that
    produced `subtask_records`, or the overlay lands in the wrong place.

    `freeze_frames`: how many times to repeat a subtask's final
    (overlaid) frame before moving on to the next subtask -- mirrors
    RolloutVideo.draw_outcome()'s "hold the last frame" pattern from the
    real eval pipeline, so a human can tell subtasks apart within one
    continuous file without needing per-subtask borders/colors (unlike
    the real pipeline, this video has no text-overlay step here -- keep
    it simple, the point is the obstacle placement, not a full outcome
    UI). `out_path` should end in ".mp4" -- libx264 needs even width/
    height (CALVIN's static camera is configured at 200x200, already
    even, see conf/cameras/cameras/static.yaml).
    """
    ImageSequenceClip = _import_image_sequence_clip()

    frames = []
    for record in subtask_records:
        for raw_frame in record["frames"]:
            overlaid = _overlay_obstacle(raw_frame, record["obstacle"], camera, base_position, base_orientation)
            frames.append(np.asarray(overlaid))
        frames.extend([frames[-1]] * freeze_frames)

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(out_path), codec="libx264", audio=False, logger=None)
