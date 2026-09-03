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


_OCCLUSION_EPSILON = 0.01  # meters -- avoids z-fighting flicker when the obstacle sits ~on a real surface
_OFFSCREEN_MARKER_RADIUS = 3.0  # pixels
_OFFSCREEN_MARKER_MARGIN = 6.0  # pixels -- keeps the clamped marker fully inside the frame, not right on the edge


def _is_occluded(depth_frame, pixel_x, pixel_y, depth, width, height):
    """True if real scene geometry (the table, the arm, ...) is actually
    closer to the camera than the obstacle at this exact pixel, per
    `depth_frame` (see calvin_experiment._camera_frame -- real per-pixel
    camera-space depth in meters, straight from calvin_env's own
    renderer). A pixel outside the frame can't be occluded by anything
    in it (nothing to look up), so this only ever returns True for
    on-screen pixels."""
    ix, iy = int(round(pixel_x)), int(round(pixel_y))
    if not (0 <= ix < width and 0 <= iy < height):
        return False
    return depth_frame[iy, ix] < depth - _OCCLUSION_EPSILON


def _overlay_obstacle(
    frame, obstacle, camera, base_position=(0.0, 0.0, 0.0), base_orientation=(0.0, 0.0, 0.0, 1.0), depth_frame=None,
):
    """`frame`: HxWx3 uint8 array (one rgb_static frame). `camera`: the
    calvin_env StaticCamera object that rendered it (read-only here --
    its .viewMatrix/.projectionMatrix/.fov/.height, never mutated).
    `base_position`/`base_orientation`: the robot's base pose in world
    coordinates (see _to_world_frame) -- default is the identity
    transform (local frame == world frame), for callers/tests that don't
    care about the offset.

    `depth_frame`: HxW float32 real camera-space depth in meters (see
    calvin_experiment._camera_frame), or None to skip the occlusion check
    entirely (always drawn, the old behavior). A privileged obstacle has
    no real presence in the PyBullet scene, so if something real (the
    table, the arm itself) is genuinely closer to the camera at that
    exact pixel, a real object at the obstacle's position would be
    hidden behind it too -- drawing the marker anyway would look like it
    were floating in front of solid geometry it should be behind.

    When the obstacle projects to a pixel outside the visible frame
    (still in front of the camera -- see project_point -- just outside
    the static camera's narrow 10-degree FOV, conf/cameras/cameras/
    static.yaml), a small marker is clamped to the nearest edge instead
    of drawing nothing, so it reads as "obstacle is nearby but out of
    view" rather than silently vanishing.

    Returns a new PIL Image with the obstacle drawn, or an unmodified
    copy if `obstacle` is None, projects behind the camera, or is
    occluded (nothing sane/visible to draw)."""
    image = Image.fromarray(frame).convert("RGB")
    if obstacle is None:
        return image

    world_center = _to_world_frame(obstacle.center, base_position, base_orientation)
    projected = project_point(world_center, camera.viewMatrix, camera.projectionMatrix, camera.width, camera.height)
    if projected is None:
        return image

    pixel_x, pixel_y, depth = projected
    if depth_frame is not None and _is_occluded(depth_frame, pixel_x, pixel_y, depth, camera.width, camera.height):
        return image

    draw = ImageDraw.Draw(image)
    pixel_r = max(2.0, project_radius(obstacle.radius, depth, camera.fov, camera.height))
    fully_offscreen = (
        pixel_x + pixel_r < 0 or pixel_x - pixel_r > camera.width
        or pixel_y + pixel_r < 0 or pixel_y - pixel_r > camera.height
    )
    if fully_offscreen:
        clamped_x = min(max(pixel_x, _OFFSCREEN_MARKER_MARGIN), camera.width - _OFFSCREEN_MARKER_MARGIN)
        clamped_y = min(max(pixel_y, _OFFSCREEN_MARKER_MARGIN), camera.height - _OFFSCREEN_MARKER_MARGIN)
        draw.ellipse(
            [clamped_x - _OFFSCREEN_MARKER_RADIUS, clamped_y - _OFFSCREEN_MARKER_RADIUS,
             clamped_x + _OFFSCREEN_MARKER_RADIUS, clamped_y + _OFFSCREEN_MARKER_RADIUS],
            fill=(255, 0, 0),
        )
    else:
        draw.ellipse(
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
    "outcome": "reached"|"violated"|"failed"}, plus optionally
    "depth_frames" (that same attempt's `depth_frames`, same length as
    "frames") to enable _overlay_obstacle's occlusion check -- omit or
    set to None to draw unconditionally (no occlusion check) instead.
    `camera`: the StaticCamera that rendered every one of these frames
    (the same viewMatrix/projectionMatrix must have produced all of
    them, or the overlay would be projected against the wrong camera
    pose).

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
        depth_frames = record.get("depth_frames")
        for step, raw_frame in enumerate(record["frames"]):
            depth_frame = depth_frames[step] if depth_frames is not None else None
            overlaid = _overlay_obstacle(
                raw_frame, record["obstacle"], camera, base_position, base_orientation, depth_frame,
            )
            frames.append(np.asarray(overlaid))
        frames.extend([frames[-1]] * freeze_frames)

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(out_path), codec="libx264", audio=False, logger=None)
