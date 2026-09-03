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


def _overlay_obstacle(frame, obstacle, camera):
    """`frame`: HxWx3 uint8 array (one rgb_static frame). `camera`: the
    calvin_env StaticCamera object that rendered it (read-only here --
    its .viewMatrix/.projectionMatrix/.fov/.height, never mutated).
    Returns a new PIL Image with the obstacle drawn as a red circle, or
    an unmodified copy if `obstacle` is None or projects behind the
    camera (nothing sane to draw)."""
    image = Image.fromarray(frame).convert("RGB")
    if obstacle is None:
        return image

    projected = project_point(obstacle.center, camera.viewMatrix, camera.projectionMatrix, camera.width, camera.height)
    if projected is None:
        return image

    pixel_x, pixel_y, depth = projected
    pixel_r = max(2.0, project_radius(obstacle.radius, depth, camera.fov, camera.height))
    ImageDraw.Draw(image).ellipse(
        [pixel_x - pixel_r, pixel_y - pixel_r, pixel_x + pixel_r, pixel_y + pixel_r],
        outline=(255, 0, 0), width=2,
    )
    return image


def save_sequence_video(subtask_records, camera, out_path, fps=10, freeze_frames=5):
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
            overlaid = _overlay_obstacle(raw_frame, record["obstacle"], camera)
            frames.append(np.asarray(overlaid))
        frames.extend([frames[-1]] * freeze_frames)

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(str(out_path), codec="libx264", audio=False, logger=None)
