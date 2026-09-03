"""Pure math: project a 3D world point into a camera's 2D pixel space,
given the same OpenGL view/projection matrices PyBullet's cameras use
(calvin_env.camera.static_camera.StaticCamera.viewMatrix/
projectionMatrix, built via pybullet.computeViewMatrix/
computeProjectionMatrixFOV). No pybullet import needed here -- this is
just 4x4 matrix algebra, kept separate from anything simulator-specific
so it stays easy to unit test with hand-built matrices, and so
shortstop.calvin_obstacle_viz (which uses this to draw the privileged
obstacle on top of the real rgb_static camera frames) never needs a live
PyBullet client of its own.
"""
import numpy as np


def project_point(world_point, view_matrix, projection_matrix, width, height):
    """`view_matrix`/`projection_matrix`: PyBullet's flat 16-tuples (their
    own computeViewMatrix/computeProjectionMatrixFOV output) -- OpenGL
    convention, column-major, so reshape(4, 4, order="F") gives the
    matrix that left-multiplies a column vector the way OpenGL intends.

    Returns (pixel_x, pixel_y, depth) -- `depth` is the point's distance
    along the camera's own viewing axis (camera-space -z, i.e. how far
    in front of the camera it is; see project_radius, which needs this
    to scale a sphere's apparent on-screen size), or `None` if the point
    is behind the camera or otherwise not sanely projectable (nothing to
    draw in that case).
    """
    view = np.asarray(view_matrix, dtype=float).reshape(4, 4, order="F")
    proj = np.asarray(projection_matrix, dtype=float).reshape(4, 4, order="F")
    point_h = np.array([*world_point, 1.0])

    camera_space = view @ point_h
    depth = -camera_space[2]  # OpenGL: camera looks down its own -z axis
    if depth <= 1e-6:
        return None

    clip = proj @ camera_space
    if clip[3] <= 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    pixel_x = (ndc[0] + 1.0) / 2.0 * width
    pixel_y = (1.0 - ndc[1]) / 2.0 * height  # NDC is y-up, image rows are y-down
    return float(pixel_x), float(pixel_y), float(depth)


def project_radius(radius, depth, fov_degrees, height):
    """Apparent on-screen radius (pixels) of a sphere of the given real
    `radius` (meters) sitting at `depth` (meters, camera-space distance
    -- see project_point) -- the standard pinhole-camera relationship for
    a vertical field-of-view `fov_degrees` (StaticCamera.fov, which
    pybullet.computeProjectionMatrixFOV takes as the *vertical* FOV)
    rendered into an image `height` pixels tall.

    Approximate, not a physically exact render: assumes the sphere's
    apparent size is small relative to `depth` (a sphere off-center
    technically projects to an ellipse, not a circle) -- good enough for
    a debug marker showing roughly how big/far the obstacle is, not a
    ground-truth silhouette.
    """
    if depth <= 1e-6:
        return 0.0
    focal_length_px = height / (2.0 * np.tan(np.radians(fov_degrees) / 2.0))
    return radius * focal_length_px / depth
