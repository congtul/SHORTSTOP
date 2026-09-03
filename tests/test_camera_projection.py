"""No pybullet needed here -- camera_projection.py is pure matrix math
over the same flat 16-tuples pybullet.computeViewMatrix/
computeProjectionMatrixFOV produce, so these tests build that exact kind
of OpenGL view/projection matrix by hand (standard lookAt + perspective
formulas) rather than depending on a live PyBullet client."""
import numpy as np

from shortstop.camera_projection import project_point, project_radius


def _look_at(eye, target, up):
    eye, target, up = np.array(eye, dtype=float), np.array(target, dtype=float), np.array(up, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    m = np.eye(4)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m.flatten(order="F")  # OpenGL/pybullet convention: column-major flat


def _perspective(fov_degrees, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_degrees) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m.flatten(order="F")


EYE = [0.0, -1.0, 0.5]
TARGET = [0.0, 0.0, 0.3]
UP = [0.0, 0.0, 1.0]
FOV = 60.0
WIDTH, HEIGHT = 200, 200
VIEW = _look_at(EYE, TARGET, UP)
PROJ = _perspective(FOV, WIDTH / HEIGHT, 0.01, 5.0)


def test_project_point_puts_the_look_at_target_at_image_center():
    result = project_point(TARGET, VIEW, PROJ, WIDTH, HEIGHT)
    assert result is not None
    pixel_x, pixel_y, depth = result
    assert np.isclose(pixel_x, WIDTH / 2, atol=1e-6)
    assert np.isclose(pixel_y, HEIGHT / 2, atol=1e-6)
    assert np.isclose(depth, np.linalg.norm(np.array(TARGET) - np.array(EYE)))


def test_project_point_returns_none_for_a_point_behind_the_camera():
    behind = [0.0, -2.0, 0.5]  # further from target than the eye itself, along -forward
    assert project_point(behind, VIEW, PROJ, WIDTH, HEIGHT) is None


def test_project_point_moves_right_for_a_point_offset_toward_camera_right():
    offset = [TARGET[0] + 0.1, TARGET[1], TARGET[2]]
    result = project_point(offset, VIEW, PROJ, WIDTH, HEIGHT)
    assert result is not None
    pixel_x, _, _ = result
    assert pixel_x > WIDTH / 2


def test_project_radius_is_positive_and_shrinks_with_depth():
    near = project_radius(0.05, depth=1.0, fov_degrees=FOV, height=HEIGHT)
    far = project_radius(0.05, depth=2.0, fov_degrees=FOV, height=HEIGHT)
    assert near > 0
    assert far > 0
    assert far < near  # farther away -> smaller on screen


def test_project_radius_is_zero_for_nonpositive_depth():
    assert project_radius(0.05, depth=0.0, fov_degrees=FOV, height=HEIGHT) == 0.0
