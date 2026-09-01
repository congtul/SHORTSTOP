import numpy as np

from shortstop.reach import Box, propagate, propagate_tube


def test_box_intersects_circle():
    box = Box(low=[0.0, 0.0], high=[1.0, 1.0])
    assert box.intersects_circle(center=np.array([1.5, 0.5]), radius=0.6)
    assert not box.intersects_circle(center=np.array([3.0, 3.0]), radius=0.5)


def test_propagate_inflates_box():
    box = Box.point([0.0, 0.0])
    new_box = propagate(box, action=[1.0, 0.0], dt=0.1, w_bar=0.02)
    assert np.allclose(new_box.center(), [0.1, 0.0])
    assert np.allclose(new_box.high - new_box.low, [0.04, 0.04])


def test_propagate_tube_length():
    tube = propagate_tube([0.0, 0.0], actions=np.zeros((8, 2)), dt=0.1, w_bar=0.02)
    assert len(tube) == 9  # R_0..R_8
