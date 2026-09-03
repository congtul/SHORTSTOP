"""Debug-only visualization for CALVIN's privileged obstacle (Stage 7b).

Renders the sphere-chain safety geometry (shortstop.robot_geometry) moving
through 3D space over one subtask attempt, with the injected X_u obstacle
drawn as a translucent wireframe sphere -- as an animated GIF, using
matplotlib's Agg backend + PillowWriter (both already pulled in by
requirements.txt's matplotlib>=3.7 dependency, no extra install needed).

This is a SEPARATE artifact from any camera frame the policy itself sees
(rgb_obs["rgb_static"]/rgb_gripper) -- never composite this into or derive
it from those frames. The obstacle is intentionally invisible to the
policy's vision input (see docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md's design
decision); this GIF exists purely for a human to sanity-check radius/
placement choices, decoupled from anything the model or the real eval
pipeline touches.
"""
import numpy as np


def _sphere_wireframe(center, radius, resolution=12):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def save_subtask_gif(trajectory, obstacle, out_path, fps=10):
    """`trajectory`: sequence of (4, 3) sphere_centers() arrays (one per
    step, index 0 = the pose before the first action -- see
    run_calvin_unshielded_subtask's `record_trajectory=True`). `obstacle`:
    a shortstop.env.Obstacle, or None (draws just the arm, no sphere).
    Writes an animated GIF to `out_path` (parent directory must already
    exist -- callers create it once up front, see
    scripts/run_calvin_unshielded.py).
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: writing a GIF needs no display
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    trajectory = np.asarray(trajectory)  # (T, 4, 3)

    bounds_points = trajectory.reshape(-1, 3)
    if obstacle is not None:
        bounds_points = np.vstack([
            bounds_points,
            obstacle.center + obstacle.radius,
            obstacle.center - obstacle.radius,
        ])
    low = bounds_points.min(axis=0) - 0.05
    high = bounds_points.max(axis=0) + 0.05

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if obstacle is not None:
        x, y, z = _sphere_wireframe(obstacle.center, obstacle.radius)
        ax.plot_wireframe(x, y, z, color="red", alpha=0.3, linewidth=0.5)

    (chain_artist,) = ax.plot([], [], [], "o-", color="tab:blue")

    def update(frame_idx):
        points = trajectory[frame_idx]
        chain_artist.set_data(points[:, 0], points[:, 1])
        chain_artist.set_3d_properties(points[:, 2])
        ax.set_title(f"step {frame_idx}/{len(trajectory) - 1}")
        return (chain_artist,)

    ani = FuncAnimation(fig, update, frames=len(trajectory), interval=1000.0 / fps, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
