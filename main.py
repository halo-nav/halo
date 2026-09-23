import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hilbert_localizability.reduced_rank_gp import ReducedRankGP

from mppi_nav.mppi import MPPI

jax.config.update("jax_platform_name", "cpu")


def main():
    data = np.loadtxt(
        "halo-data/data/mag.csv",
        delimiter=",",
        skiprows=1,
    )
    x = data[:, 0]
    y = data[:, 1]
    mag = data[:, 3]
    # normalize mag data
    mean_mag = jnp.mean(mag)
    std_mag = jnp.std(mag)
    mag = (mag - mean_mag) / std_mag
    mag = mag.reshape(-1, 1)
    test_points = jnp.meshgrid(jnp.linspace(-3, 3, 100), jnp.linspace(-2, 2, 100))
    xt = jnp.vstack([test_points[0].ravel(), test_points[1].ravel()]).T

    x_train = jnp.vstack([x, y]).T

    rrgp = ReducedRankGP(x_train, mag, num_dimension=2)
    rrgp.gp_optimize_fast()
    y_pred, y_var = rrgp.predict(xt)
    localizability_cost = lambda trajectory: rrgp.localizability(
        trajectory, sigma_R=120, sigma_y=std_mag, mean_y=mean_mag
    )
    mppi = MPPI(
        dynamics_model=None,
        cost_function=localizability_cost,
        horizon=20,
        num_samples=2000,
        loc_weight=1.0,
        vel_weight=0.5,
        lambda_=0.5,
        goal_state=jnp.array([2.5, 1.5, 0.0, 0.0]),
    )

    state = jnp.array([-2.5, -1.5, 0.0, 0.2])
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        test_points[0],
        test_points[1],
        y_pred.reshape(test_points[0].shape),
        shading="auto",
        cmap="magma",
    )
    plt.title("Sampled Trajectories")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.xlim(-3, 3)
    plt.ylim(-2, 2)
    # plt.grid()
    initial_control = jnp.array([0.0, 0.0])
    control_mean = jnp.tile(initial_control, (mppi.horizon, 1))
    import time

    while True:
        # (horizon, 2), writable
        start_time = time.time()
        localization_proxy = jnp.trace(
            rrgp.fisher_information(
                state[:2].reshape(1, 2),
                sigma_R=120,
                sigma_y=std_mag,
                mean_y=mean_mag,
                gamma=0.0,
            ),
            axis1=-2,
            axis2=-1,
        )
        controls, trajectories, costs, weights = mppi.update_control(
            initial_state=state,
            control_mean=control_mean,
        )

        # Apply first control of optimized sequence
        best_control = jnp.clip(
            controls[0],
            min=jnp.array([-1.0, -jnp.pi / 2]),
            max=jnp.array([1.0, jnp.pi / 2]),
        )
        state = mppi.dynamics_model(state, best_control, mppi.delta)

        # Warm-start: shift sequence and zero-pad
        control_mean = jnp.concatenate(
            [controls[1:], controls[-1:]],
            axis=0,
        )
        end_time = time.time()
        print(f"Time taken for one iteration: {end_time - start_time:.4f} seconds")
        print(f"Current state: {state}")
        print(f"Current localization proxy: {localization_proxy}")
        # Plot the trajectories to see what they look like

        plt.pcolormesh(
            test_points[0],
            test_points[1],
            y_pred.reshape(test_points[0].shape),
            shading="auto",
            cmap="magma",
        )
        # plt.title("Sampled Trajectories")
        # plt.xlabel("X position")
        # plt.ylabel("Y position")
        plt.xlim(-3, 3)
        plt.ylim(-2, 2)
        for i in range(trajectories.shape[0]):
            plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], alpha=0.15)

        # plt state
        plt.plot(state[0], state[1], "bo", label="Current State", markersize=20)
        plt.plot(
            trajectories[jnp.argmin(costs), :, 0],
            trajectories[jnp.argmin(costs), :, 1],
            "r-",
            label="Max Cost Trajectory",
            linewidth=5,
        )
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.clear()


if __name__ == "__main__":
    main()
