import jax
import jax.numpy as jnp


class MPPI:
    """
    Model Predictive Path Integral (MPPI) controller.
    """

    def __init__(
        self,
        dynamics_model,
        cost_function,
        horizon=100,
        num_samples=2000,
        lambda_=0.5,
        noise_sigma=jnp.array([0.5, 2.0]),
        process_noise=jnp.diag(jnp.array([0.05, 0.05, 0.005])),
        goal_state=jnp.array([1.0, 1.0, 0.0, 0.0]),
        delta=0.1,
        loc_weight=0.0,
        goal_weight=1.0,
        vel_weight=0.5,
        key=11234,
    ) -> None:
        """
        Initialization of the MPPI controller. The MPPI controller is a model predictive
        control algorithm that uses a sampling-based approach to optimize control inputs
        over a finite time horizon.

        Args:
            dynamics_model (functional): A function that takes the current state and control input and noise and returns the next state.
            cost_function (functional): A function that takes a trajectory of states and control inputs and returns a scalar cost.
            horizon (int): The number of time steps to look ahead in the future.
            num_samples (int): The number of control sequences to sample at each time step.
            lambda_ (float): The temperature parameter that controls the exploration-exploitation trade-off.
            noise_sigma (float): The standard deviation of the Gaussian noise added to the control inputs for sampling
            delta (float): The time step size for the dynamics propagation.
        """
        self.dynamics_model = (
            dynamics_model
            if dynamics_model is not None
            else self.unicycle_model
        )
        self.cost_function = cost_function
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        self.noise_sigma = noise_sigma
        self.delta = delta
        self.rng_key = jax.random.PRNGKey(key)
        self.process_noise = process_noise
        self.goal_state = goal_state
        self.loc_weight = loc_weight
        self.goal_weight = goal_weight
        self.vel_weight = vel_weight

        self._sample_compiled = jax.jit(self._sample_rollout)

    def _sample_rollout(
        self,
        key,
        control_mean,
        initial_state,
    ):
        def normalize(x):
            return (x - jnp.mean(x, axis=0)) / (jnp.std(x, axis=0) + 1e-8)

        control_dim = control_mean.shape[-1]
        noise = 1 * jax.random.normal(
            key, shape=(self.num_samples, self.horizon, control_dim)
        )
        noise = noise * self.noise_sigma[None, None, :]
        controls = control_mean[None, :, :] + noise
        controls = controls.at[:, :, 0].set(
            jnp.clip(controls[:, :, 0], -1.0, 1.0)
        )

        trajectories = jax.vmap(self.trajectory_rollout, in_axes=(None, 0))(
            initial_state, controls
        )

        positions = trajectories[:, :, :2].reshape(
            self.num_samples * self.horizon, 2
        )

        positions = trajectories[:, :, :2].reshape(
            self.num_samples * self.horizon, 2
        )
        dist_to_goal = jnp.linalg.norm(
            trajectories[:, :, :2] - self.goal_state[None, None, :2],
            axis=-1,
        )
        loc_scale = jnp.tanh(dist_to_goal / 0.5)

        loc_costs = (
            -self.cost_function(positions).reshape(
                self.num_samples, self.horizon
            )
            * loc_scale
        ).sum(axis=1)

        pos_diff = trajectories[:, :, :2] - self.goal_state[None, None, :2]
        running_goal_costs = jnp.sum(pos_diff**2, axis=-1).sum(axis=1)

        proximity_weight = jnp.exp(-dist_to_goal)
        running_vel_costs = (
            proximity_weight * trajectories[:, :, 3] ** 2
        ).sum(axis=1)

        control_costs = jnp.sum(controls**2, axis=-1).sum(axis=1)

        terminal_costs = jax.vmap(self.terminal_cost, in_axes=(0, None))(
            trajectories[:, -1, :], self.goal_state
        )
        costs = (
            self.loc_weight * normalize(loc_costs)
            + 3.0 * normalize(running_goal_costs)
            + self.vel_weight * normalize(running_vel_costs)
            + 0.1 * normalize(control_costs)
            + self.goal_weight * normalize(terminal_costs)
        )

        return noise, controls, trajectories, costs

    def sample_trajectories(self, initial_state, control_mean):
        """
        Samples control sequences and generates corresponding trajectories.

        Args:
            initial_state (np.ndarray): The initial state of the system.
            control_mean (np.ndarray): The mean control input [v,omega], usually is the previous optimized control.
        Returns:
            np.ndarray: Sampled control sequences of shape (num_samples, horizon, control_dim).
            np.ndarray: Corresponding trajectories of shape (num_samples, horizon, state_dim).
            np.ndarray: Costs associated with each trajectory of shape (num_samples,).
        """
        initial_state = jnp.asarray(initial_state)
        control_mean = jnp.asarray(control_mean)

        if control_mean.shape != (self.horizon, self.noise_sigma.size):
            raise ValueError(
                f"control_mean must have shape ({self.horizon},{self.noise_sigma.size}), but got {control_mean.shape}"
            )

        self.rng_key, sample_key = jax.random.split(self.rng_key)

        return self._sample_compiled(sample_key, control_mean, initial_state)

    def trajectory_rollout(
        self,
        initial_state: jnp.ndarray,
        controls: jnp.ndarray,
    ):
        """
        Generates a trajectory by propagating dynamics into the future for a given control sequence and initial state.

        Args:
            control (jax.numpy.ndarray): Control sequence of shape (horizon, control_dim).
            initial_state (jax.numpy.ndarray): Initial state of shape (state_dim,).
            horizon (int): Number of time steps to propagate.
        Returns:
            jax.numpy.ndarray: Predicted states of shape (horizon, state_dim).
        """

        def step(state, control):
            next_state = self.dynamics_model(state, control, self.delta)
            return next_state, next_state

        _, trajectory = jax.lax.scan(step, initial_state, controls)
        return trajectory

    @staticmethod
    def unicycle_model(state, control, delta):
        """
        Unicycle dynamics model.

        Args:
            state (np.ndarray): Current state of the system [x, y, theta,v].
            control (np.ndarray): Control input [a, omega].
            delta (float): Time step size.
        Returns:
            np.ndarray: Next state of the system [x_next, y_next, theta_next].
        """
        x = state[0]
        y = state[1]
        theta = state[2]
        v = state[3]

        a = control[0]
        omega = control[1]

        x_next = x + v * jnp.cos(theta) * delta
        y_next = y + v * jnp.sin(theta) * delta
        theta_next = jnp.arctan2(
            jnp.sin(theta + omega * delta), jnp.cos(theta + omega * delta)
        )  # Normalize angle to [-pi, pi]
        v_next = jnp.clip(v + a * delta, 0.0, 1.0)

        return jnp.stack([x_next, y_next, theta_next, v_next])

    def update_control(self, initial_state, control_mean):
        noise, controls, trajectories, costs = self.sample_trajectories(
            initial_state, control_mean
        )
        shifted_costs = costs - jnp.min(costs)
        weights = jnp.exp(-shifted_costs / self.lambda_)
        weights = weights / jnp.sum(weights)  # Normalize to sum to 1

        weighted_noise = jnp.einsum("k,khu->hu", weights, noise)
        updated_controls = control_mean + weighted_noise

        # Clip to physical limits
        updated_controls = updated_controls.at[:, 0].set(
            jnp.clip(updated_controls[:, 0], -2.0, 2.0)  # acceleration
        )
        updated_controls = updated_controls.at[:, 1].set(
            jnp.clip(updated_controls[:, 1], -2.0, 2.0)  # angular velocity
        )
        return updated_controls, trajectories, costs, weights

    def terminal_cost(self, final_trajectory_state, goal_state):
        """
        Computes the terminal cost for a given state.

        Args:
            final_trajectory_state (jax.numpy.ndarray): The state at the end of the horizon.
            goal_state (jax.numpy.ndarray): The desired goal state.
        Returns:
            float: The terminal cost associated with the state.
        """
        pos_diff = final_trajectory_state[:2] - goal_state[:2]
        vel_diff = final_trajectory_state[3] - goal_state[3]
        pos_cost = jnp.dot(pos_diff, pos_diff)
        # vel_cost = vel_diff**2
        dist = jnp.linalg.norm(pos_diff)
        vel_cost = final_trajectory_state[3] ** 2  # Only near goal
        terminal_cost = pos_cost + self.vel_weight * vel_cost
        return terminal_cost
