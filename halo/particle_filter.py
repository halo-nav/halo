import jax
import jax.numpy as jnp


def deg2rad(x):
    """This function converts degrees to radians"""
    return x * jnp.pi / 180


def rad2deg(rad):
    return rad * 180.0 / jnp.pi


class ParticleFilter:
    """Particle Filter class for localization. This class implements the particle filter algorithm for localization in an magnetic anomaly map.

    It has all the steps housed in functions, and calculates it's own likelihood. The resample methods available are SIR, Stratified, Systematic
    and Gausssian around estimated pose. The default motion model used here is a simple unicycle model. The map given is also interpolated to a
    finer grid as a magnetic map is able to be interpolated. The magnetic map is given as a csv file with columns x, y, A, mag, phi. There are functions
    present to convert a pose to a magnetic measurement and vice versa."""

    def __init__(
        self,
        gp_model,
        num_particles: int = 1000,
        init_pose: jnp.ndarray = jnp.array([[0.0, 0.0, 0.0]]).T,
        Q: jnp.ndarray = jnp.diag(jnp.array([0.01, 0.01, deg2rad(0.5)])),
        R: float = 100.0,
        sigma_y: float = 1.0,
        mean_y: float = 0.0,
        x_range: tuple = (-3.0, 3.0),
        y_range: tuple = (-2.0, 2.0),
        dt: float = 0.05,
        seed: int = 0,
    ):
        """Initialize particle filter

        Parameters:
            num_particles -> number of particles
            Q -> process noise covariance matrix
            R -> measurement noise covariance matrix
            dt -> time step
            control_input -> control input vector
        Returns: None
        """
        self.gp_model = gp_model
        self.num_particles = num_particles
        self.Q = Q
        self.R = R
        self.sigma_y = sigma_y
        self.mean_y = mean_y
        self.x_range = x_range
        self.y_range = y_range
        self.dt = dt

        self.rng_key = jax.random.PRNGKey(seed)

        self.particles, self.weights = self._initialize(
            jax.random.PRNGKey(seed), init_pose
        )

        self._predict_jit = jax.jit(self._predict)
        self._update_jit = jax.jit(self._update)
        self._resample_jit = jax.jit(self._resample)

    def _initialize(self, key, robot_pose, gaussian=True):
        if gaussian:
            particles = jax.random.multivariate_normal(
                key,
                mean=robot_pose,
                cov=self.Q**2,
                shape=(self.num_particles,),
            )
            # Add some random shift to the x and y positions
            particles = particles.at[:, 0].set(
                particles[:, 0]
                + jax.random.uniform(
                    key, shape=(self.num_particles,), minval=-0.1, maxval=0.1
                )
            )
            particles = particles.at[:, 1].set(
                particles[:, 1]
                + jax.random.uniform(
                    key, shape=(self.num_particles,), minval=-0.1, maxval=0.1
                )
            )
            particles = particles.at[:, 2].set(particles[:, 2] % (2 * jnp.pi))
            weights = jnp.ones(self.num_particles) / self.num_particles
            return particles, weights
        else:
            x = jax.random.uniform(
                key,
                shape=(self.num_particles,),
                minval=self.x_range[0],
                maxval=self.x_range[1],
            )
            y = jax.random.uniform(
                key,
                shape=(self.num_particles,),
                minval=self.y_range[0],
                maxval=self.y_range[1],
            )
            theta = jax.random.uniform(
                key, shape=(self.num_particles,), minval=0.0, maxval=2 * jnp.pi
            )
            particles = jnp.stack([x, y, theta], axis=1)
            weights = jnp.ones(self.num_particles) / self.num_particles
            return particles, weights

    @staticmethod
    def _predict(particles, control, Q, dt, key):

        v, omega = control[0], control[1]
        noise = jax.random.multivariate_normal(
            key, mean=jnp.zeros(3), cov=Q**2 * dt, shape=(particles.shape[0],)
        )

        dx = v * jnp.cos(particles[:, 2]) * dt
        dy = v * jnp.sin(particles[:, 2]) * dt
        dtheta = omega * dt

        new_particles = (
            particles
            + jnp.stack([dx, dy, jnp.full(particles.shape[0], dtheta)], axis=1)
            + noise
        )

        new_particles = new_particles.at[:, 2].set(
            new_particles[:, 2] % (2 * jnp.pi)
        )

        return new_particles

    def _update(self, particles, weights, measurement, R, sigma_y, mean_y):

        pred_mean, var_mean = self.gp_model.predict(particles[:, :2])

        pred_mean = pred_mean.ravel() * sigma_y + mean_y
        pred_var = var_mean.ravel() * sigma_y**2

        total_var = pred_var + R**2
        residual = measurement - pred_mean

        log_w = -(0.5 * residual**2 / total_var) - 0.5 * jnp.log(
            2 * jnp.pi * total_var
        )

        log_w += jnp.log(weights.ravel()) + 1e-30
        log_w = log_w - jnp.max(log_w)  # for numerical stability

        weights = jnp.exp(log_w)
        weights /= jnp.sum(weights)

        return weights.squeeze()

    @staticmethod
    def _resample(particles, weights, key):

        N = particles.shape[0]
        cumsum = jnp.cumsum(weights)

        u0 = jax.random.uniform(key, minval=0.0, maxval=1.0 / N)
        strata = u0 + jnp.arange(N) / N

        indices = jnp.searchsorted(cumsum, strata)
        indices = jnp.clip(indices, 0, N - 1)

        new_particles = particles[indices]
        new_weights = jnp.ones(N) / N

        return new_particles, new_weights

    def neff(self) -> float:
        return float(1.0 / jnp.sum(self.weights**2))

    def predict(
        self,
        u,
    ):
        """The predict step will take the particles and predict them using the control input.
        Since the control input will cause imperfect motion, we will add noise to the particles
        using the process noise covariance matrix Q.

        Parameters:
            u -> control input vector
            dt -> time step
        Returns:
            particles -> particles after prediction step
        """

        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.particles = self._predict_jit(
            self.particles, u, self.Q, self.dt, subkey
        ) + jax.random.multivariate_normal(
            subkey,
            mean=jnp.zeros(3),
            cov=self.Q**2,
            shape=(self.num_particles,),
        )

    def update(self, z_k):
        """The update step will update the weights of the particles based on the measurement z by calculating a likelihood.

        Parameters:
            particles -> particles after prediction step
            weights -> weights after prediction step
            z -> measurement vector
            z_k -> measurement at current time step
            R -> measurement noise covariance matrix
        Returns:
            weights -> weights after update step
        """

        self.weights = self._update_jit(
            self.particles,
            self.weights,
            z_k,
            self.R,
            self.sigma_y,
            self.mean_y,
        )

    def resample(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        self.particles, self.weights = self._resample_jit(
            self.particles, self.weights, subkey
        )

    def step(self, control, measurement):
        self.predict(control)
        self.update(measurement)
        if self.neff() < self.num_particles / 2:
            self.resample()

    @property
    def estimate(self):

        pose_est = jnp.sum(self.weights[:, None] * self.particles, axis=0)

        yaw = jnp.arctan2(
            jnp.sum(self.weights * jnp.sin(self.particles[:, 2])),
            jnp.sum(self.weights * jnp.cos(self.particles[:, 2])),
        ) % (2 * jnp.pi)

        return pose_est.at[2].set(yaw)

    def covariance(self):
        diff = self.particles - self.estimate[None, :]

        return jnp.einsum("n,ni,nj->ij", self.weights, diff, diff)

    def rmse(self, true_state: jnp.ndarray):
        return float(
            jnp.linalg.norm(self.estimate[:2] - true_state[:2]) / jnp.sqrt(2)
        )
