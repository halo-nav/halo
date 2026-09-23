import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)


class ReducedRankGP:
    """
    The ReducedRankGP class implements a reduced rank Gaussian Process (GP)
    model using spectral basis functions. It is designed to handle input data
    of arbitrary dimensions and provides methods for training the GP model,
    making predictions, and calculating Fisher information and ambiguity
    fields. The class is initialized with input data, output data, the number
    of basis functions, and the number of dimensions. It also includes methods
    for optimizing hyperparameters, predicting outputs, and calculating
    localizability fields based on Fisher information and ambiguity.
    """

    def __init__(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        num_basis: int = 256,
        num_dimension: int = 1,
    ):
        """
        Args:
            x: Input data of shape (N, d) where N is the number of data points and d is the number of dimensions.
            y: Output data of shape (N, 1) where N is the number of data points.
            num_basis: Number of basis functions to use in the reduced rank approximation.
            num_dimension: Number of dimensions of the input data.
        Returns:
            None
        """
        self.num_basis = num_basis
        self.d = num_dimension

        self.x = x
        self.y = y
        if self.x.shape[1] != self.d:
            raise ValueError(
                "Input data x must have dimension equal to num_dimension."
            )

        # Setting domain boundaries for the data
        rangex = jnp.max(self.x, axis=0) - jnp.min(self.x, axis=0)
        mask = rangex > 0
        pm = 0.2 * jnp.min(jnp.where(mask, rangex, 0))
        LL = jnp.stack(
            [jnp.min(self.x, axis=0) - pm, jnp.max(self.x, axis=0) + pm]
        )  # LL is the domain squares.
        self.LL_mean = jnp.mean(LL, axis=0)
        self.LL = (jnp.max(LL, axis=0) - jnp.min(LL, axis=0)) / 2
        print("Domain boundaries:", self.LL)

        (
            self.eigenval,
            self.eigenfun,
            self.eigenfun_dx,
            self.eigenfun_hess,
            self.NN,
        ) = self.domain_cartesian_dx(self.num_basis, self.d, self.LL)
        Phi = self.eigenfun(self.NN, x)
        PhiPhi = Phi.T @ Phi
        Phiy = Phi.T @ y
        self.Phi = Phi
        self.PhiPhi = PhiPhi
        self.Phiy = Phiy
        self.lambdas = self.eigenval(self.NN).T

    def domain_cartesian_dx(self, m, d, L):
        """
        Constructs index set of permutations and functions for basis generations.
        Returns both the basis function and gradient of the basis function.
        Args:
            m: `int`, Number of basis functions
            d: `int`, Dimension of cartesian space
            L: `jnp.array` Domain boundaries, x in [-L1,L1] X [-L2,L2] X [L3,L3]
        Returns:
            eigenvalues: functional eigenvalues of the basis functions
            eigenfun: functional laplace eigen functions for the basis
            eigenfun_dx: functional gradient of the laplace eigen functions for the basis
            NN: index set of permutations for the basis functions
        """

        def construct_index_set(N):
            """
            Function that constructs the index set of permutations
            Args:
                N: Size of index set
            Returns:
                NN: Array of index set of permutations
            """
            N = jnp.atleast_1d(N)
            num_rows = jnp.prod(N).astype(int)
            num_cols = len(N)
            NN = jnp.zeros((num_rows, num_cols))
            if len(N) == 1:
                NN = NN.at[:, 0].set(jnp.arange(1, int(N[0]) + 1))
            else:
                n = jnp.arange(1, int(N[0]) + 1)
                nn = construct_index_set(N[1:])
                NN = NN.at[:, 0].set(
                    jnp.kron(n, jnp.ones(jnp.prod(N[1:]).astype(int)))
                )
                NN = NN.at[:, 1:].set(jnp.tile(nn, (int(N[0]), 1)))
            return NN

        def laplace_eigen_cartesian_dirchlect(
            n: int, x: jnp.ndarray, L: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Finds the laplace eigen function over a cartesian dirchlect boundary
            Args:
                n: `int`, index of the eigen function
                x: `jnp.array`, input data
                L: `jnp.array`, domain boundaries
            Returns:
                v: `jnp.array`, value of the laplace eigen function at x
            """
            x = jnp.atleast_2d(x)

            angle = (
                jnp.pi
                * n[None, :, :]
                * (x[:, None, :] + L[None, None, :])
                / (2 * L[None, None, :])
            )
            v = (1.0 / jnp.sqrt(L[None, None, :])) * jnp.sin(angle)
            return v.prod(axis=2)

        def laplace_eigen_cartesian_dirchlect_grad(
            n: int, x: jnp.ndarray, di: int, L: jnp.ndarray
        ) -> jnp.ndarray:
            angle = (
                jnp.pi
                * n[None, :, :]
                * (x[:, None, :] + L[None, None, :])
                / (2 * L[None, None, :])
            )
            factors = (1.0 / jnp.sqrt(L[None, None, :])) * jnp.sin(angle)
            factors = factors.at[:, :, di].set(
                (jnp.pi * n[None, :, di] / (2 * L[di] * jnp.sqrt(L[di])))
                * jnp.cos(angle[:, :, di])
            )
            return factors.prod(axis=2)

        def laplace_eigen_cartesian_dirchlect_hess(
            n: int, x: jnp.ndarray, di: int, dj: int, L: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Hessian of the Laplace eigenfunctions over dimensions di,dj. Extends
            laplace_eigen_cartesian_dirchlect_grad to second order derivatives.
            Used for calculating the Laplacian of the scalar field.

            Args:
                n: jnp.array eigen function indices
                x: `jnp.array`, input data
                di: `int`, dimension of the first derivative
                dj: `int`, dimension of the second derivative
                L: `jnp.array`, domain boundaries
            Returns:
                v: `jnp.array`, value of the Hessian of the laplace eigen function at x
            """
            x = jnp.atleast_2d(x)
            angle = (
                jnp.pi
                * n[None, :, :]
                * (x[:, None, :] + L[None, None, :])
                / (2 * L[None, None, :])
            )

            factors = (1 / jnp.sqrt(L[None, None, :])) * jnp.sin(angle)

            prefactor_di = (
                jnp.pi * n[None, :, di] / (2 * L[di] * jnp.sqrt(L[di]))
            )

            if di == dj:
                # Diagonal elements of the Hessian d^2 sin/dx^2 = -k^2 sin(x)
                factors = factors.at[:, :, di].set(
                    -(prefactor_di**2) * jnp.sin(angle[:, :, di])
                )
            else:
                # Off-diagonal elements of the Hessian d^2 sin/dxdy = k1*k2 cos(x)cos(y)
                prefactor_dj = (
                    jnp.pi * n[None, :, dj] / (2 * L[dj] * jnp.sqrt(L[dj]))
                )
                factors = factors.at[:, :, di].set(
                    prefactor_di * jnp.cos(angle[:, :, di])
                )
                factors = factors.at[:, :, dj].set(
                    prefactor_dj * jnp.cos(angle[:, :, dj])
                )

            return factors.prod(axis=2)

        def eigenfun_hess(
            n: int, x: jnp.ndarray, di: int, dj: int
        ) -> jnp.ndarray:
            return laplace_eigen_cartesian_dirchlect_hess(n, x, di, dj, L)

        def eigenvalue(n: int) -> jnp.ndarray:
            return jnp.sum((jnp.pi * n / (2 * L)) ** 2, axis=1)

        def eigenfun(n: int, x: jnp.ndarray) -> jnp.ndarray:
            return laplace_eigen_cartesian_dirchlect(n, x, L)

        def eigenfun_grad(n: int, x: jnp.ndarray, di: int) -> jnp.ndarray:
            return laplace_eigen_cartesian_dirchlect_grad(n, x, di, L)

        N = jnp.ceil(m ** (1 / d) * L / jnp.min(L))
        print("N:", N)
        NN = construct_index_set(N)
        ind = jnp.argsort(eigenvalue(NN), axis=0)
        NN = NN.at[ind[:m], :].get()
        return eigenvalue, eigenfun, eigenfun_grad, eigenfun_hess, NN

    def _Sse(self, w, lengthScale, magnSigma2):
        """
        Function to calculate the spectral density of the squared exponential covariance funciton

        Args:
            w: frequencies, here the eigen values are the input to this
            lengthScale: length scale of the squared exponential covariance function
            magnSigma2: magnitude of the squared exponential covariance function

        Return:
            spectral density of the squared exponential covariance function evaluated at the frequencies w
        """

        return (
            magnSigma2
            * jnp.sqrt(2 * jnp.pi) ** self.d
            * lengthScale**self.d
            * jnp.exp(-(w**2 * lengthScale**2) / 2)
        )

    def _dSse_lscale(self, w, lengthScale, magnSigma2):
        """
        Function to calculate the derivative of the spectral density of the squared exponential covariance function with respect to the length scale

        Args:
            w: frequencies, here the eigen values are the input to this
            lengthScale: length scale of the squared exponential covariance function
            magnSigma2: magnitude of the squared exponential covariance function

        Return:
            derivative of the spectral density of the squared exponential covariance function with respect to the length scale evaluated at the frequencies w
        """
        return (self.d / lengthScale - lengthScale * w**2) * self._Sse(
            w, lengthScale, magnSigma2
        )

    def _dSse_magnSigma2(self, w, lengthScale, magnSigma2):
        """
        Function to calculate the derivative of the spectral density of the squared exponential covariance function with respect to the magnitude

        Args:
            w: frequencies, here the eigen values are the input to this
            lengthScale: length scale of the squared exponential covariance function
            magnSigma2: magnitude of the squared exponential covariance function

        Return:
            derivative of the spectral density of the squared exponential covariance function with respect to the magnitude evaluated at the frequencies w
        """
        return self._Sse(w, lengthScale, magnSigma2) / magnSigma2

    def gp_optimize_fast(self) -> None:
        """
        Function that optimizes a GP. This function builds the required optimization
        functionals and runs an optimizer to find the best parameters for the data
        given. The optimization is done using the L-BFGS-B algorithm. The
        function also prints the optimized hyperparameters and the convergence
        status of the optimization. The function stores the optimized
        hyperparameters in the `self.theta` attribute of the class.

        Args:
            None
        Returns:
            None
        """

        # Centering data
        x = self.x.reshape(-1, self.d) - self.LL_mean
        LL = self.LL

        @jax.jit
        def optimize_hyperparams(
            w,
            y,
            lambdas,
            Phiy,
            PhiPhi,
        ):
            lengthScale = jnp.exp(w[0])
            magnSigma2 = jnp.exp(w[1])
            sigma2 = jnp.exp(w[2])

            k = self._Sse(jnp.sqrt(lambdas), lengthScale, magnSigma2)
            # Filter out 0's and nans
            k = jnp.clip(k, min=1e-10, max=None)

            n, d_out = y.shape
            Phiy = jnp.atleast_2d(Phiy) if Phiy.ndim == 1 else Phiy

            m = Phiy.shape[0]

            L = jnp.linalg.cholesky(PhiPhi + jnp.diag(sigma2 / k), upper=False)

            v = jsp.linalg.solve_triangular(L, Phiy, lower=True)
            vv = jsp.linalg.solve_triangular(
                L, v, trans="T", lower=True
            ).ravel()

            yiQy = (jnp.sum(y**2) - jnp.sum(v**2)) / sigma2

            logdetQ = (
                (n - m) * jnp.log(sigma2)
                + jnp.sum(jnp.log(k))
                + 2 * jnp.sum(jnp.log(jnp.diag(L)))
            ) * d_out

            error = (
                0.5 * yiQy
                + 0.5 * logdetQ
                + 0.5 * n * d_out * jnp.log(2 * jnp.pi)
            ) / n

            L_inv_scaled = jsp.linalg.solve_triangular(
                L, jnp.diag(1.0 / k), lower=True
            )
            LLk = jsp.linalg.solve_triangular(
                L, L_inv_scaled, trans="T", lower=True
            )

            error_grad = jnp.zeros_like(w)
            dk1 = jnp.concat(
                [self._dSse_lscale(jnp.sqrt(lambdas), lengthScale, magnSigma2)]
            )
            dlogdetQ = jnp.sum(dk1 / k) - sigma2 * jnp.sum(
                jnp.diag(LLk) * dk1 / k
            )
            dyiQy = -jnp.sum((dk1 / k**2) * vv**2)

            error_grad = error_grad.at[0].set(
                jnp.exp(w[0]) * (0.5 * dlogdetQ + 0.5 * dyiQy)
            )
            dk2 = jnp.concat(
                [
                    self._dSse_magnSigma2(
                        jnp.sqrt(lambdas), lengthScale, magnSigma2
                    )
                ]
            )
            dlogdetQ = jnp.sum(dk2 / k) - sigma2 * jnp.sum(
                jnp.diag(LLk) * dk2 / k
            )
            dyiQy = -jnp.sum((dk2 / k**2) * vv**2)

            error_grad = error_grad.at[1].set(
                jnp.exp(w[1]) * (0.5 * dlogdetQ + 0.5 * dyiQy)
            )

            dlogdetQ = ((n - m) / sigma2) + jnp.sum(jnp.diag(LLk))
            dyiQy = jnp.sum((1.0 / k) * vv**2) / sigma2 - yiQy / sigma2

            error_grad = error_grad.at[-1].set(0.5 * dlogdetQ + 0.5 * dyiQy)
            error_grad = error_grad.at[-1].set(jnp.exp(w[-1]) * error_grad[-1])
            error_grad = error_grad / n
            return jnp.squeeze(error), error_grad

        # Optimize hyperparameters using minimize

        def loss_and_grad_numpy(w):
            w_jax = jnp.array(w)
            val, grad = optimize_hyperparams(
                w_jax, self.y, self.lambdas, self.Phiy, self.PhiPhi
            )
            return np.array(val, dtype=np.float64), np.array(
                grad, dtype=np.float64
            )

        result = minimize(
            loss_and_grad_numpy,
            np.log([1.0, 1.0, 1.0]),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": 1000,
                "ftol": 1e-10,
                "gtol": 1e-5,
                "maxls": 100,  # more line search steps
                "maxcor": 20,
            },
        )
        w_opt = jnp.array(result.x)
        print("Optimized hyperparameters:", jnp.exp(w_opt))
        print("Converged:", result.success)
        print("Message:", result.message)
        theta = jnp.exp(w_opt)
        self.theta = theta
        k = self._Sse(jnp.sqrt(self.lambdas), theta[0], theta[1])
        self.L = jnp.linalg.cholesky(
            self.PhiPhi + jnp.diag(theta[2] / k), upper=False
        )
        v = jsp.linalg.solve_triangular(self.L, self.Phiy, lower=True)
        self.w = jsp.linalg.solve_triangular(self.L.T, v, lower=False).ravel()

    def predict(self, x) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Function that predicts the output of the GP model given input data x and output data y.

        Args:
            x: Input data of shape (N, d) where N is the number of data points and d is the number of dimensions.
        Returns:
            Eft: Output data of shape (N, 1) where N is the number of data points.
            Varft: Variance of the output data of shape (N, 1) where N is the number of data points.
        """
        x = x.reshape(-1, self.d) - self.LL_mean
        lengthScale = self.theta[0]
        magnSigma2 = self.theta[1]
        sigma2 = self.theta[2]
        Phit = self.eigenfun(self.NN, x)
        Eft = Phit @ self.w
        Varft = sigma2 * jnp.sum(
            (jsp.linalg.solve_triangular(self.L, Phit.T)) ** 2, axis=0
        )
        return Eft, Varft

    def fisher_information(
        self,
        x: jnp.ndarray,
        sigma_R: float,
        gamma: float,
        sigma_y=None,
        mean_y=None,
    ) -> jnp.ndarray:
        """
        Function that calculates the Fisher information of the area encompassed
        by the x data. Fisher information is described as the sensitivity of
        the likelihood to the unknown parameters. Defined as the expectation
        of the outerproduct of the score function which can be simplified
        into 1/sigma^2 J J^T.

        UNIT HANDLING (changed): previously, grad_field/hess_field were scaled
        *up* to physical units by sigma_y while sigma_R stayed at its raw
        physical value -- this left Ex ~ sigma_y^2 / sigma_R^2 (normalized by
        sigma_R^2) but hess_term ~ sigma_y^2 with no equivalent normalization,
        forcing `gamma` alone to bridge a dataset-dependent scale gap that could
        span several orders of magnitude depending on the field's raw units.

        Now, grad_field/hess_field/cov_grad are left in the GP's native
        normalized (z-scored) units, and `sigma_R` is instead brought *down*
        into that same normalized space (sigma_R / sigma_y). This keeps every
        term in `fisher_info` on a comparable, roughly O(1) scale regardless of
        the raw field magnitude, which is what makes `gamma` calibration
        tractable and transferable across datasets.

        NOTE: because of this change, any previously-tuned `gamma` value
        (e.g. defaults calibrated against the old physical-unit scale) will NOT
        transfer to this version and needs to be re-derived -- the two versions
        are not numerically comparable at the same gamma.

        Args:
            x: Input data of shape (N, d) where N is the number of data points
               and d is the number of dimensions.
            sigma_R: Measurement noise std, in the SAME units as the raw
                     (unnormalized) sensor readings -- e.g. the physical sensor
                     noise spec (matches `R` in `ParticleFilter`).
            gamma: Rank recovery weight for the Hessian term. NOT the same
                   numerical scale as a `gamma` tuned against the old
                   (physical-unit) version of this function -- see note above.
            sigma_y: Optional. If provided, `sigma_R` is converted into the
                     GP's normalized units via `sigma_R / sigma_y`, and the FIM
                     is computed entirely in normalized space. If not provided,
                     `sigma_R` is assumed to already be expressed in the GP's
                     native normalized units.
            mean_y: Unused in this computation (gradient/Hessian of a field are
                    invariant to a constant additive shift) -- kept only for
                    call-site/API compatibility with `predict()`.
        Returns:
            fisher_info: Fisher information matrix of shape (N, d, d), in the
                         GP's normalized units.
        """

        x -= self.LL_mean
        dPhi = jnp.stack(
            [self.eigenfun_dx(self.NN, x, di) for di in range(self.d)], axis=-1
        )

        d2Phi = jnp.stack(
            [
                jnp.stack(
                    [
                        self.eigenfun_hess(self.NN, x, di, dj)
                        for dj in range(self.d)
                    ],
                    axis=2,
                )
                for di in range(self.d)
            ],
            axis=2,
        )
        grad_field = jnp.einsum("nmd,m->nd", dPhi, self.w)
        hess_field = jnp.einsum("nmkl,m->nkl", d2Phi, self.w)

        # Bring sigma_R into the same normalized units as grad_field/hess_field,
        # rather than rescaling the fields up to physical units. This is the
        # single normalization point for the whole function.
        if sigma_y is not None:
            sigma_R = sigma_R / sigma_y

        # cov_grad is already computed from quantities fit in the GP's
        # normalized space (self.theta[2], self.L come from gp_optimize_fast on
        # normalized y) -- no rescaling needed or applied here, unlike before.
        cov_grad = self.theta[2] * jnp.einsum(
            "nma,mk,nkb->nab",
            dPhi,
            jsp.linalg.cho_solve((self.L, True), jnp.eye(self.L.shape[0])),
            dPhi,
        )

        grad_outer = jnp.einsum("ni,nj->nij", grad_field, grad_field)
        Ex = (grad_outer + cov_grad) / sigma_R**2

        hess_term = jnp.einsum("nik,njk->nij", hess_field, hess_field)
        fisher_info = Ex + gamma * hess_term
        return fisher_info

    def ambiguity_field(self, x) -> jnp.ndarray:
        """
        This function calculates an efficient implmentation of the ambiguity field.
        It uses the spectral basis functions to calculate the Laplacian of the scalar field
        Args:
            x: Input data of shape (N, d) where N is the number of data points and d is the number of dimensions.
        """
        lengthScale = self.theta[0]
        magnSigma2 = self.theta[1]
        sigma2 = self.theta[2]
        Phit = self.eigenfun(self.NN, x)
        k = self._Sse(jnp.sqrt(self.lambdas), lengthScale, magnSigma2)
        L = jnp.linalg.cholesky(
            self.PhiPhi + jnp.diag(sigma2 / k), upper=False
        )
        v = jsp.linalg.solve_triangular(L, self.Phiy, lower=True)
        vv = jsp.linalg.solve_triangular(L.T, v, lower=False)
        diag_lambdas = jnp.diag(self.lambdas)
        ambiguity = -Phit @ diag_lambdas @ vv
        return ambiguity

    def localizability(
        self, x, sigma_R, gamma=0.5, sigma_y=None, mean_y=None, eps=0.1
    ) -> jnp.ndarray:
        fim = self.fisher_information(x, sigma_R, gamma, sigma_y, mean_y)
        eigvals = jnp.linalg.eigvalsh(fim)
        localizability_field = eigvals[..., 0]

        return localizability_field
