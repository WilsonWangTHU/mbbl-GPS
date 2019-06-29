""" This file defines code for iLQG-based trajectory optimization. """
import logging
import copy

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp

from gps.algorithm.traj_opt.config import TRAJ_OPT_LQR
from gps.algorithm.traj_opt.traj_opt import TrajOpt
from gps.algorithm.traj_opt.traj_opt_utils import \
    DGD_MAX_ITER, DGD_MAX_LS_ITER, DGD_MAX_GD_ITER, \
    ALPHA, BETA1, BETA2, EPS, \
    traj_distr_kl, traj_distr_kl_alt

from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS


LOGGER = logging.getLogger(__name__)


class TrajOptLQRPython(TrajOpt):
    """ LQR trajectory optimization, Python implementation. """

    def __init__(self, hyperparams):
        config = copy.deepcopy(TRAJ_OPT_LQR)
        config.update(hyperparams)

        TrajOpt.__init__(self, config)

        self.cons_per_step = config['cons_per_step']
        self._use_prev_distr = config['use_prev_distr']
        self._update_in_bwd_pass = config['update_in_bwd_pass']

    # TODO - Add arg and return spec on this function.
    def update(self, m, algorithm):
        """ Run dual gradient decent to optimize trajectories. """
        T = algorithm.T
        eta = algorithm.cur[m].eta
        if self.cons_per_step and type(eta) in (int, float):
            eta = np.ones(T) * eta
        step_mult = algorithm.cur[m].step_mult
        traj_info = algorithm.cur[m].traj_info

        # For MDGPS, constrain to previous NN linearization
        prev_traj_distr = algorithm.cur[m].pol_info.traj_distr()

        # Set KL-divergence step size (epsilon).
        kl_step = algorithm.base_kl_step * step_mult
        if not self.cons_per_step:
            kl_step *= T

        # We assume at min_eta, kl_div > kl_step, opposite for max_eta.
        min_eta = self._hyperparams['min_eta']
        max_eta = self._hyperparams['max_eta']
        LOGGER.debug("Running DGD for trajectory %d, eta: %f", m, eta)

        max_itr = (DGD_MAX_LS_ITER if self.cons_per_step else
                   DGD_MAX_ITER)
        for itr in range(max_itr):
            LOGGER.debug("[DEBUG] Iteration %d, bracket: (%.2e , %.2e , %.2e)", itr,
                         min_eta, eta, max_eta)

            # Run fwd/bwd pass, note that eta may be updated.
            # Compute KL divergence constraint violation.
            traj_distr, eta = self.backward(prev_traj_distr, traj_info,
                                            eta, algorithm, m)

            new_mu, new_sigma = self.forward(traj_distr, traj_info)
            kl_div = traj_distr_kl(
                new_mu, new_sigma, traj_distr, prev_traj_distr,
                tot=(not self.cons_per_step)
            )
            con = kl_div - kl_step
            LOGGER.debug("[DEBUG] KL (%.2e , %.2e)", kl_div, kl_step)

            # Convergence check - constraint satisfaction.
            if self._conv_check(con, kl_step):
                LOGGER.debug("KL: %f / %f, converged iteration %d", kl_div,
                             kl_step, itr)
                break

            # Choose new eta (bisect bracket or multiply by constant)
            if con < 0:  # Eta was too big.
                max_eta = eta
                geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
                new_eta = max(geom, 0.1 * max_eta)
                LOGGER.debug("KL: %f / %f, eta too big, new eta: %f",
                             kl_div, kl_step, new_eta)
            else:  # Eta was too small.
                min_eta = eta
                geom = np.sqrt(min_eta * max_eta)  # Geometric mean.
                new_eta = min(geom, 10.0 * min_eta)
                LOGGER.debug("KL: %f / %f, eta too small, new eta: %f",
                             kl_div, kl_step, new_eta)

            # Logarithmic mean: log_mean(x,y) = (y - x)/(log(y) - log(x))
            eta = new_eta
            print('traj {}'.format(m))
            print('eta {}'.format(eta))

        if (np.mean(kl_div) > np.mean(kl_step) and
                not self._conv_check(con, kl_step)):
            LOGGER.warning(
                "Final KL divergence after DGD convergence is too high."
            )
        return traj_distr, eta

    def estimate_cost(self, traj_distr, traj_info):
        """ Compute Laplace approximation to expected cost. """
        # Constants.
        T = traj_distr.T

        # Perform forward pass (note that we repeat this here, because
        # traj_info may have different dynamics from the ones that were
        # used to compute the distribution already saved in traj).
        mu, sigma = self.forward(traj_distr, traj_info)

        # Compute cost.
        predicted_cost = np.zeros(T)
        for t in range(T):
            predicted_cost[t] = traj_info.cc[t] + 0.5 * \
                np.sum(sigma[t, :, :] * traj_info.Cm[t, :, :]) + 0.5 * \
                mu[t, :].T.dot(traj_info.Cm[t, :, :]).dot(mu[t, :]) + \
                mu[t, :].T.dot(traj_info.cv[t, :])
        return predicted_cost

    def forward(self, traj_distr, traj_info):
        """
        Perform LQR forward pass. Computes state-action marginals from
        dynamics and policy.
        Args:
            traj_distr: A linear Gaussian policy object.
            traj_info: A TrajectoryInfo object.
        Returns:
            mu: A T x dX mean action vector.
            sigma: A T x dX x dX covariance matrix.
        """
        # Compute state-action marginals from specified conditional
        # parameters and current traj_info.
        T = traj_distr.T
        dU = traj_distr.dU
        dX = traj_distr.dX

        # Constants.
        idx_x = slice(dX)

        # Allocate space.
        sigma = np.zeros((T, dX + dU, dX + dU))
        mu = np.zeros((T, dX + dU))

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv
        dyn_covar = traj_info.dynamics.dyn_covar

        # Set initial covariance (initial mu is always zero).
        sigma[0, idx_x, idx_x] = traj_info.x0sigma
        mu[0, idx_x] = traj_info.x0mu

        for t in range(T):
            sigma[t, :, :] = np.vstack([
                np.hstack([
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(traj_distr.K[t, :, :].T)
                ]),
                np.hstack([
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]),
                    traj_distr.K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(
                        traj_distr.K[t, :, :].T
                    ) + traj_distr.pol_covar[t, :, :]
                ])
            ])
            mu[t, :] = np.hstack([
                mu[t, idx_x],
                traj_distr.K[t, :, :].dot(mu[t, idx_x]) + traj_distr.k[t, :]
            ])
            if t < T - 1:
                sigma[t + 1, idx_x, idx_x] = \
                    Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T) + \
                    dyn_covar[t, :, :]
                mu[t + 1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]
        return mu, sigma

    def backward(self, prev_traj_distr, traj_info, eta, algorithm, m):
        """
        Perform LQR backward pass. This computes a new linear Gaussian
        policy object.
        Args:
            prev_traj_distr: A linear Gaussian policy object from
                previous iteration.
            traj_info: A TrajectoryInfo object.
            eta: Dual variable.
            algorithm: Algorithm object needed to compute costs.
            m: Condition number.
        Returns:
            traj_distr: A new linear Gaussian policy.
            new_eta: The updated dual variable. Updates happen if the
                Q-function is not PD.
        """
        # Constants.
        T = prev_traj_distr.T
        dU = prev_traj_distr.dU
        dX = prev_traj_distr.dX

        traj_distr = prev_traj_distr.nans_like()

        # Store pol_wt if necessary
        if type(algorithm) == AlgorithmBADMM:
            pol_wt = algorithm.cur[m].pol_info.pol_wt

        idx_x = slice(dX)
        idx_u = slice(dX, dX + dU)

        # Pull out dynamics.
        Fm = traj_info.dynamics.Fm
        fv = traj_info.dynamics.fv

        # Non-SPD correction terms.
        del_ = self._hyperparams['del0']
        if self.cons_per_step:
            del_ = np.ones(T) * del_
        eta0 = eta

        # Run dynamic programming.
        fail = True
        while fail:
            fail = False  # Flip to true on non-symmetric PD.

            # Allocate.
            Vxx = np.zeros((T, dX, dX))
            Vx = np.zeros((T, dX))
            Qtt = np.zeros((T, dX + dU, dX + dU))
            Qt = np.zeros((T, dX + dU))

            fCm, fcv = algorithm.compute_costs(
                m, eta, augment=(not self.cons_per_step)
            )
            # import pdb; pdb.set_trace()

            # Compute state-action-state function at each time step.
            for t in range(T - 1, -1, -1):
                # Add in the cost.
                Qtt[t] = fCm[t, :, :]  # (X+U) x (X+U)
                Qt[t] = fcv[t, :]  # (X+U) x 1

                # Add in the value function from the next time step.
                if t < T - 1:
                    multiplier = 1.0
                    Qtt[t] += multiplier * \
                        Fm[t, :, :].T.dot(Vxx[t + 1, :, :]).dot(Fm[t, :, :])
                    Qt[t] += multiplier * \
                        Fm[t, :, :].T.dot(Vx[t + 1, :] +
                                          Vxx[t + 1, :, :].dot(fv[t, :]))

                # Symmetrize quadratic component.
                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                inv_term = Qtt[t, idx_u, idx_u]
                k_term = Qt[t, idx_u]
                K_term = Qtt[t, idx_u, idx_x]

                # Compute Cholesky decomposition of Q function action
                # component.
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except LinAlgError as e:
                    # Error thrown when Qtt[idx_u, idx_u] is not
                    # symmetric positive definite.
                    LOGGER.debug('LinAlgError: %s', e)
                    fail = t if self.cons_per_step else True
                    break

                # Store conditional covariance, inverse, and Cholesky.
                traj_distr.inv_pol_covar[t, :, :] = inv_term
                traj_distr.pol_covar[t, :, :] = sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, np.eye(dU), lower=True)
                )
                traj_distr.chol_pol_covar[t, :, :] = sp.linalg.cholesky(
                    traj_distr.pol_covar[t, :, :]
                )

                # Compute mean terms.
                traj_distr.k[t, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, k_term, lower=True)
                )
                traj_distr.K[t, :, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, K_term, lower=True)
                )

                # Compute value function.
                Vxx[t, :, :] = Qtt[t, idx_x, idx_x] + \
                    Qtt[t, idx_x, idx_u].dot(traj_distr.K[t, :, :])
                Vx[t, :] = Qt[t, idx_x] + \
                    Qtt[t, idx_x, idx_u].dot(traj_distr.k[t, :])
                Vxx[t, :, :] = 0.5 * (Vxx[t, :, :] + Vxx[t, :, :].T)

            # Increment eta on non-SPD Q-function.
            if fail:
                old_eta = eta
                eta = eta0 + del_
                LOGGER.debug('Increasing eta: %f -> %f', old_eta, eta)
                del_ *= 2  # Increase del_ exponentially on failure.
                fail_check = (eta >= 1e16)
                if fail_check:
                    if np.any(np.isnan(Fm)) or np.any(np.isnan(fv)):
                        raise ValueError('NaNs encountered in dynamics!')
                    raise ValueError('Failed to find PD solution even for very \
                            large eta (check that dynamics and cost are \
                            reasonably well conditioned)!')
        return traj_distr, eta

    def _conv_check(self, con, kl_step):
        """Function that checks whether dual gradient descent has converged."""
        if self.cons_per_step:
            return all([abs(con[t]) < (0.1 * kl_step[t]) for t in range(con.size)])
        return abs(con) < 0.1 * kl_step
