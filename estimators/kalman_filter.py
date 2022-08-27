from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from .sensors import Measurement


class KalmanFilter:
    """
    Implementation of a standard Kalman filter
    """

    def __init__(
        self,
        process_noise_covariance: NDArray,
        matrix_transform: NDArray,
        measurement_uncertainty: NDArray = None,
        state_transition_transform: NDArray = None,
        control_transform: NDArray = None,
    ):
        self._state_transition_transform = state_transition_transform
        self._control_transform = control_transform
        self._matrix_transform = matrix_transform
        self._process_noise_covariance = process_noise_covariance
        self._measurement_uncertainty = measurement_uncertainty
        self._state: dict[datetime, NDArray] = {}
        self._estimate_uncertainty: dict[datetime, NDArray] = {}
        self._observation: dict[datetime, NDArray] = {}
        self._control_input: dict[datetime, NDArray] = {}
        self._state_pred: dict[datetime, NDArray] = {}
        self._estimate_uncertainty_pred: dict[datetime, NDArray] = {}
        self._kalman_gain: dict[datetime, NDArray] = {}

    def init_state(
        self,
        state: NDArray,
        estimate_uncertainty: NDArray,
        t: datetime = datetime.now(),
    ):
        self._state[t] = state
        self._estimate_uncertainty[t] = estimate_uncertainty
        self._prev_t = t

    def run(
        self,
        control_input: NDArray,
        observation: NDArray,
        t: datetime = datetime.now(),
        state_transition_transform: NDArray = None,
        control_transform: NDArray = None,
        measurement_uncertainty: NDArray = None,
    ):
        self._control_input[t] = control_input
        self._observation[t] = observation
        self._predict(t, state_transition_transform, control_transform)
        self._update(t, measurement_uncertainty)
        self._prev_t = t

    def _predict(
        self,
        t: datetime = datetime.now(),
        state_transition_transform: NDArray = None,
        control_transform: NDArray = None,
    ) -> None:
        """
        Predict the state and estimate uncertainty
        """
        self._predict_state(t, state_transition_transform, control_transform)
        self._predict_uncertainty(t, state_transition_transform)

    def _predict_state(
        self,
        t: datetime,
        state_transition_transform: NDArray = None,
        control_transform: NDArray = None,
    ) -> None:
        """
        Extrapolate the state of the system at time t
        """
        state_transform = (
            state_transition_transform
            if state_transition_transform is not None
            else self._state_transition_transform
        )
        control_matrix = (
            control_transform
            if control_transform is not None
            else self._control_transform
        )
        self._state_pred[t] = (
            state_transform @ self._state[self._prev_t]
            + control_matrix @ self._control_input[t]
        )

    def _predict_uncertainty(
        self,
        t: datetime,
        state_transition_transform: NDArray = None,
    ) -> None:
        """
        Extrapolate the uncertainty of the system at time t
        """
        state_transform = (
            state_transition_transform
            if state_transition_transform is not None
            else self._state_transition_transform
        )
        self._estimate_uncertainty_pred[t] = (
            state_transform
            @ self._estimate_uncertainty[self._prev_t]
            @ state_transform.T
            + self._process_noise_covariance
        )

    def _update(
        self,
        t: datetime = datetime.now(),
        measurement_uncertainty: NDArray = None,
    ) -> None:
        """
        Update the state estimate based on a set of observations
        """
        self._update_kalman_gain(t, measurement_uncertainty)
        self._update_estimate(t)
        self._update_estimate_uncertainty(t)

    def _update_kalman_gain(
        self,
        t: datetime,
        measurement_uncertainty: NDArray = None,
    ) -> None:
        measurement_uncertainty = (
            measurement_uncertainty
            if measurement_uncertainty is not None
            else self._measurement_uncertainty
        )

        self._kalman_gain[t] = (
            self._estimate_uncertainty_pred[t]
            @ self._matrix_transform.T
            @ np.linalg.inv(
                self._matrix_transform
                @ self._estimate_uncertainty_pred[t]
                @ self._matrix_transform.T
                + measurement_uncertainty
            )
        )

    def _update_estimate(self, t: datetime) -> None:
        self._state[t] = self._state_pred[t] + self._kalman_gain[t] @ (
            self._observation[t] - self._matrix_transform @ self._state_pred[t]
        )

    def _update_estimate_uncertainty(self, t: datetime) -> None:
        n = self._estimate_uncertainty_pred[t].shape[0]
        self._estimate_uncertainty[t] = (
            np.eye(n) - self._kalman_gain[t] @ self._matrix_transform
        ) @ self._estimate_uncertainty_pred[t]

    @property
    def state(self) -> NDArray:
        return self._state[self._prev_t]

    @property
    def uncertainty(self) -> NDArray:
        return self._estimate_uncertainty[self._prev_t]

    @property
    def predicted_state(self) -> NDArray:
        return self._state_pred[self._prev_t]

    @property
    def predicted_uncertainty(self) -> NDArray:
        return self._estimate_uncertainty_pred[self._prev_t]

    @property
    def kalman_gain(self) -> NDArray:
        return self._kalman_gain[self._prev_t]

    @property
    def state_history(self) -> NDArray:
        return np.vstack([s.T for s in self._state.values()])

    @property
    def gain_history(self) -> NDArray:
        return np.array([g for g in self._kalman_gain.values()])

    @property
    def observation_history(self) -> NDArray:
        return np.vstack([o.T for o in self._observation.values()])

    @property
    def input_history(self) -> NDArray:
        return np.vstack([i.T for i in self._control_input.values()])

    @property
    def uncertainty_history(self) -> NDArray:
        return np.vstack([np.diag(s) for s in self._estimate_uncertainty.values()])

    @property
    def timesteps(self) -> list[datetime]:
        return list(self._observation.keys())
