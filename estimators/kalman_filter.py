from datetime import datetime

import numpy as np

from .sensors import Measurement


class KalmanFilter:
    """
    Implementation of a standard Kalman filter
    """

    def __init__(
        self,
        process_noise_covariance: np.array,
        matrix_transform: np.array,
        measurement_uncertainty: np.array = None,
        state_transition_transform: np.array = None,
        control_transform: np.array = None,
    ):
        self._state_transition_transform = state_transition_transform
        self._control_transform = control_transform
        self._matrix_transform = matrix_transform
        self._process_noise_covariance = process_noise_covariance
        self._measurement_uncertainty = measurement_uncertainty
        self._state: dict[datetime, np.array] = {}
        self._estimate_uncertainty: dict[datetime, np.array] = {}
        self._observation: dict[datetime, np.array] = {}
        self._control_input: dict[datetime, np.array] = {}
        self._state_pred: dict[datetime, np.array] = {}
        self._estimate_uncertainty_pred: dict[datetime, np.array] = {}
        self._kalman_gain: dict[datetime, np.array] = {}

    def init_state(
        self,
        state: np.array,
        estimate_uncertainty: np.array,
        t: datetime = datetime.now(),
    ):
        self._state[t] = state
        self._estimate_uncertainty[t] = estimate_uncertainty
        self._prev_t = t

    def run(
        self,
        control_input: np.array,
        observation: np.array,
        t: datetime = datetime.now(),
        state_transition_transform: np.array = None,
        control_transform: np.array = None,
        matrix_uncertainty: np.array = None,
    ):
        self._control_input[t] = control_input
        self._observation[t] = observation
        self._predict(t, state_transition_transform, control_transform)
        self._update(t, matrix_uncertainty)
        self._prev_t = t

    def _predict(
        self,
        t: datetime = datetime.now(),
        state_transition_transform: np.array = None,
        control_transform: np.array = None,
    ) -> None:
        """
        Predict the state and estimate uncertainty
        """
        self._predict_state(t, state_transition_transform, control_transform)
        self._predict_uncertainty(t)

    def _predict_state(
        self,
        t: datetime,
        state_transition_transform: np.array = None,
        control_transform: np.array = None,
    ) -> None:
        """
        Extrapolate the state of the system at time t
        """
        state_matrix = (
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
            state_matrix @ self._state[self._prev_t]
            + control_matrix @ self._control_input[t]
        )

    def _predict_uncertainty(
        self,
        t: datetime,
        state_transition_transform: np.array = None,
    ) -> None:
        """
        Extrapolate the uncertainty of the system at time t
        """
        state_matrix = (
            state_transition_transform
            if state_transition_transform is not None
            else self._state_transition_transform
        )
        self._estimate_uncertainty_pred[t] = np.diag(
            np.diag(
                state_matrix @ self._estimate_uncertainty[self._prev_t] @ state_matrix.T
                + self._process_noise_covariance
            )
        )

    def _update(
        self,
        t: datetime = datetime.now(),
        measurement_uncertainty: np.array = None,
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
        measurement_uncertainty: np.array = None,
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
        n = self._observation[t].shape[0]
        self._estimate_uncertainty[t] = np.diag(
            np.diag(
                (np.eye(n) - self._kalman_gain[t] @ self._matrix_transform)
                @ self._estimate_uncertainty_pred[t]
            )
        )

    @property
    def state(self) -> np.array:
        return self._state[self._prev_t]

    @property
    def uncertainty(self) -> np.array:
        return self._estimate_uncertainty[self._prev_t]

    @property
    def predicted_state(self) -> np.array:
        return self._state_pred[self._prev_t]

    @property
    def predicted_uncertainty(self) -> np.array:
        return self._estimate_uncertainty_pred[self._prev_t]

    @property
    def kalman_gain(self) -> np.array:
        return self._kalman_gain[self._prev_t]

    @property
    def state_history(self) -> tuple[datetime, np.array]:
        return self._state.items()

    @property
    def uncertainty_history(self) -> tuple[datetime, np.array]:
        return self._estimate_uncertainty.items()
