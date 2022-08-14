from datetime import datetime

import numpy as np


class KalmanFilter:
    """
    Implementation of a standard Kalman filter
    """

    def __init__(
        self,
        init_state: np.array,
        init_estimate_uncertainty: np.array,
        state_transition_transform: np.array,
        control_transform: np.array,
        matrix_transform: np.array,
        process_noise_covariance: np.array,
        measurement_uncertainty: np.array,
        init_time: datetime = datetime.now(),
    ):
        self._state = {init_time: init_state}
        self._estimate_uncertainty = {init_time: init_estimate_uncertainty}
        self._state_transition_transform = state_transition_transform
        self._control_transform = control_transform
        self._matrix_transform = matrix_transform
        self._process_noise_covariance = process_noise_covariance
        self._measurement_uncertainty = measurement_uncertainty
        self._observation = {}
        self._control_input = {}
        self._state_pred = {}
        self._estimate_uncertainty_pred = {}
        self._kalman_gain = {}
        self._prev_t = init_time

    def predict(
        self,
        control_input: np.array,
        t: datetime = datetime.now(),
    ) -> None:
        """
        Predict the state and estimate uncertainty
        """
        self._control_input[t] = control_input
        self._predict_state(t)
        self._predict_uncertainty(t)
        return self._state_pred[t], self._estimate_uncertainty_pred[t]

    def _predict_state(self, t: datetime) -> None:
        """
        Extrapolate the state of the system at time t
        """
        self._state_pred[t] = (
            self._state_transition_transform @ self._state[self._prev_t]
            + self._control_transform @ self._control_input[t]
        )

    def _predict_uncertainty(self, t: datetime) -> None:
        """
        Extrapolate the uncertainty of the system at time t
        """
        self._estimate_uncertainty_pred[t] = (
            self._state_transition_transform
            @ self._estimate_uncertainty[self._prev_t]
            @ self._state_transition_transform.T
            + self.process_noise_covariance
        )

    def update(
        self,
        observation: np.array,
        t: datetime = datetime.now(),
    ) -> None:
        """
        Update the state estimate based on a set of observations
        """
        self._observation[t] = observation
        self._update_kalman_gain(t)
        self._update_estimate(t)
        self._update_estimate_uncertainty(t)
        self._prev_t = t

    def update_kalman_gain(self, t: datetime) -> None:
        self._kalman_gain[t] = (
            self._estimate_uncertainty_pred[t]
            @ self._matrix_transform.T
            @ np.linalg.inv(
                self._matrix_transform
                @ self._estimate_uncertainty_pred[t]
                @ self._matrix_transform.T
                + self._measurement_uncertainty
            )
        )

    def update_estimate(self, t: datetime) -> None:
        self._state[t] = self._state_pred[t] + self._kalman_gain[t] @ (
            self._observation[t] - self._matrix_transform @ self._state_pred[t]
        )

    def update_estimate_uncertainty(self, t: datetime) -> None:
        n = self.observation[t].shape[0]
        self._estimate_uncertainty_pred[t] = (
            np.eye(n) - self._kalman_gain[t] @ self._matrix_transform
        ) @ self._estimate_uncertainty_pred[t]

    def get_state(self, t: datetime) -> np.array:
        return self._state[t]

    def get_uncertainty(self, t: datetime) -> np.array:
        return self._estimate_uncertainty[t]

    def get_state_history(self) -> tuple[datetime, np.array]:
        return self._state.items()

    def get_uncertainty_history(self) -> tuple[datetime, np.array]:
        return self._estimate_uncertainty.items()
