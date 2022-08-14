from datetime import datetime, timedelta

import numpy as np
from kalman_filter import KalmanFilter


def test_kalman_filter():
    x_0 = np.array([4000, 280])
    p_0 = np.array([[20**2, 20 * 5], [5 * 20, 5**2]])
    state_transition_transform = np.array([[1, 1], [0, 1]])
    control_transform = np.array([1 / 2, 1])
    matrix_transform = np.eye(p_0.shape)
    process_noise_covariance = np.zeros(p_0.shape)
    measurement_uncertainty = np.array([[25**2, 0], [0, 6**2]])
    observations = [[4000, 280], [4260, 282], [4550, 285], [4860, 286], [5111, 290]]
    t_delta = 1

    t = datetime.now()
    kf = KalmanFilter(
        init_state=x_0,
        init_estimate_uncertainty=p_0,
        state_transition_transform=state_transition_transform,
        control_transform=control_transform,
        matrix_transform=matrix_transform,
        process_noise_covariance=process_noise_covariance,
        measurement_uncertainty=measurement_uncertainty,
        init_time=t,
    )
    control_input = np.array(
        [
            2,
        ]
    )

    for obs in observations:
        t = t + timedelta(seconds=t_delta)
        kf.predict(control_input=control_input, t=t)
        kf.update(observation=np.array(obs), t=t)
        state = kf.get_state()
        print(state)
        uncertainty = kf.get_uncertainty()
        print(uncertainty)


if __name__ == "__main__":
    test_kalman_filter()
