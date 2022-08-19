from datetime import datetime, timedelta

import numpy as np

from estimators.kalman_filter import KalmanFilter


def test_kalman_filter():
    x_0 = np.array(
        [
            [
                4000,
            ],
            [
                280,
            ],
        ]
    )
    p_0 = np.array([[20**2, 0], [0, 5**2]])
    state_transition_transform = np.array([[1, 1], [0, 1]])
    control_transform = np.array(
        [
            [
                1 / 2,
            ],
            [
                1,
            ],
        ]
    )
    matrix_transform = np.eye(p_0.shape[0])
    process_noise_covariance = np.zeros(p_0.shape[0])
    measurement_uncertainty = np.array([[25**2, 0], [0, 6**2]])
    t_delta = 1

    t = datetime.now()
    kf = KalmanFilter(
        state_transition_transform=state_transition_transform,
        control_transform=control_transform,
        matrix_transform=matrix_transform,
        process_noise_covariance=process_noise_covariance,
        measurement_uncertainty=measurement_uncertainty,
    )
    kf.init_state(
        state=x_0,
        estimate_uncertainty=p_0,
        t=t,
    )
    control_input = np.array(
        [
            [
                2,
            ]
        ]
    )

    observations = [
        [
            [
                4260,
            ],
            [
                282,
            ],
        ],
        [
            [
                4550,
            ],
            [
                285,
            ],
        ],
    ]
    states = [
        [
            [
                4272.5,
            ],
            [
                282.0,
            ],
        ],
        [
            [
                4553.8,
            ],
            [
                284,
            ],
        ],
    ]
    uncertainties = [
        [
            [253, 0],
            [
                0,
                14.8,
            ],
        ],
        [[187.5, 0.0], [0.0, 10.5]],
    ]
    for obs, x, p in zip(observations, states, uncertainties):
        t = t + timedelta(seconds=t_delta)
        kf.run(control_input=control_input, observation=np.array(obs), t=t)
        assert np.allclose(kf.state, np.array(x), rtol=0.1)
        assert np.allclose(kf.uncertainty, np.array(p), rtol=0.1)


if __name__ == "__main__":
    test_kalman_filter()
