from datetime import datetime, timedelta

import numpy as np

from estimators.kalman_filter import KalmanFilter
from estimators.visualise import plot_3d_timeseries


def test_4d_kalman_filter():
    x_0 = np.array(
        [
            4000,
            280,
        ]
    ).reshape(-1, 1)
    p_0 = np.array([[20**2, 0], [0, 5**2]])
    state_transition_transform = np.array([[1, 1], [0, 1]])
    control_transform = np.array(
        [
            1 / 2,
            1,
        ]
    ).reshape(-1, 1)
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
            2,
        ]
    )

    observations = [
        [
            [
                4260,
                282,
            ],
        ],
        [
            [
                4550,
                285,
            ],
        ],
    ]
    states = [
        [
            [
                4272.5,
                282.0,
            ],
        ],
        [
            [
                4553.8,
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
        kf.run(
            control_input=np.array(control_input).reshape(-1, 1),
            observation=np.array(obs).reshape(-1, 1),
            t=t,
        )
        assert np.allclose(kf.state, np.array(x).reshape(-1, 1), rtol=0.1)
        assert np.allclose(np.diag(np.diag(kf.uncertainty)), np.array(p), rtol=0.1)


def test_6d_kalman_filter():
    dt = 1
    t = datetime.now()

    state_transition_transform = np.array(
        [
            # x, y, vx, vy, ax, ay
            [1, 0, dt, 0, 0.5 * dt**2, 0],  # x
            [0, 1, 0, dt, 0, 0.5 * dt**2],  # y
            [0, 0, 1, 0, dt, 0],  # vx
            [0, 0, 0, 1, 0, dt],  # vy
            [0, 0, 0, 0, 1, 0],  # ax
            [0, 0, 0, 0, 0, 1],  # ay
        ]
    )

    process_noise = (
        np.array(
            [
                [dt**4 / 4, 0, dt**3 / 3, 0, dt**2 / 2, 0],
                [0, dt**4 / 4, 0, dt**2, 0, dt],
                [dt**3 / 2, 0, dt**2, 0, dt, 0],
                [0, dt**3 / 2, 0, dt**2, 0, dt],
                [dt**2 / 2, 0, dt, 0, 1, 0],
                [0, dt**2 / 2, 0, dt, 0, 1],
            ]
        )
        * 0.2**2
    )

    matrix_transform = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    observation_uncertainty = np.array([[3**2, 0], [0, 3**2]])
    control_input = np.array([0, 0]).reshape(-1, 1)
    control_transform = np.array(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    )
    kf = KalmanFilter(
        process_noise_covariance=process_noise,
        matrix_transform=matrix_transform,
        control_transform=control_transform,
        measurement_uncertainty=observation_uncertainty,
        state_transition_transform=state_transition_transform,
    )

    kf.init_state(
        state=np.array([1e-5] * 6).reshape(-1, 1),  # x, y, vx, vy, ax, ay
        estimate_uncertainty=np.diag(
            [
                500,
            ]
            * 6
        ),
        t=t,
    )
    observations = [
        [-393.66, 300.4],
        [-375.93, 301.78],
        [-351.04, 295.1],
        [-328.96, 305.19],
        [-299.35, 301.06],
        [-273.36, 302.05],
        [-245.89, 300],
        [-222.58, 303.57],
    ]
    states = [
        [-390.54, 298.02, -260.36, 198.7, -86.8, 66.23],
        [-378.9, 303.9, 53.8, -22.3, 94.5, -63.6],
    ]
    uncertainties = [
        [
            [8.93, 0, 5.95, 0, 2, 0],
            [0, 8.93, 0, 5.95, 0, 2],
            [5.95, 0, 504, 0, 334.7, 0],
            [0, 5.95, 0, 504, 0, 334.7],
            [2, 0, 334.7, 0, 444.9, 0],
            [0, 2, 0, 334.7, 0, 444.9],
        ],
        [
            [8.92, 0, 11.33, 0, 5.13, 0],
            [0, 8.92, 0, 11.33, 0, 5.13],
            [11.33, 0, 61.1, 0, 75.4, 0],
            [0, 11.33, 0, 61.1, 0, 75.4],
            [5.13, 0, 75.4, 0, 126.5, 0],
            [0, 5.13, 0, 75.4, 0, 126.5],
        ],
    ]
    for i, obs in enumerate(observations):
        t = t + timedelta(seconds=dt)
        kf.run(
            control_input=np.array(control_input),
            observation=np.array(obs).reshape(-1, 1),
            t=t,
        )

        if i < 2:
            x = states[i]
            p = uncertainties[i]
            print(i)
            print(np.array(p))
            print(kf.uncertainty)
            print()
            assert np.allclose(kf.state, np.array(x).reshape(-1, 1), rtol=0.1)
            assert np.allclose(kf.uncertainty, np.array(p), rtol=0.1)
