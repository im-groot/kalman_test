import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
import math

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from numpy.linalg import norm
from scipy.linalg import block_diag

def f_velocity_radar(x, dt):
    """
    状態遷移関数
    状態ベクトルは [x, x方向の速度, y, y方向の速度]
    """

    F = np.array([[1, dt, 0, 0],
                  [0,  1, 0, 0],
                  [0,  0, 1, dt],
                  [0, 0, 0, 1]], dtype=float)
    return F @ x

def h_velocity_radar(x):
    dx = x[0] - h_velocity_radar.radar_pos[0]
    dy = x[2] - h_velocity_radar.radar_pos[1]
    slant_range = math.sqrt(dx**2 + dy**2)
    azimuth_angle = math.atan2(dy, dx)
    return [slant_range, azimuth_angle]

h_velocity_radar.radar_pos = (0, 0)

def make_ukf_velocity_filter(dt, range_std_m, elevation_angle_std_deg, q_var=0.1, p_alpha=2):
    points = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=-1)
    kf = UKF(4, 2, dt, fx=f_velocity_radar, hx=h_velocity_radar, points=points)
    
    q = Q_discrete_white_noise(dim=2, dt=dt, var=q_var)
    kf.Q = block_diag(q, q)
    
    kf.R = np.diag([range_std_m**2, elevation_angle_std_deg**2])
    kf.x = np.array([0.0, 0.0, 0.0, 0.0])
    kf.P *= p_alpha

    return kf

## velocity
def f_velocity_doppler_radar(x, dt):
    """
    状態遷移関数
    状態ベクトルは [x, x方向の速度, y, y方向の速度]
    """

    F = np.array([[1, dt, 0, 0],
                  [0,  1, 0, 0],
                  [0,  0, 1, dt],
                  [0, 0, 0, 1]], dtype=float)
    return F @ x

def h_velocity_doppler_radar(x):
    dx = x[0] - h_velocity_radar.radar_pos[0]
    dy = x[2] - h_velocity_radar.radar_pos[1]
    slant_range = math.sqrt(dx**2 + dy**2)
    azimuth_angle = math.atan2(dy, dx)
    return [slant_range, azimuth_angle, x[1], x[3]] 

def make_ukf_doppler_velocity_filter(dt, range_std_m, elevation_angle_std_deg, q_var=0.1, p_alpha=2):
    points = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=-1)
    kf = UKF(4, 2, dt, fx=f_velocity_doppler_radar, hx=h_velocity_doppler_radar, points=points)
    
    q = Q_discrete_white_noise(dim=2, dt=dt, var=q_var)
    kf.Q = block_diag(q, q)
    
    kf.R = np.diag([range_std_m**2, elevation_angle_std_deg**2])
    kf.x = np.array([0.0, 0.0, 0.0, 0.0])
    kf.P *= p_alpha

    return kf

def f_accel_radar(x, dt):
    """
    状態遷移関数
    状態ベクトルは [x, x方向の速度, x方向の加速度、　y, y方向の速度、 y方向の加速度]
    """

    accel_model = np.array([[1, dt, 0.5*dt*dt],
                             [0,  1, dt],
                               [0,  0,  1]], dtype=float)
    F = block_diag(accel_model, accel_model)
    return F @ x

def h_accel_radar(x):
    dx = x[0] - h_accel_radar.radar_pos[0]
    dy = x[3] - h_accel_radar.radar_pos[1]
    slant_range = math.sqrt(dx**2 + dy**2)
    azimuth_angle = math.atan2(dy, dx)
    return [slant_range, azimuth_angle]

h_accel_radar.radar_pos = (0, 0)

def make_ukf_accel_filter(dt, range_std_m, elevation_angle_std_deg, q_var=0.02, p_alpha=2):
    points = MerweScaledSigmaPoints(n=6, alpha=.1, beta=2., kappa=-3.)
    kf = UKF(6, 2, dt, fx=f_accel_radar, hx=h_accel_radar, points=points)
    
    q = Q_discrete_white_noise(dim=3, dt=dt, var=q_var)
    kf.Q = block_diag(q, q)
    kf.R = np.diag([range_std_m**2, elevation_angle_std_deg**2])
    kf.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    kf.P *= p_alpha

    return kf

if __name__ == "__main__":
    from radar_simulation import generate_data, generate_data_maneuver, azimuth_to_xy

    dt_sec = 0.05
    range_std_m = 0.1
    elevation_angle_std_deg = 2
    track, zs = generate_data(dt_sec=dt_sec, heading=0, v0_m_sec=1.2, range_std_m=range_std_m, elevation_angle_std_deg=elevation_angle_std_deg)
    zs = np.flipud(zs)
    
    z_r = zs[:, 0]
    z_azimuth = zs[:, 1]
    sensor_xs, sensor_ys = azimuth_to_xy(*zip(*zs))

    kf = make_ukf_velocity_filter(dt_sec, range_std_m, elevation_angle_std_deg, p_alpha=1)
    # 初期値設定
    kf.x = np.array([sensor_xs[0], -0.6, sensor_ys[0], -0.6])
    kf.P = np.eye(4) * 10
    
    xs, ys = [], []
    vxs, vys = [], []
    for i, z in enumerate(zs):
        if i == 0:
            continue
        kf.predict()
        kf.update(z)
        xs.append(kf.x[0])
        ys.append(kf.x[2])
        vxs.append(kf.x[1])
        vys.append(kf.x[3])

    plt.figure(figsize=(8,4))
    plt.subplot(121)
    plt.scatter(sensor_xs, sensor_ys, color='r', alpha=0.3)
    plt.plot(xs, ys, color="b")
    plt.plot(*zip(*track), color="b", alpha=0.5)
    plt.xlim([-2,2])
    plt.ylim([0,5])
    plt.grid()
    plt.subplot(122)
    plt.plot(vxs)
    plt.plot(vys)
    plt.ylim([-1.5, 1.5])
    plt.grid()