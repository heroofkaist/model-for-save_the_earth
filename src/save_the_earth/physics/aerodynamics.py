"""
Save the Earth — Mechanistic Aerodynamics and Numerical Integration Engine.

Implements coupled firebrand flight dynamics:
1. Quadratic air drag relative to ambient wind field:
   F_drag = -0.5 * rho * Cd * Area * ||v - u_wind|| * (v - u_wind)
2. Height-dependent atmospheric boundary layer wind profile:
   u_wind(z) = u_ref * ln(z / z0 + 1) / ln(z_ref / z0 + 1)
3. Buoyant convective updraft decay:
   w_plume(z) = w0 * exp(-z / b_plume)
4. Numerical 4th-order Runge-Kutta (RK4) integration with exact ground plane interpolation.
"""

import math
import numpy as np
from typing import Tuple, Dict, Any


DEFAULT_G = 9.80665     # Gravity (m/s^2)
DEFAULT_RHO = 1.225     # Air density (kg/m^3)


def atmospheric_wind_vector(
    x: float,
    y: float,
    z: float,
    u_ref: float,
    wind_angle_rad: float = 0.0,
    z0: float = 0.01,
    z_ref: float = 10.0,
    w0_updraft: float = 0.0,
    b_plume: float = 15.0,
    r_plume: float = 12.0
) -> np.ndarray:
    z_clamped = max(0.001, z)
    u_horiz = u_ref * math.log(z_clamped / z0 + 1.0) / math.log(z_ref / z0 + 1.0)
    
    ux = u_horiz * math.cos(wind_angle_rad)
    uy = u_horiz * math.sin(wind_angle_rad)
    
    # Plume updraft decays both vertically and horizontally away from the fire origin
    if w0_updraft > 0:
        horiz_dist_sq = x * x + y * y
        decay_xy = math.exp(-horiz_dist_sq / (2.0 * r_plume * r_plume))
        uz = w0_updraft * math.exp(-z_clamped / b_plume) * decay_xy
    else:
        uz = 0.0
        
    return np.array([ux, uy, uz], dtype=np.float64)


def particle_derivative(
    state: np.ndarray,
    mass: float,
    area: float,
    cd: float,
    u_ref: float,
    wind_angle_rad: float = 0.0,
    z0: float = 0.01,
    z_ref: float = 10.0,
    w0_updraft: float = 0.0,
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G
) -> np.ndarray:
    x = state[0]
    y = state[1]
    z = state[2]
    vel = state[3:]
    
    wind = atmospheric_wind_vector(x, y, z, u_ref, wind_angle_rad, z0, z_ref, w0_updraft)
    v_rel = vel - wind
    speed_rel = math.sqrt(v_rel[0]**2 + v_rel[1]**2 + v_rel[2]**2)
    speed_rel = min(150.0, speed_rel)  # Physical cap
    
    drag_factor = 0.5 * rho * cd * area / mass
    # Max acceleration cap to prevent numerical divergence on edge cases
    accel = -drag_factor * speed_rel * v_rel + np.array([0.0, 0.0, -g], dtype=np.float64)
    accel = np.clip(accel, -500.0, 500.0)
    
    return np.array([vel[0], vel[1], vel[2], accel[0], accel[1], accel[2]], dtype=np.float64)


def simulate_advanced_trajectory(
    release_height: float,
    u_ref: float,
    mass: float,
    area: float,
    cd: float = 1.0,
    v0: np.ndarray = None,
    wind_angle_rad: float = 0.0,
    z0: float = 0.01,
    w0_updraft: float = 0.0,
    dt: float = 0.005,
    max_t: float = 45.0,
    resample_points: int = 50
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if v0 is None:
        v0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
    state = np.array([0.0, 0.0, float(release_height), v0[0], v0[1], v0[2]], dtype=np.float64)
    t = 0.0
    
    raw_times = [0.0]
    raw_states = [state.copy()]
    
    while state[2] > 0.0 and t < max_t:
        prev_state = state.copy()
        prev_t = t
        
        # RK4 numerical integration
        k1 = particle_derivative(state, mass, area, cd, u_ref, wind_angle_rad, z0, 10.0, w0_updraft)
        k2 = particle_derivative(state + 0.5 * dt * k1, mass, area, cd, u_ref, wind_angle_rad, z0, 10.0, w0_updraft)
        k3 = particle_derivative(state + 0.5 * dt * k2, mass, area, cd, u_ref, wind_angle_rad, z0, 10.0, w0_updraft)
        k4 = particle_derivative(state + dt * k3, mass, area, cd, u_ref, wind_angle_rad, z0, 10.0, w0_updraft)
        
        state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        
        raw_times.append(t)
        raw_states.append(state.copy())
        
    # Ground plane interpolation (z = 0)
    if state[2] <= 0.0:
        z_prev = prev_state[2]
        z_curr = state[2]
        denom = z_prev - z_curr
        alpha = z_prev / denom if abs(denom) > 1e-9 else 0.5
        alpha = min(max(alpha, 0.0), 1.0)
        
        t_land = prev_t + alpha * dt
        pos_land = prev_state[:3] + alpha * (state[:3] - prev_state[:3])
        pos_land[2] = 0.0
        vel_land = prev_state[3:] + alpha * (state[3:] - prev_state[3:])
    else:
        # Reached max_t without ground impact
        t_land = t
        pos_land = state[:3].copy()
        vel_land = state[3:].copy()
    
    raw_times[-1] = t_land
    raw_states[-1] = np.array([pos_land[0], pos_land[1], pos_land[2], vel_land[0], vel_land[1], vel_land[2]])
    
    raw_times = np.array(raw_times)
    raw_states = np.array(raw_states)
    
    # Resample to uniform evaluation grid
    uniform_t = np.linspace(0.0, t_land, resample_points)
    resampled_states = np.zeros((resample_points, 6), dtype=np.float64)
    for dim in range(6):
        resampled_states[:, dim] = np.interp(uniform_t, raw_times, raw_states[:, dim])
        
    landing_dist = math.sqrt(pos_land[0]**2 + pos_land[1]**2)
    
    info = {
        "flight_time_s": float(t_land),
        "landing_x_m": float(pos_land[0]),
        "landing_y_m": float(pos_land[1]),
        "landing_dist_m": float(landing_dist),
        "landing_distance_m": float(landing_dist),
        "impact_vz_m_s": float(vel_land[2]),
        "max_height_m": float(np.max(raw_states[:, 2]))
    }
    
    return uniform_t, resampled_states, info
