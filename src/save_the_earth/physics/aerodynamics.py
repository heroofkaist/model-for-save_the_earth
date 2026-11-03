"""
Save the Earth — Mechanistic Aerodynamics and Numerical Integration Engine.

Implements coupled firebrand flight dynamics:
1. Quadratic air drag relative to ambient wind field:
   F_drag = -0.5 * rho * Cd * Area * ||v - u_wind|| * (v - u_wind)
2. Height-dependent atmospheric boundary layer wind profile:
   u_wind(z) = u_ref * ln(z / z0 + 1) / ln(z_ref / z0 + 1)
3. Buoyant convective updraft decay:
   w_plume(z) = w0 * exp(-z / b_plume)
4. Empirical aerodynamics and combustion physics from experimental literature:
   - Tohidi & Kaye (JWEIA 2017): Aspect-ratio dependent drag and tumbling lift (Cl).
   - Wong, Sullivan, Zhang (CSIRO, Proc. Combust. Inst. 2026): Effective combustion
     buoyancy enhancement (eta_buoy = 1.15 - 1.24) and vortical flow elongation.
   - Albini (1979) / Tarifa et al. (1965): Dynamic burning mass loss (dm/dt).
5. Numerical 4th-order Runge-Kutta (RK4) integration with exact ground plane interpolation.
"""

import math
import numpy as np
from typing import Tuple, Dict, Any


DEFAULT_G = 9.80665     # Gravity (m/s^2)
DEFAULT_RHO = 1.225     # Air density (kg/m^3)


def compute_tohidi_drag_lift(
    mass: float,
    area: float,
    thickness: float = 0.002
) -> Tuple[float, float]:
    """
    Computes morphology-dependent drag coefficient (Cd) and tumbling lift coefficient (Cl)
    based on Tohidi & Kaye (2017).
    Disk/plate-like and rod-like embers exhibit significant lift and higher drag under tumbling.
    """
    # Equivalent diameter and aspect ratio
    d_eq = math.sqrt(4.0 * max(1e-7, area) / math.pi)
    slenderness = d_eq / max(1e-4, thickness)

    if slenderness > 10.0:
        # Flat plate / bark strip: fluttering / autorotating
        cd = 1.15
        cl = 0.35
    elif slenderness > 3.0:
        # Elongated cylinder / twig
        cd = 0.95
        cl = 0.18
    else:
        # Compact / spherical chunk
        cd = 0.85
        cl = 0.05
    return cd, cl


def compute_combustion_buoyancy(is_burning: bool = False, eta_buoy: float = 1.18) -> float:
    """
    Computes effective gravitational settling acceleration.
    Wong, Sullivan, Zhang (CSIRO / Proc. Combust. Inst. 2026) demonstrated that
    active combustion creates a micro-thermal plume around the ember, yielding
    an effective buoyancy enhancement factor eta_buoy in [1.15, 1.24], which
    retards vertical descent rate by reducing effective gravity: g_eff = g / eta_buoy.
    """
    if not is_burning:
        return DEFAULT_G
    return DEFAULT_G / max(1.0, eta_buoy)


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
    g: float = DEFAULT_G,
    cl_lift: float = 0.0
) -> np.ndarray:
    x = state[0]
    y = state[1]
    z = state[2]
    vel = state[3:]
    
    wind = atmospheric_wind_vector(x, y, z, u_ref, wind_angle_rad, z0, z_ref, w0_updraft)
    v_rel = vel - wind
    speed_rel = math.sqrt(v_rel[0]**2 + v_rel[1]**2 + v_rel[2]**2)
    speed_rel = min(150.0, speed_rel)  # Physical cap
    
    drag_factor = 0.5 * rho * cd * area / max(1e-7, mass)
    accel = -drag_factor * speed_rel * v_rel + np.array([0.0, 0.0, -g], dtype=np.float64)

    # Tohidi & Kaye (2017) aerodynamic lift for non-spherical tumbling firebrands
    if cl_lift > 0.0 and speed_rel > 0.1:
        # Lift acts upward and cross-stream normal to relative velocity
        lift_mag = 0.5 * rho * cl_lift * area * (speed_rel**2) / max(1e-7, mass)
        # Vertical component of lift opposing gravity
        horiz_speed = math.sqrt(v_rel[0]**2 + v_rel[1]**2)
        if horiz_speed > 1e-4:
            accel[2] += min(lift_mag * (horiz_speed / speed_rel), 0.8 * g)

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
    resample_points: int = 50,
    is_burning: bool = False,
    eta_buoy: float = 1.18,
    cl_lift: float = 0.0,
    burnout_rate: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if v0 is None:
        v0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
    state = np.array([0.0, 0.0, float(release_height), v0[0], v0[1], v0[2]], dtype=np.float64)
    t = 0.0
    m_curr = float(mass)
    
    # Effective gravity accounting for Wong et al. (CSIRO 2026) combustion buoyancy
    g_eff = compute_combustion_buoyancy(is_burning=is_burning, eta_buoy=eta_buoy)
    
    raw_times = [0.0]
    raw_states = [state.copy()]
    
    while state[2] > 0.0 and t < max_t:
        prev_state = state.copy()
        prev_t = t
        
        # Albini (1979) combustion mass loss
        if burnout_rate > 0.0 and is_burning:
            m_curr = mass * max(0.1, 1.0 - burnout_rate * t)**1.5

        # RK4 numerical integration
        k1 = particle_derivative(state, m_curr, area, cd, u_ref, wind_angle_rad, z0, 10.0, w0_updraft, g=g_eff, cl_lift=cl_lift)
        k2 = particle_derivative(state + 0.5 * dt * k1, m_curr, area, cd, u_ref, wind_angle_rad, z0, 10.0, w0_updraft, g=g_eff, cl_lift=cl_lift)
        k3 = particle_derivative(state + 0.5 * dt * k2, m_curr, area, cd, u_ref, wind_angle_rad, z0, 10.0, w0_updraft, g=g_eff, cl_lift=cl_lift)
        k4 = particle_derivative(state + dt * k3, m_curr, area, cd, u_ref, wind_angle_rad, z0, 10.0, w0_updraft, g=g_eff, cl_lift=cl_lift)
        
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
