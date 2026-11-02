"""
Ember Transport Simulation (Minimal Feasibility Model)
Simulates firebrand/ember particle trajectories under gravity and quadratic air drag
in horizontal wind above flat ground.

Inputs use SI units:
- wind in m/s
- height in metres
- mass in kilograms
- projected area in m²
"""

import argparse
import csv
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Default physical constants
DEFAULT_G = 9.81        # Gravity (m/s^2)
DEFAULT_RHO = 1.225     # Air density at sea level, 15°C (kg/m^3)
DEFAULT_CD = 1.0        # Ember drag coefficient (non-spherical / irregular fragment)


def derivative(state, t, mass, area, wind_vec, cd=DEFAULT_CD, rho=DEFAULT_RHO, g=DEFAULT_G):
    """
    Computes [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt].

    Identifies:
    1. Gravity: acts downward along -z with magnitude g.
    2. Quadratic air drag: acts in the direction opposite to relative velocity:
       v_rel = v - u_wind
       F_drag = -0.5 * rho * Cd * Area * ||v_rel|| * v_rel
       a_drag = F_drag / mass
    """
    x, y, z, vx, vy, vz = state

    # Relative velocity to the ambient wind
    vrx = vx - wind_vec[0]
    vry = vy - wind_vec[1]
    vrz = vz - wind_vec[2]
    v_rel = math.sqrt(vrx**2 + vry**2 + vrz**2)

    # Drag factor: 0.5 * rho * Cd * A / m
    drag_factor = 0.5 * rho * cd * area / mass

    # Total acceleration components
    ax = -drag_factor * v_rel * vrx
    ay = -drag_factor * v_rel * vry
    az = -g - drag_factor * v_rel * vrz

    return np.array([vx, vy, vz, ax, ay, az])


def rk4_step(state, t, dt, mass, area, wind_vec, cd=DEFAULT_CD, rho=DEFAULT_RHO, g=DEFAULT_G):
    """Fourth-order Runge-Kutta numerical integration step."""
    k1 = derivative(state, t, mass, area, wind_vec, cd, rho, g)
    k2 = derivative(state + 0.5 * dt * k1, t + 0.5 * dt, mass, area, wind_vec, cd, rho, g)
    k3 = derivative(state + 0.5 * dt * k2, t + 0.5 * dt, mass, area, wind_vec, cd, rho, g)
    k4 = derivative(state + dt * k3, t + dt, mass, area, wind_vec, cd, rho, g)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_single_trajectory(release_height, wind_vec, mass, area, dt=0.005, cd=DEFAULT_CD, rho=DEFAULT_RHO, g=DEFAULT_G):
    """
    Simulates a single ember from release until ground impact (z = 0).
    Returns (times, trajectory_array, landing_info_dict).
    """
    state = np.array([0.0, 0.0, float(release_height), 0.0, 0.0, 0.0])
    t = 0.0

    times = [t]
    states = [state.copy()]

    while state[2] > 0.0:
        prev_state = state.copy()
        prev_t = t

        state = rk4_step(state, t, dt, mass, area, wind_vec, cd=cd, rho=rho, g=g)
        t += dt

        times.append(t)
        states.append(state.copy())

    # Precise linear interpolation at ground boundary (z = 0)
    z_prev = prev_state[2]
    z_curr = state[2]
    alpha = z_prev / (z_prev - z_curr) if (z_prev - z_curr) != 0 else 1.0

    t_land = prev_t + alpha * dt
    x_land = prev_state[0] + alpha * (state[0] - prev_state[0])
    y_land = prev_state[1] + alpha * (state[1] - prev_state[1])
    vx_land = prev_state[3] + alpha * (state[3] - prev_state[3])
    vy_land = prev_state[4] + alpha * (state[4] - prev_state[4])
    vz_land = prev_state[5] + alpha * (state[5] - prev_state[5])

    # Replace last step with ground impact point
    states[-1] = np.array([x_land, y_land, 0.0, vx_land, vy_land, vz_land])
    times[-1] = t_land

    landing_info = {
        'flight_time': t_land,
        'landing_x': x_land,
        'landing_y': y_land,
        'landing_distance': math.sqrt(x_land**2 + y_land**2),
        'impact_speed': math.sqrt(vx_land**2 + vy_land**2 + vz_land**2)
    }

    return np.array(times), np.array(states), landing_info


def run_verifications():
    """
    Runs verification checks:
    1. Free fall (Cd = 0): compares against analytical t = sqrt(2h/g).
    2. Terminal velocity: compares against analytical vt = sqrt(2mg / (rho*Cd*A)).
    3. Timestep convergence: compares dt=0.01 vs dt=0.001.
    4. Wind-direction symmetry: compares +u_wind vs -u_wind.
    """
    print("\n--- Running Verification Suite ---")
    dt = 0.005

    # 1. Free fall check
    h = 20.0
    t_analytic = math.sqrt(2 * h / DEFAULT_G)
    _, states_ff, info_ff = simulate_single_trajectory(h, np.zeros(3), 0.001, 0.0004, dt=dt, cd=0.0)
    err_ff = abs(info_ff['flight_time'] - t_analytic)
    assert err_ff < 1e-3, f"Free fall verification failed: err={err_ff}"
    print(f"✓ Free fall check passed: t_sim={info_ff['flight_time']:.4f}s, t_analytic={t_analytic:.4f}s (diff={err_ff:.2e}s)")

    # 2. Terminal velocity check
    vt_analytic = math.sqrt(2 * 0.001 * DEFAULT_G / (DEFAULT_RHO * 0.0004 * DEFAULT_CD))
    _, states_term, _ = simulate_single_trajectory(200.0, np.zeros(3), 0.001, 0.0004, dt=dt, cd=DEFAULT_CD)
    vt_sim = abs(states_term[-1, 5])
    err_vt = abs(vt_sim - vt_analytic)
    assert err_vt < 1e-3, f"Terminal velocity verification failed: err={err_vt}"
    print(f"✓ Terminal velocity check passed: vt_sim={vt_sim:.4f}m/s, vt_analytic={vt_analytic:.4f}m/s (diff={err_vt:.2e}m/s)")

    # 3. Timestep convergence check
    _, _, info_dt1 = simulate_single_trajectory(20.0, np.array([5.0, 0, 0]), 0.001, 0.0004, dt=0.01)
    _, _, info_dt2 = simulate_single_trajectory(20.0, np.array([5.0, 0, 0]), 0.001, 0.0004, dt=0.001)
    diff_dt = abs(info_dt1['landing_x'] - info_dt2['landing_x'])
    assert diff_dt < 0.02, f"Timestep convergence check failed: diff={diff_dt}"
    print(f"✓ Timestep convergence check passed: dt=0.01 vs 0.001 diff = {diff_dt:.4f}m")

    # 4. Wind-direction symmetry check
    _, _, info_w_pos = simulate_single_trajectory(20.0, np.array([5.0, 0, 0]), 0.001, 0.0004, dt=dt)
    _, _, info_w_neg = simulate_single_trajectory(20.0, np.array([-5.0, 0, 0]), 0.001, 0.0004, dt=dt)
    diff_sym = abs(info_w_pos['landing_x'] + info_w_neg['landing_x'])
    assert diff_sym < 1e-3, f"Wind symmetry check failed: diff={diff_sym}"
    print(f"✓ Wind-direction symmetry check passed: +5m/s={info_w_pos['landing_x']:.2f}m, -5m/s={info_w_neg['landing_x']:.2f}m")
    print("Verification: free fall, terminal velocity, timestep convergence, and wind-direction symmetry checks passed.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ember transport simulation with gravity, quadratic air drag, and wind distributions."
    )
    parser.add_argument("--wind", type=float, default=5.0, help="Horizontal wind speed in m/s (default: 5.0)")
    parser.add_argument("--height", type=float, default=20.0, help="Ember release height in metres (default: 20.0)")
    parser.add_argument("--mass", type=float, default=0.001, help="Ember mass in kilograms (default: 0.001 kg = 1 g)")
    parser.add_argument("--area", type=float, default=0.0004, help="Projected area in m² (default: 0.0004 m² = 4 cm²)")
    parser.add_argument("--particles", type=int, default=300, help="Number of sampled particles (default: 300)")
    parser.add_argument("--outdir", type=str, default="ember_results", help="Directory for output plots and data (default: ember_results)")
    parser.add_argument("--verify", action="store_true", help="Run verification checks and exit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")

    args = parser.parse_args()

    # If verification requested or run by default
    run_verifications()
    if args.verify:
        return

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"--- Running Baseline Simulation ---")
    print(f"Release height: {args.height:.2f} m")
    print(f"Wind speed:     {args.wind:.2f} m/s")
    print(f"Particle mass:  {args.mass * 1000:.2f} g ({args.mass:.5f} kg)")
    print(f"Projected area: {args.area * 10000:.2f} cm² ({args.area:.6f} m²)")
    print(f"Area-to-mass:   {args.area / args.mass:.2f} m²/kg")

    # Baseline particle (pure horizontal wind along +x)
    base_wind = np.array([args.wind, 0.0, 0.0])
    base_times, base_states, base_landing = simulate_single_trajectory(
        args.height, base_wind, args.mass, args.area
    )

    print(f"\nBaseline landing results:")
    print(f"  Flight time:      {base_landing['flight_time']:.2f} seconds")
    print(f"  Landing distance: {base_landing['landing_x']:.2f} m downwind")
    print(f"With the illustrative defaults—20 m release height, 5 m/s wind, 1 g mass, and 4 cm² area—")
    print(f"the baseline lands about {base_landing['landing_x']:.2f} m downwind after {base_landing['flight_time']:.2f} seconds.\n")

    # Sample varied particles
    num_particles = max(10, args.particles)
    print(f"Simulating {num_particles} varied particles for distribution analysis...")

    # Sample particle properties to reflect uncertainty in ember morphology and wind
    # Height varies by ~15%
    sampled_heights = np.clip(rng.normal(args.height, 0.15 * args.height, num_particles), 1.0, 500.0)
    # Mass varies log-normally (factor of ~0.5 to 2)
    sampled_masses = args.mass * np.exp(rng.normal(0.0, 0.35, num_particles))
    # Area varies log-normally
    sampled_areas = args.area * np.exp(rng.normal(0.0, 0.30, num_particles))
    # Wind speed varies by ~15%
    sampled_winds = np.clip(rng.normal(args.wind, 0.15 * max(args.wind, 1.0), num_particles), 0.1, 50.0)
    # Wind direction varies slightly between flights (+- 10 degrees std dev)
    sampled_wind_angles_rad = rng.normal(0.0, np.radians(10.0), num_particles)

    landing_records = []
    sampled_trajectories = []

    # Record baseline first
    landing_records.append({
        'particle_id': 0,
        'is_baseline': 1,
        'mass_kg': args.mass,
        'area_m2': args.area,
        'area_to_mass': args.area / args.mass,
        'release_height_m': args.height,
        'wind_speed_m_s': args.wind,
        'wind_dir_deg': 0.0,
        'flight_time_s': base_landing['flight_time'],
        'landing_x_m': base_landing['landing_x'],
        'landing_y_m': base_landing['landing_y'],
        'landing_dist_m': base_landing['landing_distance']
    })

    # Record first 50 sample trajectories for visual display
    num_to_plot = min(50, num_particles)

    for i in range(num_particles):
        h_i = sampled_heights[i]
        m_i = sampled_masses[i]
        a_i = sampled_areas[i]
        w_speed = sampled_winds[i]
        w_ang = sampled_wind_angles_rad[i]

        w_vec = np.array([w_speed * math.cos(w_ang), w_speed * math.sin(w_ang), 0.0])

        t_arr, s_arr, land_info = simulate_single_trajectory(h_i, w_vec, m_i, a_i)

        if i < num_to_plot:
            sampled_trajectories.append((t_arr, s_arr))

        landing_records.append({
            'particle_id': i + 1,
            'is_baseline': 0,
            'mass_kg': m_i,
            'area_m2': a_i,
            'area_to_mass': a_i / m_i,
            'release_height_m': h_i,
            'wind_speed_m_s': w_speed,
            'wind_dir_deg': math.degrees(w_ang),
            'flight_time_s': land_info['flight_time'],
            'landing_x_m': land_info['landing_x'],
            'landing_y_m': land_info['landing_y'],
            'landing_dist_m': land_info['landing_distance']
        })

    # Save CSV files
    landing_csv_path = os.path.join(args.outdir, "landing_data.csv")
    with open(landing_csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=landing_records[0].keys())
        writer.writeheader()
        writer.writerows(landing_records)
    print(f"Saved landing data table to: {landing_csv_path}")

    base_traj_csv_path = os.path.join(args.outdir, "baseline_trajectory.csv")
    with open(base_traj_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'x_m', 'y_m', 'z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s'])
        for t, row in zip(base_times, base_states):
            writer.writerow([f"{t:.4f}", f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}", f"{row[5]:.4f}"])
    print(f"Saved baseline trajectory to: {base_traj_csv_path}")

    # Generate Two-Panel Visualization
    print("Generating plots...")
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left Plot: Trajectories viewed from the side ---
    cmap = plt.get_cmap("viridis")
    for idx, (t_arr, s_arr) in enumerate(sampled_trajectories):
        color = cmap(idx / max(1, len(sampled_trajectories) - 1))
        ax_left.plot(s_arr[:, 0], s_arr[:, 2], color=color, alpha=0.35, linewidth=1.2)

    # Black curve: the baseline particle
    ax_left.plot(
        base_states[:, 0], base_states[:, 2],
        color="black", linewidth=2.5, label="Baseline particle"
    )
    ax_left.scatter(
        [base_landing['landing_x']], [0.0],
        color="red", zorder=5, s=50,
        label=f"Baseline landing ({base_landing['landing_x']:.2f} m, {base_landing['flight_time']:.2f} s)"
    )

    ax_left.set_title("Trajectories viewed from the side", fontsize=12, fontweight='bold')
    ax_left.set_xlabel("distance along the x-direction, in metres", fontsize=10)
    ax_left.set_ylabel("height above the ground, in metres", fontsize=10)
    ax_left.set_ylim(bottom=0.0)
    ax_left.grid(True, linestyle="--", alpha=0.5)
    ax_left.legend(loc="upper right", framealpha=0.9)

    # --- Right Plot: Landing locations viewed from above ---
    all_land_x = [rec['landing_x_m'] for rec in landing_records[1:]]
    all_land_y = [rec['landing_y_m'] for rec in landing_records[1:]]

    # Calculate 2D histogram of landing locations
    bins_x = max(10, min(30, int(math.sqrt(num_particles))))
    bins_y = max(8, min(25, int(math.sqrt(num_particles) * 0.8)))

    counts, xedges, yedges = np.histogram2d(all_land_x, all_land_y, bins=[bins_x, bins_y])
    fractions = counts / float(num_particles)  # fraction of simulated particles

    X, Y = np.meshgrid(xedges, yedges)
    mesh = ax_right.pcolormesh(X, Y, fractions.T, cmap="Reds", edgecolors='none')
    cbar = plt.colorbar(mesh, ax=ax_right)
    cbar.set_label("Fraction of particles", fontsize=10)

    # Plot baseline landing point on top
    ax_right.scatter(
        [base_landing['landing_x']], [base_landing['landing_y']],
        color="black", marker="x", s=80, linewidth=2,
        label="Baseline landing", zorder=6
    )

    ax_right.set_title("Landing locations viewed from above", fontsize=12, fontweight='bold')
    ax_right.set_xlabel("x-position on the ground, in metres", fontsize=10)
    ax_right.set_ylabel("sideways y-position, in metres", fontsize=10)
    ax_right.grid(True, linestyle="--", alpha=0.5)
    ax_right.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "trajectory_and_landing_distribution.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved plot figure to: {plot_path}\n")

    print("--- Summary of Results ---")
    print(f"Total particles simulated: {num_particles}")
    print(f"Landing distance (mean ± std): {np.mean(all_land_x):.2f} ± {np.std(all_land_x):.2f} m")
    print(f"Flight time (mean ± std):      {np.mean([r['flight_time_s'] for r in landing_records[1:]]):.2f} ± {np.std([r['flight_time_s'] for r in landing_records[1:]]):.2f} s")
    print(f"Results appear in the {args.outdir} folder.")


if __name__ == "__main__":
    main()
