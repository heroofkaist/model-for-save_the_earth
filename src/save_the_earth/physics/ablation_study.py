"""
Save the Earth — Systematic Physical Ablation Study.

Evaluates the physical contribution of individual aerodynamic and combustion mechanisms:
1. Config 1 (Baseline Ballistic): Drag only (Cd=1.0, Cl=0.0, inert, constant mass)
2. Config 2 (+ Tohidi Lift & Morphology Drag): Tohidi & Kaye (2017) aspect-ratio Cd and tumbling Cl
3. Config 3 (+ Wong Combustion Buoyancy): Config 2 + Wong et al. (CSIRO 2026) micro-plume buoyancy (eta_buoy=1.18)
4. Config 4 (+ Albini Coupled Mass & Area Loss): Config 3 + Albini (1979) burning mass & area shrinkage (beta=0.02 s^-1)

Evaluates on a factorial test suite spanning:
- Particle morphologies: flat bark flake, cylindrical needle/rod, compact wood chunk
- Wind speeds: 5.36 m/s, 11.17 m/s, 17.88 m/s
- Release heights: 1.22 m (IBHS panel center) and 2.0 m (roof eave)

Outputs:
- JSON report: reports/physics_ablation_results.json
- Publication figure: reports/figures/physics_ablation_comparison.png
"""

import json
import math
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.save_the_earth.physics.aerodynamics import (
    simulate_advanced_trajectory,
    compute_tohidi_drag_lift
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT_JSON = PROJECT_ROOT / "reports" / "physics_ablation_results.json"
OUT_FIG = PROJECT_ROOT / "reports" / "figures" / "physics_ablation_comparison.png"


def run_ablation_study():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    # 1. Morphologies to test
    particles = {
        "Flat Bark Flake (Slender)": {
            "mass": 3.0e-5,   # 30 mg
            "area": 2.5e-4,   # 2.5 cm^2
            "thickness": 0.001,
            "description": "High-aspect fluttering ribbon bark / veneer flake"
        },
        "Cylindrical Rod / Twig": {
            "mass": 5.0e-5,   # 50 mg
            "area": 1.2e-4,   # 1.2 cm^2
            "thickness": 0.003,
            "description": "Elongated twig / pine needle"
        },
        "Compact Wood Chunk": {
            "mass": 8.0e-5,   # 80 mg
            "area": 0.8e-4,   # 0.8 cm^2
            "thickness": 0.008,
            "description": "Near-isometric wood cube / knot"
        }
    }

    wind_speeds = [5.36, 11.17, 17.88]
    release_heights = [1.22, 2.0]

    configs = {
        "Config 1: Baseline Drag Only": {
            "use_tohidi": False,
            "is_burning": False,
            "eta_buoy": 1.0,
            "burnout_rate": 0.0,
            "color": "#7f8c8d"
        },
        "Config 2: + Tohidi Drag & Lift": {
            "use_tohidi": True,
            "is_burning": False,
            "eta_buoy": 1.0,
            "burnout_rate": 0.0,
            "color": "#2980b9"
        },
        "Config 3: + Wong Combustion Buoyancy": {
            "use_tohidi": True,
            "is_burning": True,
            "eta_buoy": 1.18,
            "burnout_rate": 0.0,
            "color": "#8e44ad"
        },
        "Config 4: + Albini Mass & Area Loss (Full)": {
            "use_tohidi": True,
            "is_burning": True,
            "eta_buoy": 1.18,
            "burnout_rate": 0.02,
            "color": "#e67e22"
        }
    }

    results = {}
    scenario_records = []

    for p_name, p_props in particles.items():
        results[p_name] = {}
        cd_t, cl_t = compute_tohidi_drag_lift(p_props["mass"], p_props["area"], p_props["thickness"])

        for h in release_heights:
            results[p_name][f"h_{h:.2f}m"] = {}
            for u in wind_speeds:
                results[p_name][f"h_{h:.2f}m"][f"u_{u:.2f}mps"] = {}
                
                for c_name, c_props in configs.items():
                    cd = cd_t if c_props["use_tohidi"] else 1.0
                    cl = cl_t if c_props["use_tohidi"] else 0.0

                    _, _, info = simulate_advanced_trajectory(
                        release_height=h,
                        u_ref=u,
                        mass=p_props["mass"],
                        area=p_props["area"],
                        cd=cd,
                        cl_lift=cl,
                        is_burning=c_props["is_burning"],
                        eta_buoy=c_props["eta_buoy"],
                        burnout_rate=c_props["burnout_rate"],
                        dt=0.005
                    )

                    entry = {
                        "landing_distance_m": round(info["landing_distance_m"], 3),
                        "flight_time_s": round(info["flight_time_s"], 3),
                        "impact_vz_m_s": round(info["impact_vz_m_s"], 3),
                    }
                    results[p_name][f"h_{h:.2f}m"][f"u_{u:.2f}mps"][c_name] = entry

                    scenario_records.append({
                        "particle": p_name,
                        "height_m": h,
                        "wind_m_s": u,
                        "config": c_name,
                        **entry
                    })

    # Compute aggregate shifts relative to Baseline
    summary_deltas = {c: {"delta_dist_pct": [], "delta_time_pct": []} for c in configs if c != "Config 1: Baseline Drag Only"}
    for p_name in particles:
        for h in release_heights:
            for u in wind_speeds:
                base_dist = results[p_name][f"h_{h:.2f}m"][f"u_{u:.2f}mps"]["Config 1: Baseline Drag Only"]["landing_distance_m"]
                base_time = results[p_name][f"h_{h:.2f}m"][f"u_{u:.2f}mps"]["Config 1: Baseline Drag Only"]["flight_time_s"]
                for c in summary_deltas:
                    c_dist = results[p_name][f"h_{h:.2f}m"][f"u_{u:.2f}mps"][c]["landing_distance_m"]
                    c_time = results[p_name][f"h_{h:.2f}m"][f"u_{u:.2f}mps"][c]["flight_time_s"]
                    summary_deltas[c]["delta_dist_pct"].append(100.0 * (c_dist - base_dist) / max(0.01, base_dist))
                    summary_deltas[c]["delta_time_pct"].append(100.0 * (c_time - base_time) / max(0.01, base_time))

    aggregate_summary = {
        c: {
            "mean_landing_distance_shift_pct": round(float(np.mean(summary_deltas[c]["delta_dist_pct"])), 2),
            "max_landing_distance_shift_pct": round(float(np.max(summary_deltas[c]["delta_dist_pct"])), 2),
            "mean_flight_time_shift_pct": round(float(np.mean(summary_deltas[c]["delta_time_pct"])), 2)
        }
        for c in summary_deltas
    }

    # Save JSON report
    report_data = {
        "title": "Systematic Physical Ablation Study: Firebrand Aerodynamics & Combustion",
        "description": "Quantifies the trajectory shift produced by Tohidi lift, Wong micro-plume buoyancy, and Albini mass/area shrinkage.",
        "aggregate_impact_relative_to_baseline": aggregate_summary,
        "detailed_scenarios": results
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved physics ablation JSON report to: {OUT_JSON}")

    # Plot Comparison Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharey=True)
    p_names = list(particles.keys())

    for idx, (p_name, ax) in enumerate(zip(p_names, axes)):
        x_indices = np.arange(len(wind_speeds))
        width = 0.18

        for c_idx, (c_name, c_props) in enumerate(configs.items()):
            dists = [
                results[p_name]["h_1.22m"][f"u_{u:.2f}mps"][c_name]["landing_distance_m"]
                for u in wind_speeds
            ]
            short_label = c_name.split(":")[1].strip().replace("Tohidi ", "").replace("Wong ", "").replace("Albini ", "")
            rects = ax.bar(x_indices + (c_idx - 1.5) * width, dists, width=width,
                           label=short_label if idx == 0 else "", color=c_props["color"], edgecolor="black", alpha=0.9)
            
            # Label bar values
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f"{height:.1f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 2),  # 2 points vertical offset
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"{w:.1f} m/s" for w in wind_speeds], fontsize=10, fontweight="bold")
        ax.set_xlabel("Wind Speed", fontsize=10, fontweight="bold")
        ax.set_title(p_name, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Landing Distance (meters, at h=1.22m)", fontsize=11, fontweight="bold")
    fig.suptitle("Physical Ablation Study: Cumulative Impact of Lift, Buoyancy & Mass Loss", fontsize=13, fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=True, fontsize=9.5)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()
    print(f"Saved physics ablation figure to: {OUT_FIG}")

    # Print summary table
    print("\n================== PHYSICAL ABLATION SUMMARY ==================")
    print(f"{'Mechanism Addition':<40} | {'Mean Dist Shift':<16} | {'Max Dist Shift':<15} | {'Flight Time Shift':<16}")
    print("-" * 92)
    for c, stats in aggregate_summary.items():
        print(f"{c:<40} | {stats['mean_landing_distance_shift_pct']:+6.1f} %          | {stats['max_landing_distance_shift_pct']:+6.1f} %         | {stats['mean_flight_time_shift_pct']:+6.1f} %")
    print("=" * 92)

    return report_data


if __name__ == "__main__":
    run_ablation_study()
