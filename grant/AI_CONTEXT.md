# AI_CONTEXT — scientific handoff for the KAIST Save the Earth application

**Read this first.** This document is a factual orientation for a future AI writing assistant and the four-person KAIST team. It is **not** the final application, a host invitation, or a claim of completed experimental 3D validation.

**Repository:** [heroofkaist/model-for-save_the_earth](https://github.com/heroofkaist/model-for-save_the_earth)  
**Context snapshot:** 2026-11-03; evidence values originate from the checked-in scientific reports and metrics. Recheck the current `main` when preparing the official application.  
**Program:** teamKAIST *Save the Earth* Global Challenge (overseas expedition). **Verify all eligibility, deadline, format, and budget rules against the official announcement and application attachments before submission.**  
**Team:** Four KAIST students. Their actual names, majors, contributions, roles, and approvals must come from the team. No earlier speculative itinerary or biography should be reused as a verified fact.

## 1. The central story

We began the scientific work **before** applying for travel funding. Our code investigates how burning firebrands (windborne embers) move and where they land. It combines numerical aerodynamics, simplified literature-informed combustion physics, fast neural surrogates, and deposition-distance benchmarks. We can simulate plausible behavior and identify sensitivities, but the current public data do **not** let us validate full in-flight 3D motion under measured boundary conditions. The proposed Australian expedition is meant to learn from researchers and field practitioners, assess experimental methods, seek permissible data access, and design the next **testable** validation study.

The strongest defensible pitch is **prepared work → identified measurement gap → well-defined expedition tasks → reproducible next-stage research**, not “our neural network already predicts real wildfires.”

### Clearly separate the scientific targets

1. **Transport:** airborne particle trajectory, flight time, and deposition.
2. **Thermal survival:** whether the particle remains hot or burning when it lands.
3. **Ignition and wildfire spotting:** whether receptive material ignites and the new fire spreads.

The strongest implemented and benchmarked components are **transport and landing distance**. They do not establish real-world spot-fire ignition forecasts.

## 2. What has actually been implemented

| Component | What it does | Entry point |
| --- | --- | --- |
| Mechanistic 3D solver | RK4 integration with quadratic drag, gravity, logarithmic wind profile, idealized updraft; optional literature-informed lift, combustion buoyancy and mass/area shrinkage | [physics/aerodynamics.py](../src/save_the_earth/physics/aerodynamics.py) |
| Synthetic training set | 10,000 RK4-generated trajectories plus 500 OOD scenarios | [trajectory metrics](../reports/trajectory_evaluation_metrics.json), [training report](../reports/TRAJECTORY_TRAINING_10K.md) |
| Models B and C | Neural surrogate and physics-hybrid model trained to approximate generated trajectories | [models](../src/save_the_earth/models/trajectory_models.py) |
| Run-level deposition predictors | Ridge, Huber and other regression baselines on **published experimental-run summaries**, not full 3D trajectories | [benchmark code](../train_distance_benchmarks.py), [evidence dossier](scientific_evidence.md) |
| Moment-reconstructed particle benchmark | 60,370 **synthetic** particle rows sampled from distributional moments attributed to published USDA experimental summaries | [generator](../src/save_the_earth/data/usda_particles.py), [manifest](../data/manifests/usda_particles_manifest.json) |
| Model E | Deep distance/quantile model trained on the **synthetically reconstructed particles**, not on 60,370 measured rows | [model](../src/save_the_earth/models/experimental_surrogate.py), [training metrics](../reports/experimental_training_metrics.json) |
| OpenPTV import | Reconstructs linked 3D spatial detections from short consecutive optical frames | [OpenPTV report](../reports/OPENPTV_EXPERIMENTAL_VALIDATION.md), [metadata/limits](../reports/openptv_validation_metrics.json) |
| Sensitivity & ablation | Tests how **our numerical predictions** change when assumptions or modeled mechanisms change | [ablation JSON](../reports/physics_ablation_results.json), [sensitivity code](../physics_sensitivity.py) |

The project’s lift coefficients, equivalent combustion buoyancy and mass-loss equations are **simplified phenomenological choices informed by published studies**, not a full six-degree-of-freedom (6-DOF) resolving of particle orientation, turbulent combustion, and variable shape. Actual six-DOF orientation measurements and laboratory validation remain future work.

## 3. Numerical results that may be used, with the exact qualification

**A. Synthetic 3D trajectories, compared with the RK4 generator (not real world):** 10,000 total generated; held-out simulated test set of 1,500. Recorded 3D position MAE: Model B approximately **4.736 m**, Model C approximately **1.750 m**. Model B's recorded OOD 3D MAE is approximately **113.119 m**, demonstrating fragile extrapolation. [Source](../reports/trajectory_evaluation_metrics.json). **Never report 1.750 m as measured-firebrand 3D accuracy.**

**B. USDA *run-level* deposition benchmark (actual empirical aggregate summaries):** 69 reported experimental runs; recorded Huber five-fold group-CV MAE **0.995 m**, held-out test MAE **1.763 m** on run-median landing distances. Recorded Ridge held-out test MAE **1.721 m**. [Reported results](scientific_evidence.md) and [benchmark code](../train_distance_benchmarks.py). These statistics concern **run medians**, not individual particle landing errors, full trajectories, ignition, or operational wildfire forecasting. The repo does not retain every generated benchmark artifact in git; independently rerun before asserting a fully reproduced result.

**C. USDA reconstructed particle-level benchmark (not direct experimental measurements):** 60,370 generated particle records, with 8,841 on the held-out synthetic test partition; recorded Ridge MAE **4.689 m**, Model B **5.178 m**, Model C **6.302 m**. [Recorded metrics](../reports/usda_particle_evaluation_metrics.json) and [provenance](../data/manifests/usda_particles_manifest.json). The source archives containing original individual observations were **not obtained** in the ingestion audit. The 60,370 rows are moment-matched **simulations**, not 60,370 laboratory catch records.

**D. Model E uncertainty:** Latest [training metrics](../reports/experimental_training_metrics.json) report test MAE approximately **5.248 m** and nominal 80% interval test coverage **60.1% before CQR** and **63.4% after CQR** on reconstructed particles. This is **undercoverage**, not calibrated 80% reliability. The separate [particle evaluation file](../reports/usda_particle_evaluation_metrics.json) holds a slightly older Model E test MAE (~5.240 m): do not combine these as if identical runs.

**E. OpenPTV measured geometry:** The pipeline reports **247 detections across six consecutive frames** and **68 linked multi-frame tracks**. However, camera FPS/timestamps, individual mass and shape, ambient airflow and release conditions are not established. Therefore a defensible **zero-shot experimental 3D MAE is unavailable**. [Acquisition status](../reports/REAL_DATA_ACQUISITION.md), [OpenPTV report](../reports/OPENPTV_EXPERIMENTAL_VALIDATION.md). Some older files include mutually inconsistent displacement/observation-volume figures or assumed elapsed times; do **not** quote frame-rate-dependent speeds or an exact optical volume until audited.

**F. Sensitivity/ablation (computer-model effects only):** Nominal scenario gives reported local landing-distance sensitivity to height of approximately **4.015 m per meter**. The recorded mean relative landing-distance shifts are **+7.4%** with literature-informed drag/lift, **+19.46%** with added effective combustion buoyancy, and **+19.73%** with added shrinkage, relative to the numerical baseline. These are **conditional changes of the simulation**, not measured field improvements or experimental verification. [Sensitivity report](scientific_evidence.md), [ablation results](../reports/physics_ablation_results.json).

## 4. The experimental gap, and why an Australian expedition is relevant

The missing empirical quantities are time-resolved \(t,x,y,z\), verified camera timing, particle orientation/rotation, release height and velocity, turbulence and airflow, geometry/mass (ideally over time), and burning state, recorded together for comparable cases. Public data presently used cover **either aggregated landing results or short spatial traces**, not complete matched observations.

Relevant **prospective** research organizations include [CSIRO bushfire research facilities](https://www.csiro.au/en/work-with-us/use-our-labs-facilities/Bushfire-laboratory), UNSW Canberra bushfire researchers, and university/agency wildfire practitioners. [Australian research leads and sources](../docs/AUSTRALIA_RESEARCH_PLAN.md).

**No institutional hosting, meetings, facility entry, experiment, field collection, data-sharing agreement or publication is confirmed by this repository.** Model visits, interviews, optical-system demonstrations, and access to data as requests **subject to host approval**. Field sampling requires permits/landowner consent and safety clearance. A trip without new experiments can still produce an expert-interview synthesis, verified methods review, computational analysis of shareable datasets, and a responsible experiment-validation protocol.

The point of travel is **not** that computer simulations are mathematically incapable of further improvement. It is that laboratory/field observations would be needed to **test** key uncertain model assumptions against the physical world and to establish reliable generalization.

## 5. What the four-person team must supply for the final application

These are **unresolved; do not invent**:

- Actual full names, KAIST statuses, majors, prior contributions, team lead, and agreement on division of roles.
- Official program form(s), exact evaluation rubric, date/travel window, spending rules, word/page limits and application submission instructions.
- Target host contact history, invitations/availability, and agreed access conditions. A proposed facility is not a confirmed appointment.
- Realistic transportation/accommodation quotes and permitted expense categories. Any previously suggested budget is an unverified estimate; the team must calculate a new one from current rules and quotes.
- Safety/permission path for any park visit, sample collection or experiment; visas and any restrictions.
- Final expedition objectives, deliverable counts, route, accessible fallbacks and team review/approval.

This repository is a scientific handoff, not the final application or a confirmed trip schedule. The team will prepare its own itinerary and biographies from official instructions and actual agreements.

## 6. How a future writing assistant should use this repository

Read in this order:

1. **This file**, then [CLAIM_REGISTER.md](CLAIM_REGISTER.md).
2. [scientific_evidence.md](scientific_evidence.md) and [model_limitations.md](model_limitations.md).
3. Machine-readable [synthetic trajectory metrics](../reports/trajectory_evaluation_metrics.json), [reconstructed-particle metrics](../reports/usda_particle_evaluation_metrics.json), [Model E metrics](../reports/experimental_training_metrics.json), and [ablation results](../reports/physics_ablation_results.json).
4. [raw-source audit](../reports/REAL_DATA_ACQUISITION.md), [source registry](../data/manifests/external_validation_sources.json), and [code](../src/save_the_earth/).
5. [Australian research leads](../docs/AUSTRALIA_RESEARCH_PLAN.md), followed by an itinerary developed by the team from the official KAIST application rules.

Use the repository as **supporting technical evidence**, not as proof that all proposed visits, citations, statistics, or biographies have been independently verified. Where a narrative conflicts with a machine-readable result or the actual implementation, prefer the latter and flag the discrepancy. Keep evidence, assumption, hypothesis, and proposed activity clearly separated.

### Suggested one-sentence pitch (editable, not an official claim)

> Our KAIST team has built a reproducible physics-and-ML firebrand transport prototype and used it to identify assumptions that cannot yet be checked against accessible time-resolved experimental observations; an Australian research expedition would allow us to engage with leading bushfire researchers, examine validated measurement methods, and define the next scientifically defensible validation study.

## 7. Figures for the application-writing team

Use the repository's **actual generated research graphics**, not decorative stock illustrations. Proposed starting points:

| Figure | Source | Essential caption caveat |
| --- | --- | --- |
| Physical-mechanism ablation | [physics_ablation_comparison.png](../reports/figures/physics_ablation_comparison.png) | Numerical differences between *assumed mechanisms*; not experimental validation |
| Flight sensitivity | [physics_sensitivity.png](../reports/figures/physics_sensitivity.png) | Parameter sensitivity in declared numerical scenarios |
| Particle-model error comparison | [usda_model_comparison_particle_mae.png](../reports/figures/usda_model_comparison_particle_mae.png) | Against **moment-matched reconstructed synthetic particles**, not original USDA catch logs |
| By-wind error comparison | [usda_error_by_wind_speed.png](../reports/figures/usda_error_by_wind_speed.png) | Same reconstructed-particle benchmark; not a real-world 3D test |

Verify figure labels and checkpoint versions against the associated JSON outputs before quoting precise numbers. An appropriate grant graphic should make the **unresolved measurement question** visible rather than exaggerating a synthetic accuracy score.

## 8. Workflow and provenance

- Repository tests: `python3 -m unittest discover -s tests -v`.
- Main pipeline: `python3 run_pipeline.py` (may regenerate outputs and require dependencies/raw files).
- Source bibliography: [references.bib](references.bib). **Verify each article's exact title, DOI and any numerical coefficient against the primary publication before citing in a submitted application.**
- The source of truth for the final application is **the team's decisions plus the official KAIST forms**; nothing in this handoff should be represented as prior approval from KAIST or any Australian institution.
