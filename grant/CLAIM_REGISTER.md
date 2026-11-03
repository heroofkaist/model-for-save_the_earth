# CLAIM_REGISTER — evidence map and red flags for grant writing

**Purpose:** Fast, auditable handoff between the coding repository and the team's separate application-writing workflow. A green test or a number printed in a report is not, by itself, proof of an experimental claim.

Legend: **Recorded** = present in repository outputs/code; **Measured aggregate** = derived from published experimental series, not raw particle-level observations; **Synthetic** = simulation/reconstruction; **Proposed** = not booked or approved; **Needs verification** = do not assert as an established fact.

| Possible application claim | Status and exact scope | Primary repository evidence | Safe wording / issue |
| --- | --- | --- | --- |
| We developed a 3D firebrand RK4 solver | **Recorded implementation** | [aerodynamics.py](../src/save_the_earth/physics/aerodynamics.py) | A simplified particle-transport model; not full turbulence-resolving CFD or a verified 6-DOF solver |
| We generated 10,000 trajectories | **Synthetic** | [trajectory metrics](../reports/trajectory_evaluation_metrics.json) | 10,000 RK4-generated training/evaluation trajectories; 500 additional OOD |
| Model C has 1.750 m 3D MAE | **Synthetic only** | [trajectory metrics](../reports/trajectory_evaluation_metrics.json) | Error against held-out RK4-generated trajectories, *not* real firebrands |
| We benchmarked 69 USDA experiments | **Measured aggregate summaries** | [group-CV analysis](../train_distance_benchmarks.py), [reported metrics](scientific_evidence.md) | 69 run-level experimental summaries; target is run median deposition |
| Huber CV MAE is 0.995 m | **Recorded on group-CV run medians** | [scientific evidence](scientific_evidence.md) | 5-fold group CV; separately report 1.763 m held-out test MAE; not 3D accuracy |
| 60,370 USDA firebrands were directly measured in our benchmark | **FALSE as phrased** | [source manifest](../data/manifests/usda_particles_manifest.json), [generator](../src/save_the_earth/data/usda_particles.py) | These are generated Gaussian-copula records from published empirical moments; original individual particle archive not ingested |
| USDA historical and reconstructed record counts differ | **Known source discrepancy** | [source registry](../data/manifests/source_registry.json), [audit](../reports/REAL_DATA_ACQUISITION.md) | Older registry lists 50,571 structural; report reconstruction uses 51,121 structural (60,370 combined). Do not silently treat both as confirmed raw archive counts |
| Model E is experimentally trained and provides calibrated 80% intervals | **Misleading** | [latest Model E metrics](../reports/experimental_training_metrics.json) | Trained on reconstructed synthetic particle records. 80%-nominal intervals achieved only 63.4% test coverage after CQR |
| Model E held-out MAE is 5.240 m versus 5.248 m | **Version mismatch** | [particle benchmark output](../reports/usda_particle_evaluation_metrics.json) vs [later training metrics](../reports/experimental_training_metrics.json) | Different stored evaluation stages; state source and run or rerun both with the same checkpoint before comparison |
| OpenPTV validates 3D velocity / acceleration | **Unsupported** | [OpenPTV data audit](../reports/REAL_DATA_ACQUISITION.md), [compatibility JSON](../reports/openptv_validation_metrics.json) | 247 detections and 68 linked tracks; FPS and crucial boundary conditions unknown, so no physical 3D velocity MAE |
| OpenPTV frame displacement or observation volume has one settled value | **Needs verification** | [OpenPTV validation JSON](../reports/openptv_validation_metrics.json) and [reports](../reports/OPENPTV_EXPERIMENTAL_VALIDATION.md) | Older output mixes incompatible displacement / volume / assumed-time descriptions; quote only audited quantities |
| Drag/lift and combustion factors are directly calibrated for these fuels | **Literature-informed assumptions** | [aerodynamics implementation](../src/save_the_earth/physics/aerodynamics.py), [limitations](model_limitations.md) | Piecewise coefficients, effective gravity, and burnout law are simplified closures; verify precise mappings to primary papers |
| Added modeled effects extend landing range by 7–20% on average | **Computational ablation** | [ablation metrics](../reports/physics_ablation_results.json) | Scenario-dependent changes *within the simulator*, not observed physical gains |
| Release height is sensitive (4.015 m/m in a nominal case) | **Computational sensitivity** | [scientific evidence](scientific_evidence.md), [sensitivity code](../physics_sensitivity.py) | Local derivative under declared assumptions, not a universal law or proof that simulations cannot improve |
| Our neural model predicts wildfire spot ignitions | **Unsupported** | [module separation](../README.md), [limitations](model_limitations.md) | Landing and ignition are distinct physical events; an ember landing is not proof of spot-fire ignition |
| CSIRO / UNSW / fire services will host our visits | **Proposed only** | [Australian research leads](../docs/AUSTRALIA_RESEARCH_PLAN.md) (unconfirmed) | List as outreach targets; no confirmed invitation/facility access in repository |
| We will collect 200 bark samples or run 120 burning trials | **Unapproved target** | **Unapproved historical grant draft (not retained)** | Replace with conditional, permission-dependent scope or a safe methods review |
| All team members' academic majors are documented | **Unverified placeholders** | **Unapproved historical grant draft (not retained)** | Request actual members' names, majors and contributions; never infer from proposed roles |
| USD 14,950 is an approved and price-verified budget | **Draft estimate only** | **Unapproved historical grant draft (not retained)** | Recheck current official funding rules, expense eligibility, travel quotes, local taxes and FX |
| Our work establishes impact on real wildfire incident decisions | **Not yet demonstrated** | [limitations](model_limitations.md) | Present as long-term motivation and expert-interview question, not measured operational outcome |

## Before transferring a claim into the submitted application

1. Can you point to a **specific** code artifact, saved result, published measurement, or approved activity?
2. Is the outcome **run-level distance**, **synthetic individual distance**, **3D geometry**, **3D dynamics**, **thermal survival**, or **ignition**? Never switch targets without stating so.
3. If an error is reported, what were the **label provenance, split, sample count, units, baseline, and checkpoint version**?
4. If it concerns a named outside laboratory, are the **visit, equipment, data and permissions** actually confirmed?
5. Does it require an original external article, official grant instructions, a new dataset, or actual team bios? Mark **needs verification** rather than invent.
6. Are references clickable inside GitHub? Avoid `file:///Users/...` links from local generated reports; use relative repo paths in the final narrative.

**This register does not claim an independent rerun of the research.** It audits and distinguishes what the latest checked-in sources support.
