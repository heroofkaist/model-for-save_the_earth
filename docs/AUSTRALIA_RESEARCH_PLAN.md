# Save the Earth: Australia research and fieldwork pathway

Status: **research prospecting**, not an invitation, booked trip or approved experiment.

## Scientific hypothesis and scope

We want to predict the transport and possible landing of wildfire firebrands under measured conditions, then determine which missing processes matter for predicting real spotting. Keep three distinct questions apart:

1. **Particle aerodynamics / mechanics:** trajectory, flight time and landing position.
2. **Thermal viability:** whether a particle remains hot or burning at landing.
3. **Ignition and fire spread:** whether receptive fuels ignite and the new fire subsequently propagates.

The current Python model covers only (1), and under restrictive fixed-wind, fixed-mass and flat-ground assumptions. A supervised regression from published mass/area/distance data predicts a **different observable** (flight distance), not detailed trajectories. Models for (2) and (3) must be independently evaluated on corresponding labels.

## Priority Australian research contacts

### CSIRO National Bushfire Behaviour Research Laboratory, Canberra

The laboratory operates the Pyrotron and a 10.7 m vertical wind tunnel used to investigate untethered burning firebrands. Its 2025 research update describes a three-camera firebrand tracking system with 3D trajectories and combustion phases. The research update reports a limited evaluation set; it does **not** establish the availability of a public trajectory CSV.

- Official lab: https://www.csiro.au/en/work-with-us/use-our-labs-facilities/Bushfire-laboratory
- 2025 project update: https://research.csiro.au/pyropage/issue-42-3d-firebrand-tracking-pdf/
- Research paper: Abdelaziz & Sullivan (2025), https://doi.org/10.1071/WF25153
- Official project contact listed in update: Dr Andrew Sullivan, Andrew.Sullivan@csiro.au

**Specific request:** Are anonymized 3D coordinate/velocity traces, wind tunnel settings, particle mass/area/burn-state labels and sensor calibration metadata available under a research use agreement? Would a remote discussion or possible July 2027 supervised lab visit be useful, subject to KAIST approval and host availability?

### UNSW Canberra Bushfire / ember transport research

UNSW Canberra has investigated near-ground ember entrainment, re-lofting, forest-to-urban transport, flow through canopies and ember accumulation. These are precisely the processes missing from a simple drag-only ballistic model.

- Group: https://www.unsw.edu.au/canberra/our-research/research-centres-institutes/unsw-bushfire/our-people
- Project: https://www.unsw.edu.au/canberra/our-research/research-centres-institutes/unsw-bushfire/our-research/enhancing-resilience-to-ember-attack-on-the-urban-fringe
- Relevant staff: Dr Duncan Sutherland (https://www.unsw.edu.au/staff/duncan-sutherland), Prof Jason Sharples.
- Related benchmark publication: Wadhwani et al. (2022), https://doi.org/10.1071/WF21088 (controlled non-burning and burning firebrand experiments, comparisons against drag models).

**Specific request:** Which benchmark experimental observations and computational trajectories can be shared for academic reproduction, and how should a student team separate ballistic transport from saltation/re-entrainment?

### Natural Hazards Research Australia / Bureau of Meteorology

An Australian commissioned project coupled ember transport parameterisation with the Spark operational fire model. Its cases include Kilmore East (2009), Reedy Swamp (2018) and Timbarra (2019). These provide *potential* later historical-case validation targets, not automatically downloadable ember labels.

- Project: https://naturalhazards.com.au/research/research-projects/fire-ember-transport
- Research lead listed: Dr Jeff Kepert.
- Request: Availability/permissions for input forcing, model outputs and independently observed spotting times/locations.

## Two-week milestone

1. Verify licensing, checksums, variable definitions, group identifiers and spatial resolution of the USDA/Mendeley observations.
2. Run the held-out **distance** benchmark only when real observations have the necessary recorded features; otherwise identify unobserved covariates explicitly.
3. Reproduce one controlled benchmark paper and document which experiment variables are missing.
4. Prepare concise evidence slides: physics baseline, dataset audit, measured-data benchmark or blocker, and five unresolved questions.
5. Request a faculty scoping conversation before making any commitments to host laboratories.

## Six-week milestone

- Two distinct external measurement sources if available, with full provenance.
- Calibration and held-out test for one specific task; evaluate per-fuel and per-experiment failure.
- Uncertainty accounting for source wind, measurement intervals and pan-center landing quantization.
- Controlled cold-particle camera experiment, subject to safety approval; no unsupervised burning experiments.
- A defensible Australia visit proposal stating precise experiments, host supervision, travel funding and permission requirements.

## Fieldwork controls

Do not enter an active wildfire or a controlled-burn site without explicit permission, local professional supervision, risk assessment and institutional approvals. Contact and ask for data/meetings first. Research travel and lab collaboration require host acceptance, student eligibility, KAIST/UNSW/CSIRO approvals, funding and possibly visas. Do not claim an affiliation, invitation or approved access that has not been granted.

## Draft outreach messages — NOT SENT

**CSIRO**

Subject: Student research enquiry — 3D firebrand tracking data and possible 2027 visit

Dear Dr Sullivan,

We are a small student team at KAIST exploring a reproducible model of wildfire firebrand transport. We have implemented a basic gravity-and-drag simulator and are now prioritizing independent experimental validation. Your 2025 CSIRO update on three-camera 3D tracking of burning firebrands is particularly relevant.

Would trajectory and combustion-phase measurements, experimental settings, or recommended benchmark data be available for academic study? We would also value a short remote conversation about the most meaningful validation questions. If our early results are useful, we may later ask whether a supervised research discussion or visit in July 2027 could be feasible, subject to institutional approvals and travel arrangements.

Thank you for considering our enquiry.

Save the Earth student research team

**UNSW Canberra**

Subject: Student research enquiry — ember transport benchmarks and experimental validation

Dear Dr Sutherland,

We are a small KAIST student team developing a preliminary wildfire ember transport simulator. We are looking for benchmark measurements to evaluate ballistic trajectories and landing distributions before considering more advanced near-ground re-entrainment, canopy flows and combustion.

Your UNSW Bushfire group's work on ember storms and data-driven trajectory modeling closely matches our interests. Could you recommend shareable experimental or numerical benchmarks and advise whether an initial short discussion would be useful? We are also exploring a possible July 2027 Australian research visit, without confirmed funding or institutional approval at this stage.

Kind regards,

Save the Earth student research team

Before use, replace generic team signatures with the actual authorized signatory, verify the contact details, attach a reviewed one-page summary, and confirm project claims with all four members.
