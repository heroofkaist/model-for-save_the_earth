# KAIST Save the Earth Global Challenge — Research Expedition Plan

**Project Title:** Multi-Scale Physics-Informed Learning for Wildfire Spotting and Firebrand Dispersion  
**Applying Team:** Four KAIST students; names, majors and actual responsibilities to be confirmed by team  
**Destination:** Australia (Canberra, Sydney, Melbourne)  
**Proposed Duration:** 21 days (3 weeks; travel window subject to official program and team confirmation)  
**Indicative Budget:** USD 14,950 (planning estimate, not verified quotes; confirm official funding rules)  
**Repository & Codebase:** [heroofkaist/model-for-save_the_earth](https://github.com/heroofkaist/model-for-save_the_earth)  

---

> **WORKING EXPEDITION SCENARIO — NOT A BOOKED ITINERARY OR SUBMITTED FORM.** All meetings, field visits, sample counts, team biographies, equipment access and travel prices are tentative. Consult [AI_CONTEXT.md](AI_CONTEXT.md) and [CLAIM_REGISTER.md](CLAIM_REGISTER.md). An Australian organization appears here as a possible contact, **not a confirmed host**.

## 1. Executive Summary & Central Grant Argument

Spotting by windborne burning firebrands (embers) is the dominant mechanism of catastrophic wildfire spread and structure destruction. Over the past several months, the project has developed a computational prototype for firebrand transport and deposition:
- Built a 3D coupled aerodynamic numerical simulator (`src/save_the_earth/physics/aerodynamics.py`).
- Trained neural surrogates (`Model B`, `Model C`) and deep quantile networks (`Model E`).
- Established group-validated benchmarks across 69 full-scale USDA burn tests (**0.995 m MAE** on operational run medians).
- Performed rigorous sensitivity and physical ablation analyses.

Our sensitivity analysis shows that **computed flight distances can depend strongly on unmeasured physical initial conditions**—including the chosen release-height scenario ($\partial x / \partial h = +4.015$ m/m, shifting landing distance by $\pm 70\%$) and the unmeasured 6-DOF tumbling lift and combustion micro-buoyancy of natural bark.

**We are not asking for funding to start a project from scratch.** We have already built the computational models, identified their precise physical limitations, and designed a focused, conditional three-week research expedition to Australia—the global epicenter of eucalyptus firebrand spotting—to bridge our computational framework with world-class experimental and field practices.

---

## 2. Why Australia? The World's Premier Wildfire Spotting Laboratory

Australia's native eucalyptus forests present the most extreme spotting phenomena observed on Earth:
1. **Unique Fuel Morphologies:** Eucalyptus stringybark (*Eucalyptus obliqua*) and candlebark/ribbon bark (*Eucalyptus viminalis*) detach as high-aspect-ratio aerodynamic strips that autorotate, generating significant lift and traveling 5 to 30+ km ahead of flame fronts.
2. **Pioneering Research Infrastructure:** Australian institutions—notably CSIRO, the University of New South Wales (UNSW Canberra Bushfire Dynamics Group), and the University of Melbourne—have developed specialized vertical wind tunnels, photogrammetric tracking methods, and operational fire spread systems.

---

## 3. Team Composition & Individual Roles

The following are **proposed workstreams**, not verified academic majors or assignments. The actual four team members must approve ownership and provide truthful biographies:

| Workstream (proposed) | Team member / major | Expected responsibility, subject to team approval |
| --- | --- | --- |
| Research coordination and physical modeling | **TBD by team** | Prepare questions on drag, lift, initial conditions; coordinate outreach with consent |
| Experimental measurement and data quality | **TBD by team** | Review optical tracking, timing, calibration, airflow and documentation methods |
| Statistical validation and reproducibility | **TBD by team** | Assess usable datasets, grouped evaluation, distributional shift and limitations |
| Wildfire impacts, field methods and synthesis | **TBD by team** | Connect laboratory work to practical wildfire resilience and environmental context |

---

## 4. Prospective Host Institutions & Target Collaborators

*Note on Institutional Engagement:* Access to facilities and individual consultations will be conducted through formal academic seminar requests and professional liaison. The individuals and institutions below represent our primary prospective targets:

1. **CSIRO Environment / Bushfire Dynamics Group (Black Mountain, Canberra):**
   - *Target Researchers:* Dr. Andrew L. Sullivan, Dr. Yang Zhang (authors of foundational firebrand transport and vertical wind tunnel research).
   - *Objective:* Technical consultation on vertical wind tunnel (VWT) design, airflow conditioning, and methods for observing untethered burning embers.
2. **UNSW Canberra — Bushfire Dynamics Group (ADFA Campus, Canberra):**
   - *Target Researchers:* Prof. Jason Sharples, Dr. Duncan Sutherland.
   - *Objective:* Consult on extreme wildfire behavior, vorticity-driven lateral fire spread, and pyroconvective ember lofting.
3. **University of Melbourne — School of Agriculture, Food & Ecosystem Sciences (Creswick / Parkville):**
   - *Target Researchers:* Forest fire behavior and fuel management faculty.
   - *Objective:* Investigate bark shedding biology, fuel bed sampling methods, and bark physical property databases.
4. **ACT Rural Fire Service (Canberra) & NSW Rural Fire Service (Sydney):**
   - *Target Contacts:* Operational Fire Behavior Analysts (FBANs).
   - *Objective:* Structured interviews on how spotting distances are utilized in operational incident command, identifying operational constraints.

---

## 5. Detailed 21-Day Expedition Itinerary

```
WEEK 1: CANBERRA (Science Infrastructure & Laboratory Methods)
  Day 1: Departure from Incheon (ICN) -> Transit -> Arrival in Sydney -> Coach to Canberra.
  Day 2: Team orientation, local logistics setup, review of computational presentation materials.
  Day 3: UNSW Canberra (ADFA): Academic seminar presentation by KAIST team; discussion with Bushfire Dynamics Group on vorticity and plume lofting.
  Day 4: CSIRO Black Mountain: Prospective visit and technical consultation with fire scientists on vertical wind tunnel instrumentation and stereo tracking.
  Day 5: CSIRO / ACT Emergency Services: Technical meeting on operational firebrand sampling and collection pan methodologies.
  Day 6: Technical documentation session; synthesis of Week 1 laboratory observations into codebase specifications.
  Day 7: Rest and team review; consolidation of photographic calibration notes.

WEEK 2: REGIONAL FIELD CHARACTERIZATION & SYDNEY (Fuel Sampling & Ecology)
  Day 8: Field expedition to Brindabella State Conservation Area (ACT/NSW border): Non-destructive morphological survey and physical sampling of naturally shed eucalyptus stringybark and ribbon bark.
  Day 9: Field sampling in Namadgi National Park: Specimen thickness, area, mass, and curvature measurements using digital micrometers and portable optical scanners.
  Day 10: Travel from Canberra to Blue Mountains (Katoomba / NSW): Field observation of historical high-intensity spotting terrain and post-fire canopy recovery.
  Day 11: Travel to Sydney: Processing and cataloging of harvested bark specimens (measuring aspect ratios and dry bulk density).
  Day 12: NSW Rural Fire Service (Sydney HQ): Structured interview with operational Fire Behavior Analysts on practical requirements for spot-fire warning tools.
  Day 13: University of Sydney / Western Sydney University: Prospective consultation with fire safety and fluid mechanics researchers.
  Day 14: Mid-expedition review and synthesis; overnight train / transit to Melbourne.

WEEK 3: MELBOURNE & TECHNICAL SYNTHESIS (Forestry & Experimental Roadmap)
  Day 15: Arrival in Melbourne; setup at university lodging; preparation for forestry research meetings.
  Day 16: University of Melbourne (Parkville Campus): Consultation with fire ecology researchers on bark flammability and shedding dynamics.
  Day 17: University of Melbourne (Creswick Forestry Campus): Inspection of forest fuel testing facilities; demonstration of our surrogate model.
  Day 18: Country Fire Authority (CFA Victoria HQ): Meeting on operational predictive services and historical Victoria spotting records (Black Saturday archives).
  Day 19: Full-day team integration workshop: Synthesizing experimental findings, updating mathematical formulations, and drafting the experimental validation blueprint.
  Day 20: Final presentation rehearsal; transit to Melbourne Airport (MEL).
  Day 21: Return flight: Melbourne (MEL) -> Transit -> Arrival at Incheon (ICN) -> Return to KAIST (Daejeon).
```

---

## 6. Realistic Budget Justification (Strictly Within $15,000 USD Limit)

This **illustrative budget** adds to USD 14,950 using assumptions rather than live flight or accommodation quotations. Before the official submission, the team must verify the funding ceiling, eligible spending rules, student entry requirements, exchange rates and actual prices:

| Category | Detailed Itemization | Cost (USD) |
|---|---|---:|
| **Roundtrip International Airfare** | Incheon (ICN) ⇄ Sydney/Melbourne (4 students × $1,450 economy) | $5,800 |
| **Accommodation** | 20 nights across Canberra, Blue Mountains, and Melbourne (4 students sharing modest twin/quad university dorms or serviced self-catering apartments: ~$185/night total) | $3,700 |
| **Daily Subsistence & Meals** | 4 students × 21 days × $30/day (budget self-catering and university dining) | $2,520 |
| **Domestic Intercity & Local Transit** | Regional coaches, intercity rail (Canberra-Sydney-Melbourne), airport transfers, local transit cards | $1,380 |
| **Field Sampling & Safety Tools** | Precision digital micrometers, portable flatbed scanner, sample bags, field PPE, laser rangefinder | $450 |
| **Travel Insurance & Visa Costs** | Australian Subclass 600 / ETA visas and student medical travel insurance (4 × $190) | $760 |
| **Contingency / Emergency Buffer** | Reserved for unforeseen transit adjustments or logistics | $340 |
| **TOTAL EXPEDITION BUDGET** | **Illustrative total; official program conditions must be verified** | **$14,950** |

---

## 7. Achievable, Concrete Deliverables

Rather than promising multi-year laboratory campaigns, our expedition delivers concrete, immediately verifiable outputs:

1. **Experimental Validation Blueprint & Protocol Whitepaper:**  
   A rigorous, 30-page engineering specification detailing exact stereoscopic camera placement, minimum frame rates ($\ge 500$ FPS), illumination wavelengths, and calibration dot-grid standards required to measure 6-DOF untethered firebrand trajectories without unverified timestamps.
2. **The Australian Wildland Firebrand Morphology Atlas:**  
   An open-access empirical dataset containing calibrated physical dimensions (projected area, mass, thickness, aspect ratio $\lambda$, dry density) for 200+ measured field specimens across 3 dominant Australian fuel families (*E. obliqua*, *E. viminalis*, *P. radiata*).
3. **Formal Collaborative Data-Sharing Agreements:**  
   Initiating institutional academic requests to access raw experimental archives from past CSIRO and university wind-tunnel trials.
4. **Model Architecture Upgrade Plan:**  
   Concrete engineering roadmap incorporating measured Australian bark morphological priors into our prospective Huber Regressor and Model E surrogates.

---

## 8. Risk Management & Backup Contingency Plans

| Risk Scenario | Probability | Impact | Mitigation & Backup Strategy |
|---|---|---|---|
| **Laboratory Access Restriction:** Security or facility maintenance prevents entry to specialized wind tunnels. | Medium | Moderate | Pre-schedule academic seminar exchanges at UNSW Canberra and University of Melbourne. University visits are likewise subject to permission; remote technical discussions and published method reviews are the fallback. |
| **Target Personnel Unavailable:** Key researchers deployed to active field monitoring or remote conferences. | Medium | Low | We have identified potential research contacts in Canberra, Sydney, and Melbourne, but confirmed meetings and access are still required. Secondary contacts (research fellows, lab managers, and senior doctoral candidates) are included in all meeting requests. |
| **Extreme Weather / Bushfire Restrictions:** Active fire bans restrict entry into specific forest parks during Week 2. | Low | Low | No material collection in protected areas or botanical gardens without explicit written permission. Desk-based morphology studies and observation-only visits are the fallback. |
| **Budget Fluctuation:** Foreign exchange rate shifts between USD and AUD. | Low | Low | Accommodations and intercity rail will be pre-booked upon grant award; a $340 cash contingency is built into the budget. |

---

## 9. Conclusion: Preparedness and Vision

Our KAIST team has demonstrated that we can independently conceive, build, and evaluate complex scientific software. We have already reached the boundary of what can be learned from synthetic physics simulations and indirect deposition tables. This expedition will provide the missing physical reality, establishing KAIST students as serious contributors to global wildfire resilience.
