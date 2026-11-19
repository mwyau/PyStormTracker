---
title: "PyStormTracker: A High-Performance Cyclone Tracker in Python"
tags:
  - Python
  - atmospheric science
  - cyclone tracking
  - storm tracks
  - climate
  - feature tracking
authors:
  - name: Albert M. W. Yau
    orcid: 0009-0000-7825-9627
    affiliation: "1"
    corresponding: true
  - name: Kevin I. Hodges
    orcid: 0000-0003-0894-229X
    affiliation: "2"
affiliations:
  - index: 1
    name: School of Marine and Atmospheric Sciences, Stony Brook University, Stony Brook, New York, USA
  - index: 2
    name: Department of Meteorology, University of Reading, Reading, United Kingdom
date: 15 September 2026
bibliography: paper.bib
---

# Summary

Cyclone tracking converts gridded atmospheric data into trajectories of individual weather systems. These trajectories are used to study where cyclones form, how they move and intensify, how storm tracks vary between datasets, and how circulation changes relate to weather impacts. **PyStormTracker** (PST) is an open-source Python package for cyclone detection, tracking, trajectory comparison, and storm-track analysis.

PyStormTracker provides three tracking approaches through a common interface. `SimpleTracker` follows local extrema, `HodgesTracker` implements the feature-tracking methodology developed by Hodges [@hodges1994; @hodges1995; @hodges1999], and `HealpixTracker` adapts feature tracking to an equal-area spherical grid [@gorski2005]. All three return the same `Tracks` representation, so downstream comparison and storm-track statistics do not depend on which tracker produced the trajectories. The package accepts labeled atmospheric data or files, records the tracked variable and preprocessing choices, and supports parallel execution.

# Statement of need

Objective cyclone climatologies depend on both the tracking algorithm and the field supplied to it. The IMILAST intercomparison found substantial differences among extratropical-cyclone climatologies produced by different detection and tracking methods [@neu2013]. Even within one method, the diagnostic field and spatial filtering can change the identified population. For example, filtered lower-tropospheric relative vorticity emphasizes synoptic-scale circulation differently from sea-level pressure [@hoskinsHodges2002], and cyclone-change estimates can depend on whether a large-scale pressure background is removed [@changBackground2014].

Reproducible storm-track analysis therefore requires the tracked field, filtering, feature-point definition, trajectory construction, treatment of missing points, and post-tracking filters to be specified together. PyStormTracker addresses this requirement within the Python scientific ecosystem. Tracking configuration and preprocessing are retained with the output, while trajectory comparison and Eulerian and Lagrangian statistics use a common data model. The package includes cyclone and track frequency, amplitude-based statistics, Accumulated Cyclone Activity (ACA), Accumulated Track Activity (ATA), and the CORMAX and cross-validated EOF--CCA methods used to relate storm-track variability to weather impacts [@yauChang2020].

The target users are atmospheric scientists analysing reanalyses, climate-model simulations, numerical-weather-prediction output, and other gridded atmospheric datasets. The computational requirement increases as analyses move from coarse global products to higher-resolution data and non-regular spherical grids.

# State of the field

TRACK implements the Hodges feature-tracking methodology and has been used extensively for extratropical storm-track analysis and reanalysis comparison [@hoskinsHodges2002; @hoskinsHodges2005; @hodges2003]. PyStormTracker does not treat TRACK as scientific ground truth: the published Hodges papers define the methods, while TRACK 1.5.4 provides an implementation baseline for source-parity tests. `HodgesTracker` reproduces the same methodological sequence—feature detection and refinement, trajectory construction, and trajectory optimization—within a Python/xarray workflow and the shared PyStormTracker data model.

Other open-source frameworks address related tracking problems. TempestExtremes provides configurable pointwise feature detection and tracking on structured and unstructured climate grids [@ullrichZarzycki2017]. tobac provides a Python framework for identifying, tracking, and analysing clouds and other atmospheric objects [@heikenfeld2019]. PyStormTracker has a narrower storm-track focus: its distinct contribution is a Python implementation of the Hodges methodology with TRACK 1.5.4 correspondence tests, established cyclone spectral-preprocessing configurations, and common trajectory-comparison and storm-track metrics. Contributing this functionality to a broader tracking framework would still require a separate Hodges implementation, TRACK-derived parity work, and the associated storm-track analysis model.

PyStormTracker began in 2015 as a local-extrema and nearest-neighbour tracker developed through NCAR SIParCS and presented at the 2016 AMS Annual Meeting [@yau2016]. The simple tracker was later used as an independent sensitivity test in published storm-track impact analysis [@yauChang2020]. The current package retains that lineage while adding the Hodges and HEALPix approaches under the same output and analysis interfaces.

# Software design

The central data structure is an immutable, packed `Tracks` object containing aligned arrays for trajectory identifiers, offsets, times, coordinates, and sampled variables. Individual trajectories are views of these arrays rather than persistent Python objects for every track point. This design reduces object overhead for large trajectory collections and gives all trackers the same downstream representation for serialization, comparison, variable sampling, and statistics.

Meteorological inputs are normalized as labeled xarray arrays before tracker-specific calculations [@hoyerHamman2017]. For supported rectangular Gauss--Legendre and Clenshaw--Curtis grids, PyStormTracker delegates reusable spherical-harmonic field operations to `spharmgrid` [@spharmgridZenodo]. These include spectral filtering, spectral regridding, and default wind kinematics; `ducc0` supplies the underlying spherical-harmonic transforms [@reinecke2020]. These operations first entered PyStormTracker and were later extracted into `spharmgrid` so that the atmospheric field operations could be maintained, tested, and released independently. PyStormTracker retains cyclone-specific policy, including the diagnostic field, spectral band, reconstruction grid, spatial boundary treatment, and processing metadata. Reduced-Gaussian, HEALPix, polar, regional-DCT, and explicit-`lmax` vector paths use direct `ducc0` machinery where the rectangular `spharmgrid` interface does not apply.

Preprocessing, feature detection, and refinement are independent across time steps and can execute concurrently. Trajectory optimization requires temporal context, so Hodges and HEALPix processing uses overlapping time segments that are spliced deterministically. Serial, Dask, and MPI backends preserve the same tracking configuration and canonical result definitions over the repository-tested scope. Dask supports task-based execution on a workstation or cluster, MPI provides distributed-memory execution, and native `ducc0` threads can accelerate spherical transforms. The normal in-memory workflow does not require transformed intermediate files.

# Research impact statement

PyStormTracker has public development history dating to 2015--2016 and is distributed through PyPI, conda-forge, containers, and a Zenodo software archive [@pystormtrackerZenodo]. Its original simple tracker contributed an independent tracking sensitivity test to the evaluation of storm-track metrics against precipitation and strong-wind impacts [@yauChang2020]. The current metrics modules implement the Eulerian and Lagrangian statistics, ATA, CORMAX, and cross-validated EOF--CCA analyses developed in that research program.

Repository-maintained validation compares `HodgesTracker` with TRACK 1.5.4 on a full year of 2024 ERA5 mean-sea-level pressure. For 1,464 six-hourly frames with T6--42 filtering and the common RSPLICE population, one-to-one trajectory F1 is 99.7% for F320-to-T42 reconstruction and 99.8% for F320-to-F320. Median matched-track separations are 4.1 m and 4.7 m, and 95th-percentile separations are 33.9 m and 37.9 m, respectively. These measurements quantify implementation correspondence for the stated experiments; they are not external validation of cyclone climatology.

The same controlled workload provides a performance comparison on the recorded 16-core/32-thread host. Three sequential PyStormTracker Dask repetitions, using four frame workers, four SHT threads, and four `ducc0` native threads, gave median wall times of 16.97 s for F320-to-T42 and 38.29 s for F320-to-F320. Retained five-repeat TRACK 1.5.4 medians were 59.43 s and 1997.16 s, giving descriptive TRACK/PyStormTracker ratios of 3.50 and 52.16. TRACK was not rerun for the September 2026 refresh. The comparison applies to the recorded implementations, configuration, data, and machine and does not establish asymptotic scaling. Reproduction scripts, configurations, and retained TRACK reference material are maintained in the PyStormTracker-Validation repository.

# AI usage disclosure

OpenAI GPT-5.6 models, including Luna, Terra, and Sol configurations used through ChatGPT and coding agents, assisted parts of the 2026 redevelopment. Uses included repository exploration, implementation proposals, refactoring, test scaffolding, code review, documentation editing, and manuscript drafting; the present manuscript was edited with GPT-5.6 Sol. The corresponding author defined the software scope and design decisions, selected and reviewed AI-assisted changes, and checked them against repository tests, primary literature, TRACK source comparison, and recorded parity and benchmark experiments. The authors remain responsible for the accuracy, originality, licensing, and scientific interpretation of the software and manuscript.

# Acknowledgements

The original PyStormTracker project was supported through the National Center for Atmospheric Research 2015 SIParCS program and developed with Kevin Paul and John Dennis. The storm-track analysis framework implemented by the package grew from scientific collaboration with Edmund K. M. Chang. PyStormTracker also builds on the openly available TRACK source and the broader scientific Python ecosystem.

# References
