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
date: 25 August 2026
bibliography: paper.bib
---

# Summary

Storm-track activity is commonly described using Eulerian statistics of synoptic-scale variability and Lagrangian statistics derived from tracked weather systems [@chang2012; @hoskinsHodges2002]. The objective feature-tracking methodology developed by Hodges provides a general framework for identifying atmospheric features and constructing trajectories on the sphere, and has been used in a wide range of atmospheric applications [@hodges1994; @hodges1995; @hodges1999; @hodges2003]. **PyStormTracker** (PST) is an open-source Python package for feature detection, cyclone tracking, trajectory comparison, and storm-track analysis.

PyStormTracker integrates these operations with the Python scientific ecosystem. Meteorological data are represented with `xarray`; `ducc0` performs spherical-harmonic transforms; NumPy and Numba provide array computation; SciPy provides interpolation and spline routines; and Dask or MPI provide parallel execution. The public tracking API exposes `SimpleTracker`, `HodgesTracker`, and `HealpixTracker`. Each tracker's `track()` method returns the same packed `Tracks` representation for trajectory comparison, variable sampling, serialization, and Eulerian and Lagrangian storm-track statistics. The three trackers implement, respectively, local-extrema tracking descended from the original PyStormTracker implementation, the Hodges feature-tracking methodology, and a HEALPix-topology variant for equal-area spherical grids [@gorski2005].

# Statement of need

Objective cyclone detection and tracking presents a reproducibility and comparison problem. The IMILAST project compared multiple extratropical cyclone detection and tracking methods and found substantial differences among resulting cyclone climatologies [@neu2013]. Intercomparison is complicated when methods use different preprocessing, executables, coordinate conventions, output formats, and postprocessing. Reproducing an established method requires the diagnostic field, spatial filtering, feature refinement, trajectory linking, treatment of missing points, and track filtering to be specified. The tracked field and filtering are part of the scientific definition of a tracking experiment rather than interchangeable implementation details [@hoskinsHodges2002].

The computational setting has also changed. NCEP/NCAR Reanalysis used a T62 atmospheric model (about 210 km), and widely used pressure-level products are provided on 2.5-degree grids [@kalnay1996; @noaaNcepReanalysis]. ERA-Interim increased the native atmospheric resolution to T255/N128 (about 80 km), while ERA5 HRES uses T639/N320 (about 31 km) [@eraInterimGrid; @era5Grid]. Recent machine-learning weather systems have also explored native spherical meshes such as HEALPix [@karlbauer2024]. Higher-resolution and non-regular grids increase the computational cost of spectral preprocessing, data movement, and intermediate-file I/O in tracking workflows.

PyStormTracker supports atmospheric scientists working with reanalysis, climate-model, and numerical-weather-prediction output. `SimpleTracker`, `HodgesTracker`, and `HealpixTracker` accept file paths or xarray objects and return `Tracks`. Tracking variables and filtering choices are configurable and are recorded in processing metadata. The `pystormtracker.metrics` modules provide Eulerian and Lagrangian storm-track statistics, including cyclone frequency, track frequency, cyclone amplitude, Accumulated Cyclone Activity (ACA), and Accumulated Track Activity (ATA); `compute_cormax`, `find_best_cca_truncation`, and `train_cca_model` provide CORMAX and EOF--CCA analyses.

# State of the field

Hodges developed a general objective feature-tracking methodology during the 1990s, including feature-point identification, trajectory construction on the sphere, and constrained optimization of the resulting tracks [@hodges1994; @hodges1995; @hodges1999]. The resulting TRACK system has since been used in many atmospheric applications, including studies of Northern and Southern Hemisphere storm tracks and comparisons of reanalysis datasets [@hoskinsHodges2002; @hoskinsHodges2005; @hodges2003]. Hoskins and Hodges [-@hoskinsHodges2002] compared tracking based on multiple meteorological fields and found lower-tropospheric relative vorticity particularly useful for synoptic systems because it is less dominated by the large-scale background, emphasizes smaller-scale features, and can identify developing systems earlier in their life cycle. They also noted that high-resolution vorticity can contain substantial small-scale structure, making spatial filtering or smoothing an important part of the tracking definition.

Published storm-track studies show that cyclone statistics depend on the diagnostic field and preprocessing. Chang [-@changBackground2014] showed, using a single tracking algorithm on 23 CMIP5 simulations, that projected Pacific winter cyclone changes can reverse sign depending on whether cyclones are defined from total sea-level pressure or from perturbations after removing a large-scale, low-frequency background. A later study compared sea-level-pressure-anomaly tracks, T5--T42-filtered 850-hPa relative-vorticity tracks, and 24-h pressure-change variance when assessing cyclone change and temperature impacts [@changEtAl2016]. PyStormTracker therefore records the tracked variable and preprocessing operations with the resulting tracks.

Yau and Chang [-@yauChang2020] compared storm-track metrics against precipitation and strong-wind impacts using CORMAX and cross-validated EOF--CCA frameworks. They introduced ATA, combining cyclone track frequency and amplitude; among the Lagrangian definitions evaluated over Europe, ATA based on spatially filtered 850-hPa relative-vorticity maxima performed best for both impacts. The original PyStormTracker simple tracker was also used as an independent sensitivity test in that evaluation.

The PyStormTracker software effort began in 2015 with a local-extrema and nearest-neighbor tracker, developed during NCAR SIParCS and presented at the 2016 AMS Annual Meeting [@yau2016]. That implementation separated grid handling, detection, linking, and MPI execution. `mpi4py` distributed time ranges among ranks and combined partial track sets through a tree reduction. The 2016 presentation listed comparison with established cyclone trackers, including the Hodges methodology, as future validation; it did not establish TRACK parity.

`HodgesTracker` now implements the Hodges feature-tracking methodology in PyStormTracker, including feature detection and refinement, trajectory linking, and trajectory optimization. The implementation was developed in collaboration with Hodges. The published Hodges papers define the scientific methods [@hodges1994; @hodges1995; @hodges1999]; TRACK 1.5.4 is the implementation baseline used for source-parity work; and the repository-maintained validation benchmark measures PyStormTracker--TRACK correspondence. `HodgesTracker` supports mean sea-level pressure and relative-vorticity tracking, including the 850-hPa vorticity configuration used in TRACK studies. Its outputs use the same `Tracks` model, comparison functions, and storm-track metrics as `SimpleTracker` and `HealpixTracker`.

# Software design

`SimpleTracker`, `HodgesTracker`, and `HealpixTracker` return immutable packed `Tracks` objects containing aligned NumPy arrays for trajectory identifiers, offsets, times, coordinates, and variables. Individual `Track` objects are views over these arrays. `Tracks.write()`, `load_tracks`, `save_tracks`, `sample_tracks`, `compare_tracks`, and the functions in `pystormtracker.metrics` operate on this representation without persistent object-per-point storage.

Meteorological data handling uses `xarray` [@hoyerHamman2017]. Tracker `track()` methods normalize file paths and xarray objects to labeled arrays before tracker-specific calculations. `pystormtracker.preprocessing` provides spectral filtering, tapering, regridding, and spherical wind diagnostics, and processing metadata records the operations and parameters that occurred. A T5--T42-filtered 850-hPa vorticity field and unfiltered mean sea-level pressure define different feature populations. In-memory inputs pass through preprocessing, detection, and refinement without transformed intermediate files. Dask-backed inputs keep preprocessing lazy and materialize complete spatial frames as their tasks execute; full-resolution fields are released before trajectory linking.

For spherical-harmonic preprocessing, `SHTFilter` and `SpectralRegridder` call `ducc0` transforms and implement grid geometry, analysis and synthesis, spectral tapering and truncation, and regridding [@reinecke2020]. PyStormTracker does not require NCL, SPHEREPACK, or `pyspharm` at runtime. NCL has been in maintenance mode since 2019, SPHEREPACK is among NCAR's classic Fortran libraries that are no longer under development, and the original `pyspharm` package's latest PyPI release is a source distribution from 2020 [@nclMaintenance; @spherepackClassic; @pyspharm2020]. NumPy provides the array representation, Numba compiles feature-detection, geometry, cost, and Modified Greedy Exchange hot loops, and SciPy supplies interpolation and spline routines [@virtanen2020].

Preprocessing, detection, and feature refinement can execute concurrently across independent time steps. Hodges and HEALPix trajectory optimization uses overlapping temporal segments that are spliced deterministically. Serial, Dask, and MPI execution use the same tracking configuration and canonical result definitions for the repository-tested scope. Dask schedules frame and segment tasks on a workstation or local cluster, `mpi4py` provides distributed-memory execution, and `ducc0` can use native threads within spherical transforms.

A controlled full-year 2024 ERA5 benchmark measures high-resolution execution. With source frames, T6--42 filtering, and tracking configuration held constant, changing reconstruction from T42 to F320 increased TRACK 1.5.4 wall time from 59.43 s to 1997.16 s (33.6 times), while PyStormTracker increased from 27.48 s to 57.38 s (2.09 times). These measurements apply to the recorded implementations, workload, and machine; they do not establish asymptotic scaling of the Hodges algorithm.

# Research impact statement

PyStormTracker has a public history dating to 2015--2016 and is distributed through PyPI, conda-forge, containers, and a Zenodo software archive [@pystormtrackerZenodo]. `pystormtracker.metrics.eulerian`, `pystormtracker.metrics.lagrangian`, and `pystormtracker.metrics.cross_validation` implement the Eulerian and Lagrangian statistics, ATA, CORMAX, and cross-validated EOF--CCA analyses described above. The original simple tracker has also been used in published sensitivity analysis [@yauChang2020].

Repository-maintained validation compares `HodgesTracker` with TRACK 1.5.4. In a 1,464-frame 2024 ERA5 mean-sea-level-pressure comparison using T6--42 filtering and the common RSPLICE population, the full-year one-to-one trajectory F1 is 99.7% for both F320-to-T42 and F320-to-F320 output grids, with median matched-point separations of 4.1 m and 4.9 m, respectively. On the recorded 16-core benchmark host, PyStormTracker was 2.16 times faster than TRACK for the full-year F320-to-T42 case and 34.81 times faster for F320-to-F320. Source-derived configurations and raw comparison records are maintained separately in the PyStormTracker-Validation repository. These results measure implementation correspondence and performance for the stated cases; they are not external validation of cyclone climatology.

# AI usage disclosure

OpenAI generative-AI tools from the GPT-5.6 family, including Luna, Terra, and Sol configurations used through ChatGPT and Codex-style coding agents, assisted parts of the 2026 redevelopment. Assistance included repository exploration, implementation proposals, refactoring, test scaffolding, code review, documentation editing, and manuscript drafting; the present draft was prepared with GPT-5.6 Sol. The corresponding author defined the software scope and design decisions, selected which suggestions to accept, reviewed and edited AI-assisted changes, and checked them with repository tests, primary literature, TRACK source comparison, and recorded parity and benchmark experiments. The authors remain responsible for the accuracy, originality, licensing, and scientific interpretation of the submitted material.

# Acknowledgements

The original PyStormTracker project was supported through the National Center for Atmospheric Research 2015 SIParCS program and developed with Kevin Paul and John Dennis. The storm-track analysis framework implemented by the package grew from scientific collaboration with Edmund K. M. Chang. PyStormTracker also builds on the openly available TRACK source and the broader scientific Python ecosystem.

# References
