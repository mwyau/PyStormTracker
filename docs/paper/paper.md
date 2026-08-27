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

PyStormTracker brings these tasks into the modern Python scientific ecosystem. It uses `xarray` for labeled meteorological data, `ducc0` for high-performance spherical numerical operations, NumPy and Numba for array-based computation, SciPy for established interpolation routines, and Dask or MPI for parallel execution. Three tracking families share the same packed trajectory representation: a lightweight local-extrema tracker descended from the original PyStormTracker implementation, a TRACK-compatible implementation of the Hodges feature-tracking methodology, and a HEALPix-topology variant for equal-area spherical grids [@gorski2005]. The same output model is used for trajectory comparison, variable sampling, serialization, and Eulerian and Lagrangian storm-track statistics.

# Statement of need

Objective cyclone detection and tracking remains a difficult reproducibility and comparison problem. The IMILAST project compared multiple extratropical cyclone detection and tracking methods and found substantial differences among resulting cyclone climatologies [@neu2013]. Intercomparison is scientifically valuable, but difficult when methods use different preprocessing, executables, coordinate conventions, output formats, and postprocessing. Reproducing an established method also requires implementation details such as the diagnostic field, spatial filtering, feature refinement, trajectory linking, treatment of missing points, and track filtering to be explicit. The tracked field and filtering are part of the scientific definition of a tracking experiment rather than interchangeable implementation details [@hoskinsHodges2002].

The computational setting has also changed. NCEP/NCAR Reanalysis used a T62 atmospheric model (about 210 km), and widely used pressure-level products are provided on 2.5-degree grids [@kalnay1996; @noaaNcepReanalysis]. ERA-Interim increased the native atmospheric resolution to T255/N128 (about 80 km), while ERA5 HRES uses T639/N320 (about 31 km) [@eraInterimGrid; @era5Grid]. Recent machine-learning weather systems have also explored native spherical meshes such as HEALPix [@karlbauer2024]. Higher-resolution and non-regular grids make spectral preprocessing, data movement, and intermediate-file I/O increasingly important parts of a tracking workflow.

PyStormTracker is intended for atmospheric scientists working with reanalysis, climate-model, and numerical-weather-prediction output who need reproducible cyclone tracking and storm-track diagnostics. It provides a common Python framework in which tracking algorithms share data access, preprocessing, trajectory representation, comparison tools, parallel execution, and downstream analysis. Tracking variables and filtering choices remain configurable and are recorded with the processing metadata rather than hidden in a separate preprocessing workflow. The package includes cyclone frequency, track frequency, cyclone amplitude, Accumulated Cyclone Activity (ACA), Accumulated Track Activity (ATA), Eulerian storm-track statistics, CORMAX, and EOF--CCA analysis.

# State of the field

Hodges developed a general objective feature-tracking methodology during the 1990s, including feature-point identification, trajectory construction on the sphere, and constrained optimization of the resulting tracks [@hodges1994; @hodges1995; @hodges1999]. The resulting TRACK system has since been used in many atmospheric applications, including studies of Northern and Southern Hemisphere storm tracks and comparisons of reanalysis datasets [@hoskinsHodges2002; @hoskinsHodges2005; @hodges2003]. Hoskins and Hodges [-@hoskinsHodges2002] compared tracking based on multiple meteorological fields and found lower-tropospheric relative vorticity particularly useful for synoptic systems because it is less dominated by the large-scale background, emphasizes smaller-scale features, and can identify developing systems earlier in their life cycle. They also noted that high-resolution vorticity can contain substantial small-scale structure, making spatial filtering or smoothing an important part of the tracking definition.

The scientific motivation underlying PyStormTracker predates the software. Chang [-@changBackground2014] showed, using a single tracking algorithm on 23 CMIP5 simulations, that projected Pacific winter cyclone changes can reverse sign depending on whether cyclones are defined from total sea-level pressure or from perturbations after removing a large-scale, low-frequency background. A later study compared sea-level-pressure-anomaly tracks, T5--T42-filtered 850-hPa relative-vorticity tracks, and 24-h pressure-change variance when assessing cyclone change and temperature impacts [@changEtAl2016]. These results motivate making background removal and filtering, the tracked variable, dataset provenance, and metric definition explicit.

Yau and Chang [-@yauChang2020] formalized the comparison of storm-track metrics against precipitation and strong-wind impacts using CORMAX and cross-validated EOF--CCA frameworks. They introduced ATA, combining cyclone track frequency and amplitude; among the Lagrangian definitions evaluated over Europe, ATA based on spatially filtered 850-hPa relative-vorticity maxima performed best for both impacts. The original PyStormTracker simple tracker was also used as an independent sensitivity test in that evaluation.

The PyStormTracker software effort began in 2015 with a simpler local-extrema and nearest-neighbor tracker, developed during NCAR SIParCS and presented at the 2016 AMS Annual Meeting [@yau2016]. It was designed for scalability and extensibility, separating grid handling, detection, linking, and parallel execution. `mpi4py` distributed time ranges among ranks and combined partial track sets through a tree reduction, while comparison with the Hodges methodology was identified as a validation objective.

The present PyStormTracker development is being carried out in collaboration with Hodges to implement his tracking algorithms in the Python framework and validate their correspondence with TRACK. The tracking field is not restricted to mean sea-level pressure; the current implementation also supports relative-vorticity tracking, including the 850-hPa vorticity configuration commonly used in TRACK studies. Implementing the method in PyStormTracker allows the TRACK-compatible workflow to share xarray-based data handling, Dask/MPI execution, the common `Tracks` representation, comparison tools, and storm-track statistics with alternative algorithms. TRACK 1.5.4 provides the implementation reference for parity testing, while PyStormTracker provides a Python framework for using, comparing, and extending these methods.

# Software design

PyStormTracker separates **scientific algorithms**, **data representation**, and **execution policy**. Every tracker returns an immutable packed `Tracks` object containing aligned NumPy arrays for trajectory identifiers, offsets, times, coordinates, and variables. Individual tracks are lightweight views over these arrays. This common representation is the boundary used by serialization, trajectory comparison, variable sampling, and storm-track metrics.

Meteorological data handling is built around `xarray` [@hoyerHamman2017]. File paths and xarray objects are normalized to labeled arrays before tracker-specific calculations. Preprocessing includes spectral filtering, tapering, regridding, and projection, with processing metadata recording the operations and parameters that actually occurred. This is scientifically consequential for feature tracking: for example, a T5--T42-filtered 850-hPa vorticity field defines a different feature population from unfiltered mean sea-level pressure. The data path is designed to minimize transformed intermediate files: in-memory inputs pass directly through preprocessing, detection, and refinement, while Dask-backed inputs keep preprocessing lazy and materialize complete spatial frames only as their tasks execute. Full-resolution fields are released before trajectory linking.

For spherical-harmonic preprocessing, PyStormTracker calls `ducc0` directly and implements the meteorological layer around its lower-level transforms: grid geometry, analysis and synthesis, spectral tapering and truncation, and regridding [@reinecke2020]. This avoids depending on older atmospheric SHT stacks. NCL has been in maintenance mode since 2019, SPHEREPACK is among NCAR's classic Fortran libraries that are no longer under development, and the original `pyspharm` package's latest PyPI release is a source distribution from 2020 [@nclMaintenance; @spherepackClassic; @pyspharm2020]. NumPy provides the core array representation, Numba compiles feature-detection, geometry, cost, and Modified Greedy Exchange hot loops, and SciPy supplies established interpolation and spline routines [@virtanen2020].

Parallel execution follows the scientific structure of the calculation. Independent time steps can be preprocessed, detected, and refined concurrently. Hodges and HEALPix trajectory optimization runs on overlapping temporal segments that are then spliced deterministically. The execution policy changes how independent work is scheduled without changing the tracking configuration or scientific definitions. Dask provides task-based execution on a multicore workstation or local cluster, `mpi4py` remains available for distributed-memory execution, and `ducc0` can use native threads within spherical transforms.

The same design improves high-resolution execution. In the controlled full-year 2024 ERA5 comparison, with source frames, T6--42 filtering, and tracking configuration held constant, changing reconstruction from T42 to F320 increased TRACK 1.5.4 wall time from 59.43 s to 1997.16 s (33.6 times), while PyStormTracker increased from 27.48 s to 57.38 s (2.09 times). These measurements are implementation- and machine-specific rather than an asymptotic claim about the Hodges algorithm; they show how the transformation and data path can dominate a native-resolution workflow.

# Research impact statement

PyStormTracker has a public history dating to 2015--2016 and is now a tested, documented Python package distributed through PyPI, conda-forge, containers, and Zenodo [@pystormtrackerZenodo]. Its analysis layer implements the same classes of diagnostics motivated above: Eulerian and Lagrangian storm-track statistics, ATA, CORMAX, and cross-validated EOF--CCA analysis. The original simple tracker has also been used in published sensitivity analysis [@yauChang2020].

The current implementation has quantitative comparison evidence against TRACK 1.5.4. In a 1,464-frame 2024 ERA5 mean-sea-level-pressure comparison using T6--42 filtering and the common RSPLICE population, the full-year one-to-one trajectory F1 is 99.7% for both F320-to-T42 and F320-to-F320 output grids, with median matched-point separations of 4.1 m and 4.9 m, respectively. On the recorded 16-core benchmark host, PyStormTracker was 2.16 times faster than TRACK for the full-year F320-to-T42 case and 34.81 times faster for F320-to-F320. Source-derived configurations and raw comparison records are maintained separately in the PyStormTracker-Validation repository.

# AI usage disclosure

OpenAI generative-AI tools from the GPT-5.6 family, including Luna, Terra, and Sol configurations used through ChatGPT and Codex-style coding agents, assisted parts of the 2026 modernization. Assistance included repository exploration, alternative implementation proposals, refactoring, test scaffolding, code review, documentation editing, and manuscript drafting; the present draft was prepared with GPT-5.6 Sol. The corresponding author defined the software scope and core design decisions, selected which suggestions to accept, reviewed and edited AI-assisted changes, and validated them through tests, literature and primary-source review, TRACK source comparison, and reproducible benchmark/parity experiments. The authors remain responsible for the accuracy, originality, licensing, and scientific interpretation of the submitted material.

# Acknowledgements

The original PyStormTracker project was supported through the National Center for Atmospheric Research 2015 SIParCS program and developed with Kevin Paul and John Dennis. The storm-track analysis framework implemented by the package grew from scientific collaboration with Edmund K. M. Chang. PyStormTracker also builds on the openly available TRACK source and the broader scientific Python ecosystem.

# References
