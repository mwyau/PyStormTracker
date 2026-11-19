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
date: 19 November 2026
bibliography: paper.bib
---

# Summary

Storm-track activity can be described using Eulerian statistics of atmospheric variability or Lagrangian statistics derived from individual cyclone trajectories. Tracking tropical and extratropical cyclones involves identifying cyclone centers in gridded meteorological fields and linking them between successive analysis times.

PyStormTracker is a Python package for cyclone tracking and storm-track analysis. It includes a local-extrema tracker with nearest-neighbor linking, an implementation of Hodges's feature-tracking method [@hodges1994; @hodges1995; @hodges1999], and a tracker adapted to the HEALPix grid [@gorski2005]. The package reads meteorological files and xarray data, stores cyclone tracks in a common array-backed format, and computes storm-track statistics. It supports serial, Dask, and MPI execution.

# Statement of need

Cyclone statistics depend on the field and criteria used to identify cyclones. The IMILAST intercomparison documented differences among detection and tracking algorithms [@neu2013]. Chang (2014) showed that, even for the same climate-model simulations and tracking method, defining cyclones from total sea-level pressure or from perturbations after removal of a large-scale background can lead to opposite signs of projected changes in the frequency of strong North Pacific cyclones [@changBackground2014]. Such differences matter when cyclone statistics are used to examine climate variability and weather impacts.

Comparisons between cyclone climatologies require the tracked variable, filtering, detection thresholds, feature-point location, linking constraints, and post-tracking selection criteria to be specified. PyStormTracker records the tracked variable and the preprocessing performed by the package, and uses the same trajectory representation for comparison, variable sampling, and storm-track statistics. It computes cyclone and track frequency, cyclone amplitude, Accumulated Cyclone Activity (ACA), Accumulated Track Activity (ATA), and analyses based on CORMAX and cross-validated EOF–canonical correlation analysis (CCA) for relating storm-track variability to precipitation and strong winds [@yauChang2020].

The package is intended for atmospheric scientists analysing reanalyses, climate-model simulations, and numerical-weather-prediction output, including studies comparing cyclone definitions, tracking methods, or input grids.

# State of the field

TRACK implements the feature-tracking method of Hodges [@hodges1994; @hodges1995; @hodges1999] and has been used in studies of Northern and Southern Hemisphere storm tracks and comparisons of reanalysis datasets [@hoskinsHodges2002; @hoskinsHodges2005; @hodges2003]. The method identifies feature points in a meteorological field, links them into trajectories, and optimizes the trajectories using displacement and smoothness constraints. PyStormTracker's `HodgesTracker` implements this sequence. TRACK 1.5.4 serves as a reference implementation for numerical comparison.

Other open-source tools address related problems. TempestExtremes detects and tracks features on structured and unstructured climate-model grids [@ullrichZarzycki2017], while tobac was developed to identify and follow clouds and other atmospheric objects [@heikenfeld2019]. PyStormTracker focuses on cyclone tracks and their climatological statistics. Neither of these packages implements the TRACK 1.5.4 Modified Greedy Exchange procedure and the set of cyclone-activity diagnostics used here. Incorporating both would require substantial new tracking and analysis code.

PyStormTracker was presented at the 2016 AMS Annual Meeting with cyclone centers identified from sea-level-pressure minima, nearest-neighbor track linking, and MPI-parallel detection across analysis times [@yau2016]. Its simple tracker was subsequently used in a tracking-method sensitivity comparison by Yau and Chang (2020) [@yauChang2020].

# Software design

All three trackers return `Tracks`, an immutable structure with contiguous arrays of trajectory identifiers, offsets, times, positions, and sampled variables. A trajectory is a view of these arrays; the package does not store a separate Python object for every track point. The same structure is used for TrackJSON output, track comparison, variable sampling, and gridded statistics.

Input fields are read into labeled xarray arrays [@hoyerHamman2017]. Cyclones can be identified from pressure or vorticity fields, with optional spectral filtering and regridding before detection. For supported Gauss–Legendre and Clenshaw–Curtis grids, PyStormTracker uses spharmgrid for spherical-harmonic filtering, regridding, and wind kinematics [@spharmgridZenodo]. Spherical-harmonic transforms are computed with `ducc0` [@reinecke2020]. Reduced-Gaussian, HEALPix, polar, regional discrete-cosine-transform, and explicit-`lmax` vector calculations use separate `ducc0` paths. The tracker records which preprocessing operations it performed.

Feature detection and center refinement are independent between analysis times, allowing parallel processing of frames. Hodges and HEALPix trajectory construction uses overlapping time segments because linking depends on neighboring frames. The segment trajectories are then joined in time order. PyStormTracker supports serial execution, Dask task scheduling, and MPI distributed-memory execution; the numerical tracking settings are shared across these backends.

# Research impact statement

PyStormTracker was first presented at the 2016 AMS Annual Meeting [@yau2016]. Its simple tracker was used as an independent tracking-method sensitivity test in the analysis of storm-track activity and weather impacts by Yau and Chang (2020) [@yauChang2020]. The statistical methods developed in that research were also applied in a subsequent study of projected European precipitation changes [@changYauZhang2022]. These studies establish the research use of the tracking and analysis methods; they predate the current software release. PyStormTracker is distributed through PyPI, conda-forge, containers, and a Zenodo software archive [@pystormtrackerZenodo].

An implementation comparison used 1,464 six-hourly ERA5 mean-sea-level pressure fields from 2024 on the F320 Gaussian grid, with T6–42 spectral filtering. After applying the same eight-point and 10° endpoint-displacement selection criteria (RSPLICE), one-to-one trajectory matching between PyStormTracker and TRACK 1.5.4 gave an F1 score of 99.7% for both F320-to-T42 and F320-to-F320 reconstruction. Median matched-track separations were 4.1 m and 4.9 m, respectively; the corresponding 95th-percentile separations were 33.2 m and 38.9 m. These statistics measure correspondence with TRACK for the specified cases, not independent validation of the cyclone climatology.

On an AMD Ryzen 9 5950X (16 cores, 32 threads), five sequential full-year runs gave PyStormTracker median wall times of 27.48 s (F320-to-T42) and 57.38 s (F320-to-F320), compared with 59.43 s and 1,997.16 s for TRACK. Thus PyStormTracker was 2.16 and 34.81 times faster for those cases, although TRACK was faster on the shorter January F320-to-T42 test. The recorded benchmark used PyStormTracker revision `5f680c6`, preceding the v0.8.0 spharmgrid extraction and the v0.8.1 maintenance release; it has not been repeated for v0.8.1. Configurations, input hashes, five-run timings, and trajectory comparisons are documented in the PyStormTracker-Validation repository.

# AI usage disclosure

OpenAI GPT-5.6 models (Luna, Terra, and Sol configurations) assisted with parts of the 2026 software development, including code exploration, implementation proposals, refactoring, test scaffolding, code review, and documentation and manuscript drafting. GPT-6 assisted with review and revision of this manuscript. The corresponding author made the scientific and software-design decisions, reviewed AI-assisted code and research claims against tests, published methods, TRACK source, and recorded comparisons, and is responsible for verifying the final manuscript before submission. The authors take responsibility for the accuracy, originality, licensing, and scientific interpretation of the software and paper.

# Acknowledgements

The original PyStormTracker project was supported through the National Center for Atmospheric Research 2015 SIParCS program and developed with Kevin Paul and John Dennis. The storm-track analysis framework implemented by the package grew from scientific collaboration with Edmund K. M. Chang. PyStormTracker also builds on the openly available TRACK source and the broader scientific Python ecosystem.

# References
