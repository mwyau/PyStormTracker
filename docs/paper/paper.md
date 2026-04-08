---
title: 'PyStormTracker: A High-Performance Cyclone Tracker in Python'
tags:
  - Python
  - atmospheric science
  - meteorology
  - cyclones
  - climate data
  - tracking
authors:
  - name: Albert M. W. Yau
    orcid: 0009-0000-7825-9627
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 8 April 2026
bibliography: paper.bib
---

# Summary

Understanding the movement and intensity of atmospheric cyclones is essential for weather forecasting and climate change research. PyStormTracker is an open-source Python package for the trajectory analysis of these systems, released under the BSD 3-Clause license. It implements two primary algorithms: a heuristic "Simple Tracker" optimized for high-resolution data [@Yau2020] and a global optimization-based "Hodges (TRACK)" algorithm [@Hodges1999]. The software aims for algorithmic parity with the original TRACK software, with any minor implementation differences explicitly documented in the accompanying technical notes. The software provides a vectorized framework for processing large-scale climate datasets, including the ECMWF Reanalysis v5 (ERA5). PyStormTracker integrates with the scientific Python ecosystem using `xarray` for data I/O and `dask` or `mpi4py` for parallel execution. It supports modern data standards, including the HEALPix spherical nested grid.

# Statement of need

Scientific analysis of storm track activity requires the objective identification and linking of meteorological features across time steps. Historically, researchers have relied on tools such as the TRACK program [@Hodges1994; @Hodges1995; @Hodges1999], which is written in C and Fortran. While scientifically robust, these legacy tools are often characterized by complex compilation requirements, rigid monolithic architectures, and limited integration with modern data analysis environments. Crucially, legacy workflows often require writing multiple intermediate files for spectral filtering, regridding, and kinematic calculations, which introduces significant I/O overhead and complicates data management.

PyStormTracker addresses these issues by providing a native Python implementation that streamlines the entire tracking pipeline in-memory. It maintains algorithmic parity with established methods while improving usability and performance. The project originated in 2015 as part of the Summer Internships in Parallel Computational Science (SIParCS) program at the National Center for Atmospheric Research (NCAR) Computational and Information Systems Laboratory (CISL) and was presented at the 96th AMS Annual Meeting [@Yau2016]. Development restarted in 2026 to modernize the architecture for higher-resolution climate model outputs.

# State of the field

The IMILAST project [@Neu2013] demonstrated that variations in tracking algorithms lead to significantly different cyclone climatologies. Consequently, software that allows for the comparison of multiple algorithms on identical data structures is necessary. PyStormTracker distinguishes itself by offering an open-source, extensible protocol for tracking. It natively supports coordinate-aware NetCDF, GRIB, and Zarr formats through a unified `DataLoader`, and can be easily extended to support new I/O formats, grid types (such as HEALPix), and custom detector/linker heuristics.

# Research use

The Simple Tracker algorithm implemented in this package was originally described in the Supplementary Materials of @Yau2020, where it was evaluated and compared against results from the Hodges TRACK software. This comparison ensured that the heuristic linking method maintained physical consistency with established storm track activity metrics.

# Software design

The 2026 modernization introduced several core technologies and modules to enhance performance and maintainability:

*   **Integrated Spectral & Kinematics Modules:** Built-in modules for spectral regridding and kinematic derivative calculations (e.g., relative vorticity and divergence) based on the highly optimized `ducc0` library [@ducc0_software]. This eliminates the need for external preprocessing and intermediate file I/O.
*   **Modular I/O and Data Model:** The `io` module provides a centralized `DataLoader` that handles coordinate-aware NetCDF, GRIB, and Zarr formats, including support for remote datasets. The `models` module defines the core, array-backed data structures for centers and trajectories, ensuring consistent geometry and physical attributes across different tracking algorithms.
*   **Vectorized Data Model:** Replaced nested Python objects with flat, 1D NumPy arrays for all trajectory and feature data, reducing memory overhead and improving serialization speed.
*   **JIT-Optimized Kernels:** Implemented feature detection and tracking cost functions using `Numba`, enabling C-level execution speeds and multi-threaded execution.
*   **Extensible Architecture:** Designed around a protocol-oriented architecture that allows researchers to swap or add new components—including I/O engines, grid structures, feature detectors, and trajectory linkers—without modifying the core pipeline.
*   **Scalable Backends:** Supports serial execution, Dask multi-processing, and MPI-based distributed computing, utilizing `xarray` for efficient in-memory data passing across nodes.

# Engineering Standards and Quality Control

PyStormTracker adheres to rigorous software engineering standards:

*   **Validation and Regression Testing:** The suite includes specific integration tests to ensure that the Simple algorithm outputs remain consistent with the project's legacy v0.0.2 baseline.
*   **Continuous Integration (CI):** Automated workflows via GitHub Actions execute unit and integration tests on every commit, covering multiple Python versions (3.11–3.14), operating systems (Linux, macOS, Windows), and architectures (x86_64, ARM64).
*   **Static Analysis:** Enforces strict type safety using `mypy` and code quality standards using the `ruff` linter and formatter.
*   **Containerization & Deployment:** Automated Docker builds include smoke tests. Releases are automatically published to PyPI, TestPyPI, and Conda-forge.
*   **Documentation:** Comprehensive documentation is available via Read the Docs, including technical methodology, performance benchmarks, and a project roadmap outlining future development goals.

# AI usage disclosure

Generative AI tools, specifically Google Gemini (via Gemini CLI), were used in the authoring of this paper and the refactoring of the software documentation. These tools assisted with drafting technical descriptions, organizing repository metadata, and ensuring compliance with JOSS standards. All outputs were reviewed and validated for technical accuracy by the human author.

# Acknowledgements

This project was initially developed at the National Center for Atmospheric Research (NCAR) as part of the 2015 SIParCS program. The author would like to acknowledge **Kevin Paul** (ORCID: [0000-0001-8155-8038](https://orcid.org/0000-0001-8155-8038)) and **John Dennis** (ORCID: [0000-0002-2119-8242](https://orcid.org/0000-0002-2119-8242)) at the NCAR Computational and Information Systems Laboratory (CISL) for their mentorship and support during the project's inception.

# References
