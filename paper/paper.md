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
date: 23 March 2026
bibliography: paper.bib
---

# Summary

Understanding the lifecycle, frequency, and intensity of both extratropical and tropical cyclones is crucial in climate science, meteorology, and risk assessment. Cyclones are major drivers of extreme weather events, and accurately tracking their trajectories across varied spatial and temporal scales is vital for analyzing shifts in storm track activity due to anthropogenic climate change [@Yau2020]. **PyStormTracker** [@Yau2026] is an open-source, high-performance Python package designed for large-scale cyclone trajectory analysis. It implements multiple tracking algorithms, including the "Simple Tracker" heuristic and the optimization-based "Hodges (TRACK)" algorithm with adaptive constraints. Initially prototyped as an object-oriented Python implementation at the National Center for Atmospheric Research (NCAR) [@Yau2016], PyStormTracker has evolved into a highly scalable, vectorized framework. It leverages JIT-compiled Numba kernels to identify local extrema and apply discrete Laplacian operators for measuring physical intensity. The reliable detection of these systems is a prerequisite for understanding synoptic-scale dynamics, extreme precipitation events, and wind hazards. As climate models move towards ever higher resolutions, the computational burden of tracking algorithms increases quadratically, necessitating modern, optimized software solutions. PyStormTracker processes high-resolution climate datasets, such as the ECMWF Reanalysis v5 (ERA5), utilizing coordinate-aware data structures to seamlessly integrate with the broader scientific Python ecosystem.

# Statement of need

Historically, atmospheric tracking software has been written in compiled languages like Fortran or C. One of the most prominent tools in the field is the TRACK program [@Hodges1994; @Hodges1995; @Hodges1999]. While these legacy tools have proven to be highly effective and robust over decades of rigorous scientific application, they often suffer from significant usability challenges for modern researchers. They typically involve difficult compilation processes, complex system dependencies, and a reliance on rigid, monolithic architectures. Furthermore, these older codes often lack a modern, accessible interface, making them difficult for newer generations of data scientists to easily install and integrate into modern analysis environments. As climate datasets grow exponentially in both resolution and volume, the friction of maintaining legacy data pipelines becomes a significant bottleneck to rapid scientific discovery.

`PyStormTracker` is explicitly designed to fill this critical gap by providing an architecture that is significantly easier to compile, install, deploy, and use. It presents a clear, modern Python API that integrates natively with the broader scientific Python ecosystem, notably relying on `xarray` for data structures and `dask` for parallel execution. The software accomplishes the challenging goal of maintaining strict algorithmic parity with the established TRACK algorithms while simultaneously achieving much higher computational performance. Through an innovative array-backed data model, zero-copy serialization between parallel processes, and `numba`-optimized computational kernels, PyStormTracker yields an order-of-magnitude speedup in serial workloads over traditional object-oriented tracking methods, significantly accelerating research iterations. If you use this software in your research, please cite @Yau2026.

# State of the field

The identification and tracking of extratropical cyclones is a complex problem with numerous competing methodologies. The IMILAST (Intercomparison of Mid Latitude Storm Diagnostics) project [@Neu2013] systematically evaluated fifteen different tracking algorithms, demonstrating that variations in tracking methodology can lead to significantly different climatologies, even when processing the exact same underlying meteorological fields. The project highlighted that the choice of tracking algorithm strongly influences the resulting number, intensity distribution, and lifecycle characteristics of the detected storms. Because no single tracking algorithm is universally optimal for all research questions, there is a pressing need for flexible software frameworks that allow researchers to deploy multiple algorithms on identical data structures.

Despite this diversity in algorithms, the underlying software implementations have historically been siloed. Existing tracking tools, while scientifically rigorous, are often difficult to integrate into modern Python-based data workflows. For instance, the original TRACK codebase requires offline spectral filtering steps and relies on complex, system-specific shell scripting for parallel domain splitting. Other historical tools may rely on specific legacy binary formats, necessitating cumbersome conversion scripts to process standard CF-compliant NetCDF or Zarr arrays. Additionally, many tracking codes were developed before the widespread adoption of modern version control and continuous integration practices, leading to monolithic, difficult-to-modify codebases. Modern data science demands modularity, where researchers can substitute individual components—such as the feature detection heuristic or the distance metric—without rewriting the entire application. PyStormTracker fulfills this requirement through its protocol-oriented design.

`PyStormTracker` distinguishes itself in this crowded field by offering a robust, native Python package that leverages the standard scientific Python stack without sacrificing the rigorous mathematics of established methods. It introduces a centralized `DataLoader` that handles coordinate-aware NetCDF and GRIB files directly. Furthermore, `PyStormTracker` automatically applies required spherical harmonic filtering (e.g., T42 truncation) and boundary tapering on the fly via the highly optimized `SHTns` library [@Schaeffer2013]. This completely eliminates the need for manual, error-prone external preprocessing steps, streamlining the workflow from raw data to final trajectory analysis.

# Software design

PyStormTracker is engineered for massive scale, deep extensibility, and strict mathematical parity with historical algorithmic methods. The architecture represents a fundamental shift from legacy nested-object designs, centering around a unified `Tracker` API protocol, centralized threshold management, and a highly vectorized data model.

*   **Array-Backed Data Model:** Central to PyStormTracker's performance is its data representation. Tracks and feature points are managed as flat, contiguous 1D NumPy arrays (for tracking IDs, temporal indices, latitudes, longitudes, and physical variable intensities). By strictly avoiding the creation of heavy Python objects in tight computational loops, PyStormTracker minimizes memory overhead and ensures that CPU cache utilization is optimal. Furthermore, this flat array structure allows data serialization between parallel processes to be exceptionally fast, a critical requirement for distributed computing.

*   **Numba JIT Acceleration:** The algorithms required for robust storm tracking involve heavy mathematical operations. PyStormTracker offloads these computations—such as multi-stage object-based detection, iterative Connected Component Labeling (CCL), sub-grid peak finding using local quadratic surface fitting, and the Modified Greedy Exchange (MGE) track optimization—to Just-In-Time (JIT) compiled Numba kernels. This design bypasses Python's Global Interpreter Lock (GIL) and matches or exceeds the execution speed of original C implementations while remaining entirely within a Python codebase.

*   **Gather-then-Link Parallelism:** To guarantee that parallel results remain bit-wise identical to serial execution, PyStormTracker utilizes a hybrid parallelism strategy. In meteorological feature tracking, the *Detection* phase (identifying local extrema in 3D grids) consumes over 95% of the overall runtime. PyStormTracker parallelizes this computationally intensive phase across Dask or MPI workers. Conversely, the *Linking* phase is centralized: the main process gathers the raw coordinate arrays from all workers and performs a single, sequential optimization pass. This "Gather-then-Link" orchestration completely avoids the complex boundary-merging bugs and heuristic approximations common in spatial domain-splitting strategies, ensuring perfect exactness.

*   **Adaptive Constraints:** For the Hodges algorithm parity, PyStormTracker implements dynamic tracking parameters. A global optimization approach uses a spherical cost function that penalizes changes in track direction and speed. It dynamically adjusts search radii and smoothness limits based on regional zones (e.g., extratropics versus tropics) and track velocity, faithfully replicating the logic detailed in historical literature.

*   **Extensible Tracking Protocol:** The internal `Tracker` protocol defines a standardized interface for all tracking implementations. This allows users to seamlessly switch between the `SimpleTracker` heuristic and the `HodgesTracker` global optimization algorithm using identical input arguments and configuration patterns. Researchers can easily implement custom tracking heuristics by subclassing the protocol and providing a novel `link()` method, drastically lowering the barrier to entry for algorithmic experimentation.

# Acknowledgements

This project was initially developed at the National Center for Atmospheric Research (NCAR) as part of the 2015 SIParCS program.

If you use this software in your research, please cite the following:

- **Yau, A. M. W.**, 2026: mwyau/PyStormTracker. *Zenodo*, [https://doi.org/10.5281/zenodo.18764813](https://doi.org/10.5281/zenodo.18764813).

- **Yau, A. M. W. and Chang, E. K. M.**, 2020: Finding Storm Track Activity Metrics That Are Highly Correlated with Weather Impacts. *J. Climate*, **33**, 10169–10186, [https://doi.org/10.1175/JCLI-D-20-0393.1](https://doi.org/10.1175/JCLI-D-20-0393.1).

# References

- **Yau, A. M. W., K. Paul and J. Dennis**, 2016: PyStormTracker: A Parallel Object-Oriented Cyclone Tracker in Python. *96th American Meteorological Society Annual Meeting*, New Orleans, LA. *Zenodo*, [https://doi.org/10.5281/zenodo.18868625](https://doi.org/10.5281/zenodo.18868625).

- **Neu, U., et al.**, 2013: IMILAST: A Community Effort to Intercompare Extratropical Cyclone Detection and Tracking Algorithms. *Bull. Amer. Meteor. Soc.*, **94**, 529–547, [https://doi.org/10.1175/BAMS-D-11-00154.1](https://doi.org/10.1175/BAMS-D-11-00154.1).
  - IMILAST Intercomparison Protocol: [https://proclim.scnat.ch/en/activities/project_imilast/intercomparison](https://proclim.scnat.ch/en/activities/project_imilast/intercomparison)
  - IMILAST Data Download: [https://proclim.scnat.ch/en/activities/project_imilast/data_download](https://proclim.scnat.ch/en/activities/project_imilast/data_download)

- **Schaeffer, N.**, 2013: Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations. *Geochem. Geophys. Geosyst.*, **14**, 751–758, [https://doi.org/10.1002/ggge.20071](https://doi.org/10.1002/ggge.20071).

- **Hodges, K. I.**, 1999: Adaptive Constraints for Feature Tracking. *Mon. Wea. Rev.*, **127**, 1362–1373, [https://doi.org/10.1175/1520-0493(1999)127<1362:ACFFT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1999)127<1362:ACFFT>2.0.CO;2).

- **Hodges, K. I.**, 1995: Feature Tracking on the Unit Sphere. *Mon. Wea. Rev.*, **123**, 3458–3465, [https://doi.org/10.1175/1520-0493(1995)123<3458:FTOTUS>2.0.CO;2](https://doi.org/10.1175/1520-0493(1995)123<3458:FTOTUS>2.0.CO;2).

- **Hodges, K. I.**, 1994: A General Method for Tracking Analysis and Its Application to Meteorological Data. *Mon. Wea. Rev.*, **122**, 2573–2586, [https://doi.org/10.1175/1520-0493(1994)122<2573:AGMFTA>2.0.CO;2](https://doi.org/10.1175/1520-0493(1994)122<2573:AGMFTA>2.0.CO;2).
