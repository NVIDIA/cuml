.. SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
.. SPDX-License-Identifier: Apache-2.0
.. This file is the editable template for benchmarks.rst. Run
.. docs/benchmarks/generate_cuml_accel_benchmarks.py render after editing it.

Performance and Speedups
========================

.. rst-class:: benchmark-lede

Zero-code change acceleration with ``cuml.accel`` brings familiar
scikit-learn workflows to the GPU, delivering a **4.3× median
speedup** across completed training and combined-operation comparisons on
NVIDIA RTX Pro 6000 Blackwell. Measured performance depends on the operation
and the dataset's size and shape, ranging from overhead-dominated small
inference workloads to **92× for UMAP** and
**460× for HDBSCAN**.

Speedup by operation and workload
---------------------------------

Rows are ranked by median completed speedup. Green indicates a measured gain
or a CPU-only timeout where the accelerated run completed, gray is centered at
1×, warm colors indicate a slowdown, and hatching marks every timeout.

.. rst-class:: benchmark-footnote

A timeout indicates that the complete isolated benchmark case—not a single
estimator call—exceeded its wall-clock limit: **3
minutes** for small and medium workloads and PCA's ``large`` fit-transform, or
**10 minutes** for other ``large`` workloads.

Training and combined operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

54 of 70 comparisons completed;
41 reached at least 2×.

.. raw:: html

   <div class="benchmark-heatmap" tabindex="0" aria-label="Scrollable training heatmap">
     <object data="../../_static/cuml-accel-benchmarks/training-heatmap.svg" type="image/svg+xml" aria-label="Training speedup heatmap. Operation labels link to estimator details; exact values and timeout labels are present in the graphic and detailed tables.">
       <a href="../../_static/cuml-accel-benchmarks/training-heatmap.svg">Open the training speedup heatmap</a>
     </object>
   </div>

Inference and transforms
~~~~~~~~~~~~~~~~~~~~~~~~

The completed median is **0.81×**;
33 of 65 completed comparisons are
below 1×. Neighbor and forest operations still show strong gains at suitable
scales.

.. raw:: html

   <div class="benchmark-heatmap" tabindex="0" aria-label="Scrollable inference heatmap">
     <object data="../../_static/cuml-accel-benchmarks/inference-heatmap.svg" type="image/svg+xml" aria-label="Inference and transform speedup heatmap. Operation labels link to estimator details; exact values and timeout labels are present in the graphic and detailed tables.">
       <a href="../../_static/cuml-accel-benchmarks/inference-heatmap.svg">Open the inference speedup heatmap</a>
     </object>
   </div>

Detailed benchmark results
--------------------------

Use the workload guide to interpret the heatmaps, then open an estimator for
exact shapes, wall times, slowdowns, and timeouts.

Five workload shapes, from transfer-bound to compute-heavy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every operation uses the same five labels. The first four shapes are fixed;
``large`` is operation-specific because algorithms have very different
dimensionality, memory, and runtime behavior.

.. list-table:: Workload dimensions and decimal float32 X size
   :header-rows: 1
   :class: benchmark-workload-table

   * - Label
     - Rows
     - Features
     - Input
   * - ``small.balanced``
     - 19,531
     - 128
     - 10 MB
   * - ``medium.thin``
     - 195,312
     - 16
     - 12.5 MB
   * - ``medium.balanced``
     - 195,312
     - 128
     - 100 MB
   * - ``medium.wide``
     - 195,312
     - 512
     - 400 MB
   * - ``large *``
     - 5,000–19,531,250
     - 128–2,048
     - 41 MB–10 GB

.. rst-class:: benchmark-footnote

\* The actual rows, features, input size, and timeout for ``large`` are
operation-specific. Exact values appear in each estimator table.

Results by estimator
~~~~~~~~~~~~~~~~~~~~

Open an estimator for every workload, wall time, slowdown, timeout, and actual
shape.

.. raw:: html

   <div class="benchmark-detail-controls" aria-label="Estimator detail controls">
     <button type="button" data-benchmark-details="expand">Expand all</button>
     <button type="button" data-benchmark-details="collapse">Collapse all</button>
   </div>

.. container:: benchmark-estimator-details

   .. rubric:: Linear models

   .. dropdown:: LinearRegression
      :name: benchmark-linearregression

      .. list-table:: LinearRegression results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 21.8 ms
           - 5.0 ms
           - 4.34×
         * - ``fit``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 16.5 ms
           - 3.9 ms
           - 4.28×
         * - ``fit``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 265.6 ms
           - 21.5 ms
           - 12.4×
         * - ``fit``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 1.3 s
           - 51.3 ms
           - 25.3×
         * - ``fit``
           - ``large``
           - 7,812,500
           - 128
           - 4 GB
           - 15.7 s
           - 1.73 s
           - 9.08×
         * - ``predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 0.11 ms
           - 0.40 ms
           - 0.27×
         * - ``predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 0.13 ms
           - 0.38 ms
           - 0.35×
         * - ``predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 3.0 ms
           - 1.2 ms
           - 2.49×
         * - ``predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 2.9 ms
           - 3.8 ms
           - 0.76×
         * - ``predict``
           - ``large``
           - 7,812,500
           - 128
           - 4 GB
           - 23.8 ms
           - 32.8 ms
           - 0.73×

   .. dropdown:: LogisticRegression
      :name: benchmark-logisticregression

      .. list-table:: LogisticRegression results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 30.0 ms
           - 4.9 ms
           - 6.12×
         * - ``fit``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 164.9 ms
           - 6.2 ms
           - 26.6×
         * - ``fit``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 137.0 ms
           - 13.3 ms
           - 10.3×
         * - ``fit``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 294.7 ms
           - 43.0 ms
           - 6.86×
         * - ``fit``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 11.6 s
           - 1.18 s
           - 9.81×
         * - ``predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 0.17 ms
           - 0.51 ms
           - 0.33×
         * - ``predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 0.24 ms
           - 0.61 ms
           - 0.39×
         * - ``predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 0.57 ms
           - 1.3 ms
           - 0.44×
         * - ``predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 2.9 ms
           - 3.6 ms
           - 0.81×
         * - ``predict``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 49.5 ms
           - 81.3 ms
           - 0.61×

   .. dropdown:: Ridge
      :name: benchmark-ridge

      .. list-table:: Ridge results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 8.8 ms
           - 4.1 ms
           - 2.14×
         * - ``fit``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 14.7 ms
           - 3.6 ms
           - 4.08×
         * - ``fit``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 94.3 ms
           - 22.7 ms
           - 4.16×
         * - ``fit``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 563.0 ms
           - 56.3 ms
           - 10.00×
         * - ``fit``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 5.78 s
           - 1.99 s
           - 2.90×
         * - ``predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 0.10 ms
           - 0.34 ms
           - 0.30×
         * - ``predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 0.17 ms
           - 0.45 ms
           - 0.39×
         * - ``predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 0.52 ms
           - 1.3 ms
           - 0.40×
         * - ``predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 1.9 ms
           - 3.8 ms
           - 0.50×
         * - ``predict``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 54.1 ms
           - 81.1 ms
           - 0.67×

   .. rubric:: Clustering and manifold learning

   .. dropdown:: DBSCAN
      :name: benchmark-dbscan

      .. list-table:: DBSCAN results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit_predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 142.2 ms
           - 8.3 ms
           - 17.2×
         * - ``fit_predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 1.77 s
           - 315.0 ms
           - 5.61×
         * - ``fit_predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 5.63 s
           - 587.2 ms
           - 9.59×
         * - ``fit_predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 22.7 s
           - 2.41 s
           - 9.41×
         * - ``fit_predict``
           - ``large``
           - 1,953,125
           - 128
           - 1 GB
           - —
           - 70.1 s
           - CPU timeout (10 min)

   .. dropdown:: HDBSCAN
      :name: benchmark-hdbscan

      .. list-table:: HDBSCAN results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit_predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 12.9 s
           - 28.0 ms
           - 460×
         * - ``fit_predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 15.5 s
           - 678.1 ms
           - 22.9×
         * - ``fit_predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - —
           - 1.51 s
           - CPU timeout (3 min)
         * - ``fit_predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - 4.76 s
           - CPU timeout (3 min)
         * - ``fit_predict``
           - ``large``
           - 976,562
           - 128
           - 500 MB
           - —
           - 37 s
           - CPU timeout (10 min)

   .. dropdown:: KMeans
      :name: benchmark-kmeans

      .. list-table:: KMeans results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit_predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 26.5 ms
           - 14.1 ms
           - 1.87×
         * - ``fit_predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 67.1 ms
           - 20.4 ms
           - 3.30×
         * - ``fit_predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 217.1 ms
           - 63.4 ms
           - 3.43×
         * - ``fit_predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 2.2 s
           - 202.9 ms
           - 10.9×
         * - ``fit_predict``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 59 s
           - 5.06 s
           - 11.7×
         * - ``predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 0.23 ms
           - 0.60 ms
           - 0.38×
         * - ``predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 0.80 ms
           - 0.73 ms
           - 1.10×
         * - ``predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 0.98 ms
           - 1.5 ms
           - 0.64×
         * - ``predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 2.6 ms
           - 4.1 ms
           - 0.63×
         * - ``predict``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 52.1 ms
           - 85.0 ms
           - 0.61×

   .. dropdown:: UMAP
      :name: benchmark-umap

      .. list-table:: UMAP results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit_transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 5.16 s
           - 207.7 ms
           - 24.8×
         * - ``fit_transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - —
           - 299.8 ms
           - CPU timeout (3 min)
         * - ``fit_transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - —
           - 1.14 s
           - CPU timeout (3 min)
         * - ``fit_transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - 3.05 s
           - CPU timeout (3 min)
         * - ``fit_transform``
           - ``large``
           - 1,953,125
           - 128
           - 1 GB
           - —
           - 56.4 s
           - CPU timeout (10 min)
         * - ``transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 607.6 ms
           - 11.2 ms
           - 54.5×
         * - ``transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 1.59 s
           - 17.3 ms
           - 92.2×
         * - ``transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 5.52 s
           - 72.4 ms
           - 76.2×
         * - ``transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - 277.7 ms
           - CPU timeout (3 min)
         * - ``transform``
           - ``large``
           - 1,953,125
           - 128
           - 1 GB
           - —
           - 5.48 s
           - CPU timeout (10 min)

   .. rubric:: Neighbors

   .. dropdown:: KNeighborsClassifier
      :name: benchmark-kneighborsclassifier

      .. list-table:: KNeighborsClassifier results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 27.5 ms
           - 1.5 ms
           - 18.2×
         * - ``predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 314.1 ms
           - 9.8 ms
           - 32.2×
         * - ``predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 1.15 s
           - 53.3 ms
           - 21.5×
         * - ``predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 3.58 s
           - 219.4 ms
           - 16.3×
         * - ``predict``
           - ``large``
           - 7,812,500
           - 128
           - 4 GB
           - —
           - 88.8 s
           - CPU timeout (10 min)

   .. dropdown:: NearestNeighbors
      :name: benchmark-nearestneighbors

      .. list-table:: NearestNeighbors results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``kneighbors``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 19.0 ms
           - 1.3 ms
           - 15.1×
         * - ``kneighbors``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 384.0 ms
           - 9.5 ms
           - 40.4×
         * - ``kneighbors``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 1.28 s
           - 53.2 ms
           - 24.0×
         * - ``kneighbors``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 3.59 s
           - 217.1 ms
           - 16.5×
         * - ``kneighbors``
           - ``large``
           - 7,812,500
           - 128
           - 4 GB
           - —
           - 88.7 s
           - CPU timeout (10 min)

   .. rubric:: Decomposition

   .. dropdown:: PCA
      :name: benchmark-pca

      .. list-table:: PCA results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - Components
           - CPU
           - GPU
           - Result
         * - ``fit_transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - —
           - 6.9 ms
           - 8.8 ms
           - 0.79×
         * - ``fit_transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - —
           - 8.0 ms
           - 9.5 ms
           - 0.85×
         * - ``fit_transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - —
           - 55.5 ms
           - 243.0 ms
           - 0.23×
         * - ``fit_transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - 653.5 ms
           - 1.03 s
           - 0.64×
         * - ``fit_transform``
           - ``large``
           - 5,000
           - 2,048
           - 41 MB
           - 1,024
           - 1.56 s
           - 127.3 ms
           - 12.2×
         * - ``transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - —
           - 0.17 ms
           - 0.64 ms
           - 0.26×
         * - ``transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - —
           - 0.51 ms
           - 0.74 ms
           - 0.68×
         * - ``transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - —
           - 2.1 ms
           - 3.8 ms
           - 0.56×
         * - ``transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - 3.1 ms
           - 39.9 ms
           - 0.08×
         * - ``transform``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - —
           - 81.8 ms
           - 1.39 s
           - 0.06×

   .. dropdown:: TruncatedSVD
      :name: benchmark-truncatedsvd

      .. list-table:: TruncatedSVD results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit_transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 29.4 ms
           - 6.4 ms
           - 4.60×
         * - ``fit_transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 102.7 ms
           - 7.0 ms
           - 14.7×
         * - ``fit_transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 195.7 ms
           - 131.5 ms
           - 1.49×
         * - ``fit_transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 369.6 ms
           - 524.7 ms
           - 0.70×
         * - ``fit_transform``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 19.1 s
           - 13.3 s
           - 1.43×
         * - ``transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 0.14 ms
           - 0.58 ms
           - 0.24×
         * - ``transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 0.17 ms
           - 0.79 ms
           - 0.22×
         * - ``transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 1.4 ms
           - 3.9 ms
           - 0.37×
         * - ``transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 2.8 ms
           - 37.5 ms
           - 0.08×
         * - ``transform``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 68.7 ms
           - 1.38 s
           - 0.05×

   .. rubric:: Ensembles

   .. dropdown:: RandomForestClassifier
      :name: benchmark-randomforestclassifier

      .. list-table:: RandomForestClassifier results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 143.0 ms
           - 108.5 ms
           - 1.32×
         * - ``fit``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 690.3 ms
           - 205.0 ms
           - 3.37×
         * - ``fit``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 1.73 s
           - 402.8 ms
           - 4.30×
         * - ``fit``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 3.38 s
           - 592.0 ms
           - 5.70×
         * - ``fit``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - —
           - 7.91 s
           - CPU timeout (10 min)
         * - ``predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 13.5 ms
           - 0.61 ms
           - 22.3×
         * - ``predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 13.7 ms
           - 0.63 ms
           - 21.7×
         * - ``predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 14.1 ms
           - 1.8 ms
           - 8.05×
         * - ``predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 15.5 ms
           - 5.0 ms
           - 3.14×
         * - ``predict``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 569.0 ms
           - 118.0 ms
           - 4.82×

   .. dropdown:: RandomForestRegressor
      :name: benchmark-randomforestregressor

      .. list-table:: RandomForestRegressor results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 1.04 s
           - 255.8 ms
           - 4.06×
         * - ``fit``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 1.7 s
           - 744.5 ms
           - 2.28×
         * - ``fit``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 13.9 s
           - 1.5 s
           - 9.27×
         * - ``fit``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - 4.21 s
           - CPU timeout (3 min)
         * - ``fit``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - —
           - 85.7 s
           - CPU timeout (10 min)
         * - ``predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 13.5 ms
           - 0.53 ms
           - 25.4×
         * - ``predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 13.5 ms
           - 0.48 ms
           - 28.3×
         * - ``predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 14.0 ms
           - 1.5 ms
           - 9.54×
         * - ``predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 15.4 ms
           - 4.7 ms
           - 3.24×
         * - ``predict``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - —
           - 115.6 ms
           - CPU timeout (10 min)

   .. rubric:: Preprocessing

   .. dropdown:: PolynomialFeatures
      :name: benchmark-polynomialfeatures

      .. list-table:: PolynomialFeatures results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit_transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 136.5 ms
           - 74.7 ms
           - 1.83×
         * - ``fit_transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 85.4 ms
           - 14.8 ms
           - 5.77×
         * - ``fit_transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 2.06 s
           - 593.5 ms
           - 3.48×
         * - ``fit_transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - —
           - CPU + GPU timeout (3 min)
         * - ``fit_transform``
           - ``large``
           - 1,953,125
           - 128
           - 1 GB
           - 27.8 s
           - 21.1 s
           - 1.31×
         * - ``transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 14.2 ms
           - 11.2 ms
           - 1.26×
         * - ``transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 5.1 ms
           - 2.4 ms
           - 2.16×
         * - ``transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 153.9 ms
           - 78.5 ms
           - 1.96×
         * - ``transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 2.47 s
           - 1.05 s
           - 2.35×
         * - ``transform``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 41.9 s
           - —
           - GPU timeout (10 min)

   .. dropdown:: StandardScaler
      :name: benchmark-standardscaler

      .. list-table:: StandardScaler results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit_transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 7.5 ms
           - 7.2 ms
           - 1.04×
         * - ``fit_transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 16.2 ms
           - 11.1 ms
           - 1.46×
         * - ``fit_transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 103.9 ms
           - 37.9 ms
           - 2.74×
         * - ``fit_transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 326.3 ms
           - 104.2 ms
           - 3.13×
         * - ``fit_transform``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 8.41 s
           - 3.7 s
           - 2.27×
         * - ``transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 0.19 ms
           - 0.62 ms
           - 0.30×
         * - ``transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 0.31 ms
           - 0.69 ms
           - 0.44×
         * - ``transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 2.2 ms
           - 2.1 ms
           - 1.04×
         * - ``transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 9.3 ms
           - 7.9 ms
           - 1.18×
         * - ``transform``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 222.8 ms
           - 166.9 ms
           - 1.34×

   .. dropdown:: TargetEncoder
      :name: benchmark-targetencoder

      .. list-table:: TargetEncoder results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``transform``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 17.4 ms
           - 2.03 s
           - 0.01×
         * - ``transform``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 11.1 ms
           - 89.2 ms
           - 0.12×
         * - ``transform``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - 100.4 ms
           - 2.02 s
           - 0.05×
         * - ``transform``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - 520.6 ms
           - —
           - GPU timeout (3 min)
         * - ``transform``
           - ``large``
           - 19,531,250
           - 128
           - 10 GB
           - 13.9 s
           - 10 s
           - 1.38×

   .. rubric:: Kernel methods

   .. dropdown:: SVC
      :name: benchmark-svc

      .. list-table:: SVC results for all measured operations and workloads
         :header-rows: 1
         :class: benchmark-result-table

         * - Operation
           - Workload
           - Rows
           - Features
           - Input
           - CPU
           - GPU
           - Result
         * - ``fit``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 5.96 s
           - 65.5 ms
           - 91.0×
         * - ``fit``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - —
           - 315.4 ms
           - CPU timeout (3 min)
         * - ``fit``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - —
           - 979.8 ms
           - CPU timeout (3 min)
         * - ``fit``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - 2.1 s
           - CPU timeout (3 min)
         * - ``fit``
           - ``large``
           - 1,953,125
           - 128
           - 1 GB
           - —
           - 39.7 s
           - CPU timeout (10 min)
         * - ``predict``
           - ``small.balanced``
           - 19,531
           - 128
           - 10 MB
           - 606.2 ms
           - 1.7 ms
           - 354×
         * - ``predict``
           - ``medium.thin``
           - 195,312
           - 16
           - 12.5 MB
           - 6.52 s
           - 19.9 ms
           - 327×
         * - ``predict``
           - ``medium.balanced``
           - 195,312
           - 128
           - 100 MB
           - —
           - 35.5 ms
           - CPU timeout (3 min)
         * - ``predict``
           - ``medium.wide``
           - 195,312
           - 512
           - 400 MB
           - —
           - 88.8 ms
           - CPU timeout (3 min)
         * - ``predict``
           - ``large``
           - 1,953,125
           - 128
           - 1 GB
           - —
           - 962.8 ms
           - CPU timeout (10 min)

Choose workloads that can amortize acceleration overhead
---------------------------------------------------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Best candidates

      Compute-heavy fitting, forests, nearest-neighbor search, clustering, and
      sufficiently large matrices give GPU work room to dominate dispatch and
      data-conversion costs.

   .. grid-item-card:: Measure inference separately

      A fast fit does not guarantee a fast prediction or transform. Small and
      narrow inference workloads often remain latency-bound; benchmark the
      operation used in production.

   .. grid-item-card:: Treat timeouts as unknown

      A CPU-only timeout can show that the accelerated run finished within
      policy, but it does not provide a numeric speedup. Use the detailed
      status and actual shape when planning capacity.

Methodology and reproducibility
-------------------------------

These benchmarks compare scikit-learn CPU execution with ``cuml.accel`` on
NVIDIA RTX Pro 6000 Blackwell across five workload shapes. Each isolated case
used one warmup, the median of three measured repetitions,
operation-appropriate correctness validation, and a complete-case timeout.

.. dropdown:: Test system, validation, and timing policy

   **System.** NVIDIA RTX PRO 6000 Blackwell Workstation Edition (102.0 GB), AMD Ryzen Threadripper PRO 7975WX 32-Cores, and
   134.1 GB of system memory.

   **Timing.** One warmup followed by three measured repetitions; tables use
   the median end-to-end wall time. Speedup is CPU median wall time divided by
   accelerated median wall time.

   **Scale and timeout policy.** Fixed decimal-byte small and medium shapes
   plus operation-specific large shapes. Each backend ran every case in a
   separate worker with a 3-minute wall-clock limit,
   extended to 10 minutes for large workloads other than
   PCA ``fit_transform``. The limit covered process startup, data preparation,
   estimator setup, one warmup, three measured repetitions, and correctness
   validation. The results retain 23 CPU-only, 2
   GPU-only, and 1 both-side timeouts.

   **Execution and correctness.** Successful accelerated measurements were
   instrumented to verify GPU-only execution. The benchmark runner applied
   each case's operation-appropriate parity check. A timeout means validation
   could not complete.

   **Packages.** ``cuml 26.10.0a69``, ``cupy 14.1.1``, ``hdbscan 0.8.44``, ``numpy 2.4.6``, ``scikit-learn 1.9.0``, ``scipy 1.16.3``, ``umap-learn 0.5.12``.

   **Coverage and scope.** All 145 cases are included, including all
   26 unavailable comparisons and every completed result below
   1×. These measurements describe the tested cases on this system; results
   for other workloads and systems will vary. PCA's operation-specific
   ``large`` fit-transform uses 5,000 rows, 2,048 features, and 1,024 retained
   components.

Continue with cuml.accel
------------------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: usage
      :link-type: doc

      Installation and zero-code-change usage.

   .. grid-item-card:: Compatibility
      :link: compatibility
      :link-type: doc

      Supported estimators and parameter behavior.

   .. grid-item-card:: Profiling
      :link: logging-and-profiling
      :link-type: doc

      Find GPU execution and fallback behavior.
