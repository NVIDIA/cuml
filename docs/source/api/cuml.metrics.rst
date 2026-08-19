cuml.metrics
============

.. automodule:: cuml.metrics

Classification and Distance Metrics
------------------------------------

.. currentmodule:: cuml.metrics

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   accuracy_score
   confusion_matrix
   kl_divergence
   log_loss
   roc_auc_score
   precision_recall_curve
   trustworthiness

Regression Metrics
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   mean_absolute_error
   mean_squared_error
   mean_squared_log_error
   median_absolute_error
   r2_score

Clustering Metrics
------------------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   cluster.adjusted_rand_score
   cluster.entropy
   cluster.homogeneity_score
   cluster.silhouette_score
   cluster.silhouette_samples
   cluster.completeness_score
   cluster.mutual_info_score
   cluster.v_measure_score

Pairwise Distances and Kernels
------------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   pairwise_distances
   nan_euclidean_distances
   pairwise_kernels
