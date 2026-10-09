# Comparing cuML with scikit-learn

cuML aims for comparable model quality, not identical numbers. A fixed `random_state` makes
cuML reproducible with itself, but does not make it match scikit-learn
([#8631](https://github.com/NVIDIA/cuml/issues/8631)). Before calling a difference a bug,
compare the right metric (below) and check the expected differences listed here.

## Compare quality, not fitted values

| Estimator family | Compare | Do not compare |
|---|---|---|
| Clustering (KMeans, DBSCAN, HDBSCAN, SpectralClustering) | `adjusted_rand_score`, `adjusted_mutual_info_score` | `labels_` (label permutation), `cluster_centers_` order |
| Classifiers (RF, LogisticRegression, SVC, KNN) | accuracy, `log_loss`, ROC AUC on held-out data | `coef_`, `estimators_`, `support_vectors_` |
| Regressors | `r2_score`, RMSE on held-out data | `coef_` (solvers differ) |
| PCA / TruncatedSVD | explained variance ratio; subspace angles (`scipy.linalg.subspace_angles`) | signs or order of `components_` |
| UMAP / TSNE | `sklearn.manifold.trustworthiness`; TSNE `kl_divergence_` | raw embedding coordinates |
| KernelRidge | `dual_coef_` with `np.allclose` (expected to match closely) | — |

Use the same data split for both sides, and float32 input on both sides when checking
tolerance.

## Expected differences (not bugs)

- **float32 by default.** Non-float input is converted to float32; kNN, UMAP and HDBSCAN
  compute in float32 even for float64 input. Very large float64 values can overflow when cast
  ([#8570](https://github.com/NVIDIA/cuml/issues/8570), open).
- **Parallel reductions.** Atomic adds make float32 PCA and some solvers vary slightly from run
  to run; HDBSCAN's parallel MST is not deterministic when the mutual-reachability graph has
  ties (duplicate points).
- **TSNE.** Barnes-Hut is mapped to FFT; not fully deterministic even with `random_state`. On
  some very homogeneous embeddings Barnes-Hut collapses to a line where sklearn does not; try
  `method="exact"` on a sample.
- **KMeans.** Default `init="scalable-k-means++"`, `n_init="auto"`; different RNG streams, so
  centroids and labels differ from sklearn for the same seed.
- **KFold(shuffle=True)** returns the same folds as sklearn but training indices in a
  different order.
- **TargetEncoder.fit_transform** uses different cross-validation folds (transform on new
  data matches).
- **OneHotEncoder** treats `None` and `NaN` as the same category.
- **RandomForest** uses histogram split finding (`n_bins`, default 128): trees differ from
  sklearn's exact splits.

## Fixed bugs worth knowing about (upgrade if affected)

- `KMeans.transform` returned squared distances
  ([#8536](https://github.com/NVIDIA/cuml/issues/8536)); fixed in 26.10.
- `KMeans` ignored `sample_weight` in `inertia_`/`score`
  ([#8530](https://github.com/NVIDIA/cuml/issues/8530)); fixed in 26.10.
- Multiclass `SVC` with non-uniform `class_weight` gave wrong predictions
  ([#8578](https://github.com/NVIDIA/cuml/issues/8578)); fixed in 26.10.
- Open: `cuml.compose.ColumnTransformer` with `cuml.preprocessing.Normalizer(norm="l1")`
  produces L2-normalised output ([#8577](https://github.com/NVIDIA/cuml/issues/8577)).

## When it is a bug

Exceptions, crashes, or clearly worse held-out quality than scikit-learn on the same data are
bugs: report them at https://github.com/NVIDIA/cuml/issues with a minimal reproducer and
`python -m cuml.health_checks` output.
