# cuml 26.08.00 (5 Aug 2026)

### 🚨 Breaking Changes
* Switch RandomForest max_depth default to None by @csadorf in https://github.com/NVIDIA/cuml/pull/8200
* Link to libcuvs dynamically in wheel builds by @divyegala in https://github.com/NVIDIA/cuml/pull/8250
* Cleanup pairwise distances by @jcrist in https://github.com/NVIDIA/cuml/pull/8268
* Deprecate `convert_dtype` by @jcrist in https://github.com/NVIDIA/cuml/pull/8300
* Deprecate `PorterStemmer` by @jcrist in https://github.com/NVIDIA/cuml/pull/8304
* Move `Lars` out of experimental by @jcrist in https://github.com/NVIDIA/cuml/pull/8336
* CI Extend time limit for hypothesis tests by @betatim in https://github.com/NVIDIA/cuml/pull/8346
* Deprecate cuml.tsa by @csadorf in https://github.com/NVIDIA/cuml/pull/8344
* wheels: build CUDA 13 wheels with latest CTK (13.3.0) by @jameslamb in https://github.com/NVIDIA/cuml/pull/8345
* Deprecate `cuml.datasets.make_arima` by @jcrist in https://github.com/NVIDIA/cuml/pull/8379
* Deprecate legacy output types by @jcrist in https://github.com/NVIDIA/cuml/pull/8389
### 🐛 Bug Fixes
* Fix pandas 3 string dtype compatibility and CI flakes by @csadorf in https://github.com/NVIDIA/cuml/pull/8142
* Fix KDE score_samples symbol exports by @csadorf in https://github.com/NVIDIA/cuml/pull/8173
* Fix hinge loss decision-function shape handling by @csadorf in https://github.com/NVIDIA/cuml/pull/8245
* Fix clang-tidy failures in CUDA 13.2 CI by @dantegd in https://github.com/NVIDIA/cuml/pull/8251
* Match sklearn random forest feature sampling by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8239
* Test cuml.accel upstream test suite with sklearn 1.9 by @csadorf in https://github.com/NVIDIA/cuml/pull/8207
* Update test suite to work with new `pytest` release by @dantegd in https://github.com/NVIDIA/cuml/pull/8266
* Introduce Safe Arithmetic by @divyegala in https://github.com/NVIDIA/cuml/pull/8171
* Cleanup `extract_knn_graph` by @jcrist in https://github.com/NVIDIA/cuml/pull/8272
* Fix RandomForestClassifier sklearn child metadata by @csadorf in https://github.com/NVIDIA/cuml/pull/8291
* Fallback unsupported cuml.accel PCA and KernelRidge cases to CPU by @csadorf in https://github.com/NVIDIA/cuml/pull/8290
* Fix `FastIntDiv` tparam by @aamijar in https://github.com/NVIDIA/cuml/pull/8299
* Remove sklearn example xfails by @csadorf in https://github.com/NVIDIA/cuml/pull/8090
* FIX: handle None transformers in ColumnTransformer get_params/set_params by @Husnain987 in https://github.com/NVIDIA/cuml/pull/8292
* Handle OpenML checksum failures in sklearn examples by @csadorf in https://github.com/NVIDIA/cuml/pull/8359
* Fix RF feature importances with infinite metrics by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8347
* Limit cuml.accel pytest logging to errors by @csadorf in https://github.com/NVIDIA/cuml/pull/8360
* Fix --print-status path in benchmark CLI by @fallintoplace in https://github.com/NVIDIA/cuml/pull/8320
* fix(clang-format): preserve SPDX comments by @bdice in https://github.com/NVIDIA/cuml/pull/8368
* Make LabelEncoder docstring account for cuDF update by @viclafargue in https://github.com/NVIDIA/cuml/pull/8376
* Improve multi-GPU KMeans validation by @viclafargue in https://github.com/NVIDIA/cuml/pull/8350
* Separate sklearn example compatibility and total pass rates by @csadorf in https://github.com/NVIDIA/cuml/pull/8409
* Make FFT t-SNE deterministic when random seed is set by @dantegd in https://github.com/NVIDIA/cuml/pull/8123
* Fix Dask logistic regression with non-contiguous class labels by @viclafargue in https://github.com/NVIDIA/cuml/pull/8422
### 📖 Documentation
* Modernize the Python developer docs by @csadorf in https://github.com/NVIDIA/cuml/pull/7642
* Update docs to use NVIDIA Sphinx theme by @bdice in https://github.com/NVIDIA/cuml/pull/8308
* Restructure and clarify cuml.accel documentation by @csadorf in https://github.com/NVIDIA/cuml/pull/8367
* Document list and tuple input support by @csadorf in https://github.com/NVIDIA/cuml/pull/8381
### 🚀 New Features
* Add IPython magic for setting `cuml.accel` log level by @jcrist in https://github.com/NVIDIA/cuml/pull/8231
* [FEA] Add host-streaming support to single-GPU KMeans by @tarang-jain in https://github.com/NVIDIA/cuml/pull/8248
* Isolation Forest Implementation  by @dantegd in https://github.com/NVIDIA/cuml/pull/8226
### 🛠️ Improvements
* Suppress intentional hardcoded version references by @csadorf in https://github.com/NVIDIA/cuml/pull/8113
* Forward merge release/26.06 into main by @csadorf in https://github.com/NVIDIA/cuml/pull/8138
* Improve devcontainer startup times by @trxcllnt in https://github.com/NVIDIA/cuml/pull/8135
* Fix symbol export by @vyasr in https://github.com/NVIDIA/cuml/pull/8037
* Fix a few lingering sklearn compat xfails by @jcrist in https://github.com/NVIDIA/cuml/pull/8166
* Add YAML based benchmark definitions for cuML regression runs by @dantegd in https://github.com/NVIDIA/cuml/pull/7980
* Report xfail details in Python test output by @csadorf in https://github.com/NVIDIA/cuml/pull/8140
* Add SECURITY.md by @jameslamb in https://github.com/NVIDIA/cuml/pull/8175
* merge release/26.06 into main by @jameslamb in https://github.com/NVIDIA/cuml/pull/8184
* Replace RF global quantile sort with deterministic sampled quantiles by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8111
* Large (int64 indices) sparse input support by @jcrist in https://github.com/NVIDIA/cuml/pull/8196
* RF: narrow `CountBin` to `double` for weighted-training prep by @switch527 in https://github.com/NVIDIA/cuml/pull/8132
* fix(ci): fix configuration for breaking change notification workflow by @jameslamb in https://github.com/NVIDIA/cuml/pull/8199
* TST Make `check_estimator` coverage opt-out by @betatim in https://github.com/NVIDIA/cuml/pull/8176
* Forward merge release/26.06 into main by @csadorf in https://github.com/NVIDIA/cuml/pull/8214
* Remove deprecated `sample_weight` in naive bayes/`sample_weights` in `hinge_loss` by @jcrist in https://github.com/NVIDIA/cuml/pull/8221
* Remove deprecated `probability` argument to `SVC` by @jcrist in https://github.com/NVIDIA/cuml/pull/8223
* Bump scikit-learn minimum to 1.6 by @csadorf in https://github.com/NVIDIA/cuml/pull/8091
* Avoid host sync in `_repr_mimebundle_` when possible by @jcrist in https://github.com/NVIDIA/cuml/pull/8232
* Remove deprecated `probability` argument to `LinearSVC` by @jcrist in https://github.com/NVIDIA/cuml/pull/8224
* Fix type reflection when input is an array-like by @jcrist in https://github.com/NVIDIA/cuml/pull/8234
* Update `cuml.tsa` to use new validation by @jcrist in https://github.com/NVIDIA/cuml/pull/8236
* Unify distributed RF quantile samples by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8190
* Remove numba-cuda upper bound by @brandon-b-miller in https://github.com/NVIDIA/cuml/pull/8189
* refactor: switch to `rapids-artifact-name` for consistent artifact naming by @gforsyth in https://github.com/NVIDIA/cuml/pull/8174
* Update cuML sklearn tags for scikit-learn 1.6 by @csadorf in https://github.com/NVIDIA/cuml/pull/8227
* Refactor RF objectives by objective family by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8233
* Update multi-gpu estimators to new input validation by @jcrist in https://github.com/NVIDIA/cuml/pull/8241
* Prepare RF bins and objectives for weights support by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8247
* A few fixups to multi-gpu implementations by @jcrist in https://github.com/NVIDIA/cuml/pull/8249
* Remove obsolete scikit-learn 1.5 compatibility paths by @csadorf in https://github.com/NVIDIA/cuml/pull/8253
* Remove stale sklearn example xfail by @csadorf in https://github.com/NVIDIA/cuml/pull/8259
* Build and test with CUDA 13.3.0 by @bdice in https://github.com/NVIDIA/cuml/pull/8246
* Remove `make_monotonic` and `sorted_unique_labels` by @jcrist in https://github.com/NVIDIA/cuml/pull/8269
* Cleanup `cuml.model_selection` by @jcrist in https://github.com/NVIDIA/cuml/pull/8270
* Fixup CI by @jcrist in https://github.com/NVIDIA/cuml/pull/8280
* Limit high-dimensional cosine KDE property cases by @Federicorao in https://github.com/NVIDIA/cuml/pull/8273
* Validate random forest n_streams by @fallintoplace in https://github.com/NVIDIA/cuml/pull/8263
* Use main shared-workflows branch by @bdice in https://github.com/NVIDIA/cuml/pull/8286
* Mark `test_logreg_l1[42-csr_array]` flaky by @jcrist in https://github.com/NVIDIA/cuml/pull/8289
* Use segmented scan for RF row partitioning by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8257
* Consolidate sparse utilities by @jcrist in https://github.com/NVIDIA/cuml/pull/8288
* Validate LinearSVC n_streams by @fallintoplace in https://github.com/NVIDIA/cuml/pull/8282
* Add C++ RF sample weight training support by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8258
* Center RF split thresholds across empty quantile bins by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8283
* Remove xfail markers for cuml.accel examples by @jinsolp in https://github.com/NVIDIA/cuml/pull/8297
* FIX Make `__repr__` show parameters set to non-default values by @betatim in https://github.com/NVIDIA/cuml/pull/8229
* Avoid hard-coded SHAP test outputs by @viclafargue in https://github.com/NVIDIA/cuml/pull/8264
* CI Run cuml.accel tests in the sklearn nightly test runs by @betatim in https://github.com/NVIDIA/cuml/pull/8219
* Simplify `reflect` decorator by @jcrist in https://github.com/NVIDIA/cuml/pull/8295
* Remove `cuml.internals.input_utils` and remaining legacy ingest paths by @jcrist in https://github.com/NVIDIA/cuml/pull/8293
* Remove RMM make_any_device_resource helper usage by @bdice in https://github.com/NVIDIA/cuml/pull/8197
* skip Cython 3.2.7 by @jameslamb in https://github.com/NVIDIA/cuml/pull/8301
* Update CuPy and NumPy dependency specs by @bdice in https://github.com/NVIDIA/cuml/pull/8287
* Implement weighted bootstrap sampling for RF by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8296
* Add benchmark coverage for missing cuML estimators by @viclafargue in https://github.com/NVIDIA/cuml/pull/8271
* Warn in `cuml.accel` if proxying a package outside a range of tested versions by @jcrist in https://github.com/NVIDIA/cuml/pull/8303
* `cuml.accel` SVM onnx support  by @jcrist in https://github.com/NVIDIA/cuml/pull/8307
* Thread sample weights through RandomForest Python API by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8306
* Add benchmark coverage for missing cuML estimators 2 by @viclafargue in https://github.com/NVIDIA/cuml/pull/8325
* Add benchmark coverage for missing cuML estimators 3 by @viclafargue in https://github.com/NVIDIA/cuml/pull/8341
* switch from 'cuda-python' to specific components (e.g. 'cuda-bindings') by @jameslamb in https://github.com/NVIDIA/cuml/pull/8314
* Handle RF split histogram shared-memory pressure by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8323
* Empty commit to trigger new builds due to RAFT 3052 ABI breakage by @divyegala in https://github.com/NVIDIA/cuml/pull/8353
* fix(naive_bayes): use established classes_ on later partial_fit calls by @nethum529 in https://github.com/NVIDIA/cuml/pull/8316
* Prevent cuVS types from leaking through public headers by @viclafargue in https://github.com/NVIDIA/cuml/pull/8342
* Support row-major RandomForest training inputs by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8324
* Add regression test for cluster-metric label-permutation invariance (#7199) by @aminehd in https://github.com/NVIDIA/cuml/pull/8355
* Prepare RF split counts for distributed training by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8348
* Raise TypeError for invalid sparse input types in benchmark datagen by @fallintoplace in https://github.com/NVIDIA/cuml/pull/8318
* CI Fix bertopic testing by @betatim in https://github.com/NVIDIA/cuml/pull/8298
* Unify reflection system & decorators by @jcrist in https://github.com/NVIDIA/cuml/pull/8339
* Add JSON output for benchmark results by @dantegd in https://github.com/NVIDIA/cuml/pull/8027
* Increase cython-lint line length by @jcrist in https://github.com/NVIDIA/cuml/pull/8374
* A few sklearn compat test fixes by @jcrist in https://github.com/NVIDIA/cuml/pull/8366
* Split random forest histogram building and split scoring kernels by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8370
* Re-enable RF regressor C++ benchmarks by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8371
* Support coercing host arrays in `mlfunc`/`ReflectedAttr` by @jcrist in https://github.com/NVIDIA/cuml/pull/8373
* Finish migration to new decorators/descriptors by @jcrist in https://github.com/NVIDIA/cuml/pull/8378
* Address signedness compiler warning by @chyunsu3 in https://github.com/NVIDIA/cuml/pull/8382
* Add RF histogram allreduce hook for distributed split scoring by @RAMitchell in https://github.com/NVIDIA/cuml/pull/8377
* Enable building with 26.08 nvForest by @chyunsu3 in https://github.com/NVIDIA/cuml/pull/8388
* Remove legacy array and reflection machinery by @jcrist in https://github.com/NVIDIA/cuml/pull/8385
* Update developer wiki for new machinery by @jcrist in https://github.com/NVIDIA/cuml/pull/8386
* Empty commit to trigger a build by @bdice in https://github.com/NVIDIA/cuml/pull/8399
* [FEA] Skip Device Concat for KmeansMG by @tarang-jain in https://github.com/NVIDIA/cuml/pull/8084

## New Contributors
* @Federicorao made their first contribution in https://github.com/NVIDIA/cuml/pull/8273
* @nethum529 made their first contribution in https://github.com/NVIDIA/cuml/pull/8316
* @Husnain987 made their first contribution in https://github.com/NVIDIA/cuml/pull/8292
* @aminehd made their first contribution in https://github.com/NVIDIA/cuml/pull/8355

**Full Changelog**: https://github.com/NVIDIA/cuml/compare/v26.08.00a...release/26.08

# cuml 26.06.00 (3 Jun 2026)

### 🚨 Breaking Changes
* Migrate RMM usage to CCCL MR design by @bdice in https://github.com/NVIDIA/cuml/pull/7951
* Remove unused C API by @divyegala in https://github.com/NVIDIA/cuml/pull/8011
* Cleanup and apply new validation to `cuml.naive_bayes` by @jcrist in https://github.com/NVIDIA/cuml/pull/8051
* Apply new validation to metrics.hinge_loss by @csadorf in https://github.com/NVIDIA/cuml/pull/8060
* Cleanup `LabelEncoder` by @jcrist in https://github.com/NVIDIA/cuml/pull/8039
* Use the PTDS for (most) cupy operations by @jcrist in https://github.com/NVIDIA/cuml/pull/8086
* Deprecate `probability` parameter on `SVC` and `LinearSVC` by @switch527 in https://github.com/NVIDIA/cuml/pull/8089
* Adopt nvForest for random forest inference by @hcho3 in https://github.com/NVIDIA/cuml/pull/8048
### 🐛 Bug Fixes
* Fixes for CCCL 3.4 update by @dantegd in https://github.com/NVIDIA/cuml/pull/7949
* Set unique CUDA cache path for each pytest-xdist worker by @divyegala in https://github.com/NVIDIA/cuml/pull/7948
* Fix IncrementalPCA: handle missing batch_size_ in sparse transform after partial_fit by @csadorf in https://github.com/NVIDIA/cuml/pull/8010
* Drop `__firstlineno__` on `ProxyBase` instances by @jcrist in https://github.com/NVIDIA/cuml/pull/8045
* Fix KMeans int32 index overflow check by @csadorf in https://github.com/NVIDIA/cuml/pull/8067
* Add early validation for n_clusters > n_samples in dask KMeans by @csadorf in https://github.com/NVIDIA/cuml/pull/8071
* Fix CI by changing SVC intercept selection kernel dispatch by @dantegd in https://github.com/NVIDIA/cuml/pull/8077
* Fix FIL buffer bounds assertions by @csadorf in https://github.com/NVIDIA/cuml/pull/8069
* Fix KMeans score test tolerances by @csadorf in https://github.com/NVIDIA/cuml/pull/8099
* Add vertex-parallel kernels to UMAP to enforce sequential behavior  by @jinsolp in https://github.com/NVIDIA/cuml/pull/7974
* Fix wrapping metaestimators in `Pipeline` in `cuml.accel` by @jcrist in https://github.com/NVIDIA/cuml/pull/8115
### 📖 Documentation
* DOC Add third-party app example for cuml.accel by @betatim in https://github.com/NVIDIA/cuml/pull/8094
* DOC Add hyper-parameter search example by @betatim in https://github.com/NVIDIA/cuml/pull/8095
* DOC Add note about onnxruntime bug by @betatim in https://github.com/NVIDIA/cuml/pull/8162
### 🚀 New Features
* Add LSMR solver to Ridge by @jcrist in https://github.com/NVIDIA/cuml/pull/7922
* Add LSMR solver to `LinearRegression` by @jcrist in https://github.com/NVIDIA/cuml/pull/7927
* Add sparse input support to `ElasticNet`/`Lasso` by @jcrist in https://github.com/NVIDIA/cuml/pull/7943
* Support LabelEncoder in `cuml.accel` by @jcrist in https://github.com/NVIDIA/cuml/pull/8082
* Add EmpiricalCovariance estimator by @csadorf in https://github.com/NVIDIA/cuml/pull/8074
### 🛠️ Improvements
* Remove `PoolMemoryResource` wrapper by @jcrist in https://github.com/NVIDIA/cuml/pull/7878
* fix more verify-hardcoded-version findings by @jameslamb in https://github.com/NVIDIA/cuml/pull/7891
* Forward-merge release/26.04 into main by @csadorf in https://github.com/NVIDIA/cuml/pull/7888
* Forward-merge release/26.04 into main by @csadorf in https://github.com/NVIDIA/cuml/pull/7894
* Forward merge from release/26.04 by @jcrist in https://github.com/NVIDIA/cuml/pull/7910
* Update CODEOWNERS by @jcrist in https://github.com/NVIDIA/cuml/pull/7919
* Forward-merge release/26.04 into main by @gforsyth in https://github.com/NVIDIA/cuml/pull/7914
* Forward merge `release/26.04` into `main` by @jcrist in https://github.com/NVIDIA/cuml/pull/7936
* Forward-merge release/26.04 into main by @jameslamb in https://github.com/NVIDIA/cuml/pull/7937
* Remove deprecations from 26.04 by @jcrist in https://github.com/NVIDIA/cuml/pull/7928
* update pip devcontainers' base image tags by @trxcllnt in https://github.com/NVIDIA/cuml/pull/7947
* chore: bump `codespell` version for Python 3.14+ compatibility by @gforsyth in https://github.com/NVIDIA/cuml/pull/7925
* Restart xdist workers on corrupted CUDA context by @jcrist in https://github.com/NVIDIA/cuml/pull/7954
* chore: remove comment in devcontainer.json that breaks jq by @gforsyth in https://github.com/NVIDIA/cuml/pull/7952
* A few linear model test cleanups by @jcrist in https://github.com/NVIDIA/cuml/pull/7957
* Support recent hdbscan by @jcrist in https://github.com/NVIDIA/cuml/pull/7956
* Re-enabling precomputed kNN on host for UMAP by @viclafargue in https://github.com/NVIDIA/cuml/pull/7915
* Add `-v` option to `timeout` in CI scripts by @jcrist in https://github.com/NVIDIA/cuml/pull/7966
* Mark a few cuml-accel upstream tests as flaky by @jcrist in https://github.com/NVIDIA/cuml/pull/7967
* Update to clang 20.1.8 by @bdice in https://github.com/NVIDIA/cuml/pull/7968
* Run all scikit-learn examples under cuml.accel in nightlies by @csadorf in https://github.com/NVIDIA/cuml/pull/7960
* xfail ARIMA test on 'endog_hourly_earnings_by_industry_missing_exog' dataset by @jameslamb in https://github.com/NVIDIA/cuml/pull/7972
* New input validation utilities by @jcrist in https://github.com/NVIDIA/cuml/pull/7973
* Improve timeout verbosity and graceful shutdown for pytest CI jobs by @csadorf in https://github.com/NVIDIA/cuml/pull/7979
* Use new input validation in `cuml.linear_models`/`cuml.solvers` by @jcrist in https://github.com/NVIDIA/cuml/pull/7978
* Apply new validation to `cuml.cluster` by @jcrist in https://github.com/NVIDIA/cuml/pull/7984
* Apply new validation to `cuml.random_projection` by @jcrist in https://github.com/NVIDIA/cuml/pull/8009
* Use new input validation infrastructure for cuml.decomposition. by @csadorf in https://github.com/NVIDIA/cuml/pull/8006
* Apply new input validation to `cuml.multiclass` by @mgrauer in https://github.com/NVIDIA/cuml/pull/8017
* Deprecate max_depth=16 in RandomForest and add None support by @Nzouh in https://github.com/NVIDIA/cuml/pull/7958
* Migrate _classification and _ranking metrics to new input validation infrastructure by @csadorf in https://github.com/NVIDIA/cuml/pull/8012
* CI Fix cuml.svm compatibility with scikit-learn 1.9 deprecation of `probability=` by @betatim in https://github.com/NVIDIA/cuml/pull/7981
* Use scikit-learn's array-api to accelerate `StandardScaler` by @jcrist in https://github.com/NVIDIA/cuml/pull/8020
* Apply new input validation to `cuml.kernel_ridge` by @jcrist in https://github.com/NVIDIA/cuml/pull/8007
* Apply new validation to `cuml.covariance` by @jcrist in https://github.com/NVIDIA/cuml/pull/8008
* [FIL] Validate Treelite model input to prevent integer overflow and OOB memory access by @hcho3 in https://github.com/NVIDIA/cuml/pull/8016
* Cleanup Lars, apply new validation by @jcrist in https://github.com/NVIDIA/cuml/pull/8024
* Add `MinMaxScaler`, `MaxAbsScaler`, and `PolynomialFeatures` to `cuml.accel` by @jcrist in https://github.com/NVIDIA/cuml/pull/8032
* Several `ensemble` scikit-learn compatibility improvements by @jcrist in https://github.com/NVIDIA/cuml/pull/8023
* Remove inertia_check Param from KMeans by @tarang-jain in https://github.com/NVIDIA/cuml/pull/8033
* Fix sample_weight handling in `KernelRidge` by @jcrist in https://github.com/NVIDIA/cuml/pull/8040
* Apply new validation to `cuml.svm` by @jcrist in https://github.com/NVIDIA/cuml/pull/8029
* Support subclassing proxy estimators by @jcrist in https://github.com/NVIDIA/cuml/pull/8041
* fix(ci): resolve all zizmor findings and add zizmor pre-commit checks by @gforsyth in https://github.com/NVIDIA/cuml/pull/8046
* Add `check_cudf`, and a few other validation changes by @jcrist in https://github.com/NVIDIA/cuml/pull/8038
* Fix labeler config by @jcrist in https://github.com/NVIDIA/cuml/pull/8047
* Apply new validation to `cuml.explainer` by @jcrist in https://github.com/NVIDIA/cuml/pull/8043
* Apply new validation to metrics.pairwise_kernels by @csadorf in https://github.com/NVIDIA/cuml/pull/8050
* Apply new validation to `cuml.neighbors` by @viclafargue in https://github.com/NVIDIA/cuml/pull/8019
* Apply new input validation to metrics.regression by @csadorf in https://github.com/NVIDIA/cuml/pull/8044
* Ensure output of classifier `predict` has aligned index by @jcrist in https://github.com/NVIDIA/cuml/pull/8056
* Fix explainer hypothesis test by @jcrist in https://github.com/NVIDIA/cuml/pull/8054
* Apply new validation to metrics.confusion_matrix by @csadorf in https://github.com/NVIDIA/cuml/pull/8049
* Apply new validation to metrics.kl_divergence by @csadorf in https://github.com/NVIDIA/cuml/pull/8057
* Use `token.rapids.nvidia.com` when issuing S3 bucket creds in devcontainers by @trxcllnt in https://github.com/NVIDIA/cuml/pull/8036
* Apply new input validation to cuml.metrics.cluster by @csadorf in https://github.com/NVIDIA/cuml/pull/8059
* Apply new validation to `cuml.manifold` by @jinsolp in https://github.com/NVIDIA/cuml/pull/8042
* Cleanup and apply new validation to `cuml._thirdparty` and `cuml.preprocessing` by @jcrist in https://github.com/NVIDIA/cuml/pull/8052
* Apply new validation to metrics.pairwise_distances by @csadorf in https://github.com/NVIDIA/cuml/pull/8065
* Apply new input validation to metrics.trustworthiness by @csadorf in https://github.com/NVIDIA/cuml/pull/8058
* A few validation fixups by @jcrist in https://github.com/NVIDIA/cuml/pull/8066
* Use cudaStream_t instead of cuda_stream_view in cuml Cython by @vyasr in https://github.com/NVIDIA/cuml/pull/8072
* Fix failing `IncrementalPCA` test by @jcrist in https://github.com/NVIDIA/cuml/pull/8080
* Fallback on `ensure_all_finite` in `UMAP` on `cuml.accel` by @jcrist in https://github.com/NVIDIA/cuml/pull/8076
* Cleanup TargetEncoder by @jcrist in https://github.com/NVIDIA/cuml/pull/8075
* Use static cudart by @KyleFromNVIDIA in https://github.com/NVIDIA/cuml/pull/8078
* Build and test with CUDA 13.2.0 by @bdice in https://github.com/NVIDIA/cuml/pull/8081
* Support umap-learn 0.5.12 by @csadorf in https://github.com/NVIDIA/cuml/pull/8073
* fix(ci): add explicit `actions: write` permission for `telemetry-summarize`
 by @gforsyth in https://github.com/NVIDIA/cuml/pull/8088
* ci: add informational PR release target validation by @csadorf in https://github.com/NVIDIA/cuml/pull/8083
* A few cudf fixups by @jcrist in https://github.com/NVIDIA/cuml/pull/8105
* Require CMake 4.0 by @KyleFromNVIDIA in https://github.com/NVIDIA/cuml/pull/8096
* Mark upstream umap test as flaky by @jcrist in https://github.com/NVIDIA/cuml/pull/8107
* Update `libcuml` wheel to depend on `librmm` by @jcrist in https://github.com/NVIDIA/cuml/pull/8110
* Numpy 1.x compatibility fixes by @jcrist in https://github.com/NVIDIA/cuml/pull/8118
* loosen threshold in test_mbsgd_regressor test by @jameslamb in https://github.com/NVIDIA/cuml/pull/8122
* xfail `test_onnx[RandomForestClassifier]` by @jcrist in https://github.com/NVIDIA/cuml/pull/8127
* Mark `test_precomputed_sparse_transform_on_iris` as flaky by @jcrist in https://github.com/NVIDIA/cuml/pull/8130
* A few fixes for sklearn 1.9 pre-release by @jcrist in https://github.com/NVIDIA/cuml/pull/8126
* Cleanup LabelBinarizer by @jcrist in https://github.com/NVIDIA/cuml/pull/8101
* Further relax `test_nearest_neighbors_pickle` tolerance by @jcrist in https://github.com/NVIDIA/cuml/pull/8136
* Add cuml.accel support for IncrementalPCA by @JohnZed in https://github.com/NVIDIA/cuml/pull/7785
* Add "Advanced Topics" doc by @jcrist in https://github.com/NVIDIA/cuml/pull/8134
* Avoid deprecated functionality in cupy 14.1 by @jcrist in https://github.com/NVIDIA/cuml/pull/8148
* Fix BERTopic integration tests by @jcrist in https://github.com/NVIDIA/cuml/pull/8167
* Refactor kernel_density to use less memory by @Intron7 in https://github.com/NVIDIA/cuml/pull/7833
* skip CuPy 14.1.0 by @jameslamb in https://github.com/NVIDIA/cuml/pull/8179
* Xfail sklearn example requiring plotly by @csadorf in https://github.com/NVIDIA/cuml/pull/8193
* Fixes for sklearn 1.9 release by @jcrist in https://github.com/NVIDIA/cuml/pull/8208
* Handle sklearn example OpenML network failures by @csadorf in https://github.com/NVIDIA/cuml/pull/8205
* Handle sklearn example OpenML dataset lookup failures by @csadorf in https://github.com/NVIDIA/cuml/pull/8216

## New Contributors
* @mgrauer made their first contribution in https://github.com/NVIDIA/cuml/pull/8017
* @Nzouh made their first contribution in https://github.com/NVIDIA/cuml/pull/7958

**Full Changelog**: https://github.com/NVIDIA/cuml/compare/v26.06.00a...release/26.06

# cuml 26.04.00 (8 Apr 2026)

### 🚨 Breaking Changes
* FIX Raise `TypeError` when sparse input is not supported by @betatim in https://github.com/NVIDIA/cuml/pull/7728
* Add `cuml.internals.validation`, `check_is_fitted` checks by @jcrist in https://github.com/NVIDIA/cuml/pull/7868
* Add support for `feature_names_in_` by @jcrist in https://github.com/NVIDIA/cuml/pull/7877
* Validate that X is 2 dimensional by @jcrist in https://github.com/NVIDIA/cuml/pull/7889
### 🐛 Bug Fixes
* CI Fix tests to be compatible with new stricter cudf dtype handling by @betatim in https://github.com/NVIDIA/cuml/pull/7762
* Replace cudf.Series.values_host with .to_numpy(). by @csadorf in https://github.com/NVIDIA/cuml/pull/7772
* Mark two tests as flaky by @csadorf in https://github.com/NVIDIA/cuml/pull/7787
* Mark test test_spectral_clustering_output_shape_type_and_label_count as xfail by @csadorf in https://github.com/NVIDIA/cuml/pull/7791
* Add nvjitlink to cuda-toolkit pip extras for cusparse compatibility by @bdice in https://github.com/NVIDIA/cuml/pull/7794
* Fix cupy-14 compatibility by @aamijar in https://github.com/NVIDIA/cuml/pull/7814
* Fix -Wsign-compare warning in UMAP by @aamijar in https://github.com/NVIDIA/cuml/pull/7803
* Fix thrust header by @aamijar in https://github.com/NVIDIA/cuml/pull/7817
* Xfail some of the one hot encoder dask test. by @csadorf in https://github.com/NVIDIA/cuml/pull/7828
* Fix Dask KMeans performance regression by @viclafargue in https://github.com/NVIDIA/cuml/pull/7819
* Disable codecov reports by @csadorf in https://github.com/NVIDIA/cuml/pull/7852
* Prevent unprefetchable allocation in SVM by @viclafargue in https://github.com/NVIDIA/cuml/pull/7859
* Fix UMAP inverse_transform OOH error by @viclafargue in https://github.com/NVIDIA/cuml/pull/7863
* Fix `AttributeError` when `CumlArray` index is a pandas `RangeIndex` by @viclafargue in https://github.com/NVIDIA/cuml/pull/7865
* Avoid pickling lambda in dask LabelBinarizer pytests by @aamijar in https://github.com/NVIDIA/cuml/pull/7898
* Fix OOM in Dask KMeans by collecting only one model after fit by @viclafargue in https://github.com/NVIDIA/cuml/pull/7908
* Fix Dask estimators crashing when workers receive empty partitions by @viclafargue in https://github.com/NVIDIA/cuml/pull/7807
* xfail `test_umap_outliers` by @jinsolp in https://github.com/NVIDIA/cuml/pull/7938
### 📖 Documentation
* Fix documentation for in-notebook cuml.accel activation. by @csadorf in https://github.com/NVIDIA/cuml/pull/7767
* [DOC] Add missing parameters to UMAP and NearestNeighbors docstrings by @silasmue in https://github.com/NVIDIA/cuml/pull/7632
* docs updates for new RAPIDS branching strategy by @jameslamb in https://github.com/NVIDIA/cuml/pull/7795
* DOC Restructure API documentation by @betatim in https://github.com/NVIDIA/cuml/pull/7798
* Remove deprecated `tree_method=gpu_hist` with XGBoost by @hcho3 in https://github.com/NVIDIA/cuml/pull/7858
### 🚀 New Features
* Add cuml.accel support for StandardScaler by @csadorf in https://github.com/NVIDIA/cuml/pull/7766
* Add `NearestNeighbors.radius_neighbors_graph` by @jcrist in https://github.com/NVIDIA/cuml/pull/7811
* Implement health checks. by @csadorf in https://github.com/NVIDIA/cuml/pull/7854
* `SpectralClustering` in `cuml.accel` by @aamijar in https://github.com/NVIDIA/cuml/pull/7804
### 🛠️ Improvements
* Fix int64 Spectral Embedding by @aamijar in https://github.com/NVIDIA/cuml/pull/7689
* Replace `thrust::tuple` with `cuda::std::tuple` by @miscco in https://github.com/NVIDIA/cuml/pull/7700
* CI: Standardize nightly dependencies testing by @csadorf in https://github.com/NVIDIA/cuml/pull/7684
* Expand and improve xfail list management tooling by @csadorf in https://github.com/NVIDIA/cuml/pull/7607
* CI: Make the cudf-pandas-integration job non-optional by @csadorf in https://github.com/NVIDIA/cuml/pull/7704
* CI: Make the nightly dependency tests pass by @csadorf in https://github.com/NVIDIA/cuml/pull/7703
* Add optional thread-safety guard for OpenBLAS/OpenMP on high-core systems by @dantegd in https://github.com/NVIDIA/cuml/pull/7688
* More std::tuple fixes by @vyasr in https://github.com/NVIDIA/cuml/pull/7707
* CI: Test CuPy nightlies by @csadorf in https://github.com/NVIDIA/cuml/pull/7695
* Forward-merge release/26.02 into main by @csadorf in https://github.com/NVIDIA/cuml/pull/7720
* Ignore warnings caused by sklearn/scipy incompatibility. by @csadorf in https://github.com/NVIDIA/cuml/pull/7735
* Bump the ipython test dependencies. by @csadorf in https://github.com/NVIDIA/cuml/pull/7737
* Drop Python 3.10 support by @gforsyth in https://github.com/NVIDIA/cuml/pull/7731
* tighten wheel size limits, expand CI-skipping logic, other small build changes by @jameslamb in https://github.com/NVIDIA/cuml/pull/7738
* Remove numba-cuda upper bound by @brandon-b-miller in https://github.com/NVIDIA/cuml/pull/7706
* remove pip.conf migration code in CI scripts, update CI-skipping rules by @jameslamb in https://github.com/NVIDIA/cuml/pull/7747
* Use verify-hardcoded-version pre-commit hook by @KyleFromNVIDIA in https://github.com/NVIDIA/cuml/pull/7726
* Add instructions for CodeRabbit by @csadorf in https://github.com/NVIDIA/cuml/pull/7725
* CI Use ipython 7.34.0 in "oldest" version CI by @betatim in https://github.com/NVIDIA/cuml/pull/7743
* Remove deprecations from 26.02 by @jcrist in https://github.com/NVIDIA/cuml/pull/7761
* Remove deprecated `handle` argument by @jcrist in https://github.com/NVIDIA/cuml/pull/7751
* CI Switch to strict xfail policy for common estimator checks by @betatim in https://github.com/NVIDIA/cuml/pull/7753
* CI Update xfail list by @betatim in https://github.com/NVIDIA/cuml/pull/7768
* refactor: build wheels and conda packages using Python limited API by @gforsyth in https://github.com/NVIDIA/cuml/pull/7758
* Partially revert #7715. by @csadorf in https://github.com/NVIDIA/cuml/pull/7773
* remove gitutils by @jameslamb in https://github.com/NVIDIA/cuml/pull/7784
* Use GHA id-token for `sccache-dist` auth token by @trxcllnt in https://github.com/NVIDIA/cuml/pull/7790
* FIX Raise explicit exception for input with complex dtype by @betatim in https://github.com/NVIDIA/cuml/pull/7729
* Add option to force serial behavior in UMAP optimize to avoid outliers by @jinsolp in https://github.com/NVIDIA/cuml/pull/7793
* UMAP hypothesis testing by @viclafargue in https://github.com/NVIDIA/cuml/pull/7323
* Update raft headers by @aamijar in https://github.com/NVIDIA/cuml/pull/7752
* Remove raft `math.cuh` and `matrix.cuh` by @aamijar in https://github.com/NVIDIA/cuml/pull/7797
* Replace uses of `thrust::transform_iterator` with `cuda::transform_iterator` by @miscco in https://github.com/NVIDIA/cuml/pull/7808
* check-nightly-ci: update to new version by @jameslamb in https://github.com/NVIDIA/cuml/pull/7810
* check-nightly-ci: remove testing config by @jameslamb in https://github.com/NVIDIA/cuml/pull/7818
* Remove some dead code by @jcrist in https://github.com/NVIDIA/cuml/pull/7825
* Drop `thrust/functional` it is almost not used at all by @miscco in https://github.com/NVIDIA/cuml/pull/7823
* Add warning about the use of pickle for model deserialization to README. by @csadorf in https://github.com/NVIDIA/cuml/pull/7829
* CI Fix testing references to take into account string dtype by @betatim in https://github.com/NVIDIA/cuml/pull/7830
* DOC Add basic tests for ONNX support by @betatim in https://github.com/NVIDIA/cuml/pull/7805
* refactor(limited api): add explicit `wheel.py-api` to `pyproject.toml`
 by @gforsyth in https://github.com/NVIDIA/cuml/pull/7832
* Remove workaround for old cupy.percentile bug by @wphicks in https://github.com/NVIDIA/cuml/pull/7789
* Update Cython lower bound pin to 3.2.2 by @vyasr in https://github.com/NVIDIA/cuml/pull/7836
* Optimize data transfer for `Pipeline` in `cuml.accel` by @jcrist in https://github.com/NVIDIA/cuml/pull/7835
* Allow using cuML benchmark tools on systems without cuML installed by @dantegd in https://github.com/NVIDIA/cuml/pull/7593
* Fix integer overflow in FIL by @hcho3 in https://github.com/NVIDIA/cuml/pull/7727
* Relax UMAP version constraint by @csadorf in https://github.com/NVIDIA/cuml/pull/7850
* cuml.accel's pth needs to go after the `numba_cuda` one by @betatim in https://github.com/NVIDIA/cuml/pull/7872
* Use `PrefetchResourceAdaptor` in `cuml.accel` by @jcrist in https://github.com/NVIDIA/cuml/pull/7834
* Add support for Python 3.14 by @gforsyth in https://github.com/NVIDIA/cuml/pull/7831
* Add `check_features` by @jcrist in https://github.com/NVIDIA/cuml/pull/7874
* fix verify-hardcoded-version issues by @jameslamb in https://github.com/NVIDIA/cuml/pull/7882
* Cherry-pick "Remove `PoolMemoryResource` wrapper" into release by @jcrist in https://github.com/NVIDIA/cuml/pull/7881
* Update Treelite to 4.7.0 by @hcho3 in https://github.com/NVIDIA/cuml/pull/7870
* Remove deprecation warning in `TargetEncoder` by @jcrist in https://github.com/NVIDIA/cuml/pull/7892
* Remove xfailed cudf pandas tests by @jcrist in https://github.com/NVIDIA/cuml/pull/7901
* Cap numba-cuda upper bound at <0.29.0 by @brandon-b-miller in https://github.com/NVIDIA/cuml/pull/7900
* Remove no_bad_cuml_array_check by @csadorf in https://github.com/NVIDIA/cuml/pull/7887
* Bump scipy minimum required version to 1.14.0. by @csadorf in https://github.com/NVIDIA/cuml/pull/7857
* fix(devcontainers): build with Python 3.13 while dependencies are unavailable by @gforsyth in https://github.com/NVIDIA/cuml/pull/7916
* build wheels with CUDA 13.0.x, test wheels against mix of CTK versions by @jameslamb in https://github.com/NVIDIA/cuml/pull/7907
* Fix `CDMG` xfails for numpy 2.4 by @jcrist in https://github.com/NVIDIA/cuml/pull/7918
* Use PCA and TSVD from raft by @aamijar in https://github.com/NVIDIA/cuml/pull/7802
* ci: preserve Release project field also on release/* PRs by @csadorf in https://github.com/NVIDIA/cuml/pull/7921
* Small update to cuml-accel limitations by @jcrist in https://github.com/NVIDIA/cuml/pull/7929
* Pin openblas on aarch64 by @jcrist in https://github.com/NVIDIA/cuml/pull/7933
* Disable SciPy intersphinx by @jcrist in https://github.com/NVIDIA/cuml/pull/7935
* Support unlimited_depth for the random forest estimators by @csadorf in https://github.com/NVIDIA/cuml/pull/7895

## New Contributors
* @silasmue made their first contribution in https://github.com/NVIDIA/cuml/pull/7632

**Full Changelog**: https://github.com/NVIDIA/cuml/compare/v26.04.00a...release/26.04

# cuml 26.02.00 (4 Feb 2026)

### 🚨 Breaking Changes
* Simplify type reflection implementation by @jcrist in https://github.com/NVIDIA/cuml/pull/7539
* Deprecate `TotalIters` for SVC/SVR by @jcrist in https://github.com/NVIDIA/cuml/pull/7612
* Deprecate handle from public APIs by @jcrist in https://github.com/NVIDIA/cuml/pull/7628
* Don't coerce `output_type=None` to global `output_type` in `__init__` by @jcrist in https://github.com/NVIDIA/cuml/pull/7649
* Remove unused QR decomposition MG code by @JohnZed in https://github.com/NVIDIA/cuml/pull/7687
* Use train_test_split from scikit-learn by @betatim in https://github.com/NVIDIA/cuml/pull/7667
### 🐛 Bug Fixes
* Don't link to cumlprims_mg when MG is turned off by @robertmaynard in https://github.com/NVIDIA/cuml/pull/7583
* UMAP fix int32 overflow causing illegal mem access by @aamijar in https://github.com/NVIDIA/cuml/pull/7587
* Explicitly initialize uninitialized SvmParameter fields in SVC constructor by @dantegd in https://github.com/NVIDIA/cuml/pull/7604
* Use a generated dataset in place of the actual Covertype dataset. by @csadorf in https://github.com/NVIDIA/cuml/pull/7634
* Remove reliance on remote datasets in tests by @csadorf in https://github.com/NVIDIA/cuml/pull/7637
* Use synthetic dataset in BERTopic wheel integration tests. by @csadorf in https://github.com/NVIDIA/cuml/pull/7644
* Refactor pointer comparison in check_ptr function for input validation by @csadorf in https://github.com/NVIDIA/cuml/pull/7636
* Add XGBoost back to tests by @dantegd in https://github.com/NVIDIA/cuml/pull/7653
* xfail check_estimators spectral embedding tests by @aamijar in https://github.com/NVIDIA/cuml/pull/7677
* Update RMM memory resource APIs to use ref-based equivalents by @bdice in https://github.com/NVIDIA/cuml/pull/7668
* Xfail test 'test_tsne_distance_metrics_on_sparse_input' by @csadorf in https://github.com/NVIDIA/cuml/pull/7681
* Use NCCL for Dask kNN and DBSCAN by @viclafargue in https://github.com/NVIDIA/cuml/pull/7574
* Mark some SpectralEmbedding() common test_estimators tests as flaky. by @csadorf in https://github.com/NVIDIA/cuml/pull/7686
* Handle new sklearn LARS error message in test_typeerror_input by @csadorf in https://github.com/NVIDIA/cuml/pull/7694
* Build.sh now support custom LIBCUML_BUILD_DIR value by @robertmaynard in https://github.com/NVIDIA/cuml/pull/7582
* Fix UMAP outliers when `random_state` is given by @jinsolp in https://github.com/NVIDIA/cuml/pull/7597
* Work-around sentence-transformer regression in BERTopic integration tests and do not fail test runs on integration tests by @csadorf in https://github.com/NVIDIA/cuml/pull/7715
* classlabels utils refactor by @viclafargue in https://github.com/NVIDIA/cuml/pull/7691
### 📖 Documentation
* DOC Rework LinearRegression documentation by @virchan in https://github.com/NVIDIA/cuml/pull/7218
* `k-means++` documented as a init option by @aamijar in https://github.com/NVIDIA/cuml/pull/7615
* Add general XGBoost demo notebook by @dantegd in https://github.com/NVIDIA/cuml/pull/7601
* Add security warnings to model serialization/deserialization notebook. by @csadorf in https://github.com/NVIDIA/cuml/pull/7661
### 🚀 New Features
* Use CCCL's mdspan implementation by @divyegala in https://github.com/NVIDIA/cuml/pull/7555
* `TargetEncoder` in cuml.accel by @aamijar in https://github.com/NVIDIA/cuml/pull/7476
### 🛠️ Improvements
* Forward merge `release/25.12` into `main` by @jcrist in https://github.com/NVIDIA/cuml/pull/7519
* Use strict priority in CI conda tests by @bdice in https://github.com/NVIDIA/cuml/pull/7524
* CI: Do not automatically update the release field of linked issues. by @csadorf in https://github.com/NVIDIA/cuml/pull/7521
* Remove verbose descriptor by @jcrist in https://github.com/NVIDIA/cuml/pull/7527
* Forward merge release/25.12 into main by @jcrist in https://github.com/NVIDIA/cuml/pull/7531
* CI: Improve PR target release automation by @csadorf in https://github.com/NVIDIA/cuml/pull/7511
* CI: Add or adjust job-specific timeouts for GHA test jobs by @csadorf in https://github.com/NVIDIA/cuml/pull/7533
* Forward-merge release/25.12 into main by @csadorf in https://github.com/NVIDIA/cuml/pull/7537
* CI: Fail PR auto assignment gracefully. by @csadorf in https://github.com/NVIDIA/cuml/pull/7558
* Use strict priority in CI conda tests by @bdice in https://github.com/NVIDIA/cuml/pull/7559
* Adjust CI timeouts by @jcrist in https://github.com/NVIDIA/cuml/pull/7563
* use CUDA-suffixed XGBoost wheels by @jameslamb in https://github.com/NVIDIA/cuml/pull/7550
* MNT: Remove unused legacy spectral.cu by @mani-builds in https://github.com/NVIDIA/cuml/pull/7556
* Remove alpha specs from non-RAPIDS dependencies by @bdice in https://github.com/NVIDIA/cuml/pull/7568
* Enable merge barriers by @KyleFromNVIDIA in https://github.com/NVIDIA/cuml/pull/7579
* Update to numba-cuda `>=0.22.1,<0.23.0` by @brandon-b-miller in https://github.com/NVIDIA/cuml/pull/7549
* Forward merge `release/25.12` into `main` by @jcrist in https://github.com/NVIDIA/cuml/pull/7592
* Add devcontainer fallback for C++ test location by @bdice in https://github.com/NVIDIA/cuml/pull/7596
* Add lower bound to dask-ml>=2024 by @bdice in https://github.com/NVIDIA/cuml/pull/7600
* Forward `cuml.accel` logging to subprocesses by @jcrist in https://github.com/NVIDIA/cuml/pull/7602
* Improved condense hierarchy of HDBSCAN by @jinsolp in https://github.com/NVIDIA/cuml/pull/7459
* Isolating `test_umap_outliers` from precomputed kNN issues by @jinsolp in https://github.com/NVIDIA/cuml/pull/7594
* Fix scikit-learn 1.8 test incompatibilities by @csadorf in https://github.com/NVIDIA/cuml/pull/7589
* Prepare cuml for removal of deprecated raft apis by @aamijar in https://github.com/NVIDIA/cuml/pull/7561
* Require scipy>=1.13.0. by @csadorf in https://github.com/NVIDIA/cuml/pull/7605
* Remove deprecations for 26.02 by @jcrist in https://github.com/NVIDIA/cuml/pull/7613
* Remove deprecated `normalize` parameter by @jcrist in https://github.com/NVIDIA/cuml/pull/7611
* Update cuML to C++20 by @divyegala in https://github.com/NVIDIA/cuml/pull/7619
* Avoid deprecated conversion of numpy array->scalar by @jcrist in https://github.com/NVIDIA/cuml/pull/7618
* Updating UMAP random_state doc by @viclafargue in https://github.com/NVIDIA/cuml/pull/7590
* Add warning in UMAP for nondeterministic behavior when `build_algo=nn_descent` by @jinsolp in https://github.com/NVIDIA/cuml/pull/7620
* FIX Remove custom CUDA kernels from naive bayes estimators by @betatim in https://github.com/NVIDIA/cuml/pull/7424
* Bypass data allocation to GPU given precomputed knn in UMAP by @jinsolp in https://github.com/NVIDIA/cuml/pull/7566
* Attempt to find `cmake-format-rapids-cmake.json` in `cpp/latest` by @divyegala in https://github.com/NVIDIA/cuml/pull/7626
* Optional seed spectral embedding by @aamijar in https://github.com/NVIDIA/cuml/pull/7608
* Cleanup naive bayes by @jcrist in https://github.com/NVIDIA/cuml/pull/7623
* Don't fail on `cuda.core.experimental` deprecation warnings by @jcrist in https://github.com/NVIDIA/cuml/pull/7629
* Empty commit to trigger a build by @bdice in https://github.com/NVIDIA/cuml/pull/7630
* Move cumlprims_mg into cuML by @dantegd in https://github.com/NVIDIA/cuml/pull/7585
* CI Make nightly cuml.accel integration test stricter by @betatim in https://github.com/NVIDIA/cuml/pull/7631
* Use SPDX license identifiers in pyproject.toml, bump build dependency floors by @jameslamb in https://github.com/NVIDIA/cuml/pull/7641
* Require scikit-learn>=1.5 by @csadorf in https://github.com/NVIDIA/cuml/pull/7606
* remove unused test dependency 'tenacity' by @jameslamb in https://github.com/NVIDIA/cuml/pull/7645
* Add CUDA 13.1 support by @bdice in https://github.com/NVIDIA/cuml/pull/7610
* Remove cumlprims from clang-format by @bdice in https://github.com/NVIDIA/cuml/pull/7647
* Use cuvs spectral clustering dataset api by @aamijar in https://github.com/NVIDIA/cuml/pull/7625
* CI: Add notebook tests to nightly tests by @csadorf in https://github.com/NVIDIA/cuml/pull/7567
* Address sklearn deprecation warnings by @csadorf in https://github.com/NVIDIA/cuml/pull/7371
* Remove cutlass reference in unused header by @divyegala in https://github.com/NVIDIA/cuml/pull/7655
* Add LedoitWolf covariance estimator by @will-hill in https://github.com/NVIDIA/cuml/pull/7548
* Update input_utils CI tests to account for cuDF update by @viclafargue in https://github.com/NVIDIA/cuml/pull/7660
* build and test against CUDA 13.1.0 by @jameslamb in https://github.com/NVIDIA/cuml/pull/7650
* Empty commit to trigger a build by @jameslamb in https://github.com/NVIDIA/cuml/pull/7666
* Use main shared-workflows branch by @jameslamb in https://github.com/NVIDIA/cuml/pull/7682
* Use saga solver in test_logistic_regression test function. by @csadorf in https://github.com/NVIDIA/cuml/pull/7670
* Mirror input mem type for `nn_descent` algo in HDBSCAN by @jinsolp in https://github.com/NVIDIA/cuml/pull/7614
* Avoid installing CCCL headers in wheels by @divyegala in https://github.com/NVIDIA/cuml/pull/7662
* CI Add a nightly job that uses scikit-learn's nightlies by @betatim in https://github.com/NVIDIA/cuml/pull/7621
* UMAP with `int64_t` spectral initialization by @aamijar in https://github.com/NVIDIA/cuml/pull/7586
* UMAP custom init by @viclafargue in https://github.com/NVIDIA/cuml/pull/7651
* Support older IPython versions for `cuml.accel` magics by @jcrist in https://github.com/NVIDIA/cuml/pull/7675
* wheel builds: react to changes in pip's handling of build constraints by @mmccarty in https://github.com/NVIDIA/cuml/pull/7693
* Use all-neighbors brute force in UMAP and deprecate `build_kwds` args to be consistent with HDBSCAN by @jinsolp in https://github.com/NVIDIA/cuml/pull/7698
* fix(build): build package on merge to `release/*` branch by @gforsyth in https://github.com/NVIDIA/cuml/pull/7719
* UMAP inverse transform by @viclafargue in https://github.com/NVIDIA/cuml/pull/7674
* Fallback to numba-cuda with no extra CUDA packages if 'cuda_suffixed' isn't true by @trxcllnt in https://github.com/NVIDIA/cuml/pull/7721
* Add precomputed kernel support in SVM by @viclafargue in https://github.com/NVIDIA/cuml/pull/7616
* Document support for and limitations of TargetEncoder in cuml.accel. by @csadorf in https://github.com/NVIDIA/cuml/pull/7741

## New Contributors
* @virchan made their first contribution in https://github.com/NVIDIA/cuml/pull/7218
* @mani-builds made their first contribution in https://github.com/NVIDIA/cuml/pull/7556
* @will-hill made their first contribution in https://github.com/NVIDIA/cuml/pull/7548
* @mmccarty made their first contribution in https://github.com/NVIDIA/cuml/pull/7693

**Full Changelog**: https://github.com/NVIDIA/cuml/compare/v26.02.00a...release/26.02

# cuml 25.12.00 (10 Dec 2025)

### 🚨 Breaking Changes
* Make `dask` an optional dependency by @jcrist in https://github.com/NVIDIA/cuml/pull/7303
* Using `int64_t` for HDBSCAN and dropping `int` by @jinsolp in https://github.com/NVIDIA/cuml/pull/7104
* UMAP - delay embedding allocation by @viclafargue in https://github.com/NVIDIA/cuml/pull/7313
* Don't infer dtype from model in `cuml.explainer` by @jcrist in https://github.com/NVIDIA/cuml/pull/7358
* Cleanup LinearRegression by @jcrist in https://github.com/NVIDIA/cuml/pull/7355
* Cleanup `AgglomerativeClustering` by @jcrist in https://github.com/NVIDIA/cuml/pull/7379
* Cleanup `LinearSVC`/`LinearSVR` by @jcrist in https://github.com/NVIDIA/cuml/pull/7376
* Require CUDA 12.2+ by @jakirkham in https://github.com/NVIDIA/cuml/pull/7408
* Several improvements to `Ridge` by @jcrist in https://github.com/NVIDIA/cuml/pull/7410
* Expose `n_iter_` in SVC/SVR by @jcrist in https://github.com/NVIDIA/cuml/pull/7461
* Standardize regression `predict` output dtype by @jcrist in https://github.com/NVIDIA/cuml/pull/7464
* Support non-numeric class labels everywhere by @jcrist in https://github.com/NVIDIA/cuml/pull/7480
* Rename TSNE `n_iter` to `max_iter` by @jcrist in https://github.com/NVIDIA/cuml/pull/7500
* Cleanup `cuml.multiclass` by @jcrist in https://github.com/NVIDIA/cuml/pull/7508
* Cleanup `SGD`/`MBSGDClassifier`/`MBSGDRegressor` by @jcrist in https://github.com/NVIDIA/cuml/pull/7504
### 🐛 Bug Fixes
* Avoid pyyaml runtime dependency. by @csadorf in https://github.com/NVIDIA/cuml/pull/7312
* Make conversion from scikit-learn SVM to cuML more robust by @csadorf in https://github.com/NVIDIA/cuml/pull/7324
* Revert "CI: Auto-assign priority to PRs from linked issues (#7354)" by @csadorf in https://github.com/NVIDIA/cuml/pull/7357
* Adding checks for UMAP precomputed KNN feature by @viclafargue in https://github.com/NVIDIA/cuml/pull/7300
* Unpin treelite and xfail some tests affected by the XGBoost/treelite incompatibility by @csadorf in https://github.com/NVIDIA/cuml/pull/7366
* Fix issue in Dask DBSCAN by @viclafargue in https://github.com/NVIDIA/cuml/pull/7359
* CI: Increase nightly CI check window to 14 days. by @csadorf in https://github.com/NVIDIA/cuml/pull/7392
* Fix overflow in coordinate descent by @jcrist in https://github.com/NVIDIA/cuml/pull/7399
* Fix output argument passed by value causing the copy constructor on owning CSR matrix by @achirkin in https://github.com/NVIDIA/cuml/pull/7390
* Fix Dask DBSCAN tree reduction by @viclafargue in https://github.com/NVIDIA/cuml/pull/7398
* Fixup the test_kernel_density test. by @csadorf in https://github.com/NVIDIA/cuml/pull/7404
* Fetch rapids-cmake configuration for cmake-format in CI style checks by @bdice in https://github.com/NVIDIA/cuml/pull/7406
* Fix string slicing dtype mismatch in PorterStemmer for cudf-pandas mode by @csadorf in https://github.com/NVIDIA/cuml/pull/7413
* Set the same `random_state` value for `None` in `cuml.dask.KMeans` for all workers by @divyegala in https://github.com/NVIDIA/cuml/pull/7417
* Init checks for Dask KMeans by @viclafargue in https://github.com/NVIDIA/cuml/pull/7391
* Fix `DBSCAN` overflow by @divyegala in https://github.com/NVIDIA/cuml/pull/7421
* Fix HDBSCAN nn-descent errors with default arguments (min_samples=None) by @viclafargue in https://github.com/NVIDIA/cuml/pull/7402
* CI: Xfail test_umap_outliers to mitigate flaky test failure. by @csadorf in https://github.com/NVIDIA/cuml/pull/7430
* Ensure `n_components < n_cols` in `cuml.dask.tsvd` by @divyegala in https://github.com/NVIDIA/cuml/pull/7441
* Skip test_extract_partitions_shape by @csadorf in https://github.com/NVIDIA/cuml/pull/7453
* refactored update-version.sh to handle new branching strategy by @rockhowse in https://github.com/NVIDIA/cuml/pull/7455
* [BUG] Fix RandomForest Builder Sampling by @tarang-jain in https://github.com/NVIDIA/cuml/pull/7422
* Fix the RandomForest sampling bias by @csadorf in https://github.com/NVIDIA/cuml/pull/7449
* Pin Cython pre-3.2.0 and PyTest pre-9 by @jakirkham in https://github.com/NVIDIA/cuml/pull/7472
* Remove SpectralEmbedding from xfail list by @csadorf in https://github.com/NVIDIA/cuml/pull/7487
* Fix UMAP Outliers for small data by @jinsolp in https://github.com/NVIDIA/cuml/pull/7473
* Fix PCA sign flip by @zhuxr11 in https://github.com/NVIDIA/cuml/pull/7331
* A few small fixes for multi-gpu `PCA`/`TruncatedSVD` by @jcrist in https://github.com/NVIDIA/cuml/pull/7560
* Remove unnecessary `syncthreads` in UMAP optimize by @jinsolp in https://github.com/NVIDIA/cuml/pull/7575
* Revert precomputed kNN on CPU for UMAP by @jinsolp in https://github.com/NVIDIA/cuml/pull/7576
* Pin scikit-learn<1.8 in test dependencies by @csadorf in https://github.com/NVIDIA/cuml/pull/7588
### 📖 Documentation
* Add Spectral Embedding algorithm to README.md by @aamijar in https://github.com/NVIDIA/cuml/pull/7348
* Fix and improve build docs by @csadorf in https://github.com/NVIDIA/cuml/pull/7383
* Mention pip as install path in README. by @csadorf in https://github.com/NVIDIA/cuml/pull/7414
* Use current system architecture in conda environment creation command by @bdice in https://github.com/NVIDIA/cuml/pull/7447
* Use `sklearn.datasets.fetch_covtype` in example notebook by @jcrist in https://github.com/NVIDIA/cuml/pull/7581
### 🚀 New Features
* Provide a static target for `libcuml` by @divyegala in https://github.com/NVIDIA/cuml/pull/7351
* HDBSCAN with NN Descent build option by @jinsolp in https://github.com/NVIDIA/cuml/pull/7339
* [FEA] Implement kfold. by @trivialfis in https://github.com/NVIDIA/cuml/pull/7296
* [FEA] Feature Importances for Random Forests by @tarang-jain in https://github.com/NVIDIA/cuml/pull/7275
* SpectralClustering estimator by @aamijar in https://github.com/NVIDIA/cuml/pull/7372
### 🛠️ Improvements
* Update `RAPIDS_BRANCH`, codify changes in `update-version.sh` by @KyleFromNVIDIA in https://github.com/NVIDIA/cuml/pull/7268
* Avoid usage of deprecated cudf `from_pandas` by @TomAugspurger in https://github.com/NVIDIA/cuml/pull/7284
* Branch 25.12 merge branch 25.10 by @jcrist in https://github.com/NVIDIA/cuml/pull/7291
* Update to rapids-logger 0.2 by @bdice in https://github.com/NVIDIA/cuml/pull/7301
* Cleanup TSNE python implementation by @jcrist in https://github.com/NVIDIA/cuml/pull/7311
* Cleanup `cuml.decomposition` by @jcrist in https://github.com/NVIDIA/cuml/pull/7316
* A few HDBSCAN cleanups by @jcrist in https://github.com/NVIDIA/cuml/pull/7319
* Run the scikit-learn test suite in parallel again by @csadorf in https://github.com/NVIDIA/cuml/pull/7315
* A few `cuml.neighbors` cleanups by @jcrist in https://github.com/NVIDIA/cuml/pull/7320
* Cleanup DBSCAN by @jcrist in https://github.com/NVIDIA/cuml/pull/7325
* Release GIL in `SpectralEmbedding` by @jcrist in https://github.com/NVIDIA/cuml/pull/7326
* Cleanup `*.pxd` files and `cimport`s by @jcrist in https://github.com/NVIDIA/cuml/pull/7327
* Accelerate linear model predict on C-ordered inputs by @jcrist in https://github.com/NVIDIA/cuml/pull/7329
* Remove `cuml.ensemble`/`cuml.dask.ensemble` deprecated features by @jcrist in https://github.com/NVIDIA/cuml/pull/7332
* Cleanup `Ridge` by @jcrist in https://github.com/NVIDIA/cuml/pull/7330
* Add SparseRandomProjection, AgglomerativeClustering and GaussianRandomProjection to common checks by @betatim in https://github.com/NVIDIA/cuml/pull/7307
* Enable `sccache-dist` connection pool by @trxcllnt in https://github.com/NVIDIA/cuml/pull/7344
* Use pinned_host_memory_resource instead of pinned_memory_resource. by @bdice in https://github.com/NVIDIA/cuml/pull/7340
* Propagate `Dask`/`UCX` exceptions by @viclafargue in https://github.com/NVIDIA/cuml/pull/7308
* Add `libcuml` linkage smoke test by @divyegala in https://github.com/NVIDIA/cuml/pull/7343
* Cleanup SVM by @jcrist in https://github.com/NVIDIA/cuml/pull/7347
* Remove outdated link flags from CMakeLists.txt by @hcho3 in https://github.com/NVIDIA/cuml/pull/7349
* Fix flaky `test_logistic_regression_weighting` test by @csadorf in https://github.com/NVIDIA/cuml/pull/7321
* CI: Auto-assign priority to PRs from linked issues by @csadorf in https://github.com/NVIDIA/cuml/pull/7354
* Reduce warnings verbosity in tests by @csadorf in https://github.com/NVIDIA/cuml/pull/7322
* Add xgboost to benchmark utilities by @dantegd in https://github.com/NVIDIA/cuml/pull/7350
* Use main in RAPIDS_BRANCH by @bdice in https://github.com/NVIDIA/cuml/pull/7364
* Use main shared-workflows branch by @bdice in https://github.com/NVIDIA/cuml/pull/7369
* Use SPDX for all copyright headers by @KyleFromNVIDIA in https://github.com/NVIDIA/cuml/pull/7373
* Add commit to use SPDX copyright headers to git ignore blame revs. by @csadorf in https://github.com/NVIDIA/cuml/pull/7381
* CI: Fix the auto-release assignment GA job. by @csadorf in https://github.com/NVIDIA/cuml/pull/7380
* Cleanup `ElasticNet`, `Lasso`, and `CD` by @jcrist in https://github.com/NVIDIA/cuml/pull/7382
* Add cuVS to python test dependencies by @jinsolp in https://github.com/NVIDIA/cuml/pull/7385
* Support `KernelDensity` in `cuml.accel` by @jcrist in https://github.com/NVIDIA/cuml/pull/7397
* Deprecate `normalize` option to all linear models by @jcrist in https://github.com/NVIDIA/cuml/pull/7415
* Expose `n_iter_` in `LinearSVC`/`LinearSVR` by @jcrist in https://github.com/NVIDIA/cuml/pull/7419
* Bring back `KMeansMG` by @jcrist in https://github.com/NVIDIA/cuml/pull/7420
* Remove reachability.hpp Header by @tarang-jain in https://github.com/NVIDIA/cuml/pull/7416
* Add Out-of-Bag (OOB) Score Support to RandomForest by @csadorf in https://github.com/NVIDIA/cuml/pull/7401
* Add Third-Party Integration Tests for BERTopic by @csadorf in https://github.com/NVIDIA/cuml/pull/7388
* Expose `_fit_X` and `_y` on neighbors estimators in cuml.accel by @jcrist in https://github.com/NVIDIA/cuml/pull/7437
* Refactor pytests spectral embedding by @aamijar in https://github.com/NVIDIA/cuml/pull/7435
* Remove shims for CCCL < 3.1 compatibility by @bdice in https://github.com/NVIDIA/cuml/pull/7432
* Cleanup `LogisticRegression`/`QN`/`LogisticRegressionMG` by @jcrist in https://github.com/NVIDIA/cuml/pull/7433
* Remove `double` instantiations for `cuml.explainer` kernels by @aamijar in https://github.com/NVIDIA/cuml/pull/7384
* Migrate to new CCCL memory resource interface by @bdice in https://github.com/NVIDIA/cuml/pull/7450
* Expose `n_iter_` in `Ridge` by @jcrist in https://github.com/NVIDIA/cuml/pull/7458
* Expose `n_iter_` on `ElasticNet`/`Lasso`/`CD` by @jcrist in https://github.com/NVIDIA/cuml/pull/7457
* Support of `weights` argument in KNeighbors by @viclafargue in https://github.com/NVIDIA/cuml/pull/7289
* Pin pytest<9.0 by @jcrist in https://github.com/NVIDIA/cuml/pull/7468
* Cleanup `UMAP` and `simpl_set` by @jcrist in https://github.com/NVIDIA/cuml/pull/7456
* Remove cupy allocator utils by @jcrist in https://github.com/NVIDIA/cuml/pull/7463
* Make `cuml.manifold.umap.find_ab_params` public by @jcrist in https://github.com/NVIDIA/cuml/pull/7477
* Assorted classifier improvements by @jcrist in https://github.com/NVIDIA/cuml/pull/7475
* Improve sparse support in `cuml.accel` by @jcrist in https://github.com/NVIDIA/cuml/pull/7462
* Remove Extra Test in Random Forest by @tarang-jain in https://github.com/NVIDIA/cuml/pull/7485
* Update RMM includes from `<rmm/mr/device/*>` to `<rmm/mr/*>` by @bdice in https://github.com/NVIDIA/cuml/pull/7483
* Use ruff-check, ruff-format instead of black, flake8, isort by @KyleFromNVIDIA in https://github.com/NVIDIA/cuml/pull/7429
* Add ruff formatting migration to .git-blame-ignore-revs by @csadorf in https://github.com/NVIDIA/cuml/pull/7489
* Bump Treelite to 4.6.1 by @hcho3 in https://github.com/NVIDIA/cuml/pull/7471
* Document `SVC(probability=True)` requires 5 samples per class by @jcrist in https://github.com/NVIDIA/cuml/pull/7491
* Expose `penalty_normalized` and `lbfgs_memory` in `LogisticRegression` by @jcrist in https://github.com/NVIDIA/cuml/pull/7490
* Remove caveat in cuml.accel docs about PCA/TSVD sign by @jcrist in https://github.com/NVIDIA/cuml/pull/7492
* Fixup `test_serialize` strides check by @jcrist in https://github.com/NVIDIA/cuml/pull/7497
* Use `sccache-dist` build cluster for conda and wheel builds by @trxcllnt in https://github.com/NVIDIA/cuml/pull/7440
* Remove global, contextual, and estimator-level `memory_type` settings by @jcrist in https://github.com/NVIDIA/cuml/pull/7501
* Use s3 for preprocessor cache by @trxcllnt in https://github.com/NVIDIA/cuml/pull/7510
* Update UMAP `random_state` doc by @jcrist in https://github.com/NVIDIA/cuml/pull/7518
* Remove `target_dtype` and supporting infra by @jcrist in https://github.com/NVIDIA/cuml/pull/7516
* Improved memory efficiency in UMAP given precomputed knn graphs by @jinsolp in https://github.com/NVIDIA/cuml/pull/7481
* Temporarily drop xgboost from CI by @jcrist in https://github.com/NVIDIA/cuml/pull/7526
* Improve Dask documentation and fix a few minor issues by @csadorf in https://github.com/NVIDIA/cuml/pull/7499
* Revert "Temporarily drop xgboost from CI (#7526)" by @csadorf in https://github.com/NVIDIA/cuml/pull/7534
* Add minimal support for sklearn 1.8.0rc1 by @jcrist in https://github.com/NVIDIA/cuml/pull/7546
* Bump SciPy lower bound to 1.11 by @jcrist in https://github.com/NVIDIA/cuml/pull/7551

## New Contributors
* @rockhowse made their first contribution in https://github.com/NVIDIA/cuml/pull/7455
* @zhuxr11 made their first contribution in https://github.com/NVIDIA/cuml/pull/7331

**Full Changelog**: https://github.com/NVIDIA/cuml/compare/v25.12.00a...release/25.12

# cuml 25.10.00 (8 Oct 2025)

## 🚨 Breaking Changes

- Deprecate `convert_to_*` methods in favor of `as_*` methods in `cuml.ensemble` ([#7254](https://github.com/NVIDIA/cuml/pull/7254)) [@jcrist](https://github.com/jcrist)
- `KernelDensity.score_samples` output type &amp; dtype fixes ([#7240](https://github.com/NVIDIA/cuml/pull/7240)) [@jcrist](https://github.com/jcrist)
- Remove `get_json`/`get_detailed_text`/`get_summary_text` from ensemble estimators ([#7177](https://github.com/NVIDIA/cuml/pull/7177)) [@jcrist](https://github.com/jcrist)
- Deprecate `accuracy_metric` ([#7170](https://github.com/NVIDIA/cuml/pull/7170)) [@jcrist](https://github.com/jcrist)
- Deprecate `predict_model` in `cuml.ensemble`/`cuml.dask.ensemble` ([#7155](https://github.com/NVIDIA/cuml/pull/7155)) [@jcrist](https://github.com/jcrist)
- Remove deprecated `cuml.accel` CLI options ([#7110](https://github.com/NVIDIA/cuml/pull/7110)) [@jcrist](https://github.com/jcrist)
- Remove deprecation warnings in 25.10 ([#7109](https://github.com/NVIDIA/cuml/pull/7109)) [@jcrist](https://github.com/jcrist)
- Remove deprecation warning and `data_on_host` option for UMAP ([#7099](https://github.com/NVIDIA/cuml/pull/7099)) [@jinsolp](https://github.com/jinsolp)
- Fix UMAP graph thresholding ([#6595](https://github.com/NVIDIA/cuml/pull/6595)) [@viclafargue](https://github.com/viclafargue)

## 🐛 Bug Fixes

- Set NCCL rpath for `cu13` wheels ([#7304](https://github.com/NVIDIA/cuml/pull/7304)) [@divyegala](https://github.com/divyegala)
- Add NCCL library path to CMakeLists.txt for libcuml ([#7281](https://github.com/NVIDIA/cuml/pull/7281)) [@csadorf](https://github.com/csadorf)
- Ensure finite values passed to SpectralEmbedding ([#7280](https://github.com/NVIDIA/cuml/pull/7280)) [@jcrist](https://github.com/jcrist)
- Ensure sklearn roundtrip attribute consistency ([#7278](https://github.com/NVIDIA/cuml/pull/7278)) [@jcrist](https://github.com/jcrist)
- Fix SpectralEmbedding precomputed option with 0% sparsity ([#7271](https://github.com/NVIDIA/cuml/pull/7271)) [@aamijar](https://github.com/aamijar)
- Fix test reference object in test_onehot_inverse_transform_handle_unknown ([#7246](https://github.com/NVIDIA/cuml/pull/7246)) [@mroeschke](https://github.com/mroeschke)
- Slightly increase tolerance in ridge regression test. ([#7243](https://github.com/NVIDIA/cuml/pull/7243)) [@csadorf](https://github.com/csadorf)
- Increase rtol for float32 in test_complement_partial_fit to reduce intermittent failures ([#7237](https://github.com/NVIDIA/cuml/pull/7237)) [@csadorf](https://github.com/csadorf)
- Conditionally xfail test_umap_fit_transform_score for nrows=500. ([#7232](https://github.com/NVIDIA/cuml/pull/7232)) [@csadorf](https://github.com/csadorf)
- Fixup `StandardScaler.n_samples_seen_` ([#7209](https://github.com/NVIDIA/cuml/pull/7209)) [@jcrist](https://github.com/jcrist)
- Validate `sample_weight` in `KernelDensity.fit` ([#7208](https://github.com/NVIDIA/cuml/pull/7208)) [@jcrist](https://github.com/jcrist)
- Support non-contiguous input in `input_to_host_array` ([#7207](https://github.com/NVIDIA/cuml/pull/7207)) [@jcrist](https://github.com/jcrist)
- Set correct min_samples for HDBSCAN single_linkage ([#7195](https://github.com/NVIDIA/cuml/pull/7195)) [@tarang-jain](https://github.com/tarang-jain)
- Raise Proper Exception when HDBSCAN min_samples &gt; number of samples ([#7193](https://github.com/NVIDIA/cuml/pull/7193)) [@tarang-jain](https://github.com/tarang-jain)
- Remove docs/handling of `solver=&quot;cd&quot;` in `Ridge` ([#7190](https://github.com/NVIDIA/cuml/pull/7190)) [@jcrist](https://github.com/jcrist)
- Error nicely in `KMeans` if `n_samples &lt; n_clusters` ([#7189](https://github.com/NVIDIA/cuml/pull/7189)) [@jcrist](https://github.com/jcrist)
- Further increase tolerance in test_random_seed_consistency. ([#7180](https://github.com/NVIDIA/cuml/pull/7180)) [@csadorf](https://github.com/csadorf)
- Use custom plugin to download test data early ([#7169](https://github.com/NVIDIA/cuml/pull/7169)) [@betatim](https://github.com/betatim)
- Fix UMAP outlier issue by checking for outliers and shuffling ([#7131](https://github.com/NVIDIA/cuml/pull/7131)) [@jinsolp](https://github.com/jinsolp)
- Note non-determinism in random projection transforms ([#7129](https://github.com/NVIDIA/cuml/pull/7129)) [@jcrist](https://github.com/jcrist)
- Rewrite random projection estimators ([#7119](https://github.com/NVIDIA/cuml/pull/7119)) [@jcrist](https://github.com/jcrist)
- Prevent CUDA issues when running UMAP with scikit-learn compatibility test suite ([#7107](https://github.com/NVIDIA/cuml/pull/7107)) [@viclafargue](https://github.com/viclafargue)
- Fix accel profiler 2x count of some fallback methods ([#7101](https://github.com/NVIDIA/cuml/pull/7101)) [@jcrist](https://github.com/jcrist)
- Don&#39;t call to_output on a cupy array ([#7044](https://github.com/NVIDIA/cuml/pull/7044)) [@Matt711](https://github.com/Matt711)
- Add fix for devices that do not have memory resources ([#6823](https://github.com/NVIDIA/cuml/pull/6823)) [@viclafargue](https://github.com/viclafargue)
- Fix UMAP graph thresholding ([#6595](https://github.com/NVIDIA/cuml/pull/6595)) [@viclafargue](https://github.com/viclafargue)

## 📖 Documentation

- Add warning on subprocesses in cuml.accel profilers ([#7290](https://github.com/NVIDIA/cuml/pull/7290)) [@jcrist](https://github.com/jcrist)
- Revise the cuML docs for 25.10 ([#7228](https://github.com/NVIDIA/cuml/pull/7228)) [@csadorf](https://github.com/csadorf)
- Provide GitHub issue template to report CI failures. ([#7178](https://github.com/NVIDIA/cuml/pull/7178)) [@csadorf](https://github.com/csadorf)
- Fix cuml-accel docs redirects ([#7139](https://github.com/NVIDIA/cuml/pull/7139)) [@jcrist](https://github.com/jcrist)

## 🚀 New Features

- Update cufft defines to CUDA 13 types ([#7094](https://github.com/NVIDIA/cuml/pull/7094)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Fix missed deps in update-version.sh ([#7285](https://github.com/NVIDIA/cuml/pull/7285)) [@AyodeAwe](https://github.com/AyodeAwe)
- Empty commit to trigger a build ([#7264](https://github.com/NVIDIA/cuml/pull/7264)) [@msarahan](https://github.com/msarahan)
- Enforce correct dependencies on cuVS ([#7261](https://github.com/NVIDIA/cuml/pull/7261)) [@divyegala](https://github.com/divyegala)
- CI Hardcode list of words to test porter stemmer with ([#7258](https://github.com/NVIDIA/cuml/pull/7258)) [@betatim](https://github.com/betatim)
- Deprecate `convert_to_*` methods in favor of `as_*` methods in `cuml.ensemble` ([#7254](https://github.com/NVIDIA/cuml/pull/7254)) [@jcrist](https://github.com/jcrist)
- Revert &quot;Extend nightlies breakage window ([#7137)&quot; (#7251](https://github.com/NVIDIA/cuml/pull/7137)&quot; (#7251)) [@csadorf](https://github.com/csadorf)
- Cleanup `cuml.ensemble` ([#7249](https://github.com/NVIDIA/cuml/pull/7249)) [@jcrist](https://github.com/jcrist)
- `KernelDensity.score_samples` output type &amp; dtype fixes ([#7240](https://github.com/NVIDIA/cuml/pull/7240)) [@jcrist](https://github.com/jcrist)
- Add test for `adjusted_rand_score` with small inputs ([#7235](https://github.com/NVIDIA/cuml/pull/7235)) [@jcrist](https://github.com/jcrist)
- Use `except +` everywhere in cython ([#7234](https://github.com/NVIDIA/cuml/pull/7234)) [@jcrist](https://github.com/jcrist)
- Use `rapids-xgboost` and exclude libnccl from auditwheel check ([#7231](https://github.com/NVIDIA/cuml/pull/7231)) [@jcrist](https://github.com/jcrist)
- Add `cuml.metrics.median_absolute_error` ([#7229](https://github.com/NVIDIA/cuml/pull/7229)) [@jcrist](https://github.com/jcrist)
- Configure repo for automatic release notes generation ([#7226](https://github.com/NVIDIA/cuml/pull/7226)) [@AyodeAwe](https://github.com/AyodeAwe)
- Accelerate `KernelDensity.score_samples` ([#7219](https://github.com/NVIDIA/cuml/pull/7219)) [@jcrist](https://github.com/jcrist)
- Update FIL doc to remove deprecated parameter ([#7215](https://github.com/NVIDIA/cuml/pull/7215)) [@hcho3](https://github.com/hcho3)
- use new cumlprims_mg repo ([#7213](https://github.com/NVIDIA/cuml/pull/7213)) [@jameslamb](https://github.com/jameslamb)
- update nvidia-ml-py (&gt;=12), use cuda-toolkit wheels ([#7212](https://github.com/NVIDIA/cuml/pull/7212)) [@jameslamb](https://github.com/jameslamb)
- Expose some private attrs in `cuml.accel` ([#7197](https://github.com/NVIDIA/cuml/pull/7197)) [@jcrist](https://github.com/jcrist)
- Cleanup `KMeans` python layer ([#7196](https://github.com/NVIDIA/cuml/pull/7196)) [@jcrist](https://github.com/jcrist)
- Don&#39;t use deprecated method in `StratifiedKFold` ([#7188](https://github.com/NVIDIA/cuml/pull/7188)) [@jcrist](https://github.com/jcrist)
- DOC fix table formatting for SHAP ([#7186](https://github.com/NVIDIA/cuml/pull/7186)) [@sotagg](https://github.com/sotagg)
- Use branch-25.10 again ([#7182](https://github.com/NVIDIA/cuml/pull/7182)) [@jameslamb](https://github.com/jameslamb)
- Small cleanup to `cuml.dask.metrics.confusion_matrix` ([#7181](https://github.com/NVIDIA/cuml/pull/7181)) [@jcrist](https://github.com/jcrist)
- Remove `get_json`/`get_detailed_text`/`get_summary_text` from ensemble estimators ([#7177](https://github.com/NVIDIA/cuml/pull/7177)) [@jcrist](https://github.com/jcrist)
- Move cert configuration to `pytest_configure` hook ([#7175](https://github.com/NVIDIA/cuml/pull/7175)) [@jcrist](https://github.com/jcrist)
- Remove baked in datasets from preloading plugin ([#7173](https://github.com/NVIDIA/cuml/pull/7173)) [@betatim](https://github.com/betatim)
- Optimize import times a bit ([#7171](https://github.com/NVIDIA/cuml/pull/7171)) [@jcrist](https://github.com/jcrist)
- Deprecate `accuracy_metric` ([#7170](https://github.com/NVIDIA/cuml/pull/7170)) [@jcrist](https://github.com/jcrist)
- use &#39;nvidia-ml-py&#39; instead of &#39;pynvml&#39;, declare &#39;numba-cuda&#39; dependency pins ([#7164](https://github.com/NVIDIA/cuml/pull/7164)) [@jameslamb](https://github.com/jameslamb)
- Deprecate `predict_model` in `cuml.ensemble`/`cuml.dask.ensemble` ([#7155](https://github.com/NVIDIA/cuml/pull/7155)) [@jcrist](https://github.com/jcrist)
- Remove deprecated `cupyx.scatter_add`/`cupyx.scatter_max` calls ([#7149](https://github.com/NVIDIA/cuml/pull/7149)) [@jcrist](https://github.com/jcrist)
- Fix HTML representation&#39;s fitted status ([#7146](https://github.com/NVIDIA/cuml/pull/7146)) [@betatim](https://github.com/betatim)
- Replace raft calls with cuvs in DBSCAN ([#7144](https://github.com/NVIDIA/cuml/pull/7144)) [@aamijar](https://github.com/aamijar)
- Expose `TSNE.n_iter_` fitted attribute ([#7142](https://github.com/NVIDIA/cuml/pull/7142)) [@jcrist](https://github.com/jcrist)
- Update rapids-dependency-file-generator ([#7141](https://github.com/NVIDIA/cuml/pull/7141)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- `n_streams=1` is not required for determinism in `cuml.ensemble` ([#7140](https://github.com/NVIDIA/cuml/pull/7140)) [@jcrist](https://github.com/jcrist)
- Extend nightlies breakage window ([#7137](https://github.com/NVIDIA/cuml/pull/7137)) [@jcrist](https://github.com/jcrist)
- Add DBSCAN to the common tests ([#7134](https://github.com/NVIDIA/cuml/pull/7134)) [@betatim](https://github.com/betatim)
- Remove UCX-Py ([#7130](https://github.com/NVIDIA/cuml/pull/7130)) [@pentschev](https://github.com/pentschev)
- Build and test with CUDA 13.0.0 ([#7128](https://github.com/NVIDIA/cuml/pull/7128)) [@jameslamb](https://github.com/jameslamb)
- Nicer error message for not implemented fitted attributes in `cuml.accel` ([#7127](https://github.com/NVIDIA/cuml/pull/7127)) [@jcrist](https://github.com/jcrist)
- Use gammainc from cupyx ([#7126](https://github.com/NVIDIA/cuml/pull/7126)) [@jcrist](https://github.com/jcrist)
- Support `kneighbors_graph` for approximate algos ([#7125](https://github.com/NVIDIA/cuml/pull/7125)) [@jcrist](https://github.com/jcrist)
- update gputreeshap ([#7121](https://github.com/NVIDIA/cuml/pull/7121)) [@jameslamb](https://github.com/jameslamb)
- Spectral Embedding argument `affinity={&quot;precomputed&quot;, &quot;nearest_neighbors&quot;}` ([#7117](https://github.com/NVIDIA/cuml/pull/7117)) [@aamijar](https://github.com/aamijar)
- Increase wheel size cap to 525M account for newer architectures ([#7116](https://github.com/NVIDIA/cuml/pull/7116)) [@divyegala](https://github.com/divyegala)
- Redo memory management for NearestNeighbors ([#7115](https://github.com/NVIDIA/cuml/pull/7115)) [@jcrist](https://github.com/jcrist)
- FIX Make GaussianNB more resilient ([#7113](https://github.com/NVIDIA/cuml/pull/7113)) [@betatim](https://github.com/betatim)
- Remove deprecated `cuml.accel` CLI options ([#7110](https://github.com/NVIDIA/cuml/pull/7110)) [@jcrist](https://github.com/jcrist)
- Remove deprecation warnings in 25.10 ([#7109](https://github.com/NVIDIA/cuml/pull/7109)) [@jcrist](https://github.com/jcrist)
- A few fixes to the profiler IPython magics ([#7108](https://github.com/NVIDIA/cuml/pull/7108)) [@jcrist](https://github.com/jcrist)
- Use build cluster in devcontainers ([#7106](https://github.com/NVIDIA/cuml/pull/7106)) [@trxcllnt](https://github.com/trxcllnt)
- Use rapids_cuda_enable_fatbin_compression ([#7105](https://github.com/NVIDIA/cuml/pull/7105)) [@robertmaynard](https://github.com/robertmaynard)
- Forward-merge branch-25.08 into branch-25.10 ([#7103](https://github.com/NVIDIA/cuml/pull/7103)) [@jcrist](https://github.com/jcrist)
- Remove deprecation warning and `data_on_host` option for UMAP ([#7099](https://github.com/NVIDIA/cuml/pull/7099)) [@jinsolp](https://github.com/jinsolp)
- Update rapids_config to handle user defined branch name ([#7098](https://github.com/NVIDIA/cuml/pull/7098)) [@robertmaynard](https://github.com/robertmaynard)
- Add some devtools to `all_*` environments ([#7091](https://github.com/NVIDIA/cuml/pull/7091)) [@jcrist](https://github.com/jcrist)
- Configure `--import-mode` in `pyproject.toml` ([#7090](https://github.com/NVIDIA/cuml/pull/7090)) [@jcrist](https://github.com/jcrist)
- Support `SpectralEmbedding` in `cuml.accel` ([#7087](https://github.com/NVIDIA/cuml/pull/7087)) [@aamijar](https://github.com/aamijar)
- Update rapids-build-backend to 0.4.1 ([#7085](https://github.com/NVIDIA/cuml/pull/7085)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Update HDBSCAN prediction utils to use raft::linalg::map_offset ([#7075](https://github.com/NVIDIA/cuml/pull/7075)) [@tarang-jain](https://github.com/tarang-jain)
- Improved UMAP testing and debugging ([#7073](https://github.com/NVIDIA/cuml/pull/7073)) [@viclafargue](https://github.com/viclafargue)
- Add more estimators to the compatibility test suite ([#7069](https://github.com/NVIDIA/cuml/pull/7069)) [@betatim](https://github.com/betatim)
- UMAP with new Spectral Embedding initialization ([#7063](https://github.com/NVIDIA/cuml/pull/7063)) [@aamijar](https://github.com/aamijar)
- Remove duplicated cuml.accel install code ([#7062](https://github.com/NVIDIA/cuml/pull/7062)) [@betatim](https://github.com/betatim)
- cuml RAPIDS dependency now use the new rapids-cmake branch info ([#7051](https://github.com/NVIDIA/cuml/pull/7051)) [@robertmaynard](https://github.com/robertmaynard)
- Fix input utils test case ([#7050](https://github.com/NVIDIA/cuml/pull/7050)) [@emmanuel-ferdman](https://github.com/emmanuel-ferdman)
- Propagate `cuml.accel` to subprocesses ([#7047](https://github.com/NVIDIA/cuml/pull/7047)) [@jcrist](https://github.com/jcrist)
- Support enabling `cuml.accel` with env var ([#7046](https://github.com/NVIDIA/cuml/pull/7046)) [@jcrist](https://github.com/jcrist)
- Add basic infrastructure for common tests ([#7038](https://github.com/NVIDIA/cuml/pull/7038)) [@betatim](https://github.com/betatim)
- Allow latest OS in devcontainers ([#7036](https://github.com/NVIDIA/cuml/pull/7036)) [@bdice](https://github.com/bdice)
- Update build infra to support new branching strategy ([#7027](https://github.com/NVIDIA/cuml/pull/7027)) [@robertmaynard](https://github.com/robertmaynard)
- Adding appropriate casting for a more generic `IdxT` usage in HDBSCAN ([#7023](https://github.com/NVIDIA/cuml/pull/7023)) [@jinsolp](https://github.com/jinsolp)
- Build libcuml wheel with static libcuvs ([#6959](https://github.com/NVIDIA/cuml/pull/6959)) [@hcho3](https://github.com/hcho3)
- Use GCC 14 in conda builds. ([#6918](https://github.com/NVIDIA/cuml/pull/6918)) [@vyasr](https://github.com/vyasr)
- Add capability to use different allocators in cuML Python benchmarks ([#6903](https://github.com/NVIDIA/cuml/pull/6903)) [@dantegd](https://github.com/dantegd)
- Reorganize tests and modify pytest invocation to support pytest 8 ([#6876](https://github.com/NVIDIA/cuml/pull/6876)) [@vyasr](https://github.com/vyasr)

# cuml 25.08.00 (6 Aug 2025)

## 🚨 Breaking Changes

- Deprecating `data_on_host` parameter for UMAP ([#6953](https://github.com/NVIDIA/cuml/pull/6953)) [@jinsolp](https://github.com/jinsolp)
- Deprecate hdbscan prediction functions in `cuml.cluster` namespace ([#6943](https://github.com/NVIDIA/cuml/pull/6943)) [@jcrist](https://github.com/jcrist)
- Reflect output type in `hdbscan` prediction functions ([#6939](https://github.com/NVIDIA/cuml/pull/6939)) [@jcrist](https://github.com/jcrist)
- Deprecate `connectivity` parameter to `HDBSCAN` ([#6936](https://github.com/NVIDIA/cuml/pull/6936)) [@jcrist](https://github.com/jcrist)
- Deprecate `penalty=&#39;none&#39;` in `MBSGDClassifier`/`MBSGDRegressor`/`SGD` ([#6926](https://github.com/NVIDIA/cuml/pull/6926)) [@jcrist](https://github.com/jcrist)
- Remove `experimental.fil` Python module ([#6899](https://github.com/NVIDIA/cuml/pull/6899)) [@hcho3](https://github.com/hcho3)
- Change `KMeans` `random_state` default to `None` ([#6884](https://github.com/NVIDIA/cuml/pull/6884)) [@jcrist](https://github.com/jcrist)
- Remove cuvs symbols in libcuml headers ([#6883](https://github.com/NVIDIA/cuml/pull/6883)) [@hcho3](https://github.com/hcho3)
- Migrate C++ files from `src/experimental/fil` to `src/fil` ([#6882](https://github.com/NVIDIA/cuml/pull/6882)) [@hcho3](https://github.com/hcho3)
- Add test for signatures of common methods ([#6878](https://github.com/NVIDIA/cuml/pull/6878)) [@jcrist](https://github.com/jcrist)
- Port `cuml.neighbors` to `InteropMixin`/`ProxyBase` ([#6851](https://github.com/NVIDIA/cuml/pull/6851)) [@jcrist](https://github.com/jcrist)
- Remove CUDA 11 from dependencies.yaml ([#6847](https://github.com/NVIDIA/cuml/pull/6847)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove legacy FIL from libcuml ([#6844](https://github.com/NVIDIA/cuml/pull/6844)) [@hcho3](https://github.com/hcho3)
- stop uploading packages to downloads.rapids.ai ([#6803](https://github.com/NVIDIA/cuml/pull/6803)) [@jameslamb](https://github.com/jameslamb)

## 🐛 Bug Fixes

- Fix network failure in loading mnist dataset during spectral embedding testing ([#7077](https://github.com/NVIDIA/cuml/pull/7077)) [@aamijar](https://github.com/aamijar)
- Relaxing unit tolerance for `test_tsne_distance_metrics_on_sparse_input` ([#7033](https://github.com/NVIDIA/cuml/pull/7033)) [@jinsolp](https://github.com/jinsolp)
- Disable UMAP non-determinism on small datasets ([#7004](https://github.com/NVIDIA/cuml/pull/7004)) [@viclafargue](https://github.com/viclafargue)
- Remove CuPy version constraint ([#6979](https://github.com/NVIDIA/cuml/pull/6979)) [@csadorf](https://github.com/csadorf)
- Fix test_typeerror_input test with xfail for known sklearn LARS broadcasting bug ([#6978](https://github.com/NVIDIA/cuml/pull/6978)) [@csadorf](https://github.com/csadorf)
- Add upper constraint for cupy 13.5 due to memory access issues ([#6961](https://github.com/NVIDIA/cuml/pull/6961)) [@csadorf](https://github.com/csadorf)
- Decrease nightly check window to 7 days ([#6934](https://github.com/NVIDIA/cuml/pull/6934)) [@csadorf](https://github.com/csadorf)
- Support empty categorical nodes in FIL ([#6924](https://github.com/NVIDIA/cuml/pull/6924)) [@hcho3](https://github.com/hcho3)
- Rewrite `HDBSCAN` python wrapper ([#6913](https://github.com/NVIDIA/cuml/pull/6913)) [@jcrist](https://github.com/jcrist)
- Mark test_warm_start_oob[RandomForestClassifier] as flaky ([#6906](https://github.com/NVIDIA/cuml/pull/6906)) [@csadorf](https://github.com/csadorf)
- Unxfail tests affected by numba compilation error on Python 3.13.4 ([#6905](https://github.com/NVIDIA/cuml/pull/6905)) [@csadorf](https://github.com/csadorf)
- Fix import in benchmark code in algorithms.py that was causing an error ([#6902](https://github.com/NVIDIA/cuml/pull/6902)) [@dantegd](https://github.com/dantegd)
- Adding GH_TOKEN pass-through to summarize job ([#6894](https://github.com/NVIDIA/cuml/pull/6894)) [@msarahan](https://github.com/msarahan)
- Add test for signatures of common methods ([#6878](https://github.com/NVIDIA/cuml/pull/6878)) [@jcrist](https://github.com/jcrist)
- Fix CI test failures: xfail tests for scikit-learn 1.7.0 and Python 3.13.4 compatibility ([#6865](https://github.com/NVIDIA/cuml/pull/6865)) [@csadorf](https://github.com/csadorf)
- Fix RandomForestRegressor default `max_features` ([#6862](https://github.com/NVIDIA/cuml/pull/6862)) [@jcrist](https://github.com/jcrist)
- Increase tolerance in Dask logistic regression test ([#6848](https://github.com/NVIDIA/cuml/pull/6848)) [@csadorf](https://github.com/csadorf)

## 📖 Documentation

- Document supported versions. ([#7040](https://github.com/NVIDIA/cuml/pull/7040)) [@csadorf](https://github.com/csadorf)
- Update title of `cuml.accel` documentation to &quot;Zero Code Change Acceleration&quot; ([#7030](https://github.com/NVIDIA/cuml/pull/7030)) [@csadorf](https://github.com/csadorf)
- Reorganize cuml.accel docs ([#7026](https://github.com/NVIDIA/cuml/pull/7026)) [@jcrist](https://github.com/jcrist)
- UMAP multi-gpu knn graph build docs ([#7019](https://github.com/NVIDIA/cuml/pull/7019)) [@jinsolp](https://github.com/jinsolp)
- Remove async operation section from Python developer docs ([#6980](https://github.com/NVIDIA/cuml/pull/6980)) [@csadorf](https://github.com/csadorf)
- Fix docs for FFT TSNE ([#6967](https://github.com/NVIDIA/cuml/pull/6967)) [@jinsolp](https://github.com/jinsolp)
- Revamp `cuml.accel` limitations docs page ([#6965](https://github.com/NVIDIA/cuml/pull/6965)) [@jcrist](https://github.com/jcrist)
- add docs on CI workflow inputs ([#6952](https://github.com/NVIDIA/cuml/pull/6952)) [@jameslamb](https://github.com/jameslamb)
- Fixup rst formatting of 0cc limitations doc ([#6945](https://github.com/NVIDIA/cuml/pull/6945)) [@jcrist](https://github.com/jcrist)
- Documentation Updates for cuML Python Developer Guide ([#6843](https://github.com/NVIDIA/cuml/pull/6843)) [@csadorf](https://github.com/csadorf)

## 🚀 New Features

- Spectral Embedding ([#6581](https://github.com/NVIDIA/cuml/pull/6581)) [@aamijar](https://github.com/aamijar)

## 🛠️ Improvements

- Add profilers to `cuml.accel` ([#7021](https://github.com/NVIDIA/cuml/pull/7021)) [@jcrist](https://github.com/jcrist)
- use versioned CI image tags ([#7016](https://github.com/NVIDIA/cuml/pull/7016)) [@jameslamb](https://github.com/jameslamb)
- Support inferencing with wide data in FIL ([#7014](https://github.com/NVIDIA/cuml/pull/7014)) [@hcho3](https://github.com/hcho3)
- Remove sphinx upper bound ([#7009](https://github.com/NVIDIA/cuml/pull/7009)) [@bdice](https://github.com/bdice)
- Compatibility for dask ucx protocol changes ([#7008](https://github.com/NVIDIA/cuml/pull/7008)) [@TomAugspurger](https://github.com/TomAugspurger)
- Add &quot;devcontainers&quot; dependency file key that excludes xgboost ([#7006](https://github.com/NVIDIA/cuml/pull/7006)) [@bdice](https://github.com/bdice)
- Use `cuvs::neighbors::knn_merge_parts` ([#7005](https://github.com/NVIDIA/cuml/pull/7005)) [@jcrist](https://github.com/jcrist)
- Mark `Ridge` as not supporting underdetermined systems ([#7003](https://github.com/NVIDIA/cuml/pull/7003)) [@betatim](https://github.com/betatim)
- Reduce `arima` kernels binary size ([#6997](https://github.com/NVIDIA/cuml/pull/6997)) [@jcrist](https://github.com/jcrist)
- Run the upstream HDBSCAN tests in CI ([#6995](https://github.com/NVIDIA/cuml/pull/6995)) [@jcrist](https://github.com/jcrist)
- Use `pytest.ini` for all upstream tests ([#6994](https://github.com/NVIDIA/cuml/pull/6994)) [@jcrist](https://github.com/jcrist)
- Support invoking `cuml_accel_tests` without extra pytest flags ([#6993](https://github.com/NVIDIA/cuml/pull/6993)) [@jcrist](https://github.com/jcrist)
- Fall back on unsupported metrics in TSNE ([#6992](https://github.com/NVIDIA/cuml/pull/6992)) [@jcrist](https://github.com/jcrist)
- Expand `LogisticRegression` limitations ([#6991](https://github.com/NVIDIA/cuml/pull/6991)) [@jcrist](https://github.com/jcrist)
- Expand `sklearn.ensemble` limitations ([#6990](https://github.com/NVIDIA/cuml/pull/6990)) [@jcrist](https://github.com/jcrist)
- Run `umap-learn` test suite in CI ([#6989](https://github.com/NVIDIA/cuml/pull/6989)) [@jcrist](https://github.com/jcrist)
- Add a CI run with &quot;intermediate&quot; versions of dependencies ([#6988](https://github.com/NVIDIA/cuml/pull/6988)) [@betatim](https://github.com/betatim)
- Correctly handle non-default GPU context in FIL ([#6987](https://github.com/NVIDIA/cuml/pull/6987)) [@hcho3](https://github.com/hcho3)
- Move `cuml.accel` upstream tests to the `cuml_accel_tests` directory ([#6986](https://github.com/NVIDIA/cuml/pull/6986)) [@jcrist](https://github.com/jcrist)
- Compute `components_` in `DBSCAN` ([#6976](https://github.com/NVIDIA/cuml/pull/6976)) [@jcrist](https://github.com/jcrist)
- Add the cython-lint commit to `.git-blame-ignore-revs` ([#6972](https://github.com/NVIDIA/cuml/pull/6972)) [@jcrist](https://github.com/jcrist)
- Several SVC cleanups ([#6970](https://github.com/NVIDIA/cuml/pull/6970)) [@jcrist](https://github.com/jcrist)
- Update cython-lint, fix long lines ([#6969](https://github.com/NVIDIA/cuml/pull/6969)) [@jcrist](https://github.com/jcrist)
- Enhance logging for cuml.accel proxy estimators ([#6957](https://github.com/NVIDIA/cuml/pull/6957)) [@csadorf](https://github.com/csadorf)
- Add a reason to all `UnsupportedOnGPU`/`UnsupportedOnCPU` errors ([#6954](https://github.com/NVIDIA/cuml/pull/6954)) [@jcrist](https://github.com/jcrist)
- Deprecating `data_on_host` parameter for UMAP ([#6953](https://github.com/NVIDIA/cuml/pull/6953)) [@jinsolp](https://github.com/jinsolp)
- Implement metadata routing in `cuml.accel` ([#6950](https://github.com/NVIDIA/cuml/pull/6950)) [@jcrist](https://github.com/jcrist)
- xfail some tests due to new scipy 1.16 release ([#6946](https://github.com/NVIDIA/cuml/pull/6946)) [@jcrist](https://github.com/jcrist)
- Use CUDA 12.9 in Conda, Devcontainers, Spark, GHA, etc. ([#6944](https://github.com/NVIDIA/cuml/pull/6944)) [@jakirkham](https://github.com/jakirkham)
- Deprecate hdbscan prediction functions in `cuml.cluster` namespace ([#6943](https://github.com/NVIDIA/cuml/pull/6943)) [@jcrist](https://github.com/jcrist)
- Support `set_output` and `get_feature_names_out` in `cuml.accel` ([#6942](https://github.com/NVIDIA/cuml/pull/6942)) [@jcrist](https://github.com/jcrist)
- Reflect output type in `hdbscan` prediction functions ([#6939](https://github.com/NVIDIA/cuml/pull/6939)) [@jcrist](https://github.com/jcrist)
- Remove `Base.__getattr__`/`Base.__getstate__`/`Base.__setstate__` ([#6937](https://github.com/NVIDIA/cuml/pull/6937)) [@jcrist](https://github.com/jcrist)
- Deprecate `connectivity` parameter to `HDBSCAN` ([#6936](https://github.com/NVIDIA/cuml/pull/6936)) [@jcrist](https://github.com/jcrist)
- Remove nvidia and dask channels ([#6935](https://github.com/NVIDIA/cuml/pull/6935)) [@vyasr](https://github.com/vyasr)
- Fixup KernelRidge hypothesis test ([#6931](https://github.com/NVIDIA/cuml/pull/6931)) [@jcrist](https://github.com/jcrist)
- A few fixes for CI ([#6928](https://github.com/NVIDIA/cuml/pull/6928)) [@jcrist](https://github.com/jcrist)
- Deprecate `penalty=&#39;none&#39;` in `MBSGDClassifier`/`MBSGDRegressor`/`SGD` ([#6926](https://github.com/NVIDIA/cuml/pull/6926)) [@jcrist](https://github.com/jcrist)
- Run FIL demo notebook in CI ([#6923](https://github.com/NVIDIA/cuml/pull/6923)) [@hcho3](https://github.com/hcho3)
- Restore old behavior for `threshold` in FIL ([#6922](https://github.com/NVIDIA/cuml/pull/6922)) [@hcho3](https://github.com/hcho3)
- De-Cython several modules ([#6920](https://github.com/NVIDIA/cuml/pull/6920)) [@jcrist](https://github.com/jcrist)
- Remove `UniversalBase` ([#6919](https://github.com/NVIDIA/cuml/pull/6919)) [@jcrist](https://github.com/jcrist)
- Add `KernelRidge` to `cuml.accel` ([#6917](https://github.com/NVIDIA/cuml/pull/6917)) [@jcrist](https://github.com/jcrist)
- Port `HDBSCAN` to `InteropMixin`/`ProxyBase` ([#6916](https://github.com/NVIDIA/cuml/pull/6916)) [@jcrist](https://github.com/jcrist)
- Expose `n_iter_` for `LogisticRegression` models ([#6911](https://github.com/NVIDIA/cuml/pull/6911)) [@betatim](https://github.com/betatim)
- Warn and fail on stale xfail list entries ([#6907](https://github.com/NVIDIA/cuml/pull/6907)) [@csadorf](https://github.com/csadorf)
- Handling all identical vectors in UMAP `smooth_knn_dist_kernel` ([#6904](https://github.com/NVIDIA/cuml/pull/6904)) [@jinsolp](https://github.com/jinsolp)
- refactor(shellcheck): fix all shellcheck warnings/errors ([#6901](https://github.com/NVIDIA/cuml/pull/6901)) [@gforsyth](https://github.com/gforsyth)
- Instantiate only specific RAFT linewise kernels ([#6900](https://github.com/NVIDIA/cuml/pull/6900)) [@aamijar](https://github.com/aamijar)
- Remove `experimental.fil` Python module ([#6899](https://github.com/NVIDIA/cuml/pull/6899)) [@hcho3](https://github.com/hcho3)
- Remove `GlobalSettings.device_type` ([#6895](https://github.com/NVIDIA/cuml/pull/6895)) [@jcrist](https://github.com/jcrist)
- Add xfail list auto-formatter ([#6892](https://github.com/NVIDIA/cuml/pull/6892)) [@csadorf](https://github.com/csadorf)
- Remove deprecated device selection functions ([#6888](https://github.com/NVIDIA/cuml/pull/6888)) [@jcrist](https://github.com/jcrist)
- Remove deprecations for 25.08 ([#6887](https://github.com/NVIDIA/cuml/pull/6887)) [@jcrist](https://github.com/jcrist)
- Removing tests, codepaths and other bits related to CUDA 11 ([#6885](https://github.com/NVIDIA/cuml/pull/6885)) [@dantegd](https://github.com/dantegd)
- Change `KMeans` `random_state` default to `None` ([#6884](https://github.com/NVIDIA/cuml/pull/6884)) [@jcrist](https://github.com/jcrist)
- Remove cuvs symbols in libcuml headers ([#6883](https://github.com/NVIDIA/cuml/pull/6883)) [@hcho3](https://github.com/hcho3)
- Migrate C++ files from `src/experimental/fil` to `src/fil` ([#6882](https://github.com/NVIDIA/cuml/pull/6882)) [@hcho3](https://github.com/hcho3)
- Remove invalid xfail entry and mark correct test as flaky ([#6877](https://github.com/NVIDIA/cuml/pull/6877)) [@csadorf](https://github.com/csadorf)
- A few `max_features` fixups for `cuml.ensemble` ([#6873](https://github.com/NVIDIA/cuml/pull/6873)) [@jcrist](https://github.com/jcrist)
- Loosen constrain on numpy upper version ([#6871](https://github.com/NVIDIA/cuml/pull/6871)) [@csadorf](https://github.com/csadorf)
- Correct `sys.argv` when running `cuml.accel` ([#6870](https://github.com/NVIDIA/cuml/pull/6870)) [@jcrist](https://github.com/jcrist)
- Support `LinearSVC` and `LinearSVR` in `cuml.accel` ([#6866](https://github.com/NVIDIA/cuml/pull/6866)) [@viclafargue](https://github.com/viclafargue)
- Port `cuml.ensemble` to `InteropMixin`/`ProxyBase` ([#6863](https://github.com/NVIDIA/cuml/pull/6863)) [@jcrist](https://github.com/jcrist)
- Deprecate some `cuml.accel` CLI options ([#6855](https://github.com/NVIDIA/cuml/pull/6855)) [@jcrist](https://github.com/jcrist)
- Remove deprecated args in `cuml.ensemble` ([#6854](https://github.com/NVIDIA/cuml/pull/6854)) [@jcrist](https://github.com/jcrist)
- Move deprecated positional/keyword args to keyword-only ([#6853](https://github.com/NVIDIA/cuml/pull/6853)) [@jcrist](https://github.com/jcrist)
- Support `-c` and `-` options in `cuml.accel` CLI ([#6852](https://github.com/NVIDIA/cuml/pull/6852)) [@jcrist](https://github.com/jcrist)
- Port `cuml.neighbors` to `InteropMixin`/`ProxyBase` ([#6851](https://github.com/NVIDIA/cuml/pull/6851)) [@jcrist](https://github.com/jcrist)
- Remove CUDA 11 from dependencies.yaml ([#6847](https://github.com/NVIDIA/cuml/pull/6847)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Add support for scikit-learn version 1.4 ([#6845](https://github.com/NVIDIA/cuml/pull/6845)) [@betatim](https://github.com/betatim)
- Remove legacy FIL from libcuml ([#6844](https://github.com/NVIDIA/cuml/pull/6844)) [@hcho3](https://github.com/hcho3)
- Port UMAP to InteropMixin/ProxyBase ([#6840](https://github.com/NVIDIA/cuml/pull/6840)) [@jcrist](https://github.com/jcrist)
- Move FIL to subclass from Base ([#6835](https://github.com/NVIDIA/cuml/pull/6835)) [@jcrist](https://github.com/jcrist)
- Mark `test_equivalence_components_pca_spca` as flaky ([#6834](https://github.com/NVIDIA/cuml/pull/6834)) [@jcrist](https://github.com/jcrist)
- Fix several bugs in `cuml.accel` CLI ([#6832](https://github.com/NVIDIA/cuml/pull/6832)) [@jcrist](https://github.com/jcrist)
- stop uploading packages to downloads.rapids.ai (pt 2) ([#6828](https://github.com/NVIDIA/cuml/pull/6828)) [@jameslamb](https://github.com/jameslamb)
- Remove CUDA 11 devcontainers and update CI scripts ([#6818](https://github.com/NVIDIA/cuml/pull/6818)) [@bdice](https://github.com/bdice)
- refactor(rattler): remove cuda11 options and general cleanup ([#6817](https://github.com/NVIDIA/cuml/pull/6817)) [@gforsyth](https://github.com/gforsyth)
- Add GitHub Actions workflow to assign PRs to their authors ([#6816](https://github.com/NVIDIA/cuml/pull/6816)) [@csadorf](https://github.com/csadorf)
- stop uploading packages to downloads.rapids.ai ([#6803](https://github.com/NVIDIA/cuml/pull/6803)) [@jameslamb](https://github.com/jameslamb)
- Instantiate only specific RAFT reduction kernels ([#6780](https://github.com/NVIDIA/cuml/pull/6780)) [@divyegala](https://github.com/divyegala)
- Forward-merge branch-25.06 into branch-25.08 ([#6647](https://github.com/NVIDIA/cuml/pull/6647)) [@gforsyth](https://github.com/gforsyth)
- [FEA] HDBSCAN to Migrate to cuVS cluster prims from raft::cluster ([#6560](https://github.com/NVIDIA/cuml/pull/6560)) [@tarang-jain](https://github.com/tarang-jain)

# cuml 25.06.00 (5 Jun 2025)

## 🚨 Breaking Changes

- Deprecate device selection ([#6784](https://github.com/NVIDIA/cuml/pull/6784)) [@jcrist](https://github.com/jcrist)
- Remove use of legacy FIL from Python cuML ([#6728](https://github.com/NVIDIA/cuml/pull/6728)) [@hcho3](https://github.com/hcho3)
- Use RBC from cuVS ([#6644](https://github.com/NVIDIA/cuml/pull/6644)) [@divyegala](https://github.com/divyegala)
- Map Barnes-Hut to FFT for T-SNE in cuml.accel. ([#6619](https://github.com/NVIDIA/cuml/pull/6619)) [@csadorf](https://github.com/csadorf)
- New Estimator Proxy architecture ([#6613](https://github.com/NVIDIA/cuml/pull/6613)) [@jcrist](https://github.com/jcrist)
- Disable building and uploading cuml-cpu in CI ([#6529](https://github.com/NVIDIA/cuml/pull/6529)) [@dantegd](https://github.com/dantegd)
- FIX Propagate random state to numpy rng in `make_classification` ([#6518](https://github.com/NVIDIA/cuml/pull/6518)) [@betatim](https://github.com/betatim)

## 🐛 Bug Fixes

- UMAP batch nnd test hotfix ([#6826](https://github.com/NVIDIA/cuml/pull/6826)) [@jinsolp](https://github.com/jinsolp)
- Update forest_inference_demo.ipynb for new FIL API changes ([#6824](https://github.com/NVIDIA/cuml/pull/6824)) [@dantegd](https://github.com/dantegd)
- Fix to compile FIL `infer_kernel` with CUDA arch 1210 ([#6821](https://github.com/NVIDIA/cuml/pull/6821)) [@viclafargue](https://github.com/viclafargue)
- Reduce solver test threshold for ElasticNet test. ([#6766](https://github.com/NVIDIA/cuml/pull/6766)) [@csadorf](https://github.com/csadorf)
- Use _assert_allclose function for approximate equality check. ([#6763](https://github.com/NVIDIA/cuml/pull/6763)) [@csadorf](https://github.com/csadorf)
- UMAP spectral initialization fallback to random on error ([#6750](https://github.com/NVIDIA/cuml/pull/6750)) [@aamijar](https://github.com/aamijar)
- [Fix] `include_self=False` for duplicate points (tied zero distances) in kneighbors_graph ([#6735](https://github.com/NVIDIA/cuml/pull/6735)) [@aamijar](https://github.com/aamijar)
- FIX: Set scikit-learn KMeans n_init=10 to match cuML ([#6727](https://github.com/NVIDIA/cuml/pull/6727)) [@csadorf](https://github.com/csadorf)
- Fix output dtype of `LinearSVC.predict` ([#6715](https://github.com/NVIDIA/cuml/pull/6715)) [@jcrist](https://github.com/jcrist)
- CUDA 12.9 Correctly bound updated compression flags ([#6713](https://github.com/NVIDIA/cuml/pull/6713)) [@robertmaynard](https://github.com/robertmaynard)
- Fix type and support for `PCA.noise_variance_` ([#6693](https://github.com/NVIDIA/cuml/pull/6693)) [@jcrist](https://github.com/jcrist)
- Update dask RF regressor partial_inference ([#6691](https://github.com/NVIDIA/cuml/pull/6691)) [@TomAugspurger](https://github.com/TomAugspurger)
- Properly handle degenerate trees in FIL ([#6673](https://github.com/NVIDIA/cuml/pull/6673)) [@hcho3](https://github.com/hcho3)
- Fix strict marker in scikit-learn test xfail list. ([#6661](https://github.com/NVIDIA/cuml/pull/6661)) [@csadorf](https://github.com/csadorf)
- DOC Repair docstring formatting ([#6659](https://github.com/NVIDIA/cuml/pull/6659)) [@betatim](https://github.com/betatim)
- Fix the field ids in the pr/issue status automation ([#6656](https://github.com/NVIDIA/cuml/pull/6656)) [@csadorf](https://github.com/csadorf)
- Adjust scikit-learn test failure handling ([#6646](https://github.com/NVIDIA/cuml/pull/6646)) [@csadorf](https://github.com/csadorf)
- Remove arbitrary score threshold for the test_qn test with softmax ([#6636](https://github.com/NVIDIA/cuml/pull/6636)) [@csadorf](https://github.com/csadorf)
- Do not set `n_features_in_` attribute before `fit` ([#6624](https://github.com/NVIDIA/cuml/pull/6624)) [@betatim](https://github.com/betatim)
- Map Barnes-Hut to FFT for T-SNE in cuml.accel. ([#6619](https://github.com/NVIDIA/cuml/pull/6619)) [@csadorf](https://github.com/csadorf)
- Mark sklearn test test_cross_val_predict[coo_array] as flaky. ([#6610](https://github.com/NVIDIA/cuml/pull/6610)) [@csadorf](https://github.com/csadorf)
- Do not run the scikit-learn test suite in parallel. ([#6609](https://github.com/NVIDIA/cuml/pull/6609)) [@csadorf](https://github.com/csadorf)
- Mark one kmeans and one t_sne sklearn test as flaky ([#6598](https://github.com/NVIDIA/cuml/pull/6598)) [@csadorf](https://github.com/csadorf)
- Relax threshold for failing ARIMA pytest ([#6579](https://github.com/NVIDIA/cuml/pull/6579)) [@divyegala](https://github.com/divyegala)
- Fix race condition in `_block_gemv` kernel ([#6578](https://github.com/NVIDIA/cuml/pull/6578)) [@divyegala](https://github.com/divyegala)
- Fix launch parameters for kernel shap ([#6577](https://github.com/NVIDIA/cuml/pull/6577)) [@divyegala](https://github.com/divyegala)
- Update SVC tests for CCCL update ([#6569](https://github.com/NVIDIA/cuml/pull/6569)) [@viclafargue](https://github.com/viclafargue)
- Update FIL Model Loading Parameter in FIL demo notebook ([#6562](https://github.com/NVIDIA/cuml/pull/6562)) [@csadorf](https://github.com/csadorf)
- Localise output‑dtype; remove global set_api_output_dtype calls ([#6561](https://github.com/NVIDIA/cuml/pull/6561)) [@Ofek-Haim](https://github.com/Ofek-Haim)
- Lower logistic regression digits test threshold to 0.9. ([#6552](https://github.com/NVIDIA/cuml/pull/6552)) [@csadorf](https://github.com/csadorf)
- FIX Propagate random state to numpy rng in `make_classification` ([#6518](https://github.com/NVIDIA/cuml/pull/6518)) [@betatim](https://github.com/betatim)
- Fix cudf.DataFrame construction with dict like inputs and ensure CumlArray.to_output(&#39;cudf&#39;) does not convert NaN to NA ([#6517](https://github.com/NVIDIA/cuml/pull/6517)) [@mroeschke](https://github.com/mroeschke)
- Calling gc in SVC tests ([#6514](https://github.com/NVIDIA/cuml/pull/6514)) [@viclafargue](https://github.com/viclafargue)
- Fix logger macros ([#6511](https://github.com/NVIDIA/cuml/pull/6511)) [@vyasr](https://github.com/vyasr)
- Simplicial set functions fixes and cleanups ([#6493](https://github.com/NVIDIA/cuml/pull/6493)) [@viclafargue](https://github.com/viclafargue)
- Inspect KNN graph during smooth KNN generation and fail if necessary ([#6491](https://github.com/NVIDIA/cuml/pull/6491)) [@viclafargue](https://github.com/viclafargue)
- More permanently disable BATCH_TREE_REORG ([#6374](https://github.com/NVIDIA/cuml/pull/6374)) [@wphicks](https://github.com/wphicks)

## 📖 Documentation

- Move FIL docs to user docs ([#6762](https://github.com/NVIDIA/cuml/pull/6762)) [@dantegd](https://github.com/dantegd)
- Document `cuml.accel` doesn&#39;t support `set_output` ([#6717](https://github.com/NVIDIA/cuml/pull/6717)) [@jcrist](https://github.com/jcrist)
- DOC Update zero-code-change documentation ([#6697](https://github.com/NVIDIA/cuml/pull/6697)) [@csadorf](https://github.com/csadorf)
- Document WSL2 limitation re UVM ([#6663](https://github.com/NVIDIA/cuml/pull/6663)) [@csadorf](https://github.com/csadorf)
- DOC Make `MultinomialNB` dask doctest more lax ([#6482](https://github.com/NVIDIA/cuml/pull/6482)) [@betatim](https://github.com/betatim)
- Future-proof SVC documentation ([#6481](https://github.com/NVIDIA/cuml/pull/6481)) [@viclafargue](https://github.com/viclafargue)
- DOC Add developer documentation about deprecation policy ([#6476](https://github.com/NVIDIA/cuml/pull/6476)) [@betatim](https://github.com/betatim)

## 🚀 New Features

- Multi-gpu KNN build for UMAP using all-neighbors API ([#6654](https://github.com/NVIDIA/cuml/pull/6654)) [@jinsolp](https://github.com/jinsolp)
- Add caching and object size metrics to `conda` CI builds ([#6573](https://github.com/NVIDIA/cuml/pull/6573)) [@divyegala](https://github.com/divyegala)
- Use Gram kernel from cuVS instead of Raft ([#6538](https://github.com/NVIDIA/cuml/pull/6538)) [@lowener](https://github.com/lowener)

## 🛠️ Improvements

- Migrate cuml.experimental.fil -&gt; cuml.fil ([#6799](https://github.com/NVIDIA/cuml/pull/6799)) [@hcho3](https://github.com/hcho3)
- Mark `test_uniform_grid` as flaky in sklearn tests ([#6791](https://github.com/NVIDIA/cuml/pull/6791)) [@jcrist](https://github.com/jcrist)
- Deprecate a few missed cuml-only non-keyword-only kwargs ([#6789](https://github.com/NVIDIA/cuml/pull/6789)) [@jcrist](https://github.com/jcrist)
- Delay importing of dask ([#6788](https://github.com/NVIDIA/cuml/pull/6788)) [@jcrist](https://github.com/jcrist)
- Port `DBSCAN` to `ProxyBase`/`InteropMixin` ([#6786](https://github.com/NVIDIA/cuml/pull/6786)) [@jcrist](https://github.com/jcrist)
- Deprecate device selection ([#6784](https://github.com/NVIDIA/cuml/pull/6784)) [@jcrist](https://github.com/jcrist)
- use &#39;rapids-init-pip&#39; in wheel CI, other CI changes ([#6779](https://github.com/NVIDIA/cuml/pull/6779)) [@jameslamb](https://github.com/jameslamb)
- Support `SVC` and `SVR` in `cuml.accel` ([#6778](https://github.com/NVIDIA/cuml/pull/6778)) [@jcrist](https://github.com/jcrist)
- chore: reduce ceiling of `cuml` wheel size ([#6771](https://github.com/NVIDIA/cuml/pull/6771)) [@gforsyth](https://github.com/gforsyth)
- Port KMeans to use `ProxyBase`/`InteropMixin` ([#6770](https://github.com/NVIDIA/cuml/pull/6770)) [@jcrist](https://github.com/jcrist)
- Port `sklearn.manifold` wrappers to use `ProxyBase` ([#6761](https://github.com/NVIDIA/cuml/pull/6761)) [@jcrist](https://github.com/jcrist)
- Fix dask warning in FIL demo notebook ([#6758](https://github.com/NVIDIA/cuml/pull/6758)) [@jcrist](https://github.com/jcrist)
- Remove deprecations for 25.06 ([#6754](https://github.com/NVIDIA/cuml/pull/6754)) [@jcrist](https://github.com/jcrist)
- Check that `as_sklearn`&#39;ed models can be fitted ([#6751](https://github.com/NVIDIA/cuml/pull/6751)) [@betatim](https://github.com/betatim)
- Implement __sklearn_tags__ on mixin using vendored compat module. ([#6746](https://github.com/NVIDIA/cuml/pull/6746)) [@csadorf](https://github.com/csadorf)
- Test that proxied method signatures match their CPU versions ([#6740](https://github.com/NVIDIA/cuml/pull/6740)) [@jcrist](https://github.com/jcrist)
- Remove upper bound constraint on xgboost. ([#6739](https://github.com/NVIDIA/cuml/pull/6739)) [@csadorf](https://github.com/csadorf)
- Deprecate non-keyword-only cuml-only kwargs ([#6738](https://github.com/NVIDIA/cuml/pull/6738)) [@jcrist](https://github.com/jcrist)
- Fix DistributedDataHandler.calculate_parts_to_sizes ([#6737](https://github.com/NVIDIA/cuml/pull/6737)) [@TomAugspurger](https://github.com/TomAugspurger)
- Finish CUDA 12.9 migration and use branch-25.06 workflows ([#6733](https://github.com/NVIDIA/cuml/pull/6733)) [@bdice](https://github.com/bdice)
- Remove use of legacy FIL from Python cuML ([#6728](https://github.com/NVIDIA/cuml/pull/6728)) [@hcho3](https://github.com/hcho3)
- Update Dask RandomForest classifier ([#6726](https://github.com/NVIDIA/cuml/pull/6726)) [@TomAugspurger](https://github.com/TomAugspurger)
- Update to clang 20 ([#6724](https://github.com/NVIDIA/cuml/pull/6724)) [@bdice](https://github.com/bdice)
- Use matplotlib-base in docs/notebooks environments. ([#6723](https://github.com/NVIDIA/cuml/pull/6723)) [@bdice](https://github.com/bdice)
- FIX Allow prereleases when evaluating xfail conditions ([#6722](https://github.com/NVIDIA/cuml/pull/6722)) [@betatim](https://github.com/betatim)
- Quote head_rev in conda recipes ([#6701](https://github.com/NVIDIA/cuml/pull/6701)) [@bdice](https://github.com/bdice)
- Don&#39;t setup managed memory if rmm isn&#39;t using default settings ([#6700](https://github.com/NVIDIA/cuml/pull/6700)) [@jcrist](https://github.com/jcrist)
- Expand sklearn matrix used in tests ([#6698](https://github.com/NVIDIA/cuml/pull/6698)) [@jcrist](https://github.com/jcrist)
- Port `sklearn.decomposition` to new proxy estimator structure ([#6695](https://github.com/NVIDIA/cuml/pull/6695)) [@jcrist](https://github.com/jcrist)
- fix(rattler): add librmm to host for libcuml to fix overlinking error ([#6686](https://github.com/NVIDIA/cuml/pull/6686)) [@gforsyth](https://github.com/gforsyth)
- CUDA 12.9 use updated compression flags ([#6684](https://github.com/NVIDIA/cuml/pull/6684)) [@robertmaynard](https://github.com/robertmaynard)
- Increase scikit-learn compatibility to 1.5+ ([#6666](https://github.com/NVIDIA/cuml/pull/6666)) [@csadorf](https://github.com/csadorf)
- Revert &quot;CI: Increase nightly CI check window to 14 days. ([#6628)&quot; (#6660](https://github.com/NVIDIA/cuml/pull/6628)&quot; (#6660)) [@csadorf](https://github.com/csadorf)
- Remove cuml-cpu and CPU &lt;&gt; GPU docs ([#6651](https://github.com/NVIDIA/cuml/pull/6651)) [@jcrist](https://github.com/jcrist)
- More dask==2025.4.1 compatibility ([#6645](https://github.com/NVIDIA/cuml/pull/6645)) [@TomAugspurger](https://github.com/TomAugspurger)
- Use RBC from cuVS ([#6644](https://github.com/NVIDIA/cuml/pull/6644)) [@divyegala](https://github.com/divyegala)
- Exclude librmm.so from auditwheel ([#6632](https://github.com/NVIDIA/cuml/pull/6632)) [@bdice](https://github.com/bdice)
- Enhance accelerator parity testing framework ([#6629](https://github.com/NVIDIA/cuml/pull/6629)) [@csadorf](https://github.com/csadorf)
- CI: Increase nightly CI check window to 14 days. ([#6628](https://github.com/NVIDIA/cuml/pull/6628)) [@csadorf](https://github.com/csadorf)
- Add support for Python 3.13 ([#6616](https://github.com/NVIDIA/cuml/pull/6616)) [@gforsyth](https://github.com/gforsyth)
- Compatibility with Dask 2025.4.1 ([#6614](https://github.com/NVIDIA/cuml/pull/6614)) [@TomAugspurger](https://github.com/TomAugspurger)
- New Estimator Proxy architecture ([#6613](https://github.com/NVIDIA/cuml/pull/6613)) [@jcrist](https://github.com/jcrist)
- Using cuVS NN Descent in UMAP ([#6611](https://github.com/NVIDIA/cuml/pull/6611)) [@jinsolp](https://github.com/jinsolp)
- Make scikit-learn a required dependency ([#6608](https://github.com/NVIDIA/cuml/pull/6608)) [@betatim](https://github.com/betatim)
- Speedup dask `LogisticRegression` tests ([#6607](https://github.com/NVIDIA/cuml/pull/6607)) [@jcrist](https://github.com/jcrist)
- Add GitHub Actions workflow for PR and issue project field automation ([#6602](https://github.com/NVIDIA/cuml/pull/6602)) [@csadorf](https://github.com/csadorf)
- Use `nogil` for `libcuml` calls ([#6601](https://github.com/NVIDIA/cuml/pull/6601)) [@jcrist](https://github.com/jcrist)
- Remove `cuml.common.cuda` ([#6600](https://github.com/NVIDIA/cuml/pull/6600)) [@jcrist](https://github.com/jcrist)
- Remove _most_ of `import_utils.py` ([#6599](https://github.com/NVIDIA/cuml/pull/6599)) [@jcrist](https://github.com/jcrist)
- Remove guarded imports of `scipy` ([#6596](https://github.com/NVIDIA/cuml/pull/6596)) [@jcrist](https://github.com/jcrist)
- Remove remainder of code branching on `cuml-cpu` support ([#6593](https://github.com/NVIDIA/cuml/pull/6593)) [@jcrist](https://github.com/jcrist)
- Add timeout to scikit-learn test suite ([#6591](https://github.com/NVIDIA/cuml/pull/6591)) [@jcrist](https://github.com/jcrist)
- Remove `safe_imports.py` ([#6588](https://github.com/NVIDIA/cuml/pull/6588)) [@jcrist](https://github.com/jcrist)
- Remove usage of `cudf.core.buffer` ([#6586](https://github.com/NVIDIA/cuml/pull/6586)) [@mroeschke](https://github.com/mroeschke)
- Split wheel tests into two jobs ([#6585](https://github.com/NVIDIA/cuml/pull/6585)) [@jcrist](https://github.com/jcrist)
- Only run one `cudf.pandas` enabled build per PR ([#6584](https://github.com/NVIDIA/cuml/pull/6584)) [@jcrist](https://github.com/jcrist)
- Don&#39;t run memleak tests in CI ([#6583](https://github.com/NVIDIA/cuml/pull/6583)) [@jcrist](https://github.com/jcrist)
- Documentation and Testing Infrastructure Updates ([#6580](https://github.com/NVIDIA/cuml/pull/6580)) [@csadorf](https://github.com/csadorf)
- Move Random Forest n_streams UserWarning out of execution ([#6574](https://github.com/NVIDIA/cuml/pull/6574)) [@ncclementi](https://github.com/ncclementi)
- Remove conditional compilation blocks leftover from `cuml-cpu` ([#6572](https://github.com/NVIDIA/cuml/pull/6572)) [@jcrist](https://github.com/jcrist)
- Download build artifacts from Github for CI ([#6566](https://github.com/NVIDIA/cuml/pull/6566)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- Don&#39;t exclude sklearn tests from acceleration ([#6565](https://github.com/NVIDIA/cuml/pull/6565)) [@jcrist](https://github.com/jcrist)
- Add .git-blame-ignore-revs ([#6564](https://github.com/NVIDIA/cuml/pull/6564)) [@csadorf](https://github.com/csadorf)
- Add `cuml.accel.is_proxy` ([#6559](https://github.com/NVIDIA/cuml/pull/6559)) [@jcrist](https://github.com/jcrist)
- Update KMeans test to improve stability by setting n_init to 2 ([#6553) (#6555](https://github.com/NVIDIA/cuml/pull/6553) (#6555)) [@csadorf](https://github.com/csadorf)
- Support `weights=&quot;distance&quot;` for `KNeighbors*` in `cuml.accel` ([#6554](https://github.com/NVIDIA/cuml/pull/6554)) [@jcrist](https://github.com/jcrist)
- Improve detection of scikit-learn parity regressions ([#6553](https://github.com/NVIDIA/cuml/pull/6553)) [@csadorf](https://github.com/csadorf)
- Revise cuml.accel module patching and import mechanism ([#6550](https://github.com/NVIDIA/cuml/pull/6550)) [@jcrist](https://github.com/jcrist)
- Add isort to pre-commit hooks ([#6549](https://github.com/NVIDIA/cuml/pull/6549)) [@csadorf](https://github.com/csadorf)
- Refactor dataset generation and testing utilities ([#6548](https://github.com/NVIDIA/cuml/pull/6548)) [@csadorf](https://github.com/csadorf)
- Move `cuml.accel` tests out into own directory ([#6547](https://github.com/NVIDIA/cuml/pull/6547)) [@jcrist](https://github.com/jcrist)
- refactor(rattler): enable strict channel priority for builds ([#6546](https://github.com/NVIDIA/cuml/pull/6546)) [@gforsyth](https://github.com/gforsyth)
- Remove unnecessary CudfBuffer and vendor cuDF min_signed_type function ([#6543](https://github.com/NVIDIA/cuml/pull/6543)) [@mroeschke](https://github.com/mroeschke)
- Remove `_experimental_dispatching` attribute ([#6542](https://github.com/NVIDIA/cuml/pull/6542)) [@jcrist](https://github.com/jcrist)
- Avoid private attributes on cuDF objects ([#6537](https://github.com/NVIDIA/cuml/pull/6537)) [@mroeschke](https://github.com/mroeschke)
- Reduce CI test combinations for scikit-learn acceleration tests ([#6536](https://github.com/NVIDIA/cuml/pull/6536)) [@csadorf](https://github.com/csadorf)
- Vendor RAPIDS.cmake ([#6531](https://github.com/NVIDIA/cuml/pull/6531)) [@bdice](https://github.com/bdice)
- Disable building and uploading cuml-cpu in CI ([#6529](https://github.com/NVIDIA/cuml/pull/6529)) [@dantegd](https://github.com/dantegd)
- Remove CUDA whole compilation ODR violations ([#6525](https://github.com/NVIDIA/cuml/pull/6525)) [@divyegala](https://github.com/divyegala)
- SLHC to Migrate to cuVS Agglomerative from RAFT ([#6507](https://github.com/NVIDIA/cuml/pull/6507)) [@tarang-jain](https://github.com/tarang-jain)
- Solve forward merger conflicts of 6435 ([#6501](https://github.com/NVIDIA/cuml/pull/6501)) [@csadorf](https://github.com/csadorf)
- Add minimal import check to CI ([#6484](https://github.com/NVIDIA/cuml/pull/6484)) [@jcrist](https://github.com/jcrist)
- Moving wheel builds to specified location and uploading build artifacts to Github ([#6472](https://github.com/NVIDIA/cuml/pull/6472)) [@VenkateshJaya](https://github.com/VenkateshJaya)

# cuml 25.04.00 (9 Apr 2025)

## 🚨 Breaking Changes

- Promote experimental FIL to stable ([#6464](https://github.com/NVIDIA/cuml/pull/6464)) [@wphicks](https://github.com/wphicks)
- Support non-trivial `classes_` in `LogisticRegression` ([#6346](https://github.com/NVIDIA/cuml/pull/6346)) [@jcrist](https://github.com/jcrist)
- Use new rapids-logger library ([#6289](https://github.com/NVIDIA/cuml/pull/6289)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Unblock CI for 25.04 ([#6519](https://github.com/NVIDIA/cuml/pull/6519)) [@csadorf](https://github.com/csadorf)
- Skip test_rf_classification_seed for cudf.pandas tests. ([#6500](https://github.com/NVIDIA/cuml/pull/6500)) [@csadorf](https://github.com/csadorf)
- Remove dask to sparse workarounds ([#6489](https://github.com/NVIDIA/cuml/pull/6489)) [@TomAugspurger](https://github.com/TomAugspurger)
- Fix accidental required sklearn import ([#6483](https://github.com/NVIDIA/cuml/pull/6483)) [@jcrist](https://github.com/jcrist)
- FIX Pass instances to the constructor of the meta-estimator in `SVC` ([#6471](https://github.com/NVIDIA/cuml/pull/6471)) [@betatim](https://github.com/betatim)
- Fix compiler dependency on ARM ([#6456](https://github.com/NVIDIA/cuml/pull/6456)) [@bdice](https://github.com/bdice)
- Revert &quot;Temporarily increase `max_days_without_success` ([#6390)&quot; (#6455](https://github.com/NVIDIA/cuml/pull/6390)&quot; (#6455)) [@divyegala](https://github.com/divyegala)
- Enhance argument handling in UniversalBase to accept NoneType ([#6453](https://github.com/NVIDIA/cuml/pull/6453)) [@csadorf](https://github.com/csadorf)
- Fix UMAP transform ([#6449](https://github.com/NVIDIA/cuml/pull/6449)) [@viclafargue](https://github.com/viclafargue)
- Fallback to CPU for sparse inputs for KMeans ([#6448](https://github.com/NVIDIA/cuml/pull/6448)) [@csadorf](https://github.com/csadorf)
- Limited support for array-like inputs ([#6442](https://github.com/NVIDIA/cuml/pull/6442)) [@csadorf](https://github.com/csadorf)
- Fix `test_accuracy_score` test on `cudf.pandas` build ([#6439](https://github.com/NVIDIA/cuml/pull/6439)) [@jcrist](https://github.com/jcrist)
- Support positional args in `cuml.accel` estimators ([#6423](https://github.com/NVIDIA/cuml/pull/6423)) [@jcrist](https://github.com/jcrist)
- Fix HDBSCAN python docs related to `metric` ([#6422](https://github.com/NVIDIA/cuml/pull/6422)) [@divyegala](https://github.com/divyegala)
- declare runtime dependency on &#39;packaging&#39;, update scikit-learn and hdbscan for cuml-cpu to match cuml ([#6420](https://github.com/NVIDIA/cuml/pull/6420)) [@jameslamb](https://github.com/jameslamb)
- Fix forwarding of `initial_alpha` and `learning_rate` in UMAP ([#6417](https://github.com/NVIDIA/cuml/pull/6417)) [@jcrist](https://github.com/jcrist)
- Implement Ridge .solver_ estimated attribute ([#6415](https://github.com/NVIDIA/cuml/pull/6415)) [@csadorf](https://github.com/csadorf)
- Fix multi-target prediction in linear models ([#6414](https://github.com/NVIDIA/cuml/pull/6414)) [@csadorf](https://github.com/csadorf)
- Ensure `output_type=&quot;pandas&quot;` returns user-facing pandas ([#6407](https://github.com/NVIDIA/cuml/pull/6407)) [@jcrist](https://github.com/jcrist)
- Ensure `LinearSVC` supports all input types ([#6404](https://github.com/NVIDIA/cuml/pull/6404)) [@jcrist](https://github.com/jcrist)
- Make UMAP callback pickleable ([#6402](https://github.com/NVIDIA/cuml/pull/6402)) [@jcrist](https://github.com/jcrist)
- Correctly align trees in experimental FIL ([#6397](https://github.com/NVIDIA/cuml/pull/6397)) [@wphicks](https://github.com/wphicks)
- A few log level handling cleanups ([#6393](https://github.com/NVIDIA/cuml/pull/6393)) [@jcrist](https://github.com/jcrist)
- Fix cupy error in `KernelDensity` with `epanechnikov` ([#6388](https://github.com/NVIDIA/cuml/pull/6388)) [@jcrist](https://github.com/jcrist)
- Check if both ground truth and result are NaN in MSEObjectiveTest for RF ([#6387](https://github.com/NVIDIA/cuml/pull/6387)) [@wphicks](https://github.com/wphicks)
- Support non-native endian inputs to `LabelEncoder` ([#6384](https://github.com/NVIDIA/cuml/pull/6384)) [@jcrist](https://github.com/jcrist)
- Fix hypothesis test in `test_kernel_ridge.py` ([#6382](https://github.com/NVIDIA/cuml/pull/6382)) [@jcrist](https://github.com/jcrist)
- Remove debug logging from nearest neighbors test ([#6376](https://github.com/NVIDIA/cuml/pull/6376)) [@csadorf](https://github.com/csadorf)
- Ensure FIL CPU can be run without an available GPU ([#6373](https://github.com/NVIDIA/cuml/pull/6373)) [@wphicks](https://github.com/wphicks)
- Correctly pass through unhashable objects during hyperparam lookup ([#6369](https://github.com/NVIDIA/cuml/pull/6369)) [@wphicks](https://github.com/wphicks)
- Correct translation of RandomForest criterion hyperparameter ([#6363](https://github.com/NVIDIA/cuml/pull/6363)) [@wphicks](https://github.com/wphicks)
- UMAP fallback improvements ([#6358](https://github.com/NVIDIA/cuml/pull/6358)) [@viclafargue](https://github.com/viclafargue)
- A few GPU&lt;-&gt;CPU interop fixes ([#6355](https://github.com/NVIDIA/cuml/pull/6355)) [@jcrist](https://github.com/jcrist)
- Fix output type of `KernelRidge.predict` ([#6354](https://github.com/NVIDIA/cuml/pull/6354)) [@jcrist](https://github.com/jcrist)
- Skip test_extract_partitions_shape test. ([#6338](https://github.com/NVIDIA/cuml/pull/6338)) [@csadorf](https://github.com/csadorf)
- Require sphinx&lt;8.2.0 ([#6336](https://github.com/NVIDIA/cuml/pull/6336)) [@csadorf](https://github.com/csadorf)
- Fix call to NearestNeighbors.kneighbors() without args. ([#6333](https://github.com/NVIDIA/cuml/pull/6333)) [@csadorf](https://github.com/csadorf)
- Commenting out debug print statements in UMAP ([#6332](https://github.com/NVIDIA/cuml/pull/6332)) [@viclafargue](https://github.com/viclafargue)
- Correct UMAP dispatch trigger condition ([#6330](https://github.com/NVIDIA/cuml/pull/6330)) [@viclafargue](https://github.com/viclafargue)
- Fix issue caused by None arg provided by sklearn pipeline ([#6326](https://github.com/NVIDIA/cuml/pull/6326)) [@viclafargue](https://github.com/viclafargue)
- Improvements for interoperability of Random Forest ([#6320](https://github.com/NVIDIA/cuml/pull/6320)) [@dantegd](https://github.com/dantegd)
- Improvements for PCA CPU interoperability ([#6319](https://github.com/NVIDIA/cuml/pull/6319)) [@dantegd](https://github.com/dantegd)
- Plumb `metric` and `metric_kwds` through to UMAP with `nn_descent` ([#6304](https://github.com/NVIDIA/cuml/pull/6304)) [@jcrist](https://github.com/jcrist)
- Use `L2Expanded`/`L2SqrtExpanded` instead of the `*Unexpanded` variants ([#6303](https://github.com/NVIDIA/cuml/pull/6303)) [@jcrist](https://github.com/jcrist)
- Fix UMAP issues with large inputs ([#6245](https://github.com/NVIDIA/cuml/pull/6245)) [@viclafargue](https://github.com/viclafargue)
- Fix `LogisticRegression.decision_function` output shape ([#6235](https://github.com/NVIDIA/cuml/pull/6235)) [@jcrist](https://github.com/jcrist)
- Compute `noise_variance_` in PCA implementation ([#6234](https://github.com/NVIDIA/cuml/pull/6234)) [@jcrist](https://github.com/jcrist)
- Multiple CPU interop fixes for serialization and cloning ([#6223](https://github.com/NVIDIA/cuml/pull/6223)) [@dantegd](https://github.com/dantegd)

## 📖 Documentation

- Clarify support in docs ([#6452](https://github.com/NVIDIA/cuml/pull/6452)) [@bdice](https://github.com/bdice)
- Provide docs for cuml.accel command line feature ([#6322](https://github.com/NVIDIA/cuml/pull/6322)) [@wphicks](https://github.com/wphicks)

## 🚀 New Features

- Promote experimental FIL to stable ([#6464](https://github.com/NVIDIA/cuml/pull/6464)) [@wphicks](https://github.com/wphicks)
- UMAP 32bits dispatch mechanism ([#6314](https://github.com/NVIDIA/cuml/pull/6314)) [@viclafargue](https://github.com/viclafargue)
- Add aarch64 conda environments ([#6226](https://github.com/NVIDIA/cuml/pull/6226)) [@jcrist](https://github.com/jcrist)
- SVM cpu/gpu interop ([#6089](https://github.com/NVIDIA/cuml/pull/6089)) [@viclafargue](https://github.com/viclafargue)

## 🛠️ Improvements

- Don&#39;t use private APIs in FIL notebook ([#6488](https://github.com/NVIDIA/cuml/pull/6488)) [@jcrist](https://github.com/jcrist)
- Only use released versions of dask ([#6485](https://github.com/NVIDIA/cuml/pull/6485)) [@jcrist](https://github.com/jcrist)
- Move SVM working set log to debug ([#6477](https://github.com/NVIDIA/cuml/pull/6477)) [@jcrist](https://github.com/jcrist)
- C++ `SG_RF_TEST` OOMs in 11.4 nightly ([#6474](https://github.com/NVIDIA/cuml/pull/6474)) [@divyegala](https://github.com/divyegala)
- Revert &quot;ci: pre-filter 11.4 jobs before they are enabled in shared workflows ([#6447)&quot; (#6470](https://github.com/NVIDIA/cuml/pull/6447)&quot; (#6470)) [@divyegala](https://github.com/divyegala)
- Support `__len__` in ensemble estimators ([#6468](https://github.com/NVIDIA/cuml/pull/6468)) [@jcrist](https://github.com/jcrist)
- Add deprecation warning to `cuml-cpu` builds ([#6466](https://github.com/NVIDIA/cuml/pull/6466)) [@jcrist](https://github.com/jcrist)
- Fix template argument of `DeviceSegmentedReduce` kernels ([#6459](https://github.com/NVIDIA/cuml/pull/6459)) [@miscco](https://github.com/miscco)
- Add scikit-learn acceleration tests to nightly CI tests ([#6457](https://github.com/NVIDIA/cuml/pull/6457)) [@csadorf](https://github.com/csadorf)
- Fix benchmark builds in devcontainers ([#6450](https://github.com/NVIDIA/cuml/pull/6450)) [@bdice](https://github.com/bdice)
- ci: pre-filter 11.4 jobs before they are enabled in shared workflows ([#6447](https://github.com/NVIDIA/cuml/pull/6447)) [@gforsyth](https://github.com/gforsyth)
- Reenable more UMAP tests on ARM ([#6446](https://github.com/NVIDIA/cuml/pull/6446)) [@jcrist](https://github.com/jcrist)
- Clean up after removing KMeans deprecation warning ([#6445](https://github.com/NVIDIA/cuml/pull/6445)) [@betatim](https://github.com/betatim)
- Port all conda recipes to `rattler-build` ([#6440](https://github.com/NVIDIA/cuml/pull/6440)) [@gforsyth](https://github.com/gforsyth)
- Remove the deprecation warning and switch to new default ([#6433](https://github.com/NVIDIA/cuml/pull/6433)) [@betatim](https://github.com/betatim)
- Remove old deprecations ([#6427](https://github.com/NVIDIA/cuml/pull/6427)) [@jcrist](https://github.com/jcrist)
- Fix oob_score=True handling with accelerator ([#6425](https://github.com/NVIDIA/cuml/pull/6425)) [@betatim](https://github.com/betatim)
- remove unused imports ([#6424](https://github.com/NVIDIA/cuml/pull/6424)) [@jameslamb](https://github.com/jameslamb)
- Disable codecov comments ([#6419](https://github.com/NVIDIA/cuml/pull/6419)) [@jcrist](https://github.com/jcrist)
- Require hypothesis tests have explicit examples ([#6409](https://github.com/NVIDIA/cuml/pull/6409)) [@aamijar](https://github.com/aamijar)
- Improve `accuracy_score` compatibility with `sklearn`. ([#6406](https://github.com/NVIDIA/cuml/pull/6406)) [@jcrist](https://github.com/jcrist)
- Drop `cub{Min, Max}` ([#6405](https://github.com/NVIDIA/cuml/pull/6405)) [@miscco](https://github.com/miscco)
- Move `accel` implementation fully to `cuml.accel` ([#6401](https://github.com/NVIDIA/cuml/pull/6401)) [@jcrist](https://github.com/jcrist)
- cuml-cpu: fix import issues, enable conda import tests ([#6400](https://github.com/NVIDIA/cuml/pull/6400)) [@jameslamb](https://github.com/jameslamb)
- Drop `numba_utils` ([#6396](https://github.com/NVIDIA/cuml/pull/6396)) [@jakirkham](https://github.com/jakirkham)
- Simplify `cpu_to_gpu` and `gpu_to_cpu` ([#6395](https://github.com/NVIDIA/cuml/pull/6395)) [@jcrist](https://github.com/jcrist)
- Adds get_level() ([#6392](https://github.com/NVIDIA/cuml/pull/6392)) [@Intron7](https://github.com/Intron7)
- Temporarily increase `max_days_without_success` ([#6390](https://github.com/NVIDIA/cuml/pull/6390)) [@jcrist](https://github.com/jcrist)
- Drop deprecated CCCL functors ([#6389](https://github.com/NVIDIA/cuml/pull/6389)) [@miscco](https://github.com/miscco)
- Solve forward merger conflicts of #6313 ([#6385](https://github.com/NVIDIA/cuml/pull/6385)) [@dantegd](https://github.com/dantegd)
- CI: Run extended accel mode tests ([#6377](https://github.com/NVIDIA/cuml/pull/6377)) [@csadorf](https://github.com/csadorf)
- Use conda-build instead of conda-mambabuild ([#6368](https://github.com/NVIDIA/cuml/pull/6368)) [@bdice](https://github.com/bdice)
- Don&#39;t log coverage output to terminal in CI ([#6367](https://github.com/NVIDIA/cuml/pull/6367)) [@jcrist](https://github.com/jcrist)
- Remove `KernelRidge` from `cuml.accel` ([#6361](https://github.com/NVIDIA/cuml/pull/6361)) [@jcrist](https://github.com/jcrist)
- Enable UVM in cuml.accel ([#6359](https://github.com/NVIDIA/cuml/pull/6359)) [@wphicks](https://github.com/wphicks)
- Drop deprecated `CUB` iterators in favor of `Thrust` iterators ([#6356](https://github.com/NVIDIA/cuml/pull/6356)) [@miscco](https://github.com/miscco)
- Adjust accel verbosity ([#6353](https://github.com/NVIDIA/cuml/pull/6353)) [@csadorf](https://github.com/csadorf)
- Avoid importing `umap` with `import cuml` ([#6352](https://github.com/NVIDIA/cuml/pull/6352)) [@jcrist](https://github.com/jcrist)
- Make proxy module patching skip patching if patched library is not installed ([#6348](https://github.com/NVIDIA/cuml/pull/6348)) [@dantegd](https://github.com/dantegd)
- Add new `cuml.accel` module ([#6347](https://github.com/NVIDIA/cuml/pull/6347)) [@jcrist](https://github.com/jcrist)
- Support non-trivial `classes_` in `LogisticRegression` ([#6346](https://github.com/NVIDIA/cuml/pull/6346)) [@jcrist](https://github.com/jcrist)
- Make `r2_score` compatible with Scikit-Learn ([#6337](https://github.com/NVIDIA/cuml/pull/6337)) [@jcrist](https://github.com/jcrist)
- Remove left-over code related to strict mode. ([#6335](https://github.com/NVIDIA/cuml/pull/6335)) [@csadorf](https://github.com/csadorf)
- Make sure specifying default value or nothing behaves the same in `KMeans` ([#6331](https://github.com/NVIDIA/cuml/pull/6331)) [@betatim](https://github.com/betatim)
- Make KernelRidge inherit from UniversalBase ([#6327](https://github.com/NVIDIA/cuml/pull/6327)) [@betatim](https://github.com/betatim)
- Reduce peak memory in `UMAP.fit`/`UMAP.fit_transform` ([#6323](https://github.com/NVIDIA/cuml/pull/6323)) [@jcrist](https://github.com/jcrist)
- Consolidate more Conda solves in CI ([#6321](https://github.com/NVIDIA/cuml/pull/6321)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Require CMake 3.30.4 ([#6318](https://github.com/NVIDIA/cuml/pull/6318)) [@robertmaynard](https://github.com/robertmaynard)
- Copy attributes from the original class to the proxy ([#6306](https://github.com/NVIDIA/cuml/pull/6306)) [@betatim](https://github.com/betatim)
- Use shared-workflows branch-25.04 ([#6298](https://github.com/NVIDIA/cuml/pull/6298)) [@bdice](https://github.com/bdice)
- Add build_type input field for `test.yaml` ([#6296](https://github.com/NVIDIA/cuml/pull/6296)) [@gforsyth](https://github.com/gforsyth)
- Do not use `LEGACY_PTX_ARCH` argument for `BlockReduce` ([#6290](https://github.com/NVIDIA/cuml/pull/6290)) [@miscco](https://github.com/miscco)
- Use new rapids-logger library ([#6289](https://github.com/NVIDIA/cuml/pull/6289)) [@vyasr](https://github.com/vyasr)
- disallow fallback to Make in wheel builds ([#6286](https://github.com/NVIDIA/cuml/pull/6286)) [@jameslamb](https://github.com/jameslamb)
- Forward-merge branch-25.02 to branch-25.04 ([#6283](https://github.com/NVIDIA/cuml/pull/6283)) [@bdice](https://github.com/bdice)
- Migrate to NVKS for amd64 CI runners ([#6280](https://github.com/NVIDIA/cuml/pull/6280)) [@bdice](https://github.com/bdice)
- Add `verify-codeowners` hook ([#6275](https://github.com/NVIDIA/cuml/pull/6275)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Speed up FIL and TSNE pytests in CI ([#6270](https://github.com/NVIDIA/cuml/pull/6270)) [@dantegd](https://github.com/dantegd)
- Bracnh 25.04 merge branch 25.02 ([#6268](https://github.com/NVIDIA/cuml/pull/6268)) [@vyasr](https://github.com/vyasr)
- Add shellcheck to pre-commit and fix warnings ([#6246](https://github.com/NVIDIA/cuml/pull/6246)) [@gforsyth](https://github.com/gforsyth)
- Prepare for n_init=auto in KMeans ([#6142](https://github.com/NVIDIA/cuml/pull/6142)) [@betatim](https://github.com/betatim)

# cuml 25.02.00 (13 Feb 2025)

## 🚨 Breaking Changes

- update pip devcontainers to UCX 1.18 ([#6249](https://github.com/NVIDIA/cuml/pull/6249)) [@jameslamb](https://github.com/jameslamb)

## 🐛 Bug Fixes

- Fix dask logistic regression segfault/hang due to wrongly typed variable ([#6281](https://github.com/NVIDIA/cuml/pull/6281)) [@dantegd](https://github.com/dantegd)
- Log UMAP arrays at trace verbosity level. ([#6274](https://github.com/NVIDIA/cuml/pull/6274)) [@csadorf](https://github.com/csadorf)
- Ensure all method signatures are sklearn compatible ([#6260](https://github.com/NVIDIA/cuml/pull/6260)) [@jcrist](https://github.com/jcrist)
- Fix UMAP transform illegal memory access error when data_on_host=True ([#6259](https://github.com/NVIDIA/cuml/pull/6259)) [@csadorf](https://github.com/csadorf)
- Ignore cudf&#39;s __dataframe__ deprecation. ([#6229](https://github.com/NVIDIA/cuml/pull/6229)) [@bdice](https://github.com/bdice)
- Fix for porter stemmer for cuDF change and ARIMA pytest adjustments ([#6227](https://github.com/NVIDIA/cuml/pull/6227)) [@dantegd](https://github.com/dantegd)
- Avoid duplicate log entries ([#6222](https://github.com/NVIDIA/cuml/pull/6222)) [@jcrist](https://github.com/jcrist)
- Further fixes for Scipy 1.15 update for PR and nightly CI ([#6213](https://github.com/NVIDIA/cuml/pull/6213)) [@dantegd](https://github.com/dantegd)
- Update setulb call from Scipy for updated signature of 1.15 ([#6207](https://github.com/NVIDIA/cuml/pull/6207)) [@dantegd](https://github.com/dantegd)
- Adjust test_kmeans to avoid false positive failures ([#6193](https://github.com/NVIDIA/cuml/pull/6193)) [@dantegd](https://github.com/dantegd)
- Adjust margin of logistic regression `log_proba` pytest to avoid false positive failures ([#6188](https://github.com/NVIDIA/cuml/pull/6188)) [@dantegd](https://github.com/dantegd)
- Skip flaky test of kernel_density in nightly job of CUDA 12.0.1 ([#6184](https://github.com/NVIDIA/cuml/pull/6184)) [@dantegd](https://github.com/dantegd)
- Try to reduce network usage in cuML tests. ([#6174](https://github.com/NVIDIA/cuml/pull/6174)) [@bdice](https://github.com/bdice)
- cuML dask fixes to unblock CI ([#6170](https://github.com/NVIDIA/cuml/pull/6170)) [@dantegd](https://github.com/dantegd)
- Remap BATCH_TREE_REORG to TREE_REORG in FIL ([#6161](https://github.com/NVIDIA/cuml/pull/6161)) [@wphicks](https://github.com/wphicks)

## 📖 Documentation

- Fixes github link from pyx files ([#6202](https://github.com/NVIDIA/cuml/pull/6202)) [@thomasjpfan](https://github.com/thomasjpfan)

## 🚀 New Features

- Allow CUDA ODR violations in 25.02 ([#6264](https://github.com/NVIDIA/cuml/pull/6264)) [@robertmaynard](https://github.com/robertmaynard)
- Define block size for sm_120 ([#6250](https://github.com/NVIDIA/cuml/pull/6250)) [@robertmaynard](https://github.com/robertmaynard)
- Support `alpha=0` in `Ridge` ([#6236](https://github.com/NVIDIA/cuml/pull/6236)) [@jcrist](https://github.com/jcrist)
- Add `as_sklearn` and `from_sklearn` APIs to serialize to CPU sklearn-estimators for supported models ([#6102](https://github.com/NVIDIA/cuml/pull/6102)) [@dantegd](https://github.com/dantegd)

## 🛠️ Improvements

- Use `rapids-pip-retry` in CI jobs that might need retries ([#6293](https://github.com/NVIDIA/cuml/pull/6293)) [@gforsyth](https://github.com/gforsyth)
- Avoid large device allocation in UMAP with nndescent ([#6292](https://github.com/NVIDIA/cuml/pull/6292)) [@jcrist](https://github.com/jcrist)
- Revert CUDA 12.8 shared workflow branch changes ([#6282](https://github.com/NVIDIA/cuml/pull/6282)) [@vyasr](https://github.com/vyasr)
- Build and test with CUDA 12.8.0 ([#6266](https://github.com/NVIDIA/cuml/pull/6266)) [@bdice](https://github.com/bdice)
- update pip devcontainers to UCX 1.18 ([#6249](https://github.com/NVIDIA/cuml/pull/6249)) [@jameslamb](https://github.com/jameslamb)
- Drop deprecated thrust features and replace with libcu++ ones ([#6248](https://github.com/NVIDIA/cuml/pull/6248)) [@miscco](https://github.com/miscco)
- Add upper bound to prevent usage of numba 0.61.0 ([#6244](https://github.com/NVIDIA/cuml/pull/6244)) [@galipremsagar](https://github.com/galipremsagar)
- Normalize whitespace ([#6238](https://github.com/NVIDIA/cuml/pull/6238)) [@bdice](https://github.com/bdice)
- Rename cpp/test to cpp/tests. ([#6237](https://github.com/NVIDIA/cuml/pull/6237)) [@bdice](https://github.com/bdice)
- Use cuda.bindings layout. ([#6233](https://github.com/NVIDIA/cuml/pull/6233)) [@bdice](https://github.com/bdice)
- Skip dispatching to GPU for unimplemented metrics in UMAP ([#6224](https://github.com/NVIDIA/cuml/pull/6224)) [@betatim](https://github.com/betatim)
- Use GCC 13 in CUDA 12 conda builds. ([#6221](https://github.com/NVIDIA/cuml/pull/6221)) [@bdice](https://github.com/bdice)
- declare cuda-python dependency for wheels, other small packaging changes ([#6217](https://github.com/NVIDIA/cuml/pull/6217)) [@jameslamb](https://github.com/jameslamb)
- Bump Treelite to 4.4.1 ([#6212](https://github.com/NVIDIA/cuml/pull/6212)) [@hcho3](https://github.com/hcho3)
- Support raft&#39;s logger targets ([#6208](https://github.com/NVIDIA/cuml/pull/6208)) [@vyasr](https://github.com/vyasr)
- Use rapids-cmake for the logger ([#6205](https://github.com/NVIDIA/cuml/pull/6205)) [@vyasr](https://github.com/vyasr)
- consolidate pytest config in pyproject.toml ([#6201](https://github.com/NVIDIA/cuml/pull/6201)) [@jameslamb](https://github.com/jameslamb)
- introduce libcuml wheels ([#6199](https://github.com/NVIDIA/cuml/pull/6199)) [@jameslamb](https://github.com/jameslamb)
- Check if nightlies have succeeded recently enough ([#6196](https://github.com/NVIDIA/cuml/pull/6196)) [@vyasr](https://github.com/vyasr)
- Remove sphinx pinning ([#6195](https://github.com/NVIDIA/cuml/pull/6195)) [@vyasr](https://github.com/vyasr)
- simplify wheel CI scripts, other small packaging changes ([#6190](https://github.com/NVIDIA/cuml/pull/6190)) [@jameslamb](https://github.com/jameslamb)
- Update for raft logger changes ([#6187](https://github.com/NVIDIA/cuml/pull/6187)) [@vyasr](https://github.com/vyasr)
- update telemetry actions to fluent-bit friendly style ([#6186](https://github.com/NVIDIA/cuml/pull/6186)) [@msarahan](https://github.com/msarahan)
- HDBSCAN and KMeans API improvements for improving CPU interoperability ([#6181](https://github.com/NVIDIA/cuml/pull/6181)) [@dantegd](https://github.com/dantegd)
- CPU/GPU interop with RandomForest ([#6175](https://github.com/NVIDIA/cuml/pull/6175)) [@hcho3](https://github.com/hcho3)
- Update version references in workflow ([#6172](https://github.com/NVIDIA/cuml/pull/6172)) [@AyodeAwe](https://github.com/AyodeAwe)
- gate telemetry dispatch calls on TELEMETRY_ENABLED env var ([#6171](https://github.com/NVIDIA/cuml/pull/6171)) [@msarahan](https://github.com/msarahan)
- Use rapids-logger to generate the cuml logger ([#6162](https://github.com/NVIDIA/cuml/pull/6162)) [@vyasr](https://github.com/vyasr)
- Use estimator tags to improve sparse error handling ([#6151](https://github.com/NVIDIA/cuml/pull/6151)) [@dantegd](https://github.com/dantegd)
- FEA Add support for accepting a Numpy RandomState ([#6150](https://github.com/NVIDIA/cuml/pull/6150)) [@betatim](https://github.com/betatim)
- prefer system install of UCX in pip devcontainers, update outdated RAPIDS references ([#6149](https://github.com/NVIDIA/cuml/pull/6149)) [@jameslamb](https://github.com/jameslamb)
- Improve infrastructure for experimental dispatching of non existing methods in cuML ([#6148](https://github.com/NVIDIA/cuml/pull/6148)) [@dantegd](https://github.com/dantegd)
- Adapt to rmm logger changes ([#6147](https://github.com/NVIDIA/cuml/pull/6147)) [@vyasr](https://github.com/vyasr)
- Require approval to run CI on draft PRs ([#6145](https://github.com/NVIDIA/cuml/pull/6145)) [@bdice](https://github.com/bdice)
- Add breaking change workflow trigger ([#6130](https://github.com/NVIDIA/cuml/pull/6130)) [@AyodeAwe](https://github.com/AyodeAwe)
- Switch to native traceback in `cuml` ([#6078](https://github.com/NVIDIA/cuml/pull/6078)) [@galipremsagar](https://github.com/galipremsagar)

# cuml 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- Forward merge Branch 24.10 into 24.12 ([#6106](https://github.com/NVIDIA/cuml/pull/6106)) [@divyegala](https://github.com/divyegala)

## 🐛 Bug Fixes

- Fix `scikit-learn` version specifier ([#6158](https://github.com/NVIDIA/cuml/pull/6158)) [@trxcllnt](https://github.com/trxcllnt)
- Correctly handle missing categorical data in experimental FIL ([#6132](https://github.com/NVIDIA/cuml/pull/6132)) [@wphicks](https://github.com/wphicks)
- Put a ceiling on cuda-python ([#6131](https://github.com/NVIDIA/cuml/pull/6131)) [@bdice](https://github.com/bdice)
- Don&#39;t presume pointers are mutually exclusive for device/host. ([#6128](https://github.com/NVIDIA/cuml/pull/6128)) [@robertmaynard](https://github.com/robertmaynard)
- cuml SINGLEGPU now tells cuvs to not build with nccl/mg support ([#6127](https://github.com/NVIDIA/cuml/pull/6127)) [@robertmaynard](https://github.com/robertmaynard)
- Remove type from pickle header for CumlArray ([#6120](https://github.com/NVIDIA/cuml/pull/6120)) [@wphicks](https://github.com/wphicks)
- Forward merge Branch 24.10 into 24.12 ([#6106](https://github.com/NVIDIA/cuml/pull/6106)) [@divyegala](https://github.com/divyegala)
- Fix Dask estimators serialization prior to training ([#6065](https://github.com/NVIDIA/cuml/pull/6065)) [@viclafargue](https://github.com/viclafargue)

## 🚀 New Features

- Enable HDBSCAN `gpu` training and `cpu` inference ([#6108](https://github.com/NVIDIA/cuml/pull/6108)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- Update FIL tests to use XGBoost UBJSON instead of binary ([#6153](https://github.com/NVIDIA/cuml/pull/6153)) [@hcho3](https://github.com/hcho3)
- Use sparse knn / distances from cuvs ([#6143](https://github.com/NVIDIA/cuml/pull/6143)) [@benfred](https://github.com/benfred)
- Ensure MG to have the same number of allreduce calls in mean_stddev for sparse matrix to avoid hanging ([#6141](https://github.com/NVIDIA/cuml/pull/6141)) [@lijinf2](https://github.com/lijinf2)
- Stop excluding cutlass from symbol exclusion check ([#6140](https://github.com/NVIDIA/cuml/pull/6140)) [@vyasr](https://github.com/vyasr)
- Optimize MG variance calculation for dataset standardization for logistic regression ([#6138](https://github.com/NVIDIA/cuml/pull/6138)) [@lijinf2](https://github.com/lijinf2)
- enforce wheel size limits, README formatting in CI ([#6136](https://github.com/NVIDIA/cuml/pull/6136)) [@jameslamb](https://github.com/jameslamb)
- Experimental command line interface UX ([#6135](https://github.com/NVIDIA/cuml/pull/6135)) [@dantegd](https://github.com/dantegd)
- add telemetry ([#6126](https://github.com/NVIDIA/cuml/pull/6126)) [@msarahan](https://github.com/msarahan)
- Make cuVS optional if CUML_ALGORITHMS is set ([#6125](https://github.com/NVIDIA/cuml/pull/6125)) [@hcho3](https://github.com/hcho3)
- devcontainer: replace `VAULT_HOST` with `AWS_ROLE_ARN` ([#6118](https://github.com/NVIDIA/cuml/pull/6118)) [@jjacobelli](https://github.com/jjacobelli)
- print sccache stats in builds ([#6111](https://github.com/NVIDIA/cuml/pull/6111)) [@jameslamb](https://github.com/jameslamb)
- fix version in Doxygen docs ([#6104](https://github.com/NVIDIA/cuml/pull/6104)) [@jameslamb](https://github.com/jameslamb)
- make conda installs in CI stricter ([#6103](https://github.com/NVIDIA/cuml/pull/6103)) [@jameslamb](https://github.com/jameslamb)
- Make `get_param_names` a class method on single GPU estimators to match Scikit-learn closer ([#6101](https://github.com/NVIDIA/cuml/pull/6101)) [@dantegd](https://github.com/dantegd)
- Prune workflows based on changed files ([#6094](https://github.com/NVIDIA/cuml/pull/6094)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Update all rmm imports to use pylibrmm/librmm ([#6084](https://github.com/NVIDIA/cuml/pull/6084)) [@Matt711](https://github.com/Matt711)
- Merge branch 24.10 into branch 24.12 ([#6083](https://github.com/NVIDIA/cuml/pull/6083)) [@jameslamb](https://github.com/jameslamb)

# cuml 24.10.00 (9 Oct 2024)

## 🚨 Breaking Changes

- Remove old dask-glm based logistic regression ([#6028](https://github.com/NVIDIA/cuml/pull/6028)) [@dantegd](https://github.com/dantegd)

## 🐛 Bug Fixes

- Fix train_test_split for string columns ([#6088](https://github.com/NVIDIA/cuml/pull/6088)) [@dantegd](https://github.com/dantegd)
- Stop shadowing free function ([#6076](https://github.com/NVIDIA/cuml/pull/6076)) [@vyasr](https://github.com/vyasr)
- Set default values for conftest options. ([#6067](https://github.com/NVIDIA/cuml/pull/6067)) [@bdice](https://github.com/bdice)
- Add license file to conda packages ([#6061](https://github.com/NVIDIA/cuml/pull/6061)) [@raydouglass](https://github.com/raydouglass)
- Fix np.NAN to np.nan. ([#6056](https://github.com/NVIDIA/cuml/pull/6056)) [@bdice](https://github.com/bdice)
- Reenable `pytest cuml-dask` for CUDA 12.5 wheel CI tests ([#6051](https://github.com/NVIDIA/cuml/pull/6051)) [@divyegala](https://github.com/divyegala)
- Fix for `simplicial_set_embedding` ([#6043](https://github.com/NVIDIA/cuml/pull/6043)) [@viclafargue](https://github.com/viclafargue)
- MAINT: Allow for error message to contain ``np.float32(1.0)`` ([#6030](https://github.com/NVIDIA/cuml/pull/6030)) [@seberg](https://github.com/seberg)
- Stop exporting fill_k kernel as that causes ODR violations ([#6021](https://github.com/NVIDIA/cuml/pull/6021)) [@robertmaynard](https://github.com/robertmaynard)
- Avoid cudf column APIs after cudf.Series disallows column inputs ([#6019](https://github.com/NVIDIA/cuml/pull/6019)) [@mroeschke](https://github.com/mroeschke)
- Use HDBSCAN package pin to `0.8.38` ([#5906](https://github.com/NVIDIA/cuml/pull/5906)) [@divyegala](https://github.com/divyegala)

## 📖 Documentation

- Update UMAP doc ([#6064](https://github.com/NVIDIA/cuml/pull/6064)) [@viclafargue](https://github.com/viclafargue)
- Update README in experimental FIL ([#6052](https://github.com/NVIDIA/cuml/pull/6052)) [@hcho3](https://github.com/hcho3)
- add docs for simplicial_set ([#6042](https://github.com/NVIDIA/cuml/pull/6042)) [@Intron7](https://github.com/Intron7)

## 🚀 New Features

- TSNE CPU/GPU Interop ([#6063](https://github.com/NVIDIA/cuml/pull/6063)) [@divyegala](https://github.com/divyegala)
- Enable GPU `fit` and CPU `transform` in UMAP ([#6032](https://github.com/NVIDIA/cuml/pull/6032)) [@divyegala](https://github.com/divyegala)

## 🛠️ Improvements

- Migrate to use cuVS for vector search ([#6085](https://github.com/NVIDIA/cuml/pull/6085)) [@benfred](https://github.com/benfred)
- Support all-zeroes feature vectors for MG sparse logistic regression ([#6082](https://github.com/NVIDIA/cuml/pull/6082)) [@lijinf2](https://github.com/lijinf2)
- Update update-version.sh to use packaging lib ([#6081](https://github.com/NVIDIA/cuml/pull/6081)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use CI workflow branch &#39;branch-24.10&#39; again ([#6072](https://github.com/NVIDIA/cuml/pull/6072)) [@jameslamb](https://github.com/jameslamb)
- Update fmt (to 11.0.2) and spdlog (to 1.14.1), add those libraries to libcuml conda host dependencies ([#6071](https://github.com/NVIDIA/cuml/pull/6071)) [@jameslamb](https://github.com/jameslamb)
- Update flake8 to 7.1.1. ([#6070](https://github.com/NVIDIA/cuml/pull/6070)) [@bdice](https://github.com/bdice)
- Add support for Python 3.12, update to umap-learn==0.5.6 ([#6060](https://github.com/NVIDIA/cuml/pull/6060)) [@jameslamb](https://github.com/jameslamb)
- Fix compiler warning about signed vs unsigned ints ([#6053](https://github.com/NVIDIA/cuml/pull/6053)) [@hcho3](https://github.com/hcho3)
- Update rapidsai/pre-commit-hooks ([#6048](https://github.com/NVIDIA/cuml/pull/6048)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Drop Python 3.9 support ([#6040](https://github.com/NVIDIA/cuml/pull/6040)) [@jameslamb](https://github.com/jameslamb)
- Add use_cuda_wheels matrix entry ([#6038](https://github.com/NVIDIA/cuml/pull/6038)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Switch debug build to RelWithDebInfo ([#6033](https://github.com/NVIDIA/cuml/pull/6033)) [@rongou](https://github.com/rongou)
- Remove NumPy &lt;2 pin ([#6031](https://github.com/NVIDIA/cuml/pull/6031)) [@seberg](https://github.com/seberg)
- Remove old dask-glm based logistic regression ([#6028](https://github.com/NVIDIA/cuml/pull/6028)) [@dantegd](https://github.com/dantegd)
- [FEA] UMAP API for building with batched NN Descent ([#6022](https://github.com/NVIDIA/cuml/pull/6022)) [@jinsolp](https://github.com/jinsolp)
- Enabling CPU/GPU interop for SVM, DBSCAN and KMeans ([#6020](https://github.com/NVIDIA/cuml/pull/6020)) [@viclafargue](https://github.com/viclafargue)
- Update pre-commit hooks ([#6016](https://github.com/NVIDIA/cuml/pull/6016)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Improve update-version.sh ([#6014](https://github.com/NVIDIA/cuml/pull/6014)) [@bdice](https://github.com/bdice)
- Use tool.scikit-build.cmake.version, set scikit-build-core minimum-version ([#6012](https://github.com/NVIDIA/cuml/pull/6012)) [@jameslamb](https://github.com/jameslamb)
- Merge branch-24.08 into branch-24.10 ([#5981](https://github.com/NVIDIA/cuml/pull/5981)) [@jameslamb](https://github.com/jameslamb)
- Use CUDA math wheels ([#5966](https://github.com/NVIDIA/cuml/pull/5966)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

# cuml 24.08.00 (7 Aug 2024)

## 🐛 Bug Fixes

- Fixes for encoders/transformers for cudf.pandas ([#5990](https://github.com/NVIDIA/cuml/pull/5990)) [@dantegd](https://github.com/dantegd)
- BUG: remove sample parameter from pca call to mean ([#5980](https://github.com/NVIDIA/cuml/pull/5980)) [@mfoerste4](https://github.com/mfoerste4)
- Fix segfault and other errors in ForestInference.load_from_sklearn ([#5973](https://github.com/NVIDIA/cuml/pull/5973)) [@hcho3](https://github.com/hcho3)
- Rename `.devcontainer`s for CUDA 12.5 ([#5967](https://github.com/NVIDIA/cuml/pull/5967)) [@jakirkham](https://github.com/jakirkham)
- [MNT] Small NumPy 2 related fixes ([#5954](https://github.com/NVIDIA/cuml/pull/5954)) [@seberg](https://github.com/seberg)
- CI Fix: use ld_preload to avoid libgomp issue on ARM jobs ([#5949](https://github.com/NVIDIA/cuml/pull/5949)) [@dantegd](https://github.com/dantegd)
- Fix for benchmark runner to handle parameter sweeps of multiple data types ([#5938](https://github.com/NVIDIA/cuml/pull/5938)) [@dantegd](https://github.com/dantegd)
- Avoid extra memory copy when using cp.concatenate in cuml.dask kmeans ([#5937](https://github.com/NVIDIA/cuml/pull/5937)) [@dantegd](https://github.com/dantegd)
- Assign correct `labels_` in `cuml.dask.kmeans` ([#5931](https://github.com/NVIDIA/cuml/pull/5931)) [@dantegd](https://github.com/dantegd)
- Fix nightly jobs by updating hypothesis strategies to account for sklearn change ([#5925](https://github.com/NVIDIA/cuml/pull/5925)) [@dantegd](https://github.com/dantegd)
- Fix for SVC fit_proba not using class weights ([#5912](https://github.com/NVIDIA/cuml/pull/5912)) [@pablotanner](https://github.com/pablotanner)
- Fix `cudf.pandas` failure on `test_convert_input_dtype` ([#5885](https://github.com/NVIDIA/cuml/pull/5885)) [@dantegd](https://github.com/dantegd)
- Fix `cudf.pandas` failure on  `test_convert_matrix_order_cuml_array` ([#5882](https://github.com/NVIDIA/cuml/pull/5882)) [@dantegd](https://github.com/dantegd)
- Simplify cuml array ([#5166](https://github.com/NVIDIA/cuml/pull/5166)) [@wence-](https://github.com/wence-)

## 🚀 New Features

- [FEA] Enable UMAP to build knn graph using NN Descent ([#5910](https://github.com/NVIDIA/cuml/pull/5910)) [@jinsolp](https://github.com/jinsolp)
- Allow estimators to accept any dtype ([#5888](https://github.com/NVIDIA/cuml/pull/5888)) [@dantegd](https://github.com/dantegd)

## 🛠️ Improvements

- Add support for XGBoost UBJSON in FIL ([#6009](https://github.com/NVIDIA/cuml/pull/6009)) [@hcho3](https://github.com/hcho3)
- split up CUDA-suffixed dependencies in dependencies.yaml ([#5974](https://github.com/NVIDIA/cuml/pull/5974)) [@jameslamb](https://github.com/jameslamb)
- Use workflow branch 24.08 again ([#5970](https://github.com/NVIDIA/cuml/pull/5970)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Bump Treelite to 4.3.0 ([#5968](https://github.com/NVIDIA/cuml/pull/5968)) [@hcho3](https://github.com/hcho3)
- reduce memory_footprint for sparse PCA transform ([#5964](https://github.com/NVIDIA/cuml/pull/5964)) [@Intron7](https://github.com/Intron7)
- Build and test with CUDA 12.5.1 ([#5963](https://github.com/NVIDIA/cuml/pull/5963)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Support int64 index type in MG sparse LogisticRegression ([#5962](https://github.com/NVIDIA/cuml/pull/5962)) [@lijinf2](https://github.com/lijinf2)
- Add CUDA_STATIC_MATH_LIBRARIES ([#5959](https://github.com/NVIDIA/cuml/pull/5959)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- skip CMake 3.30.0 ([#5956](https://github.com/NVIDIA/cuml/pull/5956)) [@jameslamb](https://github.com/jameslamb)
- Make `ci/run_cuml_dask_pytests.sh` environment-agnostic again ([#5950](https://github.com/NVIDIA/cuml/pull/5950)) [@trxcllnt](https://github.com/trxcllnt)
- Use verify-alpha-spec hook ([#5948](https://github.com/NVIDIA/cuml/pull/5948)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- nest cuml one level deeper in python ([#5944](https://github.com/NVIDIA/cuml/pull/5944)) [@msarahan](https://github.com/msarahan)
- resolve dependency-file-generator warning, other rapids-build-backend followup ([#5928](https://github.com/NVIDIA/cuml/pull/5928)) [@jameslamb](https://github.com/jameslamb)
- Adopt CI/packaging codeowners ([#5923](https://github.com/NVIDIA/cuml/pull/5923)) [@bdice](https://github.com/bdice)
- Remove text builds of documentation ([#5921](https://github.com/NVIDIA/cuml/pull/5921)) [@vyasr](https://github.com/vyasr)
- Fix conflict of forward-merge #5905 of branch-24.06 into branch-24.08 ([#5911](https://github.com/NVIDIA/cuml/pull/5911)) [@dantegd](https://github.com/dantegd)
- Bump Treelite to 4.2.1 ([#5908](https://github.com/NVIDIA/cuml/pull/5908)) [@hcho3](https://github.com/hcho3)
- remove unnecessary &#39;setuptools&#39; dependency ([#5901](https://github.com/NVIDIA/cuml/pull/5901)) [@jameslamb](https://github.com/jameslamb)
- [FEA] PCA Initialization for TSNE ([#5897](https://github.com/NVIDIA/cuml/pull/5897)) [@aamijar](https://github.com/aamijar)
- Use rapids-build-backend ([#5804](https://github.com/NVIDIA/cuml/pull/5804)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

# cuml 24.06.00 (5 Jun 2024)

## 🐛 Bug Fixes

- [HOTFIX] Fix import of sklearn by using cpu_only_import ([#5914](https://github.com/NVIDIA/cuml/pull/5914)) [@dantegd](https://github.com/dantegd)
- Fix label binarize for binary class ([#5900](https://github.com/NVIDIA/cuml/pull/5900)) [@jinsolp](https://github.com/jinsolp)
- Fix RandomForestClassifier return type ([#5896](https://github.com/NVIDIA/cuml/pull/5896)) [@jinsolp](https://github.com/jinsolp)
- Fix nightly CI: remove deprecated creation of columns by using explicit dtype ([#5880](https://github.com/NVIDIA/cuml/pull/5880)) [@dantegd](https://github.com/dantegd)
- Fix DBSCAN allocates rbc index even if deactivated ([#5859](https://github.com/NVIDIA/cuml/pull/5859)) [@mfoerste4](https://github.com/mfoerste4)
- Remove gtest from dependencies.yaml ([#5854](https://github.com/NVIDIA/cuml/pull/5854)) [@robertmaynard](https://github.com/robertmaynard)
- Support expression-based Dask Dataframe API ([#5835](https://github.com/NVIDIA/cuml/pull/5835)) [@rjzamora](https://github.com/rjzamora)
- Mark all kernels with internal linkage ([#5764](https://github.com/NVIDIA/cuml/pull/5764)) [@robertmaynard](https://github.com/robertmaynard)
- Fix build.sh clean command ([#5730](https://github.com/NVIDIA/cuml/pull/5730)) [@csadorf](https://github.com/csadorf)

## 📖 Documentation

- Update the developer&#39;s guide with new copyright hook ([#5848](https://github.com/NVIDIA/cuml/pull/5848)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)

## 🚀 New Features

- Always use a static gtest and gbench ([#5847](https://github.com/NVIDIA/cuml/pull/5847)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- Support double precision in MNMG Logistic Regression ([#5898](https://github.com/NVIDIA/cuml/pull/5898)) [@lijinf2](https://github.com/lijinf2)
- Reduce and rename cudf.pandas integrations jobs ([#5890](https://github.com/NVIDIA/cuml/pull/5890)) [@dantegd](https://github.com/dantegd)
- Fix building cuml with CCCL main ([#5886](https://github.com/NVIDIA/cuml/pull/5886)) [@trxcllnt](https://github.com/trxcllnt)
- Add optional CI job for integration tests with cudf.pandas ([#5881](https://github.com/NVIDIA/cuml/pull/5881)) [@dantegd](https://github.com/dantegd)
- Enable pytest failures on FutureWarnings/DeprecationWarnings ([#5877](https://github.com/NVIDIA/cuml/pull/5877)) [@mroeschke](https://github.com/mroeschke)
- Remove return in test_lbfgs ([#5875](https://github.com/NVIDIA/cuml/pull/5875)) [@mroeschke](https://github.com/mroeschke)
- Avoid dask_cudf.core imports ([#5874](https://github.com/NVIDIA/cuml/pull/5874)) [@bdice](https://github.com/bdice)
- Support CPU object for `train_test_split` ([#5873](https://github.com/NVIDIA/cuml/pull/5873)) [@isVoid](https://github.com/isVoid)
- Only use functions in the limited API ([#5871](https://github.com/NVIDIA/cuml/pull/5871)) [@vyasr](https://github.com/vyasr)
- Replace deprecated disutils.version with packaging.version ([#5868](https://github.com/NVIDIA/cuml/pull/5868)) [@mroeschke](https://github.com/mroeschke)
- Adjust deprecated cupy.sparse usage ([#5867](https://github.com/NVIDIA/cuml/pull/5867)) [@mroeschke](https://github.com/mroeschke)
- Fix numpy 2.0 deprecations ([#5866](https://github.com/NVIDIA/cuml/pull/5866)) [@mroeschke](https://github.com/mroeschke)
- Fix deprecated positional arg usage ([#5865](https://github.com/NVIDIA/cuml/pull/5865)) [@mroeschke](https://github.com/mroeschke)
- Use int instead of float in random.randint ([#5864](https://github.com/NVIDIA/cuml/pull/5864)) [@mroeschke](https://github.com/mroeschke)
- Migrate to `{{ stdlib(&quot;c&quot;) }}` ([#5863](https://github.com/NVIDIA/cuml/pull/5863)) [@hcho3](https://github.com/hcho3)
- Avoid deprecated API in notebook ([#5862](https://github.com/NVIDIA/cuml/pull/5862)) [@rjzamora](https://github.com/rjzamora)
- Add dedicated handling for cudf.pandas wrapped Numpy arrays ([#5861](https://github.com/NVIDIA/cuml/pull/5861)) [@betatim](https://github.com/betatim)
- Prepend devcontainer name with the username ([#5860](https://github.com/NVIDIA/cuml/pull/5860)) [@trxcllnt](https://github.com/trxcllnt)
- add --rm and --name to devcontainer run args ([#5857](https://github.com/NVIDIA/cuml/pull/5857)) [@trxcllnt](https://github.com/trxcllnt)
- Update pip devcontainers to UCX v1.15.0 ([#5856](https://github.com/NVIDIA/cuml/pull/5856)) [@trxcllnt](https://github.com/trxcllnt)
- Replace rmm::mr::device_memory_resource* with rmm::device_async_resource_ref ([#5853](https://github.com/NVIDIA/cuml/pull/5853)) [@harrism](https://github.com/harrism)
- Update scikit-learn to 1.4 ([#5851](https://github.com/NVIDIA/cuml/pull/5851)) [@betatim](https://github.com/betatim)
- Prevent undefined behavior when passing handle from Treelite to cuML FIL ([#5849](https://github.com/NVIDIA/cuml/pull/5849)) [@hcho3](https://github.com/hcho3)
- Adds missing files to `update-version.sh` ([#5830](https://github.com/NVIDIA/cuml/pull/5830)) [@AyodeAwe](https://github.com/AyodeAwe)
- Enable all tests for `arm` arch ([#5824](https://github.com/NVIDIA/cuml/pull/5824)) [@galipremsagar](https://github.com/galipremsagar)
- Address PytestReturnNotNoneWarning in cuml tests ([#5819](https://github.com/NVIDIA/cuml/pull/5819)) [@mroeschke](https://github.com/mroeschke)
- Handle binary classifier with all-0 labels ([#5810](https://github.com/NVIDIA/cuml/pull/5810)) [@hcho3](https://github.com/hcho3)
- Use pytest_cases.fixture to fix warnings. ([#5798](https://github.com/NVIDIA/cuml/pull/5798)) [@bdice](https://github.com/bdice)
- Enable Dask tests with UCX-Py/UCXX in CI ([#5697](https://github.com/NVIDIA/cuml/pull/5697)) [@pentschev](https://github.com/pentschev)

# cuML 24.04.00 (10 Apr 2024)

## 🐛 Bug Fixes

- Update pre-commit-hooks to v0.0.3 ([#5816](https://github.com/NVIDIA/cuml/pull/5816)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Correct and adjust tolerances of mnmg logreg pytests ([#5812](https://github.com/NVIDIA/cuml/pull/5812)) [@dantegd](https://github.com/dantegd)
- Remove use of cudf.core.column.full. ([#5794](https://github.com/NVIDIA/cuml/pull/5794)) [@bdice](https://github.com/bdice)
- Suppress all HealthChecks on test_split_datasets. ([#5791](https://github.com/NVIDIA/cuml/pull/5791)) [@bdice](https://github.com/bdice)
- Suppress a hypothesis HealthCheck on test_split_datasets that fails in nightly CI. ([#5790](https://github.com/NVIDIA/cuml/pull/5790)) [@bdice](https://github.com/bdice)
- [BUG] Fix `MAX_THREADS_PER_SM` on sm 89. ([#5785](https://github.com/NVIDIA/cuml/pull/5785)) [@trivialfis](https://github.com/trivialfis)
- fix device to host copy not sync stream in logistic regression mg ([#5766](https://github.com/NVIDIA/cuml/pull/5766)) [@lijinf2](https://github.com/lijinf2)
- Use cudf.Index instead of cudf.GenericIndex. ([#5738](https://github.com/NVIDIA/cuml/pull/5738)) [@bdice](https://github.com/bdice)
- update RAPIDS dependencies to 24.4, refactor dependencies.yaml ([#5726](https://github.com/NVIDIA/cuml/pull/5726)) [@jameslamb](https://github.com/jameslamb)

## 🚀 New Features

- Support CUDA 12.2 ([#5711](https://github.com/NVIDIA/cuml/pull/5711)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- Use `conda env create --yes` instead of `--force` ([#5822](https://github.com/NVIDIA/cuml/pull/5822)) [@bdice](https://github.com/bdice)
- Bump Treelite to 4.1.2 ([#5814](https://github.com/NVIDIA/cuml/pull/5814)) [@hcho3](https://github.com/hcho3)
- Support standardization for sparse vectors in logistic regression MG ([#5806](https://github.com/NVIDIA/cuml/pull/5806)) [@lijinf2](https://github.com/lijinf2)
- Update script input name ([#5802](https://github.com/NVIDIA/cuml/pull/5802)) [@AyodeAwe](https://github.com/AyodeAwe)
- Add upper bound to prevent usage of NumPy 2 ([#5797](https://github.com/NVIDIA/cuml/pull/5797)) [@bdice](https://github.com/bdice)
- Enable pytest failures on warnings from cudf ([#5796](https://github.com/NVIDIA/cuml/pull/5796)) [@mroeschke](https://github.com/mroeschke)
- Use public cudf APIs where possible ([#5795](https://github.com/NVIDIA/cuml/pull/5795)) [@mroeschke](https://github.com/mroeschke)
- Remove hard-coding of RAPIDS version where possible ([#5793](https://github.com/NVIDIA/cuml/pull/5793)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Switch `pytest-xdist` algorithm to `worksteal` ([#5792](https://github.com/NVIDIA/cuml/pull/5792)) [@bdice](https://github.com/bdice)
- Automate C++ include file grouping and ordering using clang-format ([#5787](https://github.com/NVIDIA/cuml/pull/5787)) [@harrism](https://github.com/harrism)
- Add support for Python 3.11, require NumPy 1.23+ ([#5786](https://github.com/NVIDIA/cuml/pull/5786)) [@jameslamb](https://github.com/jameslamb)
- [ENH] Let cuDF handle input types for label encoder. ([#5783](https://github.com/NVIDIA/cuml/pull/5783)) [@trivialfis](https://github.com/trivialfis)
- Install test dependencies at the same time as cuml packages. ([#5781](https://github.com/NVIDIA/cuml/pull/5781)) [@bdice](https://github.com/bdice)
- Update devcontainers to CUDA Toolkit 12.2 ([#5778](https://github.com/NVIDIA/cuml/pull/5778)) [@trxcllnt](https://github.com/trxcllnt)
- target branch-24.04 for GitHub Actions workflows ([#5776](https://github.com/NVIDIA/cuml/pull/5776)) [@jameslamb](https://github.com/jameslamb)
- Add environment-agnostic scripts for running ctests and pytests ([#5761](https://github.com/NVIDIA/cuml/pull/5761)) [@trxcllnt](https://github.com/trxcllnt)
- Pandas 2.x support ([#5758](https://github.com/NVIDIA/cuml/pull/5758)) [@dantegd](https://github.com/dantegd)
- Update ops-bot.yaml ([#5752](https://github.com/NVIDIA/cuml/pull/5752)) [@AyodeAwe](https://github.com/AyodeAwe)
- Forward-merge branch-24.02 to branch-24.04 ([#5735](https://github.com/NVIDIA/cuml/pull/5735)) [@bdice](https://github.com/bdice)
- Replace local copyright check with pre-commit-hooks verify-copyright ([#5732](https://github.com/NVIDIA/cuml/pull/5732)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- DBSCAN utilize rbc eps_neighbors ([#5728](https://github.com/NVIDIA/cuml/pull/5728)) [@mfoerste4](https://github.com/mfoerste4)

# cuML 24.02.00 (12 Feb 2024)

## 🚨 Breaking Changes

- Update to CCCL 2.2.0. ([#5702](https://github.com/NVIDIA/cuml/pull/5702)) [@bdice](https://github.com/bdice)
- Switch to scikit-build-core ([#5693](https://github.com/NVIDIA/cuml/pull/5693)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- [Hotfix] Fix FIL gtest ([#5755](https://github.com/NVIDIA/cuml/pull/5755)) [@hcho3](https://github.com/hcho3)
- Exclude tests from builds ([#5754](https://github.com/NVIDIA/cuml/pull/5754)) [@vyasr](https://github.com/vyasr)
- Fix ctest directory to ensure tests are executed ([#5753](https://github.com/NVIDIA/cuml/pull/5753)) [@bdice](https://github.com/bdice)
- Synchronize stream in SVC memory test ([#5729](https://github.com/NVIDIA/cuml/pull/5729)) [@wphicks](https://github.com/wphicks)
- Fix shared-workflows repo name ([#5723](https://github.com/NVIDIA/cuml/pull/5723)) [@raydouglass](https://github.com/raydouglass)
- Fix cupy dependency in pyproject.toml ([#5705](https://github.com/NVIDIA/cuml/pull/5705)) [@vyasr](https://github.com/vyasr)
- Only cufft offers a static_nocallback version of the library ([#5703](https://github.com/NVIDIA/cuml/pull/5703)) [@robertmaynard](https://github.com/robertmaynard)

## 🛠️ Improvements

- [Hotfix] Update GPUTreeSHAP to fix ARM build ([#5747](https://github.com/NVIDIA/cuml/pull/5747)) [@hcho3](https://github.com/hcho3)
- Disable HistGradientBoosting support for now ([#5744](https://github.com/NVIDIA/cuml/pull/5744)) [@hcho3](https://github.com/hcho3)
- Disable hnswlib feature in RAFT; pin pytest ([#5733](https://github.com/NVIDIA/cuml/pull/5733)) [@hcho3](https://github.com/hcho3)
- [LogisticRegressionMG] Support standardization with no data modification ([#5724](https://github.com/NVIDIA/cuml/pull/5724)) [@lijinf2](https://github.com/lijinf2)
- Remove usages of rapids-env-update ([#5716](https://github.com/NVIDIA/cuml/pull/5716)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Remove extraneous SKBUILD_BUILD_OPTIONS ([#5714](https://github.com/NVIDIA/cuml/pull/5714)) [@vyasr](https://github.com/vyasr)
- refactor CUDA versions in dependencies.yaml ([#5712](https://github.com/NVIDIA/cuml/pull/5712)) [@jameslamb](https://github.com/jameslamb)
- Update to CCCL 2.2.0. ([#5702](https://github.com/NVIDIA/cuml/pull/5702)) [@bdice](https://github.com/bdice)
- Migrate to Treelite 4.0 ([#5701](https://github.com/NVIDIA/cuml/pull/5701)) [@hcho3](https://github.com/hcho3)
- Use cuda::proclaim_return_type on device lambdas. ([#5696](https://github.com/NVIDIA/cuml/pull/5696)) [@bdice](https://github.com/bdice)
- move _process_generic to base_return_types, avoid circular import ([#5695](https://github.com/NVIDIA/cuml/pull/5695)) [@dcolinmorgan](https://github.com/dcolinmorgan)
- Switch to scikit-build-core ([#5693](https://github.com/NVIDIA/cuml/pull/5693)) [@vyasr](https://github.com/vyasr)
- Fix all deprecated function calls in TUs where warnings are errors ([#5692](https://github.com/NVIDIA/cuml/pull/5692)) [@vyasr](https://github.com/vyasr)
- Remove CUML_BUILD_WHEELS and standardize Python builds ([#5689](https://github.com/NVIDIA/cuml/pull/5689)) [@vyasr](https://github.com/vyasr)
- Forward-merge branch-23.12 to branch-24.02 ([#5657](https://github.com/NVIDIA/cuml/pull/5657)) [@bdice](https://github.com/bdice)
- Add cuML devcontainers ([#5568](https://github.com/NVIDIA/cuml/pull/5568)) [@trxcllnt](https://github.com/trxcllnt)

# cuML 23.12.00 (6 Dec 2023)

## 🚨 Breaking Changes

- [LogisticRegressionMG] Support sparse vectors ([#5632](https://github.com/NVIDIA/cuml/pull/5632)) [@lijinf2](https://github.com/lijinf2)

## 🐛 Bug Fixes

- Update actions/labeler to v4 ([#5686](https://github.com/NVIDIA/cuml/pull/5686)) [@raydouglass](https://github.com/raydouglass)
- updated docs around `make_column_transformer` change from `.preprocessing` to `.compose` ([#5680](https://github.com/NVIDIA/cuml/pull/5680)) [@taureandyernv](https://github.com/taureandyernv)
- Skip dask pytest NN hang in CUDA 11.4 CI ([#5665](https://github.com/NVIDIA/cuml/pull/5665)) [@dantegd](https://github.com/dantegd)
- Avoid hard import of sklearn in base module. ([#5663](https://github.com/NVIDIA/cuml/pull/5663)) [@csadorf](https://github.com/csadorf)
- CI: Pin clang-tidy to 15.0.7. ([#5661](https://github.com/NVIDIA/cuml/pull/5661)) [@csadorf](https://github.com/csadorf)
- Adjust assumption regarding valid cudf.Series dimensional input. ([#5654](https://github.com/NVIDIA/cuml/pull/5654)) [@csadorf](https://github.com/csadorf)
- Flatten cupy array before feeding to cudf.Series ([#5651](https://github.com/NVIDIA/cuml/pull/5651)) [@vyasr](https://github.com/vyasr)
- CI: Fix expected ValueError and dask-glm incompatibility ([#5644](https://github.com/NVIDIA/cuml/pull/5644)) [@csadorf](https://github.com/csadorf)
- Use drop_duplicates instead of unique for cudf&#39;s pandas compatibility mode ([#5639](https://github.com/NVIDIA/cuml/pull/5639)) [@vyasr](https://github.com/vyasr)
- Temporarily avoid pydata-sphinx-theme version 0.14.2. ([#5629](https://github.com/NVIDIA/cuml/pull/5629)) [@csadorf](https://github.com/csadorf)
- Fix type hint in split function. ([#5625](https://github.com/NVIDIA/cuml/pull/5625)) [@trivialfis](https://github.com/trivialfis)
- Fix trying to get pointer to None in svm/linear.pyx ([#5615](https://github.com/NVIDIA/cuml/pull/5615)) [@yosider](https://github.com/yosider)
- Reduce parallelism to avoid OOMs in wheel tests ([#5611](https://github.com/NVIDIA/cuml/pull/5611)) [@vyasr](https://github.com/vyasr)

## 📖 Documentation

- Update interoperability docs ([#5633](https://github.com/NVIDIA/cuml/pull/5633)) [@beckernick](https://github.com/beckernick)
- Update instructions for creating a conda build environment ([#5628](https://github.com/NVIDIA/cuml/pull/5628)) [@csadorf](https://github.com/csadorf)

## 🚀 New Features

- Basic implementation of `OrdinalEncoder`. ([#5646](https://github.com/NVIDIA/cuml/pull/5646)) [@trivialfis](https://github.com/trivialfis)

## 🛠️ Improvements

- Build concurrency for nightly and merge triggers ([#5658](https://github.com/NVIDIA/cuml/pull/5658)) [@bdice](https://github.com/bdice)
- [LogisticRegressionMG][FEA] Support training when dataset contains only one class ([#5655](https://github.com/NVIDIA/cuml/pull/5655)) [@lijinf2](https://github.com/lijinf2)
- Use new `rapids-dask-dependency` metapackage for managing `dask` versions ([#5649](https://github.com/NVIDIA/cuml/pull/5649)) [@galipremsagar](https://github.com/galipremsagar)
- Simplify some logic in LabelEncoder ([#5648](https://github.com/NVIDIA/cuml/pull/5648)) [@vyasr](https://github.com/vyasr)
- Increase `Nanny` close timeout in `LocalCUDACluster` tests ([#5636](https://github.com/NVIDIA/cuml/pull/5636)) [@pentschev](https://github.com/pentschev)
- [LogisticRegressionMG] Support sparse vectors ([#5632](https://github.com/NVIDIA/cuml/pull/5632)) [@lijinf2](https://github.com/lijinf2)
- Add rich HTML representation to estimators ([#5630](https://github.com/NVIDIA/cuml/pull/5630)) [@betatim](https://github.com/betatim)
- Unpin `dask` and `distributed` for `23.12` development ([#5627](https://github.com/NVIDIA/cuml/pull/5627)) [@galipremsagar](https://github.com/galipremsagar)
- Update `shared-action-workflows` references ([#5621](https://github.com/NVIDIA/cuml/pull/5621)) [@AyodeAwe](https://github.com/AyodeAwe)
- Use branch-23.12 workflows. ([#5618](https://github.com/NVIDIA/cuml/pull/5618)) [@bdice](https://github.com/bdice)
- Update rapids-cmake functions to non-deprecated signatures ([#5616](https://github.com/NVIDIA/cuml/pull/5616)) [@robertmaynard](https://github.com/robertmaynard)
- Allow nightly dependencies and set up consistent nightly versions for conda and pip packages ([#5607](https://github.com/NVIDIA/cuml/pull/5607)) [@vyasr](https://github.com/vyasr)
- Forward-merge branch-23.10 to branch-23.12 ([#5596](https://github.com/NVIDIA/cuml/pull/5596)) [@bdice](https://github.com/bdice)
- Build CUDA 12.0 ARM conda packages. ([#5595](https://github.com/NVIDIA/cuml/pull/5595)) [@bdice](https://github.com/bdice)
- Enable multiclass svm for sparse input ([#5588](https://github.com/NVIDIA/cuml/pull/5588)) [@mfoerste4](https://github.com/mfoerste4)

# cuML 23.10.00 (11 Oct 2023)

## 🚨 Breaking Changes

- add sample_weight parameter to dbscan.fit ([#5574](https://github.com/NVIDIA/cuml/pull/5574)) [@mfoerste4](https://github.com/mfoerste4)
- Update to Cython 3.0.0 ([#5506](https://github.com/NVIDIA/cuml/pull/5506)) [@vyasr](https://github.com/vyasr)

## 🐛 Bug Fixes

- Fix accidental unsafe cupy import ([#5613](https://github.com/NVIDIA/cuml/pull/5613)) [@dantegd](https://github.com/dantegd)
- Fixes for CPU package ([#5599](https://github.com/NVIDIA/cuml/pull/5599)) [@dantegd](https://github.com/dantegd)
- Fixes for timeouts in tests ([#5598](https://github.com/NVIDIA/cuml/pull/5598)) [@dantegd](https://github.com/dantegd)

## 🚀 New Features

- Enable cuml-cpu nightly ([#5585](https://github.com/NVIDIA/cuml/pull/5585)) [@dantegd](https://github.com/dantegd)
- add sample_weight parameter to dbscan.fit ([#5574](https://github.com/NVIDIA/cuml/pull/5574)) [@mfoerste4](https://github.com/mfoerste4)

## 🛠️ Improvements

- cuml-cpu notebook, docs and cluster models ([#5597](https://github.com/NVIDIA/cuml/pull/5597)) [@dantegd](https://github.com/dantegd)
- Pin `dask` and `distributed` for `23.10` release ([#5592](https://github.com/NVIDIA/cuml/pull/5592)) [@galipremsagar](https://github.com/galipremsagar)
- Add changes for early experimental support for dataframe interchange protocol API ([#5591](https://github.com/NVIDIA/cuml/pull/5591)) [@dantegd](https://github.com/dantegd)
- [FEA] Support L1 regularization and ElasticNet in MNMG Dask LogisticRegression ([#5587](https://github.com/NVIDIA/cuml/pull/5587)) [@lijinf2](https://github.com/lijinf2)
- Update image names ([#5586](https://github.com/NVIDIA/cuml/pull/5586)) [@AyodeAwe](https://github.com/AyodeAwe)
- Update to clang 16.0.6. ([#5583](https://github.com/NVIDIA/cuml/pull/5583)) [@bdice](https://github.com/bdice)
- Upgrade to Treelite 3.9.1 ([#5581](https://github.com/NVIDIA/cuml/pull/5581)) [@hcho3](https://github.com/hcho3)
- Update to doxygen 1.9.1. ([#5580](https://github.com/NVIDIA/cuml/pull/5580)) [@bdice](https://github.com/bdice)
- [REVIEW] Adding a few of datasets for benchmarking ([#5573](https://github.com/NVIDIA/cuml/pull/5573)) [@vinaydes](https://github.com/vinaydes)
- Allow cuML MNMG estimators to be serialized ([#5571](https://github.com/NVIDIA/cuml/pull/5571)) [@viclafargue](https://github.com/viclafargue)
- [FEA] Support multiple classes in multi-node-multi-gpu logistic regression, from C++, Cython, to Dask Python class ([#5565](https://github.com/NVIDIA/cuml/pull/5565)) [@lijinf2](https://github.com/lijinf2)
- Use `copy-pr-bot` ([#5563](https://github.com/NVIDIA/cuml/pull/5563)) [@ajschmidt8](https://github.com/ajschmidt8)
- Unblock CI for branch-23.10 ([#5561](https://github.com/NVIDIA/cuml/pull/5561)) [@csadorf](https://github.com/csadorf)
- Fix CPU-only build for new FIL ([#5559](https://github.com/NVIDIA/cuml/pull/5559)) [@hcho3](https://github.com/hcho3)
- [FEA] Support no regularization in MNMG LogisticRegression ([#5558](https://github.com/NVIDIA/cuml/pull/5558)) [@lijinf2](https://github.com/lijinf2)
- Unpin `dask` and `distributed` for `23.10` development ([#5557](https://github.com/NVIDIA/cuml/pull/5557)) [@galipremsagar](https://github.com/galipremsagar)
- Branch 23.10 merge 23.08 ([#5547](https://github.com/NVIDIA/cuml/pull/5547)) [@vyasr](https://github.com/vyasr)
- Use Python builtins to prep benchmark `tmp_dir` ([#5537](https://github.com/NVIDIA/cuml/pull/5537)) [@jakirkham](https://github.com/jakirkham)
- Branch 23.10 merge 23.08 ([#5522](https://github.com/NVIDIA/cuml/pull/5522)) [@vyasr](https://github.com/vyasr)
- Update to Cython 3.0.0 ([#5506](https://github.com/NVIDIA/cuml/pull/5506)) [@vyasr](https://github.com/vyasr)

# cuML 23.08.00 (9 Aug 2023)

## 🚨 Breaking Changes

- Stop using setup.py in build.sh ([#5500](https://github.com/NVIDIA/cuml/pull/5500)) [@vyasr](https://github.com/vyasr)
- Add `copy_X` parameter to `LinearRegression` ([#5495](https://github.com/NVIDIA/cuml/pull/5495)) [@viclafargue](https://github.com/viclafargue)

## 🐛 Bug Fixes

- Update dependencies.yaml test_notebooks to include dask_ml ([#5545](https://github.com/NVIDIA/cuml/pull/5545)) [@taureandyernv](https://github.com/taureandyernv)
- Fix cython-lint issues. ([#5536](https://github.com/NVIDIA/cuml/pull/5536)) [@bdice](https://github.com/bdice)
- Skip rf_memleak tests ([#5529](https://github.com/NVIDIA/cuml/pull/5529)) [@dantegd](https://github.com/dantegd)
- Pin hdbscan to fix pytests in CI ([#5515](https://github.com/NVIDIA/cuml/pull/5515)) [@dantegd](https://github.com/dantegd)
- Fix UMAP and simplicial set functions metric ([#5490](https://github.com/NVIDIA/cuml/pull/5490)) [@viclafargue](https://github.com/viclafargue)
- Fix test_masked_column_mode ([#5480](https://github.com/NVIDIA/cuml/pull/5480)) [@viclafargue](https://github.com/viclafargue)
- Use fit_predict rather than fit for KNeighborsClassifier and KNeighborsRegressor in benchmark utility ([#5460](https://github.com/NVIDIA/cuml/pull/5460)) [@beckernick](https://github.com/beckernick)
- Modify HDBSCAN membership_vector batch_size check ([#5455](https://github.com/NVIDIA/cuml/pull/5455)) [@tarang-jain](https://github.com/tarang-jain)

## 🚀 New Features

- Use rapids-cmake testing to run tests in parallel ([#5487](https://github.com/NVIDIA/cuml/pull/5487)) [@robertmaynard](https://github.com/robertmaynard)
- [FEA] Update MST Reduction Op ([#5386](https://github.com/NVIDIA/cuml/pull/5386)) [@tarang-jain](https://github.com/tarang-jain)
- cuml: Build CUDA 12 packages ([#5318](https://github.com/NVIDIA/cuml/pull/5318)) [@vyasr](https://github.com/vyasr)
- CI: Add custom GitHub Actions job to run clang-tidy ([#5235](https://github.com/NVIDIA/cuml/pull/5235)) [@csadorf](https://github.com/csadorf)

## 🛠️ Improvements

- Pin `dask` and `distributed` for `23.08` release ([#5541](https://github.com/NVIDIA/cuml/pull/5541)) [@galipremsagar](https://github.com/galipremsagar)
- Remove Dockerfile. ([#5534](https://github.com/NVIDIA/cuml/pull/5534)) [@bdice](https://github.com/bdice)
- Improve temporary directory handling in cuML ([#5527](https://github.com/NVIDIA/cuml/pull/5527)) [@jakirkham](https://github.com/jakirkham)
- Support init arguments in MNMG LogisticRegression ([#5519](https://github.com/NVIDIA/cuml/pull/5519)) [@lijinf2](https://github.com/lijinf2)
- Support predict in MNMG Logistic Regression ([#5516](https://github.com/NVIDIA/cuml/pull/5516)) [@lijinf2](https://github.com/lijinf2)
- Remove unused matrix.cuh and math.cuh headers to eliminate deprecation warnings. ([#5513](https://github.com/NVIDIA/cuml/pull/5513)) [@bdice](https://github.com/bdice)
- Update gputreeshap to use rapids-cmake. ([#5512](https://github.com/NVIDIA/cuml/pull/5512)) [@bdice](https://github.com/bdice)
- Remove raft specializations includes. ([#5509](https://github.com/NVIDIA/cuml/pull/5509)) [@bdice](https://github.com/bdice)
- Revert CUDA 12.0 CI workflows to branch-23.08. ([#5508](https://github.com/NVIDIA/cuml/pull/5508)) [@bdice](https://github.com/bdice)
- Enable wheels CI scripts to run locally ([#5507](https://github.com/NVIDIA/cuml/pull/5507)) [@divyegala](https://github.com/divyegala)
- Default to nproc for PARALLEL_LEVEL in build.sh. ([#5505](https://github.com/NVIDIA/cuml/pull/5505)) [@csadorf](https://github.com/csadorf)
- Fixed potential overflows in SVM, minor adjustments to nvtx ranges ([#5504](https://github.com/NVIDIA/cuml/pull/5504)) [@mfoerste4](https://github.com/mfoerste4)
- Stop using setup.py in build.sh ([#5500](https://github.com/NVIDIA/cuml/pull/5500)) [@vyasr](https://github.com/vyasr)
- Fix PCA test ([#5498](https://github.com/NVIDIA/cuml/pull/5498)) [@viclafargue](https://github.com/viclafargue)
- Update build dependencies ([#5496](https://github.com/NVIDIA/cuml/pull/5496)) [@csadorf](https://github.com/csadorf)
- Add `copy_X` parameter to `LinearRegression` ([#5495](https://github.com/NVIDIA/cuml/pull/5495)) [@viclafargue](https://github.com/viclafargue)
- Sparse pca patch ([#5493](https://github.com/NVIDIA/cuml/pull/5493)) [@Intron7](https://github.com/Intron7)
- Restrict HDBSCAN metric options to L2 #5415 ([#5492](https://github.com/NVIDIA/cuml/pull/5492)) [@Rvch7](https://github.com/Rvch7)
- Fix typos. ([#5481](https://github.com/NVIDIA/cuml/pull/5481)) [@bdice](https://github.com/bdice)
- Add multi-node-multi-gpu Logistic Regression in C++ ([#5477](https://github.com/NVIDIA/cuml/pull/5477)) [@lijinf2](https://github.com/lijinf2)
- Add missing stream argument to cub calls in workingset ([#5476](https://github.com/NVIDIA/cuml/pull/5476)) [@mfoerste4](https://github.com/mfoerste4)
- Update to CMake 3.26.4 ([#5464](https://github.com/NVIDIA/cuml/pull/5464)) [@vyasr](https://github.com/vyasr)
- use rapids-upload-docs script ([#5457](https://github.com/NVIDIA/cuml/pull/5457)) [@AyodeAwe](https://github.com/AyodeAwe)
- Unpin `dask` and `distributed` for development ([#5452](https://github.com/NVIDIA/cuml/pull/5452)) [@galipremsagar](https://github.com/galipremsagar)
- Remove documentation build scripts for Jenkins ([#5450](https://github.com/NVIDIA/cuml/pull/5450)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix update version and pinnings for 23.08. ([#5440](https://github.com/NVIDIA/cuml/pull/5440)) [@bdice](https://github.com/bdice)
- Add cython-lint configuration. ([#5439](https://github.com/NVIDIA/cuml/pull/5439)) [@bdice](https://github.com/bdice)
- Unpin scikit-build upper bound ([#5438](https://github.com/NVIDIA/cuml/pull/5438)) [@vyasr](https://github.com/vyasr)
- Fix some deprecation warnings in tests. ([#5436](https://github.com/NVIDIA/cuml/pull/5436)) [@bdice](https://github.com/bdice)
- Update `raft::sparse::distance::pairwise_distance` to new API ([#5428](https://github.com/NVIDIA/cuml/pull/5428)) [@divyegala](https://github.com/divyegala)

# cuML 23.06.00 (7 Jun 2023)

## 🚨 Breaking Changes

- Dropping Python 3.8 ([#5385](https://github.com/NVIDIA/cuml/pull/5385)) [@divyegala](https://github.com/divyegala)
- Support sparse input for SVC and SVR ([#5273](https://github.com/NVIDIA/cuml/pull/5273)) [@mfoerste4](https://github.com/mfoerste4)

## 🐛 Bug Fixes

- Fixes for nightly GHA runs ([#5446](https://github.com/NVIDIA/cuml/pull/5446)) [@dantegd](https://github.com/dantegd)
- Add missing RAFT cusolver_macros import and changes for recent cuDF updates ([#5434](https://github.com/NVIDIA/cuml/pull/5434)) [@dantegd](https://github.com/dantegd)
- Fix kmeans pytest to correctly compute fp output error ([#5426](https://github.com/NVIDIA/cuml/pull/5426)) [@mdoijade](https://github.com/mdoijade)
- Add missing `raft/matrix/matrix.cuh` include ([#5411](https://github.com/NVIDIA/cuml/pull/5411)) [@benfred](https://github.com/benfred)
- Fix path to cumlprims_mg in build workflow ([#5406](https://github.com/NVIDIA/cuml/pull/5406)) [@divyegala](https://github.com/divyegala)
- Fix path to cumlprims in build workflow ([#5405](https://github.com/NVIDIA/cuml/pull/5405)) [@vyasr](https://github.com/vyasr)
- Pin to scikit-build&lt;17.2 ([#5400](https://github.com/NVIDIA/cuml/pull/5400)) [@vyasr](https://github.com/vyasr)
- Fix forward merge #5383 ([#5384](https://github.com/NVIDIA/cuml/pull/5384)) [@dantegd](https://github.com/dantegd)
- Correct buffer move assignment in experimental FIL ([#5372](https://github.com/NVIDIA/cuml/pull/5372)) [@wphicks](https://github.com/wphicks)
- Avoid invalid memory access in experimental FIL for large output size ([#5365](https://github.com/NVIDIA/cuml/pull/5365)) [@wphicks](https://github.com/wphicks)
- Fix forward merge #5336 ([#5345](https://github.com/NVIDIA/cuml/pull/5345)) [@dantegd](https://github.com/dantegd)

## 📖 Documentation

- Fix HDBSCAN docs and add membership_vector to cuml.cluster.hdbscan namespace ([#5378](https://github.com/NVIDIA/cuml/pull/5378)) [@beckernick](https://github.com/beckernick)
- Small doc fix ([#5375](https://github.com/NVIDIA/cuml/pull/5375)) [@tarang-jain](https://github.com/tarang-jain)

## 🚀 New Features

- Provide method for auto-optimization of FIL parameters ([#5368](https://github.com/NVIDIA/cuml/pull/5368)) [@wphicks](https://github.com/wphicks)

## 🛠️ Improvements

- Fix documentation source code links ([#5449](https://github.com/NVIDIA/cuml/pull/5449)) [@ajschmidt8](https://github.com/ajschmidt8)
- Drop seaborn dependency. ([#5437](https://github.com/NVIDIA/cuml/pull/5437)) [@bdice](https://github.com/bdice)
- Make all nvtx usage go through safe imports ([#5424](https://github.com/NVIDIA/cuml/pull/5424)) [@dantegd](https://github.com/dantegd)
- run docs nightly too ([#5423](https://github.com/NVIDIA/cuml/pull/5423)) [@AyodeAwe](https://github.com/AyodeAwe)
- Switch back to using primary shared-action-workflows branch ([#5420](https://github.com/NVIDIA/cuml/pull/5420)) [@vyasr](https://github.com/vyasr)
- Add librmm to libcuml dependencies. ([#5410](https://github.com/NVIDIA/cuml/pull/5410)) [@bdice](https://github.com/bdice)
- Update recipes to GTest version &gt;=1.13.0 ([#5408](https://github.com/NVIDIA/cuml/pull/5408)) [@bdice](https://github.com/bdice)
- Remove cudf from libcuml `meta.yaml` ([#5407](https://github.com/NVIDIA/cuml/pull/5407)) [@divyegala](https://github.com/divyegala)
- Support CUDA 12.0 for pip wheels ([#5404](https://github.com/NVIDIA/cuml/pull/5404)) [@divyegala](https://github.com/divyegala)
- Support for gtest 1.11+ changes ([#5403](https://github.com/NVIDIA/cuml/pull/5403)) [@dantegd](https://github.com/dantegd)
- Update cupy dependency ([#5401](https://github.com/NVIDIA/cuml/pull/5401)) [@vyasr](https://github.com/vyasr)
- Build wheels using new single image workflow ([#5394](https://github.com/NVIDIA/cuml/pull/5394)) [@vyasr](https://github.com/vyasr)
- Revert shared-action-workflows pin ([#5391](https://github.com/NVIDIA/cuml/pull/5391)) [@divyegala](https://github.com/divyegala)
- Fix logic for concatenating Treelite objects ([#5387](https://github.com/NVIDIA/cuml/pull/5387)) [@hcho3](https://github.com/hcho3)
- Dropping Python 3.8 ([#5385](https://github.com/NVIDIA/cuml/pull/5385)) [@divyegala](https://github.com/divyegala)
- Remove usage of rapids-get-rapids-version-from-git ([#5379](https://github.com/NVIDIA/cuml/pull/5379)) [@jjacobelli](https://github.com/jjacobelli)
- [ENH] Add missing includes of rmm/mr/device/per_device_resource.hpp ([#5369](https://github.com/NVIDIA/cuml/pull/5369)) [@ahendriksen](https://github.com/ahendriksen)
- Remove wheel pytest verbosity ([#5367](https://github.com/NVIDIA/cuml/pull/5367)) [@sevagh](https://github.com/sevagh)
- support parameter &#39;class_weight&#39; and method &#39;decision_function&#39; in LinearSVC ([#5364](https://github.com/NVIDIA/cuml/pull/5364)) [@mfoerste4](https://github.com/mfoerste4)
- Update clang-format to 16.0.1. ([#5361](https://github.com/NVIDIA/cuml/pull/5361)) [@bdice](https://github.com/bdice)
- Implement apply() in FIL ([#5358](https://github.com/NVIDIA/cuml/pull/5358)) [@hcho3](https://github.com/hcho3)
- Use ARC V2 self-hosted runners for GPU jobs ([#5356](https://github.com/NVIDIA/cuml/pull/5356)) [@jjacobelli](https://github.com/jjacobelli)
- Try running silhouette test ([#5353](https://github.com/NVIDIA/cuml/pull/5353)) [@vyasr](https://github.com/vyasr)
- Remove uses-setup-env-vars ([#5344](https://github.com/NVIDIA/cuml/pull/5344)) [@vyasr](https://github.com/vyasr)
- Resolve auto-merger conflicts between `branch-23.04` &amp; `branch-23.06` ([#5340](https://github.com/NVIDIA/cuml/pull/5340)) [@galipremsagar](https://github.com/galipremsagar)
- Solve merge conflict of PR #5327 ([#5329](https://github.com/NVIDIA/cuml/pull/5329)) [@dantegd](https://github.com/dantegd)
- Branch 23.06 merge 23.04 ([#5315](https://github.com/NVIDIA/cuml/pull/5315)) [@vyasr](https://github.com/vyasr)
- Support sparse input for SVC and SVR ([#5273](https://github.com/NVIDIA/cuml/pull/5273)) [@mfoerste4](https://github.com/mfoerste4)
- Delete outdated versions.json. ([#5229](https://github.com/NVIDIA/cuml/pull/5229)) [@bdice](https://github.com/bdice)

# cuML 23.04.00 (6 Apr 2023)

## 🚨 Breaking Changes

- Pin `dask` and `distributed` for release ([#5333](https://github.com/NVIDIA/cuml/pull/5333)) [@galipremsagar](https://github.com/galipremsagar)

## 🐛 Bug Fixes

- Skip pickle notebook during nbsphinx ([#5342](https://github.com/NVIDIA/cuml/pull/5342)) [@dantegd](https://github.com/dantegd)
- Avoid race condition in FIL predict_per_tree ([#5334](https://github.com/NVIDIA/cuml/pull/5334)) [@wphicks](https://github.com/wphicks)
- Ensure experimental FIL shmem usage is below device limits ([#5326](https://github.com/NVIDIA/cuml/pull/5326)) [@wphicks](https://github.com/wphicks)
- Update cuda architectures for threads per sm restriction ([#5323](https://github.com/NVIDIA/cuml/pull/5323)) [@wphicks](https://github.com/wphicks)
- Run experimental FIL tests in CI ([#5316](https://github.com/NVIDIA/cuml/pull/5316)) [@wphicks](https://github.com/wphicks)
- Run memory leak pytests without parallelism to avoid sporadic test failures ([#5313](https://github.com/NVIDIA/cuml/pull/5313)) [@dantegd](https://github.com/dantegd)
- Update cupy version for pip wheels ([#5311](https://github.com/NVIDIA/cuml/pull/5311)) [@dantegd](https://github.com/dantegd)
- Fix for raising attributeerors erroneously for ipython methods ([#5299](https://github.com/NVIDIA/cuml/pull/5299)) [@dantegd](https://github.com/dantegd)
- Fix cuml local cpp docs build ([#5297](https://github.com/NVIDIA/cuml/pull/5297)) [@galipremsagar](https://github.com/galipremsagar)
- Don&#39;t run dask tests twice when testing wheels ([#5279](https://github.com/NVIDIA/cuml/pull/5279)) [@benfred](https://github.com/benfred)
- Remove MANIFEST.in use auto-generated one for sdists and package_data for wheels ([#5278](https://github.com/NVIDIA/cuml/pull/5278)) [@vyasr](https://github.com/vyasr)
- Removing remaining include of `raft/distance/distance_type.hpp` ([#5264](https://github.com/NVIDIA/cuml/pull/5264)) [@cjnolet](https://github.com/cjnolet)
- Enable hypothesis testing for nightly test runs. ([#5244](https://github.com/NVIDIA/cuml/pull/5244)) [@csadorf](https://github.com/csadorf)
- Support numeric, boolean, and string keyword arguments to class methods during CPU dispatching ([#5236](https://github.com/NVIDIA/cuml/pull/5236)) [@beckernick](https://github.com/beckernick)
- Allowing large data in kmeans ([#5228](https://github.com/NVIDIA/cuml/pull/5228)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- Fix docs build to be `pydata-sphinx-theme=0.13.0` compatible ([#5259](https://github.com/NVIDIA/cuml/pull/5259)) [@galipremsagar](https://github.com/galipremsagar)
- Add supported CPU/GPU operators to API docs and update docstrings ([#5239](https://github.com/NVIDIA/cuml/pull/5239)) [@beckernick](https://github.com/beckernick)
- Fix documentation author ([#5126](https://github.com/NVIDIA/cuml/pull/5126)) [@bdice](https://github.com/bdice)

## 🚀 New Features

- Modify default batch size in HDBSCAN soft clustering ([#5335](https://github.com/NVIDIA/cuml/pull/5335)) [@tarang-jain](https://github.com/tarang-jain)
- reduce memory pressure in membership vector computation ([#5268](https://github.com/NVIDIA/cuml/pull/5268)) [@tarang-jain](https://github.com/tarang-jain)
- membership_vector for HDBSCAN ([#5247](https://github.com/NVIDIA/cuml/pull/5247)) [@tarang-jain](https://github.com/tarang-jain)
- Provide FIL implementation for both CPU and GPU ([#4890](https://github.com/NVIDIA/cuml/pull/4890)) [@wphicks](https://github.com/wphicks)

## 🛠️ Improvements

- Remove deprecated Treelite CI API from FIL ([#5348](https://github.com/NVIDIA/cuml/pull/5348)) [@hcho3](https://github.com/hcho3)
- Updated forest inference to new dask worker api for 23.04 ([#5347](https://github.com/NVIDIA/cuml/pull/5347)) [@taureandyernv](https://github.com/taureandyernv)
- Pin `dask` and `distributed` for release ([#5333](https://github.com/NVIDIA/cuml/pull/5333)) [@galipremsagar](https://github.com/galipremsagar)
- Pin cupy in wheel tests to supported versions ([#5312](https://github.com/NVIDIA/cuml/pull/5312)) [@vyasr](https://github.com/vyasr)
- Drop `pickle5` ([#5310](https://github.com/NVIDIA/cuml/pull/5310)) [@jakirkham](https://github.com/jakirkham)
- Remove CUDA_CHECK macro ([#5308](https://github.com/NVIDIA/cuml/pull/5308)) [@hcho3](https://github.com/hcho3)
- Revert faiss removal pinned tag ([#5306](https://github.com/NVIDIA/cuml/pull/5306)) [@cjnolet](https://github.com/cjnolet)
- Upgrade to Treelite 3.2.0 ([#5304](https://github.com/NVIDIA/cuml/pull/5304)) [@hcho3](https://github.com/hcho3)
- Implement predict_per_tree() in FIL ([#5303](https://github.com/NVIDIA/cuml/pull/5303)) [@hcho3](https://github.com/hcho3)
- remove faiss from cuml ([#5293](https://github.com/NVIDIA/cuml/pull/5293)) [@benfred](https://github.com/benfred)
- Stop setting package version attribute in wheels ([#5285](https://github.com/NVIDIA/cuml/pull/5285)) [@vyasr](https://github.com/vyasr)
- Add libfaiss runtime dependency to libcuml. ([#5284](https://github.com/NVIDIA/cuml/pull/5284)) [@bdice](https://github.com/bdice)
- Move faiss_mr from raft ([#5281](https://github.com/NVIDIA/cuml/pull/5281)) [@benfred](https://github.com/benfred)
- Generate pyproject dependencies with dfg ([#5275](https://github.com/NVIDIA/cuml/pull/5275)) [@vyasr](https://github.com/vyasr)
- Updating cuML to use consolidated RAFT libs ([#5272](https://github.com/NVIDIA/cuml/pull/5272)) [@cjnolet](https://github.com/cjnolet)
- Add codespell as a linter ([#5265](https://github.com/NVIDIA/cuml/pull/5265)) [@benfred](https://github.com/benfred)
- Pass `AWS_SESSION_TOKEN` and `SCCACHE_S3_USE_SSL` vars to conda build ([#5263](https://github.com/NVIDIA/cuml/pull/5263)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update to GCC 11 ([#5258](https://github.com/NVIDIA/cuml/pull/5258)) [@bdice](https://github.com/bdice)
- Drop Python 3.7 handling for pickle protocol 4 ([#5256](https://github.com/NVIDIA/cuml/pull/5256)) [@jakirkham](https://github.com/jakirkham)
- Migrate as much as possible to pyproject.toml ([#5251](https://github.com/NVIDIA/cuml/pull/5251)) [@vyasr](https://github.com/vyasr)
- Adapt to rapidsai/rmm#1221 which moves allocator callbacks ([#5249](https://github.com/NVIDIA/cuml/pull/5249)) [@wence-](https://github.com/wence-)
- Add dfg as a pre-commit hook. ([#5246](https://github.com/NVIDIA/cuml/pull/5246)) [@vyasr](https://github.com/vyasr)
- Stop using versioneer to manage versions ([#5245](https://github.com/NVIDIA/cuml/pull/5245)) [@vyasr](https://github.com/vyasr)
- Enhance cuML benchmark utility and refactor hdbscan import utilities ([#5242](https://github.com/NVIDIA/cuml/pull/5242)) [@beckernick](https://github.com/beckernick)
- Fix GHA build workflow ([#5241](https://github.com/NVIDIA/cuml/pull/5241)) [@AjayThorve](https://github.com/AjayThorve)
- Support innerproduct distance in the pairwise_distance API ([#5230](https://github.com/NVIDIA/cuml/pull/5230)) [@benfred](https://github.com/benfred)
- Enable hypothesis for 23.04 ([#5221](https://github.com/NVIDIA/cuml/pull/5221)) [@csadorf](https://github.com/csadorf)
- Reduce error handling verbosity in CI tests scripts ([#5219](https://github.com/NVIDIA/cuml/pull/5219)) [@AjayThorve](https://github.com/AjayThorve)
- Bump pinned pip wheel deps to 23.4 ([#5217](https://github.com/NVIDIA/cuml/pull/5217)) [@sevagh](https://github.com/sevagh)
- Update shared workflow branches ([#5215](https://github.com/NVIDIA/cuml/pull/5215)) [@ajschmidt8](https://github.com/ajschmidt8)
- Unpin `dask` and `distributed` for development ([#5209](https://github.com/NVIDIA/cuml/pull/5209)) [@galipremsagar](https://github.com/galipremsagar)
- Remove gpuCI scripts. ([#5208](https://github.com/NVIDIA/cuml/pull/5208)) [@bdice](https://github.com/bdice)
- Move date to build string in `conda` recipe ([#5190](https://github.com/NVIDIA/cuml/pull/5190)) [@ajschmidt8](https://github.com/ajschmidt8)
- Kernel shap improvements ([#5187](https://github.com/NVIDIA/cuml/pull/5187)) [@vinaydes](https://github.com/vinaydes)
- test out the raft bfknn replacement ([#5186](https://github.com/NVIDIA/cuml/pull/5186)) [@benfred](https://github.com/benfred)
- Forward merge 23.02 into 23.04 ([#5182](https://github.com/NVIDIA/cuml/pull/5182)) [@vyasr](https://github.com/vyasr)
- Add `detail` namespace for linear models ([#5107](https://github.com/NVIDIA/cuml/pull/5107)) [@lowener](https://github.com/lowener)
- Add pre-commit configuration ([#4983](https://github.com/NVIDIA/cuml/pull/4983)) [@csadorf](https://github.com/csadorf)

# cuML 23.02.00 (9 Feb 2023)

## 🚨 Breaking Changes

- Use ivf_pq and ivf_flat from raft ([#5119](https://github.com/NVIDIA/cuml/pull/5119)) [@benfred](https://github.com/benfred)
- Estimators adaptation toward CPU/GPU interoperability ([#4918](https://github.com/NVIDIA/cuml/pull/4918)) [@viclafargue](https://github.com/viclafargue)
- Provide host CumlArray and associated infrastructure ([#4908](https://github.com/NVIDIA/cuml/pull/4908)) [@wphicks](https://github.com/wphicks)
- Improvements of UMAP/TSNE precomputed KNN feature ([#4865](https://github.com/NVIDIA/cuml/pull/4865)) [@viclafargue](https://github.com/viclafargue)

## 🐛 Bug Fixes

- Fix for creation of CUDA context at import time ([#5211](https://github.com/NVIDIA/cuml/pull/5211)) [@dantegd](https://github.com/dantegd)
- Correct arguments to load_from_treelite_model after classmethod conversion ([#5210](https://github.com/NVIDIA/cuml/pull/5210)) [@wphicks](https://github.com/wphicks)
- Use workaround to avoid staticmethod 3.10/Cython issue ([#5202](https://github.com/NVIDIA/cuml/pull/5202)) [@wphicks](https://github.com/wphicks)
- Increase margin for flaky FIL test ([#5194](https://github.com/NVIDIA/cuml/pull/5194)) [@wphicks](https://github.com/wphicks)
- Increase margin for flaky FIL test ([#5174](https://github.com/NVIDIA/cuml/pull/5174)) [@wphicks](https://github.com/wphicks)
- Fix gather_if raft update ([#5149](https://github.com/NVIDIA/cuml/pull/5149)) [@lowener](https://github.com/lowener)
- Add `_predict_model_on_cpu` for `RandomForestClassifier` ([#5148](https://github.com/NVIDIA/cuml/pull/5148)) [@lowener](https://github.com/lowener)
- Fix for hdbscan model serialization ([#5128](https://github.com/NVIDIA/cuml/pull/5128)) [@cjnolet](https://github.com/cjnolet)
- build.sh switch to use `RAPIDS` magic value ([#5124](https://github.com/NVIDIA/cuml/pull/5124)) [@robertmaynard](https://github.com/robertmaynard)
- Fix `Lasso` interop issue ([#5116](https://github.com/NVIDIA/cuml/pull/5116)) [@viclafargue](https://github.com/viclafargue)
- Remove nvcc conda package and add compiler/ninja to dev envs ([#5113](https://github.com/NVIDIA/cuml/pull/5113)) [@dantegd](https://github.com/dantegd)
- Add missing job dependency for new PR jobs check ([#5112](https://github.com/NVIDIA/cuml/pull/5112)) [@dantegd](https://github.com/dantegd)
- Skip RAFT docstring test in cuML ([#5088](https://github.com/NVIDIA/cuml/pull/5088)) [@dantegd](https://github.com/dantegd)
- Restore KNN metric attribute ([#5087](https://github.com/NVIDIA/cuml/pull/5087)) [@viclafargue](https://github.com/viclafargue)
- Check `sklearn` presence before importing the `Pipeline` ([#5072](https://github.com/NVIDIA/cuml/pull/5072)) [@viclafargue](https://github.com/viclafargue)
- Provide workaround for kernel ridge solver ([#5064](https://github.com/NVIDIA/cuml/pull/5064)) [@wphicks](https://github.com/wphicks)
- Keep verbosity level in KMeans OPG ([#5063](https://github.com/NVIDIA/cuml/pull/5063)) [@viclafargue](https://github.com/viclafargue)
- Transmit verbosity level to Dask workers ([#5062](https://github.com/NVIDIA/cuml/pull/5062)) [@viclafargue](https://github.com/viclafargue)
- Ensure consistent order for nearest neighbor tests ([#5059](https://github.com/NVIDIA/cuml/pull/5059)) [@wphicks](https://github.com/wphicks)
- Add `workers` argument to dask `make_blobs` ([#5057](https://github.com/NVIDIA/cuml/pull/5057)) [@viclafargue](https://github.com/viclafargue)
- Fix indexing type for ridge and linear models ([#4996](https://github.com/NVIDIA/cuml/pull/4996)) [@lowener](https://github.com/lowener)

## 📖 Documentation

- Adding benchmark notebook for hdbscan soft clustering ([#5103](https://github.com/NVIDIA/cuml/pull/5103)) [@cjnolet](https://github.com/cjnolet)
- Fix doc for solver in LogisticRegression ([#5097](https://github.com/NVIDIA/cuml/pull/5097)) [@viclafargue](https://github.com/viclafargue)
- Fix docstring of `HashingVectorizer` ([#5041](https://github.com/NVIDIA/cuml/pull/5041)) [@lowener](https://github.com/lowener)
- expose text, text.{CountVectorizer,HashingVectorizer,Tfidf{Transformer,Vectorizer}} from feature_extraction&#39;s public api ([#5028](https://github.com/NVIDIA/cuml/pull/5028)) [@mattf](https://github.com/mattf)
- Add Dask LabelEncoder to the documentation ([#5023](https://github.com/NVIDIA/cuml/pull/5023)) [@beckernick](https://github.com/beckernick)

## 🚀 New Features

- HDBSCAN CPU/GPU Interop ([#5137](https://github.com/NVIDIA/cuml/pull/5137)) [@divyegala](https://github.com/divyegala)
- Make all CPU/GPU only imports &quot;safe&quot; for respective package ([#5117](https://github.com/NVIDIA/cuml/pull/5117)) [@wphicks](https://github.com/wphicks)
- Pickling for HBDSCAN ([#5102](https://github.com/NVIDIA/cuml/pull/5102)) [@divyegala](https://github.com/divyegala)
- Break up silhouette score into 3 units to improve compilation time ([#5061](https://github.com/NVIDIA/cuml/pull/5061)) [@wphicks](https://github.com/wphicks)
- Provide host CumlArray and associated infrastructure ([#4908](https://github.com/NVIDIA/cuml/pull/4908)) [@wphicks](https://github.com/wphicks)

## 🛠️ Improvements

- Pin `dask` and `distributed` for release ([#5198](https://github.com/NVIDIA/cuml/pull/5198)) [@galipremsagar](https://github.com/galipremsagar)
- Update shared workflow branches ([#5197](https://github.com/NVIDIA/cuml/pull/5197)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pin wheel dependencies to same RAPIDS release ([#5183](https://github.com/NVIDIA/cuml/pull/5183)) [@sevagh](https://github.com/sevagh)
- Reverting RAFT pin ([#5178](https://github.com/NVIDIA/cuml/pull/5178)) [@cjnolet](https://github.com/cjnolet)
- Remove `faiss` from `libcuml` ([#5175](https://github.com/NVIDIA/cuml/pull/5175)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update location of `import_utils` from `common` to `internals` for Forest notebook ([#5171](https://github.com/NVIDIA/cuml/pull/5171)) [@taureandyernv](https://github.com/taureandyernv)
- Disable hypothesis tests for 23.02 burndown. ([#5168](https://github.com/NVIDIA/cuml/pull/5168)) [@csadorf](https://github.com/csadorf)
- Use CTK 118/cp310 branch of wheel workflows ([#5163](https://github.com/NVIDIA/cuml/pull/5163)) [@sevagh](https://github.com/sevagh)
- Add docs build GH ([#5155](https://github.com/NVIDIA/cuml/pull/5155)) [@AjayThorve](https://github.com/AjayThorve)
- Adapt to changes in `cudf.core.buffer.Buffer` ([#5154](https://github.com/NVIDIA/cuml/pull/5154)) [@galipremsagar](https://github.com/galipremsagar)
- Upgrade Treelite to 3.1.0 ([#5146](https://github.com/NVIDIA/cuml/pull/5146)) [@hcho3](https://github.com/hcho3)
- Replace cpdef variables with cdef variables. ([#5145](https://github.com/NVIDIA/cuml/pull/5145)) [@bdice](https://github.com/bdice)
- Update Scikit-learn compatibility to 1.2 ([#5141](https://github.com/NVIDIA/cuml/pull/5141)) [@dantegd](https://github.com/dantegd)
- Replace deprecated raft headers ([#5134](https://github.com/NVIDIA/cuml/pull/5134)) [@lowener](https://github.com/lowener)
- Execution device interoperability documentation ([#5130](https://github.com/NVIDIA/cuml/pull/5130)) [@viclafargue](https://github.com/viclafargue)
- Remove outdated macOS deployment target from build script. ([#5125](https://github.com/NVIDIA/cuml/pull/5125)) [@bdice](https://github.com/bdice)
- Build CUDA 11.8 and Python 3.10 Packages ([#5120](https://github.com/NVIDIA/cuml/pull/5120)) [@bdice](https://github.com/bdice)
- Use ivf_pq and ivf_flat from raft ([#5119](https://github.com/NVIDIA/cuml/pull/5119)) [@benfred](https://github.com/benfred)
- Update workflows for nightly tests ([#5110](https://github.com/NVIDIA/cuml/pull/5110)) [@ajschmidt8](https://github.com/ajschmidt8)
- Build pip wheels alongside conda CI ([#5109](https://github.com/NVIDIA/cuml/pull/5109)) [@sevagh](https://github.com/sevagh)
- Remove PROJECT_FLASH from libcuml conda build environment. ([#5108](https://github.com/NVIDIA/cuml/pull/5108)) [@bdice](https://github.com/bdice)
- Enable `Recently Updated` Check ([#5105](https://github.com/NVIDIA/cuml/pull/5105)) [@ajschmidt8](https://github.com/ajschmidt8)
- Ensure `pytest` is run from relevant directories in GH Actions ([#5101](https://github.com/NVIDIA/cuml/pull/5101)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove C++ Kmeans test ([#5098](https://github.com/NVIDIA/cuml/pull/5098)) [@lowener](https://github.com/lowener)
- Slightly lower the test_mbsgd_regressor expected min score. ([#5092](https://github.com/NVIDIA/cuml/pull/5092)) [@csadorf](https://github.com/csadorf)
- Skip all hypothesis health checks by default in CI runs. ([#5090](https://github.com/NVIDIA/cuml/pull/5090)) [@csadorf](https://github.com/csadorf)
- Reduce Naive Bayes test time ([#5082](https://github.com/NVIDIA/cuml/pull/5082)) [@lowener](https://github.com/lowener)
- Remove unused `.conda` folder ([#5078](https://github.com/NVIDIA/cuml/pull/5078)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix conflicts in #5045 ([#5077](https://github.com/NVIDIA/cuml/pull/5077)) [@ajschmidt8](https://github.com/ajschmidt8)
- Add GitHub Actions Workflows ([#5075](https://github.com/NVIDIA/cuml/pull/5075)) [@csadorf](https://github.com/csadorf)
- Skip test_linear_regression_model_default test. ([#5074](https://github.com/NVIDIA/cuml/pull/5074)) [@csadorf](https://github.com/csadorf)
- Fix link. ([#5067](https://github.com/NVIDIA/cuml/pull/5067)) [@bdice](https://github.com/bdice)
- Expand hypothesis testing for linear models ([#5065](https://github.com/NVIDIA/cuml/pull/5065)) [@csadorf](https://github.com/csadorf)
- Update xgb version in GPU CI 23.02 to 1.7.1 and unblocking CI ([#5051](https://github.com/NVIDIA/cuml/pull/5051)) [@dantegd](https://github.com/dantegd)
- Remove direct UCX and NCCL dependencies ([#5038](https://github.com/NVIDIA/cuml/pull/5038)) [@vyasr](https://github.com/vyasr)
- Move single test from `test` to `tests` ([#5037](https://github.com/NVIDIA/cuml/pull/5037)) [@vyasr](https://github.com/vyasr)
- Support using `CountVectorizer` &amp; `TfidVectorizer` in `cuml.pipeline.Pipeline` ([#5034](https://github.com/NVIDIA/cuml/pull/5034)) [@lasse-it](https://github.com/lasse-it)
- Refactor API decorators ([#5026](https://github.com/NVIDIA/cuml/pull/5026)) [@csadorf](https://github.com/csadorf)
- Implement hypothesis strategies and tests for arrays ([#5017](https://github.com/NVIDIA/cuml/pull/5017)) [@csadorf](https://github.com/csadorf)
- Add dependencies.yaml for rapids-dependency-file-generator ([#5003](https://github.com/NVIDIA/cuml/pull/5003)) [@beckernick](https://github.com/beckernick)
- Improved CPU/GPU interoperability ([#5001](https://github.com/NVIDIA/cuml/pull/5001)) [@viclafargue](https://github.com/viclafargue)
- Estimators adaptation toward CPU/GPU interoperability ([#4918](https://github.com/NVIDIA/cuml/pull/4918)) [@viclafargue](https://github.com/viclafargue)
- Improvements of UMAP/TSNE precomputed KNN feature ([#4865](https://github.com/NVIDIA/cuml/pull/4865)) [@viclafargue](https://github.com/viclafargue)

# cuML 22.12.00 (8 Dec 2022)

## 🚨 Breaking Changes

- Change docs theme to `pydata-sphinx` theme ([#4985](https://github.com/NVIDIA/cuml/pull/4985)) [@galipremsagar](https://github.com/galipremsagar)
- Remove &quot;Open In Colab&quot; link from Estimator Intro notebook. ([#4980](https://github.com/NVIDIA/cuml/pull/4980)) [@bdice](https://github.com/bdice)
- Remove `CumlArray.copy()` ([#4958](https://github.com/NVIDIA/cuml/pull/4958)) [@madsbk](https://github.com/madsbk)

## 🐛 Bug Fixes

- Remove cupy.cusparse custom serialization ([#5024](https://github.com/NVIDIA/cuml/pull/5024)) [@dantegd](https://github.com/dantegd)
- Restore `LinearRegression` documentation ([#5020](https://github.com/NVIDIA/cuml/pull/5020)) [@viclafargue](https://github.com/viclafargue)
- Don&#39;t use CMake 3.25.0 as it has a FindCUDAToolkit show stopping bug ([#5007](https://github.com/NVIDIA/cuml/pull/5007)) [@robertmaynard](https://github.com/robertmaynard)
- verifying cusparse wrapper revert passes CI ([#4990](https://github.com/NVIDIA/cuml/pull/4990)) [@cjnolet](https://github.com/cjnolet)
- Use rapdsi_cpm_find(COMPONENTS ) for proper component tracking ([#4989](https://github.com/NVIDIA/cuml/pull/4989)) [@robertmaynard](https://github.com/robertmaynard)
- Fix integer overflow in AutoARIMA due to bool-to-int cub scan ([#4971](https://github.com/NVIDIA/cuml/pull/4971)) [@Nyrio](https://github.com/Nyrio)
- Add missing includes ([#4947](https://github.com/NVIDIA/cuml/pull/4947)) [@vyasr](https://github.com/vyasr)
- Fix the CMake option for disabling deprecation warnings. ([#4946](https://github.com/NVIDIA/cuml/pull/4946)) [@vyasr](https://github.com/vyasr)
- Make doctest resilient to changes in cupy reprs ([#4945](https://github.com/NVIDIA/cuml/pull/4945)) [@vyasr](https://github.com/vyasr)
- Assign python/ sub-directory to python-codeowners ([#4940](https://github.com/NVIDIA/cuml/pull/4940)) [@csadorf](https://github.com/csadorf)
- Fix for non-contiguous strides ([#4736](https://github.com/NVIDIA/cuml/pull/4736)) [@viclafargue](https://github.com/viclafargue)

## 📖 Documentation

- Change docs theme to `pydata-sphinx` theme ([#4985](https://github.com/NVIDIA/cuml/pull/4985)) [@galipremsagar](https://github.com/galipremsagar)
- Remove &quot;Open In Colab&quot; link from Estimator Intro notebook. ([#4980](https://github.com/NVIDIA/cuml/pull/4980)) [@bdice](https://github.com/bdice)
- Updating build instructions ([#4979](https://github.com/NVIDIA/cuml/pull/4979)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Reenable copy_prs. ([#5010](https://github.com/NVIDIA/cuml/pull/5010)) [@vyasr](https://github.com/vyasr)
- Add wheel builds ([#5009](https://github.com/NVIDIA/cuml/pull/5009)) [@vyasr](https://github.com/vyasr)
- LinearRegression: add support for multiple targets ([#4988](https://github.com/NVIDIA/cuml/pull/4988)) [@ahendriksen](https://github.com/ahendriksen)
- CPU/GPU interoperability POC ([#4874](https://github.com/NVIDIA/cuml/pull/4874)) [@viclafargue](https://github.com/viclafargue)

## 🛠️ Improvements

- Upgrade Treelite to 3.0.1 ([#5018](https://github.com/NVIDIA/cuml/pull/5018)) [@hcho3](https://github.com/hcho3)
- fix addition of nan_euclidean_distances to public api ([#5015](https://github.com/NVIDIA/cuml/pull/5015)) [@mattf](https://github.com/mattf)
- Fixing raft pin to 22.12 ([#5000](https://github.com/NVIDIA/cuml/pull/5000)) [@cjnolet](https://github.com/cjnolet)
- Pin `dask` and `distributed` for release ([#4999](https://github.com/NVIDIA/cuml/pull/4999)) [@galipremsagar](https://github.com/galipremsagar)
- Update `dask` nightly install command in CI ([#4978](https://github.com/NVIDIA/cuml/pull/4978)) [@galipremsagar](https://github.com/galipremsagar)
- Improve error message for array_equal asserts. ([#4973](https://github.com/NVIDIA/cuml/pull/4973)) [@csadorf](https://github.com/csadorf)
- Use new rapids-cmake functionality for rpath handling. ([#4966](https://github.com/NVIDIA/cuml/pull/4966)) [@vyasr](https://github.com/vyasr)
- Impl. `CumlArray.deserialize()` ([#4965](https://github.com/NVIDIA/cuml/pull/4965)) [@madsbk](https://github.com/madsbk)
- Update `cuda-python` dependency to 11.7.1 ([#4961](https://github.com/NVIDIA/cuml/pull/4961)) [@galipremsagar](https://github.com/galipremsagar)
- Add check for nsys utility version in the `nvtx_benchmarks.py` script ([#4959](https://github.com/NVIDIA/cuml/pull/4959)) [@viclafargue](https://github.com/viclafargue)
- Remove `CumlArray.copy()` ([#4958](https://github.com/NVIDIA/cuml/pull/4958)) [@madsbk](https://github.com/madsbk)
- Implement hypothesis-based tests for linear models ([#4952](https://github.com/NVIDIA/cuml/pull/4952)) [@csadorf](https://github.com/csadorf)
- Switch to using rapids-cmake for gbench. ([#4950](https://github.com/NVIDIA/cuml/pull/4950)) [@vyasr](https://github.com/vyasr)
- Remove stale labeler ([#4949](https://github.com/NVIDIA/cuml/pull/4949)) [@raydouglass](https://github.com/raydouglass)
- Fix url in python/setup.py setuptools metadata. ([#4937](https://github.com/NVIDIA/cuml/pull/4937)) [@csadorf](https://github.com/csadorf)
- Updates to fix cuml build ([#4928](https://github.com/NVIDIA/cuml/pull/4928)) [@cjnolet](https://github.com/cjnolet)
- Documenting hdbscan module to add prediction functions ([#4925](https://github.com/NVIDIA/cuml/pull/4925)) [@cjnolet](https://github.com/cjnolet)
- Unpin `dask` and `distributed` for development ([#4912](https://github.com/NVIDIA/cuml/pull/4912)) [@galipremsagar](https://github.com/galipremsagar)
- Use KMeans from Raft ([#4713](https://github.com/NVIDIA/cuml/pull/4713)) [@lowener](https://github.com/lowener)
- Update cuml raft header extensions ([#4599](https://github.com/NVIDIA/cuml/pull/4599)) [@cjnolet](https://github.com/cjnolet)
- Reconciling primitives moved to RAFT ([#4583](https://github.com/NVIDIA/cuml/pull/4583)) [@cjnolet](https://github.com/cjnolet)

# cuML 22.10.00 (12 Oct 2022)

## 🐛 Bug Fixes

- Skipping some hdbscan tests when cuda version is &lt;= 11.2. ([#4916](https://github.com/NVIDIA/cuml/pull/4916)) [@cjnolet](https://github.com/cjnolet)
- Fix HDBSCAN python namespace ([#4895](https://github.com/NVIDIA/cuml/pull/4895)) [@cjnolet](https://github.com/cjnolet)
- Cupy 11 fixes ([#4889](https://github.com/NVIDIA/cuml/pull/4889)) [@dantegd](https://github.com/dantegd)
- Fix small fp precision failure in linear regression doctest test ([#4884](https://github.com/NVIDIA/cuml/pull/4884)) [@lowener](https://github.com/lowener)
- Remove unused cuDF imports ([#4873](https://github.com/NVIDIA/cuml/pull/4873)) [@beckernick](https://github.com/beckernick)
- Update for thrust 1.17 and fixes to accommodate for cuDF Buffer refactor ([#4871](https://github.com/NVIDIA/cuml/pull/4871)) [@dantegd](https://github.com/dantegd)
- Use rapids-cmake 22.10 best practice for RAPIDS.cmake location ([#4862](https://github.com/NVIDIA/cuml/pull/4862)) [@robertmaynard](https://github.com/robertmaynard)
- Patch for nightly test&amp;bench ([#4840](https://github.com/NVIDIA/cuml/pull/4840)) [@viclafargue](https://github.com/viclafargue)
- Fixed Large memory requirements for SimpleImputer strategy median #4794 ([#4817](https://github.com/NVIDIA/cuml/pull/4817)) [@erikrene](https://github.com/erikrene)
- Transforms RandomForest estimators non-consecutive labels to consecutive labels where appropriate ([#4780](https://github.com/NVIDIA/cuml/pull/4780)) [@VamsiTallam95](https://github.com/VamsiTallam95)

## 📖 Documentation

- Document that minimum required CMake version is now 3.23.1 ([#4899](https://github.com/NVIDIA/cuml/pull/4899)) [@robertmaynard](https://github.com/robertmaynard)
- Update KMeans notebook for clarity ([#4886](https://github.com/NVIDIA/cuml/pull/4886)) [@beckernick](https://github.com/beckernick)

## 🚀 New Features

- Allow cupy 11 ([#4880](https://github.com/NVIDIA/cuml/pull/4880)) [@galipremsagar](https://github.com/galipremsagar)
- Add `sample_weight` to Coordinate Descent solver (Lasso and ElasticNet) ([#4867](https://github.com/NVIDIA/cuml/pull/4867)) [@lowener](https://github.com/lowener)
- Import treelite models into FIL in a different precision ([#4839](https://github.com/NVIDIA/cuml/pull/4839)) [@canonizer](https://github.com/canonizer)
- #4783 Added nan_euclidean distance metric to pairwise_distances ([#4797](https://github.com/NVIDIA/cuml/pull/4797)) [@Sreekiran096](https://github.com/Sreekiran096)
- `PowerTransformer`, `QuantileTransformer` and `KernelCenterer` ([#4755](https://github.com/NVIDIA/cuml/pull/4755)) [@viclafargue](https://github.com/viclafargue)
- Add &quot;median&quot; to TargetEncoder ([#4722](https://github.com/NVIDIA/cuml/pull/4722)) [@daxiongshu](https://github.com/daxiongshu)
- New Feature StratifiedKFold ([#3109](https://github.com/NVIDIA/cuml/pull/3109)) [@daxiongshu](https://github.com/daxiongshu)

## 🛠️ Improvements

- Updating python to use pylibraft ([#4887](https://github.com/NVIDIA/cuml/pull/4887)) [@cjnolet](https://github.com/cjnolet)
- Upgrade Treelite to 3.0.0 ([#4885](https://github.com/NVIDIA/cuml/pull/4885)) [@hcho3](https://github.com/hcho3)
- Statically link all CUDA toolkit libraries ([#4881](https://github.com/NVIDIA/cuml/pull/4881)) [@trxcllnt](https://github.com/trxcllnt)
- approximate_predict function for HDBSCAN ([#4872](https://github.com/NVIDIA/cuml/pull/4872)) [@tarang-jain](https://github.com/tarang-jain)
- Pin `dask` and `distributed` for release ([#4859](https://github.com/NVIDIA/cuml/pull/4859)) [@galipremsagar](https://github.com/galipremsagar)
- Remove Raft deprecated headers ([#4858](https://github.com/NVIDIA/cuml/pull/4858)) [@lowener](https://github.com/lowener)
- Fix forward-merge conflicts ([#4857](https://github.com/NVIDIA/cuml/pull/4857)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update the NVTX bench helper for the new nsys utility ([#4826](https://github.com/NVIDIA/cuml/pull/4826)) [@viclafargue](https://github.com/viclafargue)
- All points membership vector for HDBSCAN ([#4800](https://github.com/NVIDIA/cuml/pull/4800)) [@tarang-jain](https://github.com/tarang-jain)
- TSNE and UMAP allow several distance types ([#4779](https://github.com/NVIDIA/cuml/pull/4779)) [@tarang-jain](https://github.com/tarang-jain)
- Convert fp32 datasets to fp64 in ARIMA and AutoARIMA + update notebook to avoid deprecation warnings with positional parameters ([#4195](https://github.com/NVIDIA/cuml/pull/4195)) [@Nyrio](https://github.com/Nyrio)

# cuML 22.08.00 (17 Aug 2022)

## 🚨 Breaking Changes

- Update Python build to scikit-build ([#4818](https://github.com/NVIDIA/cuml/pull/4818)) [@dantegd](https://github.com/dantegd)
- Bump `xgboost` to `1.6.0` from `1.5.2` ([#4777](https://github.com/NVIDIA/cuml/pull/4777)) [@galipremsagar](https://github.com/galipremsagar)

## 🐛 Bug Fixes

- Revert &quot;Allow CuPy 11&quot; ([#4847](https://github.com/NVIDIA/cuml/pull/4847)) [@galipremsagar](https://github.com/galipremsagar)
- Fix RAFT_NVTX option not set ([#4825](https://github.com/NVIDIA/cuml/pull/4825)) [@achirkin](https://github.com/achirkin)
- Fix KNN error message. ([#4782](https://github.com/NVIDIA/cuml/pull/4782)) [@trivialfis](https://github.com/trivialfis)
- Update raft pinnings in dev yml files ([#4778](https://github.com/NVIDIA/cuml/pull/4778)) [@galipremsagar](https://github.com/galipremsagar)
- Bump `xgboost` to `1.6.0` from `1.5.2` ([#4777](https://github.com/NVIDIA/cuml/pull/4777)) [@galipremsagar](https://github.com/galipremsagar)
- Fixes exception when using predict_proba on fitted Pipeline object with a ColumnTransformer step ([#4774](https://github.com/NVIDIA/cuml/pull/4774)) [@VamsiTallam95](https://github.com/VamsiTallam95)
- Regression errors failing with mixed data type combinations ([#4770](https://github.com/NVIDIA/cuml/pull/4770)) [@shaswat-indian](https://github.com/shaswat-indian)

## 📖 Documentation

- Use common code in python docs and defer `js` loading ([#4852](https://github.com/NVIDIA/cuml/pull/4852)) [@galipremsagar](https://github.com/galipremsagar)
- Centralize common css &amp; js code in docs ([#4844](https://github.com/NVIDIA/cuml/pull/4844)) [@galipremsagar](https://github.com/galipremsagar)
- Add ComplementNB to the documentation ([#4805](https://github.com/NVIDIA/cuml/pull/4805)) [@lowener](https://github.com/lowener)
- Fix forward-merge branch-22.06 to branch-22.08 ([#4789](https://github.com/NVIDIA/cuml/pull/4789)) [@divyegala](https://github.com/divyegala)

## 🚀 New Features

- Update Python build to scikit-build ([#4818](https://github.com/NVIDIA/cuml/pull/4818)) [@dantegd](https://github.com/dantegd)
- Vectorizers to accept Pandas Series as input ([#4811](https://github.com/NVIDIA/cuml/pull/4811)) [@shaswat-indian](https://github.com/shaswat-indian)
- Cython wrapper for v-measure ([#4785](https://github.com/NVIDIA/cuml/pull/4785)) [@shaswat-indian](https://github.com/shaswat-indian)

## 🛠️ Improvements

- Pin `dask` &amp; `distributed` for release ([#4850](https://github.com/NVIDIA/cuml/pull/4850)) [@galipremsagar](https://github.com/galipremsagar)
- Allow CuPy 11 ([#4837](https://github.com/NVIDIA/cuml/pull/4837)) [@jakirkham](https://github.com/jakirkham)
- Remove duplicate adj_to_csr implementation ([#4829](https://github.com/NVIDIA/cuml/pull/4829)) [@ahendriksen](https://github.com/ahendriksen)
- Update conda environment files to UCX 1.13.0 ([#4813](https://github.com/NVIDIA/cuml/pull/4813)) [@pentschev](https://github.com/pentschev)
- Update conda recipes to UCX 1.13.0 ([#4809](https://github.com/NVIDIA/cuml/pull/4809)) [@pentschev](https://github.com/pentschev)
- Fix #3414: remove naive versions dbscan algorithms ([#4804](https://github.com/NVIDIA/cuml/pull/4804)) [@ahendriksen](https://github.com/ahendriksen)
- Accelerate adjacency matrix to CSR conversion for DBSCAN ([#4803](https://github.com/NVIDIA/cuml/pull/4803)) [@ahendriksen](https://github.com/ahendriksen)
- Pin max version of `cuda-python` to `11.7.0` ([#4793](https://github.com/NVIDIA/cuml/pull/4793)) [@Ethyling](https://github.com/Ethyling)
- Allow cosine distance metric in dbscan ([#4776](https://github.com/NVIDIA/cuml/pull/4776)) [@tarang-jain](https://github.com/tarang-jain)
- Unpin `dask` &amp; `distributed` for development ([#4771](https://github.com/NVIDIA/cuml/pull/4771)) [@galipremsagar](https://github.com/galipremsagar)
- Clean up Thrust includes. ([#4675](https://github.com/NVIDIA/cuml/pull/4675)) [@bdice](https://github.com/bdice)
- Improvements in feature sampling ([#4278](https://github.com/NVIDIA/cuml/pull/4278)) [@vinaydes](https://github.com/vinaydes)

# cuML 22.06.00 (7 Jun 2022)

## 🐛 Bug Fixes

- Fix sg benchmark build. ([#4766](https://github.com/NVIDIA/cuml/pull/4766)) [@trivialfis](https://github.com/trivialfis)
- Resolve KRR hypothesis test failure ([#4761](https://github.com/NVIDIA/cuml/pull/4761)) [@RAMitchell](https://github.com/RAMitchell)
- Fix `KBinsDiscretizer` `bin_edges_` ([#4735](https://github.com/NVIDIA/cuml/pull/4735)) [@viclafargue](https://github.com/viclafargue)
- FIX Accept small floats in RandomForest ([#4717](https://github.com/NVIDIA/cuml/pull/4717)) [@thomasjpfan](https://github.com/thomasjpfan)
- Remove import of `scalar_broadcast_to` from stemmer ([#4706](https://github.com/NVIDIA/cuml/pull/4706)) [@viclafargue](https://github.com/viclafargue)
- Replace 22.04.x with 22.06.x in yaml files ([#4692](https://github.com/NVIDIA/cuml/pull/4692)) [@daxiongshu](https://github.com/daxiongshu)
- Replace cudf.logical_not with ~ ([#4669](https://github.com/NVIDIA/cuml/pull/4669)) [@canonizer](https://github.com/canonizer)

## 📖 Documentation

- Fix docs builds ([#4733](https://github.com/NVIDIA/cuml/pull/4733)) [@ajschmidt8](https://github.com/ajschmidt8)
- Change &quot;principals&quot; to &quot;principles&quot; ([#4695](https://github.com/NVIDIA/cuml/pull/4695)) [@cakiki](https://github.com/cakiki)
- Update pydoc and promote `ColumnTransformer` out of experimental ([#4509](https://github.com/NVIDIA/cuml/pull/4509)) [@viclafargue](https://github.com/viclafargue)

## 🚀 New Features

- float64 support in FIL functions ([#4655](https://github.com/NVIDIA/cuml/pull/4655)) [@canonizer](https://github.com/canonizer)
- float64 support in FIL core ([#4646](https://github.com/NVIDIA/cuml/pull/4646)) [@canonizer](https://github.com/canonizer)
- Allow &quot;LabelEncoder&quot; to accept cupy and numpy arrays as input. ([#4620](https://github.com/NVIDIA/cuml/pull/4620)) [@daxiongshu](https://github.com/daxiongshu)
- MNMG Logistic Regression (dask-glm wrapper) ([#3512](https://github.com/NVIDIA/cuml/pull/3512)) [@daxiongshu](https://github.com/daxiongshu)

## 🛠️ Improvements

- Pin `dask` &amp; `distributed` for release ([#4758](https://github.com/NVIDIA/cuml/pull/4758)) [@galipremsagar](https://github.com/galipremsagar)
- Simplicial set functions ([#4756](https://github.com/NVIDIA/cuml/pull/4756)) [@viclafargue](https://github.com/viclafargue)
- Upgrade Treelite to 2.4.0 ([#4752](https://github.com/NVIDIA/cuml/pull/4752)) [@hcho3](https://github.com/hcho3)
- Simplify recipes ([#4749](https://github.com/NVIDIA/cuml/pull/4749)) [@Ethyling](https://github.com/Ethyling)
- Inference for float64 random forests using FIL ([#4739](https://github.com/NVIDIA/cuml/pull/4739)) [@canonizer](https://github.com/canonizer)
- MNT Removes unused optim_batch_size from UMAP&#39;s docstring ([#4732](https://github.com/NVIDIA/cuml/pull/4732)) [@thomasjpfan](https://github.com/thomasjpfan)
- Require UCX 1.12.1+ ([#4720](https://github.com/NVIDIA/cuml/pull/4720)) [@jakirkham](https://github.com/jakirkham)
- Allow enabling raft NVTX markers when raft is installed ([#4718](https://github.com/NVIDIA/cuml/pull/4718)) [@achirkin](https://github.com/achirkin)
- Fix identifier collision ([#4716](https://github.com/NVIDIA/cuml/pull/4716)) [@viclafargue](https://github.com/viclafargue)
- Use raft::span in TreeExplainer ([#4714](https://github.com/NVIDIA/cuml/pull/4714)) [@hcho3](https://github.com/hcho3)
- Expose simplicial set functions ([#4711](https://github.com/NVIDIA/cuml/pull/4711)) [@viclafargue](https://github.com/viclafargue)
- Refactor `tests` in `cuml` ([#4703](https://github.com/NVIDIA/cuml/pull/4703)) [@galipremsagar](https://github.com/galipremsagar)
- Use conda to build python packages during GPU tests ([#4702](https://github.com/NVIDIA/cuml/pull/4702)) [@Ethyling](https://github.com/Ethyling)
- Update pinning to allow newer CMake versions. ([#4698](https://github.com/NVIDIA/cuml/pull/4698)) [@vyasr](https://github.com/vyasr)
- TreeExplainer extensions ([#4697](https://github.com/NVIDIA/cuml/pull/4697)) [@RAMitchell](https://github.com/RAMitchell)
- Add sample_weight for Ridge ([#4696](https://github.com/NVIDIA/cuml/pull/4696)) [@lowener](https://github.com/lowener)
- Unpin `dask` &amp; `distributed` for development ([#4693](https://github.com/NVIDIA/cuml/pull/4693)) [@galipremsagar](https://github.com/galipremsagar)
- float64 support in treelite-&gt;FIL import and Python layer ([#4690](https://github.com/NVIDIA/cuml/pull/4690)) [@canonizer](https://github.com/canonizer)
- Enable building static libs ([#4673](https://github.com/NVIDIA/cuml/pull/4673)) [@trxcllnt](https://github.com/trxcllnt)
- Treeshap hypothesis tests ([#4671](https://github.com/NVIDIA/cuml/pull/4671)) [@RAMitchell](https://github.com/RAMitchell)
- float64 support in multi-sum and child_index() ([#4648](https://github.com/NVIDIA/cuml/pull/4648)) [@canonizer](https://github.com/canonizer)
- Add libcuml-tests package ([#4635](https://github.com/NVIDIA/cuml/pull/4635)) [@Ethyling](https://github.com/Ethyling)
- Random ball cover algorithm for 3D data ([#4582](https://github.com/NVIDIA/cuml/pull/4582)) [@cjnolet](https://github.com/cjnolet)
- Use conda compilers ([#4577](https://github.com/NVIDIA/cuml/pull/4577)) [@Ethyling](https://github.com/Ethyling)
- Build packages using mambabuild ([#4542](https://github.com/NVIDIA/cuml/pull/4542)) [@Ethyling](https://github.com/Ethyling)

# cuML 22.04.00 (6 Apr 2022)

## 🚨 Breaking Changes

- Moving more ling prims to raft ([#4567](https://github.com/NVIDIA/cuml/pull/4567)) [@cjnolet](https://github.com/cjnolet)
- Refactor QN solver: pass parameters via a POD struct ([#4511](https://github.com/NVIDIA/cuml/pull/4511)) [@achirkin](https://github.com/achirkin)

## 🐛 Bug Fixes

- Fix single-GPU build by separating multi-GPU decomposition utils from single GPU ([#4645](https://github.com/NVIDIA/cuml/pull/4645)) [@dantegd](https://github.com/dantegd)
- RF: fix stream bug causing performance regressions ([#4644](https://github.com/NVIDIA/cuml/pull/4644)) [@venkywonka](https://github.com/venkywonka)
- XFail test_hinge_loss temporarily ([#4621](https://github.com/NVIDIA/cuml/pull/4621)) [@lowener](https://github.com/lowener)
- cuml now supports building non static treelite ([#4598](https://github.com/NVIDIA/cuml/pull/4598)) [@robertmaynard](https://github.com/robertmaynard)
- Fix mean_squared_error with cudf series ([#4584](https://github.com/NVIDIA/cuml/pull/4584)) [@daxiongshu](https://github.com/daxiongshu)
- Fix for nightly CI tests: Use CUDA_REL variable in gpu build.sh script ([#4581](https://github.com/NVIDIA/cuml/pull/4581)) [@dantegd](https://github.com/dantegd)
- Fix the TargetEncoder when transforming dataframe/series with custom index ([#4578](https://github.com/NVIDIA/cuml/pull/4578)) [@daxiongshu](https://github.com/daxiongshu)
- Removing sign from pca assertions for now. ([#4559](https://github.com/NVIDIA/cuml/pull/4559)) [@cjnolet](https://github.com/cjnolet)
- Fix compatibility of OneHotEncoder fit ([#4544](https://github.com/NVIDIA/cuml/pull/4544)) [@lowener](https://github.com/lowener)
- Fix worker streams in OLS-eig executing in an unsafe order ([#4539](https://github.com/NVIDIA/cuml/pull/4539)) [@achirkin](https://github.com/achirkin)
- Remove xfail from test_hinge_loss ([#4504](https://github.com/NVIDIA/cuml/pull/4504)) [@Nanthini10](https://github.com/Nanthini10)
- Fix automerge #4501 ([#4502](https://github.com/NVIDIA/cuml/pull/4502)) [@dantegd](https://github.com/dantegd)
- Remove classmethod of SimpleImputer ([#4439](https://github.com/NVIDIA/cuml/pull/4439)) [@lowener](https://github.com/lowener)

## 📖 Documentation

- RF: Fix improper documentation in dask-RF ([#4666](https://github.com/NVIDIA/cuml/pull/4666)) [@venkywonka](https://github.com/venkywonka)
- Add doctest ([#4618](https://github.com/NVIDIA/cuml/pull/4618)) [@lowener](https://github.com/lowener)
- Fix document layouts in Parameters sections ([#4609](https://github.com/NVIDIA/cuml/pull/4609)) [@Yosshi999](https://github.com/Yosshi999)
- Updates to consistency of MNMG PCA/TSVD solvers (docs + code consolidation) ([#4556](https://github.com/NVIDIA/cuml/pull/4556)) [@cjnolet](https://github.com/cjnolet)

## 🚀 New Features

- Add a dummy argument `deep` to `TargetEncoder.get_params()` ([#4601](https://github.com/NVIDIA/cuml/pull/4601)) [@daxiongshu](https://github.com/daxiongshu)
- Add Complement Naive Bayes ([#4595](https://github.com/NVIDIA/cuml/pull/4595)) [@lowener](https://github.com/lowener)
- Add get_params() to TargetEncoder ([#4588](https://github.com/NVIDIA/cuml/pull/4588)) [@daxiongshu](https://github.com/daxiongshu)
- Target Encoder with variance statistics ([#4483](https://github.com/NVIDIA/cuml/pull/4483)) [@daxiongshu](https://github.com/daxiongshu)
- Interruptible execution ([#4463](https://github.com/NVIDIA/cuml/pull/4463)) [@achirkin](https://github.com/achirkin)
- Configurable libcuml++ per algorithm ([#4296](https://github.com/NVIDIA/cuml/pull/4296)) [@dantegd](https://github.com/dantegd)

## 🛠️ Improvements

- Adding some prints when hdbscan assertion fails ([#4656](https://github.com/NVIDIA/cuml/pull/4656)) [@cjnolet](https://github.com/cjnolet)
- Temporarily disable new `ops-bot` functionality ([#4652](https://github.com/NVIDIA/cuml/pull/4652)) [@ajschmidt8](https://github.com/ajschmidt8)
- Use CPMFindPackage to retrieve `cumlprims_mg` ([#4649](https://github.com/NVIDIA/cuml/pull/4649)) [@trxcllnt](https://github.com/trxcllnt)
- Pin `dask` &amp; `distributed` versions ([#4647](https://github.com/NVIDIA/cuml/pull/4647)) [@galipremsagar](https://github.com/galipremsagar)
- Remove RAFT MM includes ([#4637](https://github.com/NVIDIA/cuml/pull/4637)) [@viclafargue](https://github.com/viclafargue)
- Add option to build RAFT artifacts statically into libcuml++ ([#4633](https://github.com/NVIDIA/cuml/pull/4633)) [@dantegd](https://github.com/dantegd)
- Upgrade `dask` &amp; `distributed` minimum version ([#4632](https://github.com/NVIDIA/cuml/pull/4632)) [@galipremsagar](https://github.com/galipremsagar)
- Add `.github/ops-bot.yaml` config file ([#4630](https://github.com/NVIDIA/cuml/pull/4630)) [@ajschmidt8](https://github.com/ajschmidt8)
- Small fixes for certain test failures ([#4628](https://github.com/NVIDIA/cuml/pull/4628)) [@vinaydes](https://github.com/vinaydes)
- Templatizing FIL types to add float64 support ([#4625](https://github.com/NVIDIA/cuml/pull/4625)) [@canonizer](https://github.com/canonizer)
- Fitsne as default tsne method ([#4597](https://github.com/NVIDIA/cuml/pull/4597)) [@lowener](https://github.com/lowener)
- Add `get_feature_names` to OneHotEncoder ([#4596](https://github.com/NVIDIA/cuml/pull/4596)) [@viclafargue](https://github.com/viclafargue)
- Fix OOM and cudaContext crash in C++ benchmarks ([#4594](https://github.com/NVIDIA/cuml/pull/4594)) [@RAMitchell](https://github.com/RAMitchell)
- Using Pyraft and automatically cloning when raft pin changes ([#4593](https://github.com/NVIDIA/cuml/pull/4593)) [@cjnolet](https://github.com/cjnolet)
- Upgrade Treelite to 2.3.0 ([#4590](https://github.com/NVIDIA/cuml/pull/4590)) [@hcho3](https://github.com/hcho3)
- Sphinx warnings as errors ([#4585](https://github.com/NVIDIA/cuml/pull/4585)) [@RAMitchell](https://github.com/RAMitchell)
- Adding missing FAISS license ([#4579](https://github.com/NVIDIA/cuml/pull/4579)) [@cjnolet](https://github.com/cjnolet)
- Add QN solver to ElasticNet and Lasso models ([#4576](https://github.com/NVIDIA/cuml/pull/4576)) [@achirkin](https://github.com/achirkin)
- Move remaining stats prims to raft ([#4568](https://github.com/NVIDIA/cuml/pull/4568)) [@cjnolet](https://github.com/cjnolet)
- Moving more ling prims to raft ([#4567](https://github.com/NVIDIA/cuml/pull/4567)) [@cjnolet](https://github.com/cjnolet)
- Adding libraft conda dependencies ([#4564](https://github.com/NVIDIA/cuml/pull/4564)) [@cjnolet](https://github.com/cjnolet)
- Fix RF integer overflow ([#4563](https://github.com/NVIDIA/cuml/pull/4563)) [@RAMitchell](https://github.com/RAMitchell)
- Add CMake `install` rules for tests ([#4551](https://github.com/NVIDIA/cuml/pull/4551)) [@ajschmidt8](https://github.com/ajschmidt8)
- Faster GLM preprocessing by fusing kernels ([#4549](https://github.com/NVIDIA/cuml/pull/4549)) [@achirkin](https://github.com/achirkin)
- RAFT API updates for lap, label, cluster, and spectral apis ([#4548](https://github.com/NVIDIA/cuml/pull/4548)) [@cjnolet](https://github.com/cjnolet)
- Moving cusparse wrappers to detail API in RAFT. ([#4547](https://github.com/NVIDIA/cuml/pull/4547)) [@cjnolet](https://github.com/cjnolet)
- Unpin max `dask` and `distributed` versions ([#4546](https://github.com/NVIDIA/cuml/pull/4546)) [@galipremsagar](https://github.com/galipremsagar)
- Kernel density estimation ([#4545](https://github.com/NVIDIA/cuml/pull/4545)) [@RAMitchell](https://github.com/RAMitchell)
- Update `xgboost` version in CI ([#4541](https://github.com/NVIDIA/cuml/pull/4541)) [@ajschmidt8](https://github.com/ajschmidt8)
- replaces `ccache` with `sccache` ([#4534](https://github.com/NVIDIA/cuml/pull/4534)) [@AyodeAwe](https://github.com/AyodeAwe)
- Remove RAFT memory management (2/2) ([#4526](https://github.com/NVIDIA/cuml/pull/4526)) [@viclafargue](https://github.com/viclafargue)
- Updating RAFT linalg headers ([#4515](https://github.com/NVIDIA/cuml/pull/4515)) [@divyegala](https://github.com/divyegala)
- Refactor QN solver: pass parameters via a POD struct ([#4511](https://github.com/NVIDIA/cuml/pull/4511)) [@achirkin](https://github.com/achirkin)
- Kernel ridge regression ([#4492](https://github.com/NVIDIA/cuml/pull/4492)) [@RAMitchell](https://github.com/RAMitchell)
- QN solvers: Use different gradient norms for different for different loss functions. ([#4491](https://github.com/NVIDIA/cuml/pull/4491)) [@achirkin](https://github.com/achirkin)
- RF: Variable binning and other minor refactoring ([#4479](https://github.com/NVIDIA/cuml/pull/4479)) [@venkywonka](https://github.com/venkywonka)
- Rewrite CD solver using more BLAS ([#4446](https://github.com/NVIDIA/cuml/pull/4446)) [@achirkin](https://github.com/achirkin)
- Add support for sample_weights in LinearRegression ([#4428](https://github.com/NVIDIA/cuml/pull/4428)) [@lowener](https://github.com/lowener)
- Nightly automated benchmark ([#4414](https://github.com/NVIDIA/cuml/pull/4414)) [@viclafargue](https://github.com/viclafargue)
- Use FAISS with RMM ([#4297](https://github.com/NVIDIA/cuml/pull/4297)) [@viclafargue](https://github.com/viclafargue)
- Split C++ tests into separate binaries ([#4295](https://github.com/NVIDIA/cuml/pull/4295)) [@dantegd](https://github.com/dantegd)

# cuML 22.02.00 (2 Feb 2022)

## 🚨 Breaking Changes

- Move NVTX range helpers to raft ([#4445](https://github.com/NVIDIA/cuml/pull/4445)) [@achirkin](https://github.com/achirkin)

## 🐛 Bug Fixes

- Always upload libcuml ([#4530](https://github.com/NVIDIA/cuml/pull/4530)) [@raydouglass](https://github.com/raydouglass)
- Fix RAFT pin to main branch ([#4508](https://github.com/NVIDIA/cuml/pull/4508)) [@dantegd](https://github.com/dantegd)
- Pin `dask` &amp; `distributed` ([#4505](https://github.com/NVIDIA/cuml/pull/4505)) [@galipremsagar](https://github.com/galipremsagar)
- Replace use of RMM provided CUDA bindings with CUDA Python ([#4499](https://github.com/NVIDIA/cuml/pull/4499)) [@shwina](https://github.com/shwina)
- Dataframe Index as columns in ColumnTransformer ([#4481](https://github.com/NVIDIA/cuml/pull/4481)) [@viclafargue](https://github.com/viclafargue)
- Support compilation with Thrust 1.15 ([#4469](https://github.com/NVIDIA/cuml/pull/4469)) [@robertmaynard](https://github.com/robertmaynard)
- fix minor ASAN issues in UMAPAlgo::Optimize::find_params_ab() ([#4405](https://github.com/NVIDIA/cuml/pull/4405)) [@yitao-li](https://github.com/yitao-li)

## 📖 Documentation

- Remove comment numerical warning ([#4408](https://github.com/NVIDIA/cuml/pull/4408)) [@viclafargue](https://github.com/viclafargue)
- Fix docstring for npermutations in PermutationExplainer ([#4402](https://github.com/NVIDIA/cuml/pull/4402)) [@hcho3](https://github.com/hcho3)

## 🚀 New Features

- Combine and expose SVC&#39;s support vectors when fitting multi-class data ([#4454](https://github.com/NVIDIA/cuml/pull/4454)) [@NV-jpt](https://github.com/NV-jpt)
- Accept fold index for TargetEncoder ([#4453](https://github.com/NVIDIA/cuml/pull/4453)) [@daxiongshu](https://github.com/daxiongshu)
- Move NVTX range helpers to raft ([#4445](https://github.com/NVIDIA/cuml/pull/4445)) [@achirkin](https://github.com/achirkin)

## 🛠️ Improvements

- Fix packages upload ([#4517](https://github.com/NVIDIA/cuml/pull/4517)) [@Ethyling](https://github.com/Ethyling)
- Testing split fused l2 knn compilation units ([#4514](https://github.com/NVIDIA/cuml/pull/4514)) [@cjnolet](https://github.com/cjnolet)
- Prepare upload scripts for Python 3.7 removal ([#4500](https://github.com/NVIDIA/cuml/pull/4500)) [@Ethyling](https://github.com/Ethyling)
- Renaming macros with their RAFT counterparts ([#4496](https://github.com/NVIDIA/cuml/pull/4496)) [@divyegala](https://github.com/divyegala)
- Allow CuPy 10 ([#4487](https://github.com/NVIDIA/cuml/pull/4487)) [@jakirkham](https://github.com/jakirkham)
- Upgrade Treelite to 2.2.1 ([#4484](https://github.com/NVIDIA/cuml/pull/4484)) [@hcho3](https://github.com/hcho3)
- Unpin `dask` and `distributed` ([#4482](https://github.com/NVIDIA/cuml/pull/4482)) [@galipremsagar](https://github.com/galipremsagar)
- Support categorical splits in in TreeExplainer ([#4473](https://github.com/NVIDIA/cuml/pull/4473)) [@hcho3](https://github.com/hcho3)
- Remove RAFT memory management ([#4468](https://github.com/NVIDIA/cuml/pull/4468)) [@viclafargue](https://github.com/viclafargue)
- Add missing imports tests ([#4452](https://github.com/NVIDIA/cuml/pull/4452)) [@Ethyling](https://github.com/Ethyling)
- Update CUDA 11.5 conda environment to use 22.02 pinnings. ([#4450](https://github.com/NVIDIA/cuml/pull/4450)) [@bdice](https://github.com/bdice)
- Support cuML / scikit-learn RF classifiers in TreeExplainer ([#4447](https://github.com/NVIDIA/cuml/pull/4447)) [@hcho3](https://github.com/hcho3)
- Remove `IncludeCategories` from `.clang-format` ([#4438](https://github.com/NVIDIA/cuml/pull/4438)) [@codereport](https://github.com/codereport)
- Simplify perplexity normalization in t-SNE ([#4425](https://github.com/NVIDIA/cuml/pull/4425)) [@zbjornson](https://github.com/zbjornson)
- Unify dense and sparse tests ([#4417](https://github.com/NVIDIA/cuml/pull/4417)) [@levsnv](https://github.com/levsnv)
- Update ucx-py version on release using rvc ([#4411](https://github.com/NVIDIA/cuml/pull/4411)) [@Ethyling](https://github.com/Ethyling)
- Universal Treelite tree walk function for FIL ([#4407](https://github.com/NVIDIA/cuml/pull/4407)) [@levsnv](https://github.com/levsnv)
- Update to UCX-Py 0.24 ([#4396](https://github.com/NVIDIA/cuml/pull/4396)) [@pentschev](https://github.com/pentschev)
- Using sparse public API functions from RAFT ([#4389](https://github.com/NVIDIA/cuml/pull/4389)) [@cjnolet](https://github.com/cjnolet)
- Add a warning to prefer LinearSVM over SVM(kernel=&#39;linear&#39;) ([#4382](https://github.com/NVIDIA/cuml/pull/4382)) [@achirkin](https://github.com/achirkin)
- Hiding cusparse deprecation warnings ([#4373](https://github.com/NVIDIA/cuml/pull/4373)) [@cjnolet](https://github.com/cjnolet)
- Unify dense and sparse import in FIL ([#4328](https://github.com/NVIDIA/cuml/pull/4328)) [@levsnv](https://github.com/levsnv)
- Integrating RAFT handle updates ([#4313](https://github.com/NVIDIA/cuml/pull/4313)) [@divyegala](https://github.com/divyegala)
- Use RAFT template instantations for distances ([#4302](https://github.com/NVIDIA/cuml/pull/4302)) [@cjnolet](https://github.com/cjnolet)
- RF: code re-organization to enhance build parallelism ([#4299](https://github.com/NVIDIA/cuml/pull/4299)) [@venkywonka](https://github.com/venkywonka)
- Add option to build faiss and treelite shared libs, inherit common dependencies from raft ([#4256](https://github.com/NVIDIA/cuml/pull/4256)) [@trxcllnt](https://github.com/trxcllnt)

# cuML 21.12.00 (9 Dec 2021)

## 🚨 Breaking Changes

- Fix indexing of PCA to use safer types ([#4255](https://github.com/NVIDIA/cuml/pull/4255)) [@lowener](https://github.com/lowener)
- RF: Add Gamma and Inverse Gaussian loss criteria ([#4216](https://github.com/NVIDIA/cuml/pull/4216)) [@venkywonka](https://github.com/venkywonka)
- update RF docs ([#4138](https://github.com/NVIDIA/cuml/pull/4138)) [@venkywonka](https://github.com/venkywonka)

## 🐛 Bug Fixes

- Update conda recipe to have explicit libcusolver ([#4392](https://github.com/NVIDIA/cuml/pull/4392)) [@dantegd](https://github.com/dantegd)
- Restore FIL convention of inlining code ([#4366](https://github.com/NVIDIA/cuml/pull/4366)) [@levsnv](https://github.com/levsnv)
- Fix SVR intercept AttributeError ([#4358](https://github.com/NVIDIA/cuml/pull/4358)) [@lowener](https://github.com/lowener)
- Fix `is_stable_build` logic for CI scripts ([#4350](https://github.com/NVIDIA/cuml/pull/4350)) [@ajschmidt8](https://github.com/ajschmidt8)
- Temporarily disable rmm devicebuffer in array.py ([#4333](https://github.com/NVIDIA/cuml/pull/4333)) [@dantegd](https://github.com/dantegd)
- Fix categorical test in python ([#4326](https://github.com/NVIDIA/cuml/pull/4326)) [@levsnv](https://github.com/levsnv)
- Revert &quot;Merge pull request #4319 from AyodeAwe/branch-21.12&quot; ([#4325](https://github.com/NVIDIA/cuml/pull/4325)) [@ajschmidt8](https://github.com/ajschmidt8)
- Preserve indexing in methods when applied to DataFrame and Series objects ([#4317](https://github.com/NVIDIA/cuml/pull/4317)) [@dantegd](https://github.com/dantegd)
- Fix potential CUDA context poison when negative (invalid) categories provided to FIL model ([#4314](https://github.com/NVIDIA/cuml/pull/4314)) [@levsnv](https://github.com/levsnv)
- Using sparse expanded distances where possible ([#4310](https://github.com/NVIDIA/cuml/pull/4310)) [@cjnolet](https://github.com/cjnolet)
- Fix for `mean_squared_error` ([#4287](https://github.com/NVIDIA/cuml/pull/4287)) [@viclafargue](https://github.com/viclafargue)
- Fix for Categorical Naive Bayes sparse handling ([#4277](https://github.com/NVIDIA/cuml/pull/4277)) [@lowener](https://github.com/lowener)
- Throw an explicit excpetion if the input array is empty in DBSCAN.fit #4273 ([#4275](https://github.com/NVIDIA/cuml/pull/4275)) [@viktorkovesd](https://github.com/viktorkovesd)
- Fix KernelExplainer returning TypeError for certain input ([#4272](https://github.com/NVIDIA/cuml/pull/4272)) [@Nanthini10](https://github.com/Nanthini10)
- Remove most warnings from pytest suite ([#4196](https://github.com/NVIDIA/cuml/pull/4196)) [@dantegd](https://github.com/dantegd)

## 📖 Documentation

- Add experimental GPUTreeSHAP to API doc ([#4398](https://github.com/NVIDIA/cuml/pull/4398)) [@hcho3](https://github.com/hcho3)
- Fix GLM typo on device/host pointer ([#4320](https://github.com/NVIDIA/cuml/pull/4320)) [@lowener](https://github.com/lowener)
- update RF docs ([#4138](https://github.com/NVIDIA/cuml/pull/4138)) [@venkywonka](https://github.com/venkywonka)

## 🚀 New Features

- Add GPUTreeSHAP to cuML explainer module (experimental) ([#4351](https://github.com/NVIDIA/cuml/pull/4351)) [@hcho3](https://github.com/hcho3)
- Enable training single GPU cuML models using Dask DataFrames and Series ([#4300](https://github.com/NVIDIA/cuml/pull/4300)) [@ChrisJar](https://github.com/ChrisJar)
- LinearSVM using QN solvers ([#4268](https://github.com/NVIDIA/cuml/pull/4268)) [@achirkin](https://github.com/achirkin)
- Add support for exogenous variables to ARIMA ([#4221](https://github.com/NVIDIA/cuml/pull/4221)) [@Nyrio](https://github.com/Nyrio)
- Use opt-in shared memory carveout for FIL ([#3759](https://github.com/NVIDIA/cuml/pull/3759)) [@levsnv](https://github.com/levsnv)
- Symbolic Regression/Classification C/C++ ([#3638](https://github.com/NVIDIA/cuml/pull/3638)) [@vimarsh6739](https://github.com/vimarsh6739)

## 🛠️ Improvements

- Fix Changelog Merge Conflicts for `branch-21.12` ([#4393](https://github.com/NVIDIA/cuml/pull/4393)) [@ajschmidt8](https://github.com/ajschmidt8)
- Pin max `dask` and `distributed` to `2012.11.2` ([#4390](https://github.com/NVIDIA/cuml/pull/4390)) [@galipremsagar](https://github.com/galipremsagar)
- Fix forward merge #4349 ([#4374](https://github.com/NVIDIA/cuml/pull/4374)) [@dantegd](https://github.com/dantegd)
- Upgrade `clang` to `11.1.0` ([#4372](https://github.com/NVIDIA/cuml/pull/4372)) [@galipremsagar](https://github.com/galipremsagar)
- Update clang-format version in docs; allow unanchored version string ([#4365](https://github.com/NVIDIA/cuml/pull/4365)) [@zbjornson](https://github.com/zbjornson)
- Add CUDA 11.5 developer environment ([#4364](https://github.com/NVIDIA/cuml/pull/4364)) [@dantegd](https://github.com/dantegd)
- Fix aliasing violation in t-SNE ([#4363](https://github.com/NVIDIA/cuml/pull/4363)) [@zbjornson](https://github.com/zbjornson)
- Promote FITSNE from experimental ([#4361](https://github.com/NVIDIA/cuml/pull/4361)) [@lowener](https://github.com/lowener)
- Fix unnecessary f32/f64 conversions in t-SNE KL calc ([#4331](https://github.com/NVIDIA/cuml/pull/4331)) [@zbjornson](https://github.com/zbjornson)
- Update rapids-cmake version ([#4330](https://github.com/NVIDIA/cuml/pull/4330)) [@dantegd](https://github.com/dantegd)
- rapids-cmake version update to 21.12 ([#4327](https://github.com/NVIDIA/cuml/pull/4327)) [@dantegd](https://github.com/dantegd)
- Use compute-sanitizer instead of cuda-memcheck ([#4324](https://github.com/NVIDIA/cuml/pull/4324)) [@teju85](https://github.com/teju85)
- Ability to pass fp64 type to cuml benchmarks ([#4323](https://github.com/NVIDIA/cuml/pull/4323)) [@teju85](https://github.com/teju85)
- Split treelite fil import from `forest` object definition ([#4306](https://github.com/NVIDIA/cuml/pull/4306)) [@levsnv](https://github.com/levsnv)
- update xgboost version ([#4301](https://github.com/NVIDIA/cuml/pull/4301)) [@msadang](https://github.com/msadang)
- Accounting for RAFT updates to matrix, stats, and random implementations in detail ([#4294](https://github.com/NVIDIA/cuml/pull/4294)) [@divyegala](https://github.com/divyegala)
- Update cudf matrix calls for to_numpy and to_cupy ([#4293](https://github.com/NVIDIA/cuml/pull/4293)) [@dantegd](https://github.com/dantegd)
- Update `conda` recipes for Enhanced Compatibility effort ([#4288](https://github.com/NVIDIA/cuml/pull/4288)) [@ajschmidt8](https://github.com/ajschmidt8)
- Increase parallelism from 4 to 8 jobs in CI ([#4286](https://github.com/NVIDIA/cuml/pull/4286)) [@dantegd](https://github.com/dantegd)
- RAFT distance prims public API update ([#4280](https://github.com/NVIDIA/cuml/pull/4280)) [@cjnolet](https://github.com/cjnolet)
- Update to UCX-Py 0.23 ([#4274](https://github.com/NVIDIA/cuml/pull/4274)) [@pentschev](https://github.com/pentschev)
- In FIL, clip blocks_per_sm to one wave instead of asserting ([#4271](https://github.com/NVIDIA/cuml/pull/4271)) [@levsnv](https://github.com/levsnv)
- Update of &quot;Gracefully accept &#39;n_jobs&#39;, a common sklearn parameter, in NearestNeighbors Estimator&quot; ([#4267](https://github.com/NVIDIA/cuml/pull/4267)) [@NV-jpt](https://github.com/NV-jpt)
- Improve numerical stability of the Kalman filter for ARIMA ([#4259](https://github.com/NVIDIA/cuml/pull/4259)) [@Nyrio](https://github.com/Nyrio)
- Fix indexing of PCA to use safer types ([#4255](https://github.com/NVIDIA/cuml/pull/4255)) [@lowener](https://github.com/lowener)
- Change calculation of ARIMA confidence intervals ([#4248](https://github.com/NVIDIA/cuml/pull/4248)) [@Nyrio](https://github.com/Nyrio)
- Unpin `dask` &amp; `distributed` in CI ([#4235](https://github.com/NVIDIA/cuml/pull/4235)) [@galipremsagar](https://github.com/galipremsagar)
- RF: Add Gamma and Inverse Gaussian loss criteria ([#4216](https://github.com/NVIDIA/cuml/pull/4216)) [@venkywonka](https://github.com/venkywonka)
- Exposing KL divergence in TSNE ([#4208](https://github.com/NVIDIA/cuml/pull/4208)) [@viclafargue](https://github.com/viclafargue)
- Unify template parameter dispatch for FIL inference and shared memory footprint estimation ([#4013](https://github.com/NVIDIA/cuml/pull/4013)) [@levsnv](https://github.com/levsnv)

# cuML 21.10.00 (7 Oct 2021)

## 🚨 Breaking Changes

- RF: python api behaviour refactor ([#4207](https://github.com/NVIDIA/cuml/pull/4207)) [@venkywonka](https://github.com/venkywonka)
- Implement vector leaf for random forest ([#4191](https://github.com/NVIDIA/cuml/pull/4191)) [@RAMitchell](https://github.com/RAMitchell)
- Random forest refactoring ([#4166](https://github.com/NVIDIA/cuml/pull/4166)) [@RAMitchell](https://github.com/RAMitchell)
- RF: Add Poisson deviance impurity criterion ([#4156](https://github.com/NVIDIA/cuml/pull/4156)) [@venkywonka](https://github.com/venkywonka)
- avoid paramsSolver::{n_rows,n_cols} shadowing their base class counterparts ([#4130](https://github.com/NVIDIA/cuml/pull/4130)) [@yitao-li](https://github.com/yitao-li)
- Apply modifications to account for RAFT changes ([#4077](https://github.com/NVIDIA/cuml/pull/4077)) [@viclafargue](https://github.com/viclafargue)

## 🐛 Bug Fixes

- Update scikit-learn version in conda dev envs to 0.24 ([#4241](https://github.com/NVIDIA/cuml/pull/4241)) [@dantegd](https://github.com/dantegd)
- Using pinned host memory for Random Forest and DBSCAN ([#4215](https://github.com/NVIDIA/cuml/pull/4215)) [@divyegala](https://github.com/divyegala)
- Make sure we keep the rapids-cmake and cuml cal version in sync ([#4213](https://github.com/NVIDIA/cuml/pull/4213)) [@robertmaynard](https://github.com/robertmaynard)
- Add thrust_create_target to install export in CMakeLists ([#4209](https://github.com/NVIDIA/cuml/pull/4209)) [@dantegd](https://github.com/dantegd)
- Change the error type to match sklearn. ([#4198](https://github.com/NVIDIA/cuml/pull/4198)) [@achirkin](https://github.com/achirkin)
- Fixing remaining hdbscan bug ([#4179](https://github.com/NVIDIA/cuml/pull/4179)) [@cjnolet](https://github.com/cjnolet)
- Fix for cuDF changes to cudf.core ([#4168](https://github.com/NVIDIA/cuml/pull/4168)) [@dantegd](https://github.com/dantegd)
- Fixing UMAP reproducibility pytest failures in 11.4 by using random init for now ([#4152](https://github.com/NVIDIA/cuml/pull/4152)) [@cjnolet](https://github.com/cjnolet)
- avoid paramsSolver::{n_rows,n_cols} shadowing their base class counterparts ([#4130](https://github.com/NVIDIA/cuml/pull/4130)) [@yitao-li](https://github.com/yitao-li)
- Use the new RAPIDS.cmake to fetch rapids-cmake ([#4102](https://github.com/NVIDIA/cuml/pull/4102)) [@robertmaynard](https://github.com/robertmaynard)

## 📖 Documentation

- Expose train_test_split in API doc ([#4234](https://github.com/NVIDIA/cuml/pull/4234)) [@hcho3](https://github.com/hcho3)
- Adding docs for `.get_feature_names()` inside `TfidfVectorizer` ([#4226](https://github.com/NVIDIA/cuml/pull/4226)) [@mayankanand007](https://github.com/mayankanand007)
- Removing experimental flag from hdbscan description in docs ([#4211](https://github.com/NVIDIA/cuml/pull/4211)) [@cjnolet](https://github.com/cjnolet)
- updated build instructions ([#4200](https://github.com/NVIDIA/cuml/pull/4200)) [@shaneding](https://github.com/shaneding)
- Forward-merge branch-21.08 to branch-21.10 ([#4171](https://github.com/NVIDIA/cuml/pull/4171)) [@jakirkham](https://github.com/jakirkham)

## 🚀 New Features

- Experimental option to build libcuml++ only with FIL ([#4225](https://github.com/NVIDIA/cuml/pull/4225)) [@dantegd](https://github.com/dantegd)
- FIL to import categorical models from treelite ([#4173](https://github.com/NVIDIA/cuml/pull/4173)) [@levsnv](https://github.com/levsnv)
- Add hamming, jensen-shannon, kl-divergence, correlation and russellrao distance metrics ([#4155](https://github.com/NVIDIA/cuml/pull/4155)) [@mdoijade](https://github.com/mdoijade)
- Add Categorical Naive Bayes ([#4150](https://github.com/NVIDIA/cuml/pull/4150)) [@lowener](https://github.com/lowener)
- FIL to infer categorical forests and generate them in C++ tests ([#4092](https://github.com/NVIDIA/cuml/pull/4092)) [@levsnv](https://github.com/levsnv)
- Add Gaussian Naive Bayes ([#4079](https://github.com/NVIDIA/cuml/pull/4079)) [@lowener](https://github.com/lowener)
- ARIMA - Add support for missing observations and padding ([#4058](https://github.com/NVIDIA/cuml/pull/4058)) [@Nyrio](https://github.com/Nyrio)

## 🛠️ Improvements

- Pin max `dask` and `distributed` versions to 2021.09.1 ([#4229](https://github.com/NVIDIA/cuml/pull/4229)) [@galipremsagar](https://github.com/galipremsagar)
- Fea/umap refine ([#4228](https://github.com/NVIDIA/cuml/pull/4228)) [@AjayThorve](https://github.com/AjayThorve)
- Upgrade Treelite to 2.1.0 ([#4220](https://github.com/NVIDIA/cuml/pull/4220)) [@hcho3](https://github.com/hcho3)
- Add option to clone RAFT even if it is in the environment ([#4217](https://github.com/NVIDIA/cuml/pull/4217)) [@dantegd](https://github.com/dantegd)
- RF: python api behaviour refactor ([#4207](https://github.com/NVIDIA/cuml/pull/4207)) [@venkywonka](https://github.com/venkywonka)
- Pytest updates for Scikit-learn 0.24 ([#4205](https://github.com/NVIDIA/cuml/pull/4205)) [@dantegd](https://github.com/dantegd)
- Faster glm ols-via-eigendecomposition algorithm ([#4201](https://github.com/NVIDIA/cuml/pull/4201)) [@achirkin](https://github.com/achirkin)
- Implement vector leaf for random forest ([#4191](https://github.com/NVIDIA/cuml/pull/4191)) [@RAMitchell](https://github.com/RAMitchell)
- Refactor kmeans sampling code ([#4190](https://github.com/NVIDIA/cuml/pull/4190)) [@Nanthini10](https://github.com/Nanthini10)
- Gracefully accept &#39;n_jobs&#39;, a common sklearn parameter, in NearestNeighbors Estimator ([#4178](https://github.com/NVIDIA/cuml/pull/4178)) [@NV-jpt](https://github.com/NV-jpt)
- Update with rapids cmake new features ([#4175](https://github.com/NVIDIA/cuml/pull/4175)) [@robertmaynard](https://github.com/robertmaynard)
- Update to UCX-Py 0.22 ([#4174](https://github.com/NVIDIA/cuml/pull/4174)) [@pentschev](https://github.com/pentschev)
- Random forest refactoring ([#4166](https://github.com/NVIDIA/cuml/pull/4166)) [@RAMitchell](https://github.com/RAMitchell)
- Fix log level for dask tree_reduce ([#4163](https://github.com/NVIDIA/cuml/pull/4163)) [@lowener](https://github.com/lowener)
- Add CUDA 11.4 development environment ([#4160](https://github.com/NVIDIA/cuml/pull/4160)) [@dantegd](https://github.com/dantegd)
- RF: Add Poisson deviance impurity criterion ([#4156](https://github.com/NVIDIA/cuml/pull/4156)) [@venkywonka](https://github.com/venkywonka)
- Split FIL infer_k into phases to speed up compilation (when a patch is applied) ([#4148](https://github.com/NVIDIA/cuml/pull/4148)) [@levsnv](https://github.com/levsnv)
- RF node queue rewrite ([#4125](https://github.com/NVIDIA/cuml/pull/4125)) [@RAMitchell](https://github.com/RAMitchell)
- Remove max version pin for `dask` &amp; `distributed` on development branch ([#4118](https://github.com/NVIDIA/cuml/pull/4118)) [@galipremsagar](https://github.com/galipremsagar)
- Correct name of a cmake function in get_spdlog.cmake ([#4106](https://github.com/NVIDIA/cuml/pull/4106)) [@robertmaynard](https://github.com/robertmaynard)
- Apply modifications to account for RAFT changes ([#4077](https://github.com/NVIDIA/cuml/pull/4077)) [@viclafargue](https://github.com/viclafargue)
- Warnings are errors ([#4075](https://github.com/NVIDIA/cuml/pull/4075)) [@harrism](https://github.com/harrism)
- ENH Replace gpuci_conda_retry with gpuci_mamba_retry ([#4065](https://github.com/NVIDIA/cuml/pull/4065)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Changes to NearestNeighbors to call 2d random ball cover ([#4003](https://github.com/NVIDIA/cuml/pull/4003)) [@cjnolet](https://github.com/cjnolet)
- support space in workspace ([#3752](https://github.com/NVIDIA/cuml/pull/3752)) [@jolorunyomi](https://github.com/jolorunyomi)

# cuML 21.08.00 (4 Aug 2021)

## 🚨 Breaking Changes

- Remove deprecated target_weights in UMAP ([#4081](https://github.com/NVIDIA/cuml/pull/4081)) [@lowener](https://github.com/lowener)
- Upgrade Treelite to 2.0.0 ([#4072](https://github.com/NVIDIA/cuml/pull/4072)) [@hcho3](https://github.com/hcho3)
- RF/DT cleanup ([#4005](https://github.com/NVIDIA/cuml/pull/4005)) [@venkywonka](https://github.com/venkywonka)
- RF: memset and batch size optimization for computing splits ([#4001](https://github.com/NVIDIA/cuml/pull/4001)) [@venkywonka](https://github.com/venkywonka)
- Remove old RF backend ([#3868](https://github.com/NVIDIA/cuml/pull/3868)) [@RAMitchell](https://github.com/RAMitchell)
- Enable warp-per-tree inference in FIL for regression and binary classification ([#3760](https://github.com/NVIDIA/cuml/pull/3760)) [@levsnv](https://github.com/levsnv)

## 🐛 Bug Fixes

- Disabling umap reproducibility tests for cuda 11.4 ([#4128](https://github.com/NVIDIA/cuml/pull/4128)) [@cjnolet](https://github.com/cjnolet)
- Fix for crash in RF when `max_leaves` parameter is specified ([#4126](https://github.com/NVIDIA/cuml/pull/4126)) [@vinaydes](https://github.com/vinaydes)
- Running umap mnmg test twice ([#4112](https://github.com/NVIDIA/cuml/pull/4112)) [@cjnolet](https://github.com/cjnolet)
- Minimal fix for `SparseRandomProjection` ([#4100](https://github.com/NVIDIA/cuml/pull/4100)) [@viclafargue](https://github.com/viclafargue)
- Creating copy of `components` in PCA transform and inverse transform ([#4099](https://github.com/NVIDIA/cuml/pull/4099)) [@divyegala](https://github.com/divyegala)
- Fix SVM model parameter handling in case n_support=0 ([#4097](https://github.com/NVIDIA/cuml/pull/4097)) [@tfeher](https://github.com/tfeher)
- Fix set_params for linear models ([#4096](https://github.com/NVIDIA/cuml/pull/4096)) [@lowener](https://github.com/lowener)
- Fix train test split pytest comparison ([#4062](https://github.com/NVIDIA/cuml/pull/4062)) [@dantegd](https://github.com/dantegd)
- Fix fit_transform on KMeans ([#4055](https://github.com/NVIDIA/cuml/pull/4055)) [@lowener](https://github.com/lowener)
- Fixing -1 key access in 1nn reduce op in HDBSCAN ([#4052](https://github.com/NVIDIA/cuml/pull/4052)) [@divyegala](https://github.com/divyegala)
- Disable installing gbench to avoid container permission issues ([#4049](https://github.com/NVIDIA/cuml/pull/4049)) [@dantegd](https://github.com/dantegd)
- Fix double fit crash in preprocessing models ([#4040](https://github.com/NVIDIA/cuml/pull/4040)) [@viclafargue](https://github.com/viclafargue)
- Always add `faiss` library alias if it&#39;s missing ([#4028](https://github.com/NVIDIA/cuml/pull/4028)) [@trxcllnt](https://github.com/trxcllnt)
- Fixing intermittent HBDSCAN pytest failure in CI ([#4025](https://github.com/NVIDIA/cuml/pull/4025)) [@divyegala](https://github.com/divyegala)
- HDBSCAN bug on A100 ([#4024](https://github.com/NVIDIA/cuml/pull/4024)) [@divyegala](https://github.com/divyegala)
- Add treelite include paths to treelite targets ([#4023](https://github.com/NVIDIA/cuml/pull/4023)) [@trxcllnt](https://github.com/trxcllnt)
- Add Treelite_BINARY_DIR include to `cuml++` build interface include paths ([#4018](https://github.com/NVIDIA/cuml/pull/4018)) [@trxcllnt](https://github.com/trxcllnt)
- Small ARIMA-related bug fixes in Hessenberg reduction and make_arima ([#4017](https://github.com/NVIDIA/cuml/pull/4017)) [@Nyrio](https://github.com/Nyrio)
- Update setup.py ([#4015](https://github.com/NVIDIA/cuml/pull/4015)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update `treelite` version in `get_treelite.cmake` ([#4014](https://github.com/NVIDIA/cuml/pull/4014)) [@ajschmidt8](https://github.com/ajschmidt8)
- Fix build with latest RAFT branch-21.08 ([#4012](https://github.com/NVIDIA/cuml/pull/4012)) [@trxcllnt](https://github.com/trxcllnt)
- Skipping hdbscan pytests when gpu is a100 ([#4007](https://github.com/NVIDIA/cuml/pull/4007)) [@cjnolet](https://github.com/cjnolet)
- Using 64-bit array lengths to increase scale of pca &amp; tsvd ([#3983](https://github.com/NVIDIA/cuml/pull/3983)) [@cjnolet](https://github.com/cjnolet)
- Fix MNMG test in Dask RF ([#3964](https://github.com/NVIDIA/cuml/pull/3964)) [@hcho3](https://github.com/hcho3)
- Use nested include in destination of install headers to avoid docker permission issues ([#3962](https://github.com/NVIDIA/cuml/pull/3962)) [@dantegd](https://github.com/dantegd)
- Fix automerge #3939 ([#3952](https://github.com/NVIDIA/cuml/pull/3952)) [@dantegd](https://github.com/dantegd)
- Update UCX-Py version to 0.21 ([#3950](https://github.com/NVIDIA/cuml/pull/3950)) [@pentschev](https://github.com/pentschev)
- Fix kernel and line info in cmake ([#3941](https://github.com/NVIDIA/cuml/pull/3941)) [@dantegd](https://github.com/dantegd)
- Fix for multi GPU PCA compute failing bug after transform and added error handling when n_components is not passed ([#3912](https://github.com/NVIDIA/cuml/pull/3912)) [@akaanirban](https://github.com/akaanirban)
- Tolerate QN linesearch failures when it&#39;s harmless ([#3791](https://github.com/NVIDIA/cuml/pull/3791)) [@achirkin](https://github.com/achirkin)

## 📖 Documentation

- Improve docstrings for silhouette score metrics. ([#4026](https://github.com/NVIDIA/cuml/pull/4026)) [@bdice](https://github.com/bdice)
- Update CHANGELOG.md link ([#3956](https://github.com/NVIDIA/cuml/pull/3956)) [@Salonijain27](https://github.com/Salonijain27)
- Update documentation build examples to be generator agnostic ([#3909](https://github.com/NVIDIA/cuml/pull/3909)) [@robertmaynard](https://github.com/robertmaynard)
- Improve FIL code readability and documentation ([#3056](https://github.com/NVIDIA/cuml/pull/3056)) [@levsnv](https://github.com/levsnv)

## 🚀 New Features

- Add Multinomial and Bernoulli Naive Bayes variants ([#4053](https://github.com/NVIDIA/cuml/pull/4053)) [@lowener](https://github.com/lowener)
- Add weighted K-Means sampling for SHAP ([#4051](https://github.com/NVIDIA/cuml/pull/4051)) [@Nanthini10](https://github.com/Nanthini10)
- Use chebyshev, canberra, hellinger and minkowski distance metrics ([#3990](https://github.com/NVIDIA/cuml/pull/3990)) [@mdoijade](https://github.com/mdoijade)
- Implement vector leaf prediction for fil. ([#3917](https://github.com/NVIDIA/cuml/pull/3917)) [@RAMitchell](https://github.com/RAMitchell)
- change TargetEncoder&#39;s smooth argument from ratio to count ([#3876](https://github.com/NVIDIA/cuml/pull/3876)) [@daxiongshu](https://github.com/daxiongshu)
- Enable warp-per-tree inference in FIL for regression and binary classification ([#3760](https://github.com/NVIDIA/cuml/pull/3760)) [@levsnv](https://github.com/levsnv)

## 🛠️ Improvements

- Remove clang/clang-tools from conda recipe ([#4109](https://github.com/NVIDIA/cuml/pull/4109)) [@dantegd](https://github.com/dantegd)
- Pin dask version ([#4108](https://github.com/NVIDIA/cuml/pull/4108)) [@galipremsagar](https://github.com/galipremsagar)
- ANN warnings/tests updates ([#4101](https://github.com/NVIDIA/cuml/pull/4101)) [@viclafargue](https://github.com/viclafargue)
- Removing local memory operations from computeSplitKernel and other optimizations ([#4083](https://github.com/NVIDIA/cuml/pull/4083)) [@vinaydes](https://github.com/vinaydes)
- Fix libfaiss dependency to not expressly depend on conda-forge ([#4082](https://github.com/NVIDIA/cuml/pull/4082)) [@Ethyling](https://github.com/Ethyling)
- Remove deprecated target_weights in UMAP ([#4081](https://github.com/NVIDIA/cuml/pull/4081)) [@lowener](https://github.com/lowener)
- Upgrade Treelite to 2.0.0 ([#4072](https://github.com/NVIDIA/cuml/pull/4072)) [@hcho3](https://github.com/hcho3)
- Optimize dtype conversion for FIL ([#4070](https://github.com/NVIDIA/cuml/pull/4070)) [@dantegd](https://github.com/dantegd)
- Adding quick notes to HDBSCAN public API docs as to why discrepancies may occur between cpu and gpu impls. ([#4061](https://github.com/NVIDIA/cuml/pull/4061)) [@cjnolet](https://github.com/cjnolet)
- Update `conda` environment name for CI ([#4039](https://github.com/NVIDIA/cuml/pull/4039)) [@ajschmidt8](https://github.com/ajschmidt8)
- Rewrite random forest gtests ([#4038](https://github.com/NVIDIA/cuml/pull/4038)) [@RAMitchell](https://github.com/RAMitchell)
- Updating Clang Version to 11.0.0 ([#4029](https://github.com/NVIDIA/cuml/pull/4029)) [@codereport](https://github.com/codereport)
- Raise ARIMA parameter limits from 4 to 8 ([#4022](https://github.com/NVIDIA/cuml/pull/4022)) [@Nyrio](https://github.com/Nyrio)
- Testing extract clusters in HDBSCAN ([#4009](https://github.com/NVIDIA/cuml/pull/4009)) [@divyegala](https://github.com/divyegala)
- ARIMA - Kalman loop rewrite: single megakernel instead of host loop ([#4006](https://github.com/NVIDIA/cuml/pull/4006)) [@Nyrio](https://github.com/Nyrio)
- RF/DT cleanup ([#4005](https://github.com/NVIDIA/cuml/pull/4005)) [@venkywonka](https://github.com/venkywonka)
- Exposing condensed hierarchy through cython for easier unit-level testing ([#4004](https://github.com/NVIDIA/cuml/pull/4004)) [@cjnolet](https://github.com/cjnolet)
- Use the 21.08 branch of rapids-cmake as rmm requires it ([#4002](https://github.com/NVIDIA/cuml/pull/4002)) [@robertmaynard](https://github.com/robertmaynard)
- RF: memset and batch size optimization for computing splits ([#4001](https://github.com/NVIDIA/cuml/pull/4001)) [@venkywonka](https://github.com/venkywonka)
- Reducing cluster size to number of selected clusters. Returning stability scores ([#3987](https://github.com/NVIDIA/cuml/pull/3987)) [@cjnolet](https://github.com/cjnolet)
- HDBSCAN: Lazy-loading (and caching) condensed &amp; single-linkage tree objects ([#3986](https://github.com/NVIDIA/cuml/pull/3986)) [@cjnolet](https://github.com/cjnolet)
- Fix `21.08` forward-merge conflicts ([#3982](https://github.com/NVIDIA/cuml/pull/3982)) [@ajschmidt8](https://github.com/ajschmidt8)
- Update Dask/Distributed version ([#3978](https://github.com/NVIDIA/cuml/pull/3978)) [@pentschev](https://github.com/pentschev)
- Use clang-tools on x86 only ([#3969](https://github.com/NVIDIA/cuml/pull/3969)) [@jakirkham](https://github.com/jakirkham)
- Promote `trustworthiness_score` to public header, add missing includes, update dependencies ([#3968](https://github.com/NVIDIA/cuml/pull/3968)) [@trxcllnt](https://github.com/trxcllnt)
- Moving FAISS ANN wrapper to raft ([#3963](https://github.com/NVIDIA/cuml/pull/3963)) [@cjnolet](https://github.com/cjnolet)
- Add MG weighted k-means ([#3959](https://github.com/NVIDIA/cuml/pull/3959)) [@lowener](https://github.com/lowener)
- Remove unused code in UMAP. ([#3931](https://github.com/NVIDIA/cuml/pull/3931)) [@trivialfis](https://github.com/trivialfis)
- Fix automerge #3900 and correct package versions in meta packages ([#3918](https://github.com/NVIDIA/cuml/pull/3918)) [@dantegd](https://github.com/dantegd)
- Adaptive stress tests when GPU memory capacity is insufficient ([#3916](https://github.com/NVIDIA/cuml/pull/3916)) [@lowener](https://github.com/lowener)
- Fix merge conflicts ([#3892](https://github.com/NVIDIA/cuml/pull/3892)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove old RF backend ([#3868](https://github.com/NVIDIA/cuml/pull/3868)) [@RAMitchell](https://github.com/RAMitchell)
- Refactor to extract random forest objectives ([#3854](https://github.com/NVIDIA/cuml/pull/3854)) [@RAMitchell](https://github.com/RAMitchell)

# cuML 21.06.00 (9 Jun 2021)

## 🚨 Breaking Changes

- Remove Base.enable_rmm_pool method as it is no longer needed ([#3875](https://github.com/NVIDIA/cuml/pull/3875)) [@teju85](https://github.com/teju85)
- RF: Make experimental-backend default for regression tasks and deprecate old-backend. ([#3872](https://github.com/NVIDIA/cuml/pull/3872)) [@venkywonka](https://github.com/venkywonka)
- Deterministic UMAP with floating point rounding. ([#3848](https://github.com/NVIDIA/cuml/pull/3848)) [@trivialfis](https://github.com/trivialfis)
- Fix RF regression performance ([#3845](https://github.com/NVIDIA/cuml/pull/3845)) [@RAMitchell](https://github.com/RAMitchell)
- Add feature to print forest shape in FIL upon importing ([#3763](https://github.com/NVIDIA/cuml/pull/3763)) [@levsnv](https://github.com/levsnv)
- Remove &#39;seed&#39; and &#39;output_type&#39; deprecated features ([#3739](https://github.com/NVIDIA/cuml/pull/3739)) [@lowener](https://github.com/lowener)

## 🐛 Bug Fixes

- Disable UMAP deterministic test on CTK11.2 ([#3942](https://github.com/NVIDIA/cuml/pull/3942)) [@trivialfis](https://github.com/trivialfis)
- Revert #3869 ([#3933](https://github.com/NVIDIA/cuml/pull/3933)) [@hcho3](https://github.com/hcho3)
- RF: fix the bug in `pdf_to_cdf` device function that causes hang when `n_bins &gt; TPB &amp;&amp; n_bins % TPB != 0` ([#3921](https://github.com/NVIDIA/cuml/pull/3921)) [@venkywonka](https://github.com/venkywonka)
- Fix number of permutations in pytest and getting handle for cuml models ([#3920](https://github.com/NVIDIA/cuml/pull/3920)) [@dantegd](https://github.com/dantegd)
- Fix typo in umap `target_weight` parameter ([#3914](https://github.com/NVIDIA/cuml/pull/3914)) [@lowener](https://github.com/lowener)
- correct compliation of cuml c library ([#3908](https://github.com/NVIDIA/cuml/pull/3908)) [@robertmaynard](https://github.com/robertmaynard)
- Correct install path for include folder to avoid double nesting ([#3901](https://github.com/NVIDIA/cuml/pull/3901)) [@dantegd](https://github.com/dantegd)
- Add type check for y in train_test_split ([#3886](https://github.com/NVIDIA/cuml/pull/3886)) [@Nanthini10](https://github.com/Nanthini10)
- Fix for MNMG test_rf_classification_dask_fil_predict_proba ([#3831](https://github.com/NVIDIA/cuml/pull/3831)) [@lowener](https://github.com/lowener)
- Fix MNMG test test_rf_regression_dask_fil ([#3830](https://github.com/NVIDIA/cuml/pull/3830)) [@hcho3](https://github.com/hcho3)
- AgglomerativeClustering support single cluster and ignore only zero distances from self-loops ([#3824](https://github.com/NVIDIA/cuml/pull/3824)) [@cjnolet](https://github.com/cjnolet)

## 📖 Documentation

- Small doc fixes for 21.06 release ([#3936](https://github.com/NVIDIA/cuml/pull/3936)) [@dantegd](https://github.com/dantegd)
- Document ability to export cuML RF to predict on other machines ([#3890](https://github.com/NVIDIA/cuml/pull/3890)) [@hcho3](https://github.com/hcho3)

## 🚀 New Features

- Deterministic UMAP with floating point rounding. ([#3848](https://github.com/NVIDIA/cuml/pull/3848)) [@trivialfis](https://github.com/trivialfis)
- HDBSCAN ([#3821](https://github.com/NVIDIA/cuml/pull/3821)) [@cjnolet](https://github.com/cjnolet)
- Add feature to print forest shape in FIL upon importing ([#3763](https://github.com/NVIDIA/cuml/pull/3763)) [@levsnv](https://github.com/levsnv)

## 🛠️ Improvements

- Pin dask ot 2021.5.1 for 21.06 release ([#3937](https://github.com/NVIDIA/cuml/pull/3937)) [@dantegd](https://github.com/dantegd)
- Upgrade xgboost to 1.4.2 ([#3925](https://github.com/NVIDIA/cuml/pull/3925)) [@dantegd](https://github.com/dantegd)
- Use UCX-Py 0.20 ([#3911](https://github.com/NVIDIA/cuml/pull/3911)) [@jakirkham](https://github.com/jakirkham)
- Upgrade NCCL to 2.9.9 ([#3902](https://github.com/NVIDIA/cuml/pull/3902)) [@dantegd](https://github.com/dantegd)
- Update conda developer environments ([#3898](https://github.com/NVIDIA/cuml/pull/3898)) [@viclafargue](https://github.com/viclafargue)
- ARIMA: pre-allocation of temporary memory to reduce latencies ([#3895](https://github.com/NVIDIA/cuml/pull/3895)) [@Nyrio](https://github.com/Nyrio)
- Condense TSNE parameters into a struct ([#3884](https://github.com/NVIDIA/cuml/pull/3884)) [@lowener](https://github.com/lowener)
- Update `CHANGELOG.md` links for calver ([#3883](https://github.com/NVIDIA/cuml/pull/3883)) [@ajschmidt8](https://github.com/ajschmidt8)
- Make sure `__init__` is called in graph callback. ([#3881](https://github.com/NVIDIA/cuml/pull/3881)) [@trivialfis](https://github.com/trivialfis)
- Update docs build script ([#3877](https://github.com/NVIDIA/cuml/pull/3877)) [@ajschmidt8](https://github.com/ajschmidt8)
- Remove Base.enable_rmm_pool method as it is no longer needed ([#3875](https://github.com/NVIDIA/cuml/pull/3875)) [@teju85](https://github.com/teju85)
- RF: Make experimental-backend default for regression tasks and deprecate old-backend. ([#3872](https://github.com/NVIDIA/cuml/pull/3872)) [@venkywonka](https://github.com/venkywonka)
- Enable probability output from RF binary classifier (alternative implementaton) ([#3869](https://github.com/NVIDIA/cuml/pull/3869)) [@hcho3](https://github.com/hcho3)
- CI test speed improvement ([#3851](https://github.com/NVIDIA/cuml/pull/3851)) [@lowener](https://github.com/lowener)
- Fix RF regression performance ([#3845](https://github.com/NVIDIA/cuml/pull/3845)) [@RAMitchell](https://github.com/RAMitchell)
- Update to CMake 3.20 features, `rapids-cmake` and `CPM` ([#3844](https://github.com/NVIDIA/cuml/pull/3844)) [@dantegd](https://github.com/dantegd)
- Support sparse input features in QN solvers and Logistic Regression ([#3827](https://github.com/NVIDIA/cuml/pull/3827)) [@achirkin](https://github.com/achirkin)
- Trustworthiness score improvements ([#3826](https://github.com/NVIDIA/cuml/pull/3826)) [@viclafargue](https://github.com/viclafargue)
- Performance optimization of RF split kernels by removing empty cycles ([#3818](https://github.com/NVIDIA/cuml/pull/3818)) [@vinaydes](https://github.com/vinaydes)
- Correct deprecate positional args decorator for CalVer ([#3784](https://github.com/NVIDIA/cuml/pull/3784)) [@lowener](https://github.com/lowener)
- ColumnTransformer &amp; FunctionTransformer ([#3745](https://github.com/NVIDIA/cuml/pull/3745)) [@viclafargue](https://github.com/viclafargue)
- Remove &#39;seed&#39; and &#39;output_type&#39; deprecated features ([#3739](https://github.com/NVIDIA/cuml/pull/3739)) [@lowener](https://github.com/lowener)

# cuML 0.19.0 (21 Apr 2021)

## 🚨 Breaking Changes

- Use the new RF backend by default for classification ([#3686](https://github.com/NVIDIA/cuml/pull/3686)) [@hcho3](https://github.com/hcho3)
- Deprecating quantile-per-tree and removing three previously deprecated Random Forest parameters ([#3667](https://github.com/NVIDIA/cuml/pull/3667)) [@vinaydes](https://github.com/vinaydes)
- Update predict() / predict_proba() of RF to match sklearn ([#3609](https://github.com/NVIDIA/cuml/pull/3609)) [@hcho3](https://github.com/hcho3)
- Upgrade FAISS to 1.7.x ([#3509](https://github.com/NVIDIA/cuml/pull/3509)) [@viclafargue](https://github.com/viclafargue)
- cuML&#39;s estimator Base class for preprocessing models ([#3270](https://github.com/NVIDIA/cuml/pull/3270)) [@viclafargue](https://github.com/viclafargue)

## 🐛 Bug Fixes

- Fix brute force KNN distance metric issue ([#3755](https://github.com/NVIDIA/cuml/pull/3755)) [@viclafargue](https://github.com/viclafargue)
- Fix min_max_axis ([#3735](https://github.com/NVIDIA/cuml/pull/3735)) [@viclafargue](https://github.com/viclafargue)
- Fix NaN errors observed with ARIMA in CUDA 11.2 builds ([#3730](https://github.com/NVIDIA/cuml/pull/3730)) [@Nyrio](https://github.com/Nyrio)
- Fix random state generator ([#3716](https://github.com/NVIDIA/cuml/pull/3716)) [@viclafargue](https://github.com/viclafargue)
- Fixes the out of memory access issue for computeSplit kernels ([#3715](https://github.com/NVIDIA/cuml/pull/3715)) [@vinaydes](https://github.com/vinaydes)
- Fixing umap gtest failure under cuda 11.2. ([#3696](https://github.com/NVIDIA/cuml/pull/3696)) [@cjnolet](https://github.com/cjnolet)
- Fix irreproducibility issue in RF classification ([#3693](https://github.com/NVIDIA/cuml/pull/3693)) [@vinaydes](https://github.com/vinaydes)
- BUG fix BatchedLevelAlgo DtClsTest &amp; DtRegTest failing tests ([#3690](https://github.com/NVIDIA/cuml/pull/3690)) [@venkywonka](https://github.com/venkywonka)
- Restore the functionality of RF score() ([#3685](https://github.com/NVIDIA/cuml/pull/3685)) [@hcho3](https://github.com/hcho3)
- Use main build.sh to build docs in docs CI ([#3681](https://github.com/NVIDIA/cuml/pull/3681)) [@dantegd](https://github.com/dantegd)
- Revert &quot;Update conda recipes pinning of repo dependencies&quot; ([#3680](https://github.com/NVIDIA/cuml/pull/3680)) [@raydouglass](https://github.com/raydouglass)
- Skip tests that fail on CUDA 11.2 ([#3679](https://github.com/NVIDIA/cuml/pull/3679)) [@dantegd](https://github.com/dantegd)
- Dask KNN Cl&amp;Re 1D labels ([#3668](https://github.com/NVIDIA/cuml/pull/3668)) [@viclafargue](https://github.com/viclafargue)
- Update conda recipes pinning of repo dependencies ([#3666](https://github.com/NVIDIA/cuml/pull/3666)) [@mike-wendt](https://github.com/mike-wendt)
- OOB access in GLM SoftMax ([#3642](https://github.com/NVIDIA/cuml/pull/3642)) [@divyegala](https://github.com/divyegala)
- SilhouetteScore C++ tests seed ([#3640](https://github.com/NVIDIA/cuml/pull/3640)) [@divyegala](https://github.com/divyegala)
- SimpleImputer fix ([#3624](https://github.com/NVIDIA/cuml/pull/3624)) [@viclafargue](https://github.com/viclafargue)
- Silhouette Score `make_monotonic` for non-monotonic label set ([#3619](https://github.com/NVIDIA/cuml/pull/3619)) [@divyegala](https://github.com/divyegala)
- Fixing support for empty rows in sparse Jaccard / Cosine ([#3612](https://github.com/NVIDIA/cuml/pull/3612)) [@cjnolet](https://github.com/cjnolet)
- Fix train_test_split with stratify option ([#3611](https://github.com/NVIDIA/cuml/pull/3611)) [@Nanthini10](https://github.com/Nanthini10)
- Update predict() / predict_proba() of RF to match sklearn ([#3609](https://github.com/NVIDIA/cuml/pull/3609)) [@hcho3](https://github.com/hcho3)
- Change dask and distributed branch to main ([#3593](https://github.com/NVIDIA/cuml/pull/3593)) [@dantegd](https://github.com/dantegd)
- Fixes memory allocation for experimental backend and improves quantile computations ([#3586](https://github.com/NVIDIA/cuml/pull/3586)) [@vinaydes](https://github.com/vinaydes)
- Add ucx-proc package back that got lost during an auto merge conflict ([#3550](https://github.com/NVIDIA/cuml/pull/3550)) [@dantegd](https://github.com/dantegd)
- Fix failing Hellinger gtest ([#3549](https://github.com/NVIDIA/cuml/pull/3549)) [@cjnolet](https://github.com/cjnolet)
- Directly invoke make for non-CMake docs target ([#3534](https://github.com/NVIDIA/cuml/pull/3534)) [@wphicks](https://github.com/wphicks)
- Fix Codecov.io Coverage Upload for Branch Builds ([#3524](https://github.com/NVIDIA/cuml/pull/3524)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Ensure global_output_type is thread-safe ([#3497](https://github.com/NVIDIA/cuml/pull/3497)) [@wphicks](https://github.com/wphicks)
- List as input for SimpleImputer ([#3489](https://github.com/NVIDIA/cuml/pull/3489)) [@viclafargue](https://github.com/viclafargue)

## 📖 Documentation

- Add sparse docstring comments ([#3712](https://github.com/NVIDIA/cuml/pull/3712)) [@JohnZed](https://github.com/JohnZed)
- FIL and Dask demo ([#3698](https://github.com/NVIDIA/cuml/pull/3698)) [@miroenev](https://github.com/miroenev)
- Deprecating quantile-per-tree and removing three previously deprecated Random Forest parameters ([#3667](https://github.com/NVIDIA/cuml/pull/3667)) [@vinaydes](https://github.com/vinaydes)
- Fixing Indentation for Docstring Generators ([#3650](https://github.com/NVIDIA/cuml/pull/3650)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Update doc to indicate ExtraTree support ([#3635](https://github.com/NVIDIA/cuml/pull/3635)) [@hcho3](https://github.com/hcho3)
- Update doc, now that FIL supports multi-class classification ([#3634](https://github.com/NVIDIA/cuml/pull/3634)) [@hcho3](https://github.com/hcho3)
- Document model_type=&#39;xgboost_json&#39; in FIL ([#3633](https://github.com/NVIDIA/cuml/pull/3633)) [@hcho3](https://github.com/hcho3)
- Including log loss metric to the documentation website ([#3617](https://github.com/NVIDIA/cuml/pull/3617)) [@lowener](https://github.com/lowener)
- Update the build doc regarding the use of GCC 7.5 ([#3605](https://github.com/NVIDIA/cuml/pull/3605)) [@hcho3](https://github.com/hcho3)
- Update One-Hot Encoder doc ([#3600](https://github.com/NVIDIA/cuml/pull/3600)) [@lowener](https://github.com/lowener)
- Fix documentation of KMeans ([#3595](https://github.com/NVIDIA/cuml/pull/3595)) [@lowener](https://github.com/lowener)

## 🚀 New Features

- Reduce the size of the cuml libraries ([#3702](https://github.com/NVIDIA/cuml/pull/3702)) [@robertmaynard](https://github.com/robertmaynard)
- Use ninja as default CMake generator ([#3664](https://github.com/NVIDIA/cuml/pull/3664)) [@wphicks](https://github.com/wphicks)
- Single-Linkage Hierarchical Clustering Python Wrapper ([#3631](https://github.com/NVIDIA/cuml/pull/3631)) [@cjnolet](https://github.com/cjnolet)
- Support for precomputed distance matrix in DBSCAN ([#3585](https://github.com/NVIDIA/cuml/pull/3585)) [@Nyrio](https://github.com/Nyrio)
- Adding haversine to brute force knn ([#3579](https://github.com/NVIDIA/cuml/pull/3579)) [@cjnolet](https://github.com/cjnolet)
- Support for sample_weight parameter in LogisticRegression ([#3572](https://github.com/NVIDIA/cuml/pull/3572)) [@viclafargue](https://github.com/viclafargue)
- Provide &quot;--ccache&quot; flag for build.sh ([#3566](https://github.com/NVIDIA/cuml/pull/3566)) [@wphicks](https://github.com/wphicks)
- Eliminate unnecessary includes discovered by cppclean ([#3564](https://github.com/NVIDIA/cuml/pull/3564)) [@wphicks](https://github.com/wphicks)
- Single-linkage Hierarchical Clustering C++ ([#3545](https://github.com/NVIDIA/cuml/pull/3545)) [@cjnolet](https://github.com/cjnolet)
- Expose sparse distances via semiring to Python API ([#3516](https://github.com/NVIDIA/cuml/pull/3516)) [@lowener](https://github.com/lowener)
- Use cmake --build in build.sh to facilitate switching build tools ([#3487](https://github.com/NVIDIA/cuml/pull/3487)) [@wphicks](https://github.com/wphicks)
- Add cython hinge_loss ([#3409](https://github.com/NVIDIA/cuml/pull/3409)) [@Nanthini10](https://github.com/Nanthini10)
- Adding CodeCov Info for Dask Tests ([#3338](https://github.com/NVIDIA/cuml/pull/3338)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Add predict_proba() to XGBoost-style models in FIL C++ ([#2894](https://github.com/NVIDIA/cuml/pull/2894)) [@levsnv](https://github.com/levsnv)

## 🛠️ Improvements

- Updating docs, readme, and umap param tests for 0.19 ([#3731](https://github.com/NVIDIA/cuml/pull/3731)) [@cjnolet](https://github.com/cjnolet)
- Locking RAFT hash for 0.19 ([#3721](https://github.com/NVIDIA/cuml/pull/3721)) [@cjnolet](https://github.com/cjnolet)
- Upgrade to Treelite 1.1.0 ([#3708](https://github.com/NVIDIA/cuml/pull/3708)) [@hcho3](https://github.com/hcho3)
- Update to XGBoost 1.4.0rc1 ([#3699](https://github.com/NVIDIA/cuml/pull/3699)) [@hcho3](https://github.com/hcho3)
- Use the new RF backend by default for classification ([#3686](https://github.com/NVIDIA/cuml/pull/3686)) [@hcho3](https://github.com/hcho3)
- Update LogisticRegression documentation ([#3677](https://github.com/NVIDIA/cuml/pull/3677)) [@viclafargue](https://github.com/viclafargue)
- Preprocessing out of experimental ([#3676](https://github.com/NVIDIA/cuml/pull/3676)) [@viclafargue](https://github.com/viclafargue)
- ENH Decision Tree new backend `computeSplit*Kernel` histogram calculation optimization ([#3674](https://github.com/NVIDIA/cuml/pull/3674)) [@venkywonka](https://github.com/venkywonka)
- Remove `check_cupy8` ([#3669](https://github.com/NVIDIA/cuml/pull/3669)) [@viclafargue](https://github.com/viclafargue)
- Use custom conda build directory for ccache integration ([#3658](https://github.com/NVIDIA/cuml/pull/3658)) [@dillon-cullinan](https://github.com/dillon-cullinan)
- Disable three flaky tests ([#3657](https://github.com/NVIDIA/cuml/pull/3657)) [@hcho3](https://github.com/hcho3)
- CUDA 11.2 developer environment ([#3648](https://github.com/NVIDIA/cuml/pull/3648)) [@dantegd](https://github.com/dantegd)
- Store data frequencies in tree nodes of RF ([#3647](https://github.com/NVIDIA/cuml/pull/3647)) [@hcho3](https://github.com/hcho3)
- Row major Gram matrices ([#3639](https://github.com/NVIDIA/cuml/pull/3639)) [@tfeher](https://github.com/tfeher)
- Converting all Estimator Constructors to Keyword Arguments ([#3636](https://github.com/NVIDIA/cuml/pull/3636)) [@mdemoret-nv](https://github.com/mdemoret-nv)
- Adding make_pipeline + test score with pipeline ([#3632](https://github.com/NVIDIA/cuml/pull/3632)) [@viclafargue](https://github.com/viclafargue)
- ENH Decision Tree new backend `computeSplitClassificationKernel` histogram calculation and occupancy optimization ([#3616](https://github.com/NVIDIA/cuml/pull/3616)) [@venkywonka](https://github.com/venkywonka)
- Revert &quot;ENH Fix stale GHA and prevent duplicates &quot; ([#3614](https://github.com/NVIDIA/cuml/pull/3614)) [@mike-wendt](https://github.com/mike-wendt)
- ENH Fix stale GHA and prevent duplicates ([#3613](https://github.com/NVIDIA/cuml/pull/3613)) [@mike-wendt](https://github.com/mike-wendt)
- KNN from RAFT ([#3603](https://github.com/NVIDIA/cuml/pull/3603)) [@viclafargue](https://github.com/viclafargue)
- Update Changelog Link ([#3601](https://github.com/NVIDIA/cuml/pull/3601)) [@ajschmidt8](https://github.com/ajschmidt8)
- Move SHAP explainers out of experimental ([#3596](https://github.com/NVIDIA/cuml/pull/3596)) [@dantegd](https://github.com/dantegd)
- Fixing compatibility issue with CUDA array interface ([#3594](https://github.com/NVIDIA/cuml/pull/3594)) [@lowener](https://github.com/lowener)
- Remove cutlass usage in row major input for euclidean exp/unexp, cosine and L1 distance matrix ([#3589](https://github.com/NVIDIA/cuml/pull/3589)) [@mdoijade](https://github.com/mdoijade)
- Test FIL probabilities with absolute error thresholds in python ([#3582](https://github.com/NVIDIA/cuml/pull/3582)) [@levsnv](https://github.com/levsnv)
- Removing sparse prims and fused l2 nn prim from cuml ([#3578](https://github.com/NVIDIA/cuml/pull/3578)) [@cjnolet](https://github.com/cjnolet)
- Prepare Changelog for Automation ([#3570](https://github.com/NVIDIA/cuml/pull/3570)) [@ajschmidt8](https://github.com/ajschmidt8)
- Print debug message if SVM convergence is poor ([#3562](https://github.com/NVIDIA/cuml/pull/3562)) [@tfeher](https://github.com/tfeher)
- Fix merge conflicts in 3552 ([#3557](https://github.com/NVIDIA/cuml/pull/3557)) [@ajschmidt8](https://github.com/ajschmidt8)
- Additional distance metrics for ANN ([#3533](https://github.com/NVIDIA/cuml/pull/3533)) [@viclafargue](https://github.com/viclafargue)
- Improve warning message when QN solver reaches max_iter ([#3515](https://github.com/NVIDIA/cuml/pull/3515)) [@tfeher](https://github.com/tfeher)
- Fix merge conflicts in 3502 ([#3513](https://github.com/NVIDIA/cuml/pull/3513)) [@ajschmidt8](https://github.com/ajschmidt8)
- Upgrade FAISS to 1.7.x ([#3509](https://github.com/NVIDIA/cuml/pull/3509)) [@viclafargue](https://github.com/viclafargue)
- ENH Pass ccache variables to conda recipe &amp; use Ninja in CI ([#3508](https://github.com/NVIDIA/cuml/pull/3508)) [@Ethyling](https://github.com/Ethyling)
- Fix forward-merger conflicts in #3502 ([#3506](https://github.com/NVIDIA/cuml/pull/3506)) [@dantegd](https://github.com/dantegd)
- Sklearn meta-estimators into namespace ([#3493](https://github.com/NVIDIA/cuml/pull/3493)) [@viclafargue](https://github.com/viclafargue)
- Add flexibility to copyright checker ([#3466](https://github.com/NVIDIA/cuml/pull/3466)) [@lowener](https://github.com/lowener)
- Update sparse KNN to use rmm device buffer ([#3460](https://github.com/NVIDIA/cuml/pull/3460)) [@lowener](https://github.com/lowener)
- Fix forward-merger conflicts in #3444 ([#3455](https://github.com/NVIDIA/cuml/pull/3455)) [@ajschmidt8](https://github.com/ajschmidt8)
- Replace ML::MetricType with raft::distance::DistanceType ([#3389](https://github.com/NVIDIA/cuml/pull/3389)) [@lowener](https://github.com/lowener)
- RF param initialization cython and C++ layer cleanup ([#3358](https://github.com/NVIDIA/cuml/pull/3358)) [@venkywonka](https://github.com/venkywonka)
- MNMG RF broadcast feature ([#3349](https://github.com/NVIDIA/cuml/pull/3349)) [@viclafargue](https://github.com/viclafargue)
- cuML&#39;s estimator Base class for preprocessing models ([#3270](https://github.com/NVIDIA/cuml/pull/3270)) [@viclafargue](https://github.com/viclafargue)
- Make `_get_tags` a class/static method ([#3257](https://github.com/NVIDIA/cuml/pull/3257)) [@dantegd](https://github.com/dantegd)
- NVTX Markers for RF and RF-backend ([#3014](https://github.com/NVIDIA/cuml/pull/3014)) [@venkywonka](https://github.com/venkywonka)

# cuML 0.18.0 (24 Feb 2021)

## Breaking Changes 🚨

- cuml.experimental SHAP improvements (#3433) @dantegd
- Enable feature sampling for the experimental backend of Random Forest (#3364) @vinaydes
- re-enable cuML&#39;s copyright checker script (#3363) @teju85
- Batched Silhouette Score (#3362) @divyegala
- Update failing MNMG tests (#3348) @viclafargue
- Rename print_summary() of Dask RF to get_summary_text(); it now returns string to the client (#3341) @hcho3
- Rename dump_as_json() -&gt; get_json(); expose it from Dask RF (#3340) @hcho3
- MNMG KNN consolidation (#3307) @viclafargue
- Return confusion matrix as int unless float weights are used (#3275) @lowener
- Approximate Nearest Neighbors (#2780) @viclafargue

## Bug Fixes 🐛

- HOTFIX Add ucx-proc package back that got lost during an auto merge conflict (#3551) @dantegd
- Non project-flash CI ml test 18.04 issue debugging and bugfixing (#3495) @dantegd
- Temporarily xfail KBinsDiscretizer uniform tests (#3494) @wphicks
- Fix illegal memory accesses when NITEMS &gt; 1, and nrows % NITEMS != 0. (#3480) @canonizer
- Update call to dask client persist (#3474) @dantegd
- Adding warning for IVFPQ (#3472) @viclafargue
- Fix failing sparse NN test in CI by allowing small number of index discrepancies (#3454) @cjnolet
- Exempting thirdparty code from copyright checks (#3453) @lowener
- Relaxing Batched SilhouetteScore Test Constraint (#3452) @divyegala
- Mark kbinsdiscretizer quantile tests as xfail (#3450) @wphicks
- Fixing documentation on SimpleImputer (#3447) @lowener
- Skipping IVFPQ (#3429) @viclafargue
- Adding tol to dask test_kmeans (#3426) @lowener
- Fix memory bug for SVM with large n_rows (#3420) @tfeher
- Allow linear regression for  with CUDA &gt;=11.0 (#3417) @wphicks
- Fix vectorizer tests by restoring sort behavior in groupby (#3416) @JohnZed
- Ensure make_classification respects output type (#3415) @wphicks
- Clean Up `#include` Dependencies (#3402) @mdemoret-nv
- Fix Nearest Neighbor Stress Test (#3401) @lowener
- Fix array_equal in tests (#3400) @viclafargue
- Improving Copyright Check When Not Running in CI (#3398) @mdemoret-nv
- Also xfail zlib errors when downloading newsgroups data (#3393) @JohnZed
- Fix for ANN memory release bug (#3391) @viclafargue
- XFail Holt Winters test where statsmodels has known issues with gcc 9.3.0 (#3385) @JohnZed
- FIX Update cupy to &gt;= 7.8 and remove unused build.sh script (#3378) @dantegd
- re-enable cuML&#39;s copyright checker script (#3363) @teju85
- Update failing MNMG tests (#3348) @viclafargue
- Rename print_summary() of Dask RF to get_summary_text(); it now returns string to the client (#3341) @hcho3
- Fixing `make_blobs` to Respect the Global Output Type (#3339) @mdemoret-nv
- Fix permutation explainer (#3332) @RAMitchell
- k-means bug fix in debug build (#3321) @akkamesh
- Fix for default arguments of PCA (#3320) @lowener
- Provide workaround for cupy.percentile bug (#3315) @wphicks
- Fix SVR unit test parameter (#3294) @tfeher
- Add xfail on fetching 20newsgroup dataset (test_naive_bayes) (#3291) @lowener
- Remove unused keyword in PorterStemmer code (#3289) @wphicks
- Remove static specifier in DecisionTree unit test for C++14 compliance (#3281) @wphicks
- Correct pure virtual declaration in manifold_inputs_t (#3279) @wphicks

## Documentation 📖

- Correct import path in docs for experimental preprocessing features (#3488) @wphicks
- Minor doc updates for 0.18 (#3475) @JohnZed
- Improve Python Docs with Default Role (#3445) @mdemoret-nv
- Fixing Python Documentation Errors and Warnings (#3428) @mdemoret-nv
- Remove outdated references to changelog in CONTRIBUTING.md (#3328) @wphicks
- Adding highlighting to bibtex in readme (#3296) @cjnolet

## New Features 🚀

- Improve runtime performance of RF to Treelite conversion (#3410) @wphicks
- Parallelize Treelite to FIL conversion over trees (#3396) @wphicks
- Parallelize RF to Treelite conversion over trees (#3395) @wphicks
- Allow saving Dask RandomForest models immediately after training (fixes #3331) (#3388) @jameslamb
- genetic programming initial structures (#3387) @teju85
- MNMG DBSCAN (#3382) @Nyrio
- FIL to use L1 cache when input columns don&#39;t fit into shared memory (#3370) @levsnv
- Enable feature sampling for the experimental backend of Random Forest (#3364) @vinaydes
- Batched Silhouette Score (#3362) @divyegala
- Rename dump_as_json() -&gt; get_json(); expose it from Dask RF (#3340) @hcho3
- Exposing model_selection in a similar way to scikit-learn (#3329) @ptartan21
- Promote IncrementalPCA from experimental in 0.18 release (#3327) @lowener
- Create labeler.yml (#3324) @jolorunyomi
- Add slow high-precision mode to KNN (#3304) @wphicks
- Sparse TSNE (#3293) @divyegala
- Sparse Generalized SPMV (semiring) Primitive (#3146) @cjnolet
- Multiclass meta estimator wrappers and multiclass SVC (#3092) @tfeher
- Approximate Nearest Neighbors (#2780) @viclafargue
- Add KNN parameter to t-SNE (#2592) @aleksficek

## Improvements 🛠️

- Update stale GHA with exemptions &amp; new labels (#3507) @mike-wendt
- Add GHA to mark issues/prs as stale/rotten (#3500) @Ethyling
- Fix naive bayes inputs (#3448) @cjnolet
- Prepare Changelog for Automation (#3442) @ajschmidt8
- cuml.experimental SHAP improvements (#3433) @dantegd
- Speed up knn tests (#3411) @JohnZed
- Replacing sklearn functions with cuml in RF MNMG notebook (#3408) @lowener
- Auto-label PRs based on their content (#3407) @jolorunyomi
- Use stable 1.0.0 version of Treelite (#3394) @hcho3
- API update to match RAFT PR #120 (#3386) @drobison00
- Update linear models to use RMM memory allocation (#3365) @lowener
- Updating dense pairwise distance enum names (#3352) @cjnolet
- Upgrade Treelite module (#3316) @hcho3
- Removed FIL node types with `_t` suffix (#3314) @canonizer
- MNMG KNN consolidation (#3307) @viclafargue
- Updating PyTests to Stay Below 4 Gb Limit (#3306) @mdemoret-nv
- Refactoring: move internal FIL interface to a separate file (#3292) @canonizer
- Return confusion matrix as int unless float weights are used (#3275) @lowener
- 018 add unfitted error pca &amp; tests on IPCA (#3272) @lowener
- Linear models predict function consolidation (#3256) @dantegd
- Preparing sparse primitives for movement to RAFT (#3157) @cjnolet

# cuML 0.17.0 (10 Dec 2020)

## New Features
- PR #3164: Expose silhouette score in Python
- PR #3160: Least Angle Regression (experimental)
- PR #2659: Add initial max inner product sparse knn
- PR #3092: Multiclass meta estimator wrappers and multiclass SVC
- PR #2836: Refactor UMAP to accept sparse inputs
- PR #2894: predict_proba in FIL C++ for XGBoost-style multi-class models
- PR #3126: Experimental versions of GPU accelerated Kernel and Permutation SHAP

## Improvements
- PR #3077: Improve runtime for test_kmeans
- PR #3070: Speed up dask/test_datasets tests
- PR #3075: Speed up test_linear_model tests
- PR #3078: Speed up test_incremental_pca tests
- PR #2902: `matrix/matrix.cuh` in RAFT namespacing
- PR #2903: Moving linalg's gemm, gemv, transpose to RAFT namespaces
- PR #2905: `stats` prims `mean_center`, `sum` to RAFT namespaces
- PR #2904: Moving `linalg` basic math ops to RAFT namespaces
- PR #2956: Follow cuML array conventions in ARIMA and remove redundancy
- PR #3000: Pin cmake policies to cmake 3.17 version, bump project version to 0.17
- PR #3083: Improving test_make_blobs testing time
- PR #3223: Increase default SVM kernel cache to 2000 MiB
- PR #2906: Moving `linalg` decomp to RAFT namespaces
- PR #2988: FIL: use tree-per-class reduction for GROVE_PER_CLASS_FEW_CLASSES
- PR #2996: Removing the max_depth restriction for switching to the batched backend
- PR #3004: Remove Single Process Multi GPU (SPMG) code
- PR #3032: FIL: Add optimization parameter `blocks_per_sm` that will help all but tiniest models
- PR #3044: Move leftover `linalg` and `stats` to RAFT namespaces
- PR #3067: Deleting prims moved to RAFT and updating header paths
- PR #3074: Reducing dask coordinate descent test runtime
- PR #3096: Avoid memory transfers in CSR WeakCC for DBSCAN
- PR #3088: More readable and robust FIL C++ test management
- PR #3052: Speeding up MNMG KNN Cl&Re testing
- PR #3115: Speeding up MNMG UMAP testing
- PR #3112: Speed test_array
- PR #3111: Adding Cython to Code Coverage
- PR #3129: Update notebooks README
- PR #3002: Update flake8 Config To With Per File Settings
- PR #3135: Add QuasiNewton tests
- PR #3040: Improved Array Conversion with CumlArrayDescriptor and Decorators
- PR #3134: Improving the Deprecation Message Formatting in Documentation
- PR #3154: Adding estimator pickling demo notebooks (and docs)
- PR #3151: MNMG Logistic Regression via dask-glm
- PR #3113: Add tags and prefered memory order tags to estimators
- PR #3137: Reorganize Pytest Config and Add Quick Run Option
- PR #3144: Adding Ability to Set Arbitrary Cmake Flags in ./build.sh
- PR #3155: Eliminate unnecessary warnings from random projection test
- PR #3176: Add probabilistic SVM tests with various input array types
- PR #3180: FIL: `blocks_per_sm` support in Python
- PR #3186: Add gain to RF JSON dump
- PR #3219: Update CI to use XGBoost 1.3.0 RCs
- PR #3221: Update contributing doc for label support
- PR #3177: Make Multinomial Naive Bayes inherit from `ClassifierMixin` and use it for score
- PR #3241: Updating RAFT to latest
- PR #3240: Minor doc updates
- PR #3275: Return confusion matrix as int unless float weights are used

## Bug Fixes
- PR #3218: Specify dependency branches in conda dev environment to avoid pip resolver issue
- PR #3196: Disable ascending=false path for sortColumnsPerRow
- PR #3051: MNMG KNN Cl&Re fix + multiple improvements
- PR #3179: Remove unused metrics.cu file
- PR #3069: Prevent conversion of DataFrames to Series in preprocessing
- PR #3065: Refactoring prims metrics function names from camelcase to underscore format
- PR #3033: Splitting ml metrics to individual files
- PR #3072: Fusing metrics and score directories in src_prims
- PR #3037: Avoid logging deadlock in multi-threaded C code
- PR #2983: Fix seeding of KISS99 RNG
- PR #3011: Fix unused initialize_embeddings parameter in Barnes-Hut t-SNE
- PR #3008: Check number of columns in check_array validator
- PR #3012: Increasing learning rate for SGD log loss and invscaling pytests
- PR #2950: Fix includes in UMAP
- PR #3194: Fix cuDF to cuPy conversion (missing value)
- PR #3021: Fix a hang in cuML RF experimental backend
- PR #3039: Update RF and decision tree parameter initializations in benchmark codes
- PR #3060: Speed up test suite `test_fil`
- PR #3061: Handle C++ exception thrown from FIL predict
- PR #3073: Update mathjax CDN URL for documentation
- PR #3062: Bumping xgboost version to match cuml version
- PR #3084: Fix artifacts in t-SNE results
- PR #3086: Reverting FIL Notebook Testing
- PR #3192: Enable pipeline usage for OneHotEncoder and LabelEncoder
- PR #3114: Fixed a typo in SVC's predict_proba AttributeError
- PR #3117: Fix two crashes in experimental RF backend
- PR #3119: Fix memset args for benchmark
- PR #3130: Return Python string from `dump_as_json()` of RF
- PR #3132: Add `min_samples_split` + Rename `min_rows_per_node` -> `min_samples_leaf`
- PR #3136: Fix stochastic gradient descent example
- PR #3152: Fix access to attributes of individual NB objects in dask NB
- PR #3156: Force local conda artifact install
- PR #3162: Removing accidentally checked in debug file
- PR #3191: Fix __repr__ function for preprocessing models
- PR #3175: Fix gtest pinned cmake version for build from source option
- PR #3182: Fix a bug in MSE metric calculation
- PR #3187: Update docstring to document behavior of `bootstrap=False`
- PR #3215: Add a missing `__syncthreads()`
- PR #3246: Fix MNMG KNN doc (adding batch_size)
- PR #3185: Add documentation for Distributed TFIDF Transformer
- PR #3190: Fix Attribute error on ICPA #3183 and PCA input type
- PR #3208: Fix EXITCODE override in notebook test script
- PR #3250: Fixing label binarizer bug with multiple partitions
- PR #3214: Correct flaky silhouette score test by setting atol
- PR #3216: Ignore splits that do not satisfy constraints
- PR #3239: Fix intermittent dask random forest failure
- PR #3243: Avoid unnecessary split for degenerate case where all labels are identical
- PR #3245: Rename `rows_sample` -> `max_samples` to be consistent with sklearn's RF
- PR #3282: Add secondary test to kernel explainer pytests for stability in Volta

# cuML 0.16.0 (23 Oct 2020)

## New Features
- PR #2922: Install RAFT headers with cuML
- PR #2909: Update allgatherv for compatibility with latest RAFT
- PR #2677: Ability to export RF trees as JSON
- PR #2698: Distributed TF-IDF transformer
- PR #2476: Porter Stemmer
- PR #2789: Dask LabelEncoder
- PR #2152: add FIL C++ benchmark
- PR #2638: Improve cython build with custom `build_ext`
- PR #2866: Support XGBoost-style multiclass models (gradient boosted decision trees) in FIL C++
- PR #2874: Issue warning for degraded accuracy with float64 models in Treelite
- PR #2881: Introduces experimental batched backend for random forest
- PR #2916: Add SKLearn multi-class GBDT model support in FIL

## Improvements
- PR #2947: Add more warnings for accuracy degradation with 64-bit models
- PR #2873: Remove empty marker kernel code for NVTX markers
- PR #2796: Remove tokens of length 1 by default for text vectorizers
- PR #2741: Use rapids build packages in conda environments
- PR #2735: Update seed to random_state in random forest and associated tests
- PR #2739: Use cusparse_wrappers.h from RAFT
- PR #2729: Replace `cupy.sparse` with `cupyx.scipy.sparse`
- PR #2749: Correct docs for python version used in cuml_dev conda environment
- PR #2747: Adopting raft::handle_t and raft::comms::comms_t in cuML
- PR #2762: Fix broken links and provide minor edits to docs
- PR #2723: Support and enable convert_dtype in estimator predict
- PR #2758: Match sklearn's default n_components behavior for PCA
- PR #2770: Fix doxygen version during cmake
- PR #2766: Update default RandomForestRegressor score function to use r2
- PR #2775: Enablinbg mg gtests w/ raft mpi comms
- PR #2783: Add pytest that will fail when GPU IDs in Dask cluster are not unique
- PR #2784: Add SparseCumlArray container for sparse index/data arrays
- PR #2785: Add in cuML-specific dev conda dependencies
- PR #2778: Add README for FIL
- PR #2799: Reenable lightgbm test with lower (1%) proba accuracy
- PR #2800: Align cuML's spdlog version with RMM's
- PR #2824: Make data conversions warnings be debug level
- PR #2835: Rng prims, utils, and dependencies in RAFT
- PR #2541: Improve Documentation Examples and Source Linking
- PR #2837: Make the FIL node reorder loop more obvious
- PR #2849: make num_classes significant in FLOAT_SCALAR case
- PR #2792: Project flash (new build process) script changes
- PR #2850: Clean up unused params in paramsPCA
- PR #2871: Add timing function to utils
- PR #2863: in FIL, rename leaf_value_t enums to more descriptive
- PR #2867: improve stability of FIL benchmark measurements
- PR #2798: Add python tests for FIL multiclass classification of lightgbm models
- PR #2892: Update ci/local/README.md
- PR #2910: Adding Support for CuPy 8.x
- PR #2914: Add tests for XGBoost multi-class models in FIL
- PR #2622: Simplify tSNE perplexity search
- PR #2930: Pin libfaiss to <=1.6.3
- PR #2928: Updating Estimators Derived from Base for Consistency
- PR #2942: Adding `cuml.experimental` to the Docs
- PR #3010: Improve gpuCI Scripts
- PR #3141: Move DistanceType enum to RAFT

## Bug Fixes
- PR #2973: Allow data imputation for nan values
- PR #2982: Adjust kneighbors classifier test threshold to avoid intermittent failure
- PR #2885: Changing test target for NVTX wrapper test
- PR #2882: Allow import on machines without GPUs
- PR #2875: Bug fix to enable colorful NVTX markers
- PR #2744: Supporting larger number of classes in KNeighborsClassifier
- PR #2769: Remove outdated doxygen options for 1.8.20
- PR #2787: Skip lightgbm test for version 3 and above temporarily
- PR #2805: Retain index in stratified splitting for dataframes
- PR #2781: Use Python print to correctly redirect spdlogs when sys.stdout is changed
- PR #2787: Skip lightgbm test for version 3 and above temporarily
- PR #2813: Fix memory access in generation of non-row-major random blobs
- PR #2810: Update Rf MNMG threshold to prevent sporadic test failure
- PR #2808: Relax Doxygen version required in CMake to coincide with integration repo
- PR #2818: Fix parsing of singlegpu option in build command
- PR #2827: Force use of whole dataset when sample bootstrapping is disabled
- PR #2829: Fixing description for labels in docs and removing row number constraint from PCA xform/inverse_xform
- PR #2832: Updating stress tests that fail with OOM
- PR #2831: Removing repeated capture and parameter in lambda function
- PR #2847: Workaround for TSNE lockup, change caching preference.
- PR #2842: KNN index preprocessors were using incorrect n_samples
- PR #2848: Fix typo in Python docstring for UMAP
- PR #2856: Fix LabelEncoder for filtered input
- PR #2855: Updates for RMM being header only
- PR #2844: Fix for OPG KNN Classifier & Regressor
- PR #2880: Fix bugs in Auto-ARIMA when s==None
- PR #2877: TSNE exception for n_components > 2
- PR #2879: Update unit test for LabelEncoder on filtered input
- PR #2932: Marking KBinsDiscretizer pytests as xfail
- PR #2925: Fixing Owner Bug When Slicing CumlArray Objects
- PR #2931: Fix notebook error handling in gpuCI
- PR #2941: Fixing dask tsvd stress test failure
- PR #2943: Remove unused shuffle_features parameter
- PR #2940: Correcting labels meta dtype for `cuml.dask.make_classification`
- PR #2965: Notebooks update
- PR #2955: Fix for conftest for singlegpu build
- PR #2968: Remove shuffle_features from RF param names
- PR #2957: Fix ols test size for stability
- PR #2972: Upgrade Treelite to 0.93
- PR #2981: Prevent unguarded import of sklearn in SVC
- PR #2984: Fix GPU test scripts gcov error
- PR #2990: Reduce MNMG kneighbors regressor test threshold
- PR #2997: Changing ARIMA `get/set_params` to `get/set_fit_params`

# cuML 0.15.0 (26 Aug 2020)

## New Features
- PR #2581: Added model persistence via joblib in each section of estimator_intro.ipynb
- PR #2554: Hashing Vectorizer and general vectorizer improvements
- PR #2240: Making Dask models pickleable
- PR #2267: CountVectorizer estimator
- PR #2261: Exposing new FAISS metrics through Python API
- PR #2287: Single-GPU TfidfTransformer implementation
- PR #2289: QR SVD solver for MNMG PCA
- PR #2312: column-major support for make_blobs
- PR #2172: Initial support for auto-ARIMA
- PR #2394: Adding cosine & correlation distance for KNN
- PR #2392: PCA can accept sparse inputs, and sparse prim for computing covariance
- PR #2465: Support pandas 1.0+
- PR #2550: Single GPU Target Encoder
- PR #2519: Precision recall curve using cupy
- PR #2500: Replace UMAP functionality dependency on nvgraph with RAFT Spectral Clustering
- PR #2502: cuML Implementation of `sklearn.metrics.pairwise_distances`
- PR #2520: TfidfVectorizer estimator
- PR #2211: MNMG KNN Classifier & Regressor
- PR #2461: Add KNN Sparse Output Functionality
- PR #2615: Incremental PCA
- PR #2594: Confidence intervals for ARIMA forecasts
- PR #2607: Add support for probability estimates in SVC
- PR #2618: SVM class and sample weights
- PR #2635: Decorator to generate docstrings with autodetection of parameters
- PR #2270: Multi class MNMG RF
- PR #2661: CUDA-11 support for single-gpu code
- PR #2322: Sparse FIL forests with 8-byte nodes
- PR #2675: Update conda recipes to support CUDA 11
- PR #2645: Add experimental, sklearn-based preprocessing

## Improvements
- PR #2336: Eliminate `rmm.device_array` usage
- PR #2262: Using fully shared PartDescriptor in MNMG decomposiition, linear models, and solvers
- PR #2310: Pinning ucx-py to 0.14 to make 0.15 CI pass
- PR #1945: enable clang tidy
- PR #2339: umap performance improvements
- PR #2308: Using fixture for Dask client to eliminate possibility of not closing
- PR #2345: make C++ logger level definition to be the same as python layer
- PR #2329: Add short commit hash to conda package name
- PR #2362: Implement binary/multi-classification log loss with cupy
- PR #2363: Update threshold and make other changes for stress tests
- PR #2371: Updating MBSGD tests to use larger batches
- PR #2380: Pinning libcumlprims version to ease future updates
- PR #2405: Remove references to deprecated RMM headers.
- PR #2340: Import ARIMA in the root init file and fix the `test_fit_function` test
- PR #2408: Install meta packages for dependencies
- PR #2417: Move doc customization scripts to Jenkins
- PR #2427: Moving MNMG decomposition to cuml
- PR #2433: Add libcumlprims_mg to CMake
- PR #2420: Add and set convert_dtype default to True in estimator fit methods
- PR #2411: Refactor Mixin classes and use in classifier/regressor estimators
- PR #2442: fix setting RAFT_DIR from the RAFT_PATH env var
- PR #2469: Updating KNN c-api to document all arguments
- PR #2453: Add CumlArray to API doc
- PR #2440: Use Treelite Conda package
- PR #2403: Support for input and output type consistency in logistic regression predict_proba
- PR #2473: Add metrics.roc_auc_score to API docs. Additional readability and minor docs bug fixes
- PR #2468: Add `_n_features_in_` attribute to all single GPU estimators that implement fit
- PR #2489: Removing explicit FAISS build and adding dependency on libfaiss conda package
- PR #2480: Moving MNMG glm and solvers to cuml
- PR #2490: Moving MNMG KMeans to cuml
- PR #2483: Moving MNMG KNN to cuml
- PR #2492: Adding additional assertions to mnmg nearest neighbors pytests
- PR #2439: Update dask RF code to have print_detailed function
- PR #2431: Match output of classifier predict with target dtype
- PR #2237: Refactor RF cython code
- PR #2513: Fixing LGTM Analysis Issues
- PR #2099: Raise an error when float64 data is used with dask RF
- PR #2522: Renaming a few arguments in KNeighbors* to be more readable
- PR #2499: Provide access to `cuml.DBSCAN` core samples
- PR #2526: Removing PCA TSQR as a solver due to scalability issues
- PR #2536: Update conda upload versions for new supported CUDA/Python
- PR #2538: Remove Protobuf dependency
- PR #2553: Test pickle protocol 5 support
- PR #2570: Accepting single df or array input in train_test_split
- PR #2566: Remove deprecated cuDF from_gpu_matrix calls
- PR #2583: findpackage.cmake.in template for cmake dependencies
- PR #2577: Fully removing NVGraph dependency for CUDA 11 compatibility
- PR #2575: Speed up TfidfTransformer
- PR #2584: Removing dependency on sklearn's NotFittedError
- PR #2591: Generate benchmark datsets using `cuml.datasets`
- PR #2548: Fix limitation on number of rows usable with tSNE and refactor memory allocation
- PR #2589: including cuda-11 build fixes into raft
- PR #2599: Add Stratified train_test_split
- PR #2487: Set classes_ attribute during classifier fit
- PR #2605: Reduce memory usage in tSNE
- PR #2611: Adding building doxygen docs to gpu ci
- PR #2631: Enabling use of gtest conda package for build
- PR #2623: Fixing kmeans score() API to be compatible with Scikit-learn
- PR #2629: Add naive_bayes api docs
- PR #2643: 'dense' and 'sparse' values of `storage_type` for FIL
- PR #2691: Generic Base class attribute setter
- PR #2666: Update MBSGD documentation to mention that the model is experimental
- PR #2687: Update xgboost version to 1.2.0dev.rapidsai0.15
- PR #2684: CUDA 11 conda development environment yml and faiss patch
- PR #2648: Replace CNMeM with `rmm::mr::pool_memory_resource`.
- PR #2686: Improve SVM tests
- PR #2692: Changin LBFGS log level
- PR #2705: Add sum operator and base operator overloader functions to cumlarray
- PR #2701: Updating README + Adding ref to UMAP paper
- PR #2721: Update API docs
- PR #2730: Unpin cumlprims in conda recipes for release

## Bug Fixes
- PR #2369: Update RF code to fix set_params memory leak
- PR #2364: Fix for random projection
- PR #2373: Use Treelite Pip package in GPU testing
- PR #2376: Update documentation Links
- PR #2407: fixed batch count in DBScan for integer overflow case
- PR #2413: CumlArray and related methods updates to account for cuDF.Buffer contiguity update
- PR #2424: --singlegpu flag fix on build.sh script
- PR #2432: Using correct algo_name for UMAP in benchmark tests
- PR #2445: Restore access to coef_ property of Lasso
- PR #2441: Change p2p_enabled definition to work without ucx
- PR #2447: Drop `nvstrings`
- PR #2450: Update local build to use new gpuCI image
- PR #2454: Mark RF memleak test as XFAIL, because we can't detect memleak reliably
- PR #2455: Use correct field to store data type in `LabelEncoder.fit_transform`
- PR #2475: Fix typo in build.sh
- PR #2496: Fixing indentation for simulate_data in test_fil.py
- PR #2494: Set QN regularization strength consistent with scikit-learn
- PR #2486: Fix cupy input to kmeans init
- PR #2497: Changes to accommodate cuDF unsigned categorical changes
- PR #2209: Fix FIL benchmark for gpuarray-c input
- PR #2507: Import `treelite.sklearn`
- PR #2521: Fixing invalid smem calculation in KNeighborsCLassifier
- PR #2515: Increase tolerance for LogisticRegression test
- PR #2532: Updating doxygen in new MG headers
- PR #2521: Fixing invalid smem calculation in KNeighborsCLassifier
- PR #2515: Increase tolerance for LogisticRegression test
- PR #2545: Fix documentation of n_iter_without_progress in tSNE Python bindings
- PR #2543: Improve numerical stability of QN solver
- PR #2544: Fix Barnes-Hut tSNE not using specified post_learning_rate
- PR #2558: Disabled a long-running FIL test
- PR #2540: Update default value for n_epochs in UMAP to match documentation & sklearn API
- PR #2535: Fix issue with incorrect docker image being used in local build script
- PR #2542: Fix small memory leak in TSNE
- PR #2552: Fixed the length argument of updateDevice calls in RF test
- PR #2565: Fix cell allocation code to avoid loops in quad-tree. Prevent NaNs causing infinite descent
- PR #2563: Update scipy call for arima gradient test
- PR #2569: Fix for cuDF update
- PR #2508: Use keyword parameters in sklearn.datasets.make_* functions
- PR #2587: Attributes for estimators relying on solvers
- PR #2586: Fix SVC decision function data type
- PR #2573: Considering managed memory as device type on checking for KMeans
- PR #2574: Fixing include path in `tsvd_mg.pyx`
- PR #2506: Fix usage of CumlArray attributes on `cuml.common.base.Base`
- PR #2593: Fix inconsistency in train_test_split
- PR #2609: Fix small doxygen issues
- PR #2610: Remove cuDF tolist call
- PR #2613: Removing thresholds from kmeans score tests (SG+MG)
- PR #2616: Small test code fix for pandas dtype tests
- PR #2617: Fix floating point precision error in tSNE
- PR #2625: Update Estimator notebook to resolve errors
- PR #2634: singlegpu build option fixes
- PR #2641: [Breaking] Make `max_depth` in RF compatible with scikit-learn
- PR #2650: Make max_depth behave consistently for max_depth > 14
- PR #2651: AutoARIMA Python bug fix
- PR #2654: Fix for vectorizer concatenations
- PR #2655: Fix C++ RF predict function access of rows/samples array
- PR #2649: Cleanup sphinx doc warnings for 0.15
- PR #2668: Order conversion improvements to account for cupy behavior changes
- PR #2669: Revert PR 2655 Revert "Fixes C++ RF predict function"
- PR #2683: Fix incorrect "Bad CumlArray Use" error messages on test failures
- PR #2695: Fix debug build issue due to incorrect host/device method setup
- PR #2709: Fixing OneHotEncoder Overflow Error
- PR #2710: Fix SVC doc statement about predic_proba
- PR #2726: Return correct output type in QN
- PR #2711: Fix Dask RF failure intermittently
- PR #2718: Fix temp directory for py.test
- PR #2719: Set KNeighborsRegressor output dtype according to training target dtype
- PR #2720: Updates to outdated links
- PR #2722: Getting cuML covariance test passing w/ Cupy 7.8 & CUDA 11

# cuML 0.14.0 (03 Jun 2020)

## New Features
- PR #1994: Support for distributed OneHotEncoder
- PR #1892: One hot encoder implementation with cupy
- PR #1655: Adds python bindings for homogeneity score
- PR #1704: Adds python bindings for completeness score
- PR #1687: Adds python bindings for mutual info score
- PR #1980: prim: added a new write-only unary op prim
- PR #1867: C++: add logging interface support in cuML based spdlog
- PR #1902: Multi class inference in FIL C++ and importing multi-class forests from treelite
- PR #1906: UMAP MNMG
- PR #2067: python: wrap logging interface in cython
- PR #2083: Added dtype, order, and use_full_low_rank to MNMG `make_regression`
- PR #2074: SG and MNMG `make_classification`
- PR #2127: Added order to SG `make_blobs`, and switch from C++ to cupy based implementation
- PR #2057: Weighted k-means
- PR #2256: Add a `make_arima` generator
- PR #2245: ElasticNet, Lasso and Coordinate Descent MNMG
- PR #2242: Pandas input support with output as NumPy arrays by default
- PR #2551: Add cuML RF multiclass prediction using FIL from python
- PR #1728: Added notebook testing to gpuCI gpu build

## Improvements
- PR #1931: C++: enabled doxygen docs for all of the C++ codebase
- PR #1944: Support for dask_cudf.core.Series in _extract_partitions
- PR #1947: Cleaning up cmake
- PR #1927: Use Cython's `new_build_ext` (if available)
- PR #1946: Removed zlib dependency from cmake
- PR #1988: C++: cpp bench refactor
- PR #1873: Remove usage of nvstring and nvcat from LabelEncoder
- PR #1968: Update SVC SVR with cuML Array
- PR #1972: updates to our flow to use conda-forge's clang and clang-tools packages
- PR #1974: Reduce ARIMA testing time
- PR #1984: Enable Ninja build
- PR #1985: C++ UMAP parametrizable tests
- PR #2005: Adding missing algorithms to cuml benchmarks and notebook
- PR #2016: Add capability to setup.py and build.sh to fully clean all cython build files and artifacts
- PR #2044: A cuda-memcheck helper wrapper for devs
- PR #2018: Using `cuml.dask.part_utils.extract_partitions` and removing similar, duplicated code
- PR #2019: Enable doxygen build in our nightly doc build CI script
- PR #1996: Cythonize in parallel
- PR #2032: Reduce number of tests for MBSGD to improve CI running time
- PR #2031: Encapsulating UCX-py interactions in singleton
- PR #2029: Add C++ ARIMA log-likelihood benchmark
- PR #2085: Convert TSNE to use CumlArray
- PR #2051: Reduce the time required to run dask pca and dask tsvd tests
- PR #1981: Using CumlArray in kNN and DistributedDataHandler in dask kNN
- PR #2053: Introduce verbosity level in C++ layer instead of boolean `verbose` flag
- PR #2047: Make internal streams non-blocking w.r.t. NULL stream
- PR #2048: Random forest testing speedup
- PR #2058: Use CumlArray in Random Projection
- PR #2068: Updating knn class probabilities to use make_monotonic instead of binary search
- PR #2062: Adding random state to UMAP mnmg tests
- PR #2064: Speed-up K-Means test
- PR #2015: Renaming .h to .cuh in solver, dbscan and svm
- PR #2080: Improved import of sparse FIL forests from treelite
- PR #2090: Upgrade C++ build to C++14 standard
- PR #2089: CI: enabled cuda-memcheck on ml-prims unit-tests during nightly build
- PR #2128: Update Dask RF code to reduce the time required for GPU predict to run
- PR #2125: Build infrastructure to use RAFT
- PR #2131: Update Dask RF fit to use DistributedDataHandler
- PR #2055: Update the metrics notebook to use important cuML models
- PR #2095: Improved import of src_prims/utils.h, making it less ambiguous
- PR #2118: Updating SGD & mini-batch estimators to use CumlArray
- PR #2120: Speeding up dask RandomForest tests
- PR #1883: Use CumlArray in ARIMA
- PR #877: Adding definition of done criteria to wiki
- PR #2135: A few optimizations to UMAP fuzzy simplicial set
- PR #1914: Change the meaning of ARIMA's intercept to match the literature
- PR #2098: Renaming .h to .cuh in decision_tree, glm, pca
- PR #2150: Remove deprecated RMM calls in RMM allocator adapter
- PR #2146: Remove deprecated kalman filter
- PR #2151: Add pytest duration and pytest timeout
- PR #2156: Add Docker 19 support to local gpuci build
- PR #2178: Reduce duplicated code in RF
- PR #2124: Expand tutorial docs and sample notebook
- PR #2175: Allow CPU-only and dataset params for benchmark sweeps
- PR #2186: Refactor cython code to build OPG structs in common utils file
- PR #2180: Add fully single GPU singlegpu python build
- PR #2187: CMake improvements to manage conda environment dependencies
- PR #2185: Add has_sklearn function and use it in datasets/classification.
- PR #2193: Order-independent local shuffle in `cuml.dask.make_regression`
- PR #2204: Update python layer to use the logger interface
- PR #2184: Refoctor headers for holtwinters, rproj, tsvd, tsne, umap
- PR #2199: Remove unncessary notebooks
- PR #2195: Separating fit and transform calls in SG, MNMG PCA to save transform array memory consumption
- PR #2201: Re-enabling UMAP repro tests
- PR #2132: Add SVM C++ benchmarks
- PR #2196: Updates to benchmarks. Moving notebook
- PR #2208: Coordinate Descent, Lasso and ElasticNet CumlArray updates
- PR #2210: Updating KNN tests to evaluate multiple index partitions
- PR #2205: Use timeout to add 2 hour hard limit to dask tests
- PR #2212: Improve DBScan batch count / memory estimation
- PR #2213: Standardized include statements across all cpp source files, updated copyright on all modified files
- PR #2214: Remove utils folder and refactor to common folder
- PR #2220: Final refactoring of all src_prims header files following rules as specified in #1675
- PR #2225: input_to_cuml_array keep order option, test updates and cleanup
- PR #2244: Re-enable slow ARIMA tests as stress tests
- PR #2231: Using OPG structs from `cuml.common` in decomposition algorithms
- PR #2257: Update QN and LogisticRegression to use CumlArray
- PR #2259: Add CumlArray support to Naive Bayes
- PR #2252: Add benchmark for the Gram matrix prims
- PR #2263: Faster serialization for Treelite objects with RF
- PR #2264: Reduce build time for cuML by using make_blobs from libcuml++ interface
- PR #2269: Add docs targets to build.sh and fix python cuml.common docs
- PR #2271: Clarify doc for `_unique` default implementation in OneHotEncoder
- PR #2272: Add docs build.sh script to repository
- PR #2276: Ensure `CumlArray` provided `dtype` conforms
- PR #2281: Rely on cuDF's `Serializable` in `CumlArray`
- PR #2284: Reduce dataset size in SG RF notebook to reduce run time of sklearn
- PR #2285: Increase the threshold for elastic_net test in dask/test_coordinate_descent
- PR #2314: Update FIL default values, documentation and test
- PR #2316: 0.14 release docs additions and fixes
- PR #2320: Add prediction notes to RF docs
- PR #2323: Change verbose levels and parameter name to match Scikit-learn API
- PR #2324: Raise an error if n_bins > number of training samples in RF
- PR #2335: Throw a warning if treelite cannot be imported and `load_from_sklearn` is used

## Bug Fixes
- PR #1939: Fix syntax error in cuml.common.array
- PR #1941: Remove c++ cuda flag that was getting duplicated in CMake
- PR #1971: python: Correctly honor --singlegpu option and CUML_BUILD_PATH env variable
- PR #1969: Update libcumlprims to 0.14
- PR #1973: Add missing mg files for setup.py --singlegpu flag
- PR #1993: Set `umap_transform_reproducibility` tests to xfail
- PR #2004: Refactoring the arguments to `plant()` call
- PR #2017: Fixing memory issue in weak cc prim
- PR #2028: Skipping UMAP knn reproducibility tests until we figure out why its failing in CUDA 10.2
- PR #2024: Fixed cuda-memcheck errors with sample-without-replacement prim
- PR #1540: prims: support for custom math-type used for computation inside adjusted rand index prim
- PR #2077: dask-make blobs arguments to match sklearn
- PR #2059: Make all Scipy imports conditional
- PR #2078: Ignore negative cache indices in get_vecs
- PR #2084: Fixed cuda-memcheck errors with COO unit-tests
- PR #2087: Fixed cuda-memcheck errors with dispersion prim
- PR #2096: Fixed syntax error with nightly build command for memcheck unit-tests
- PR #2115: Fixed contingency matrix prim unit-tests for computing correct golden values
- PR #2107: Fix PCA transform
- PR #2109: input_to_cuml_array __cuda_array_interface__ bugfix
- PR #2117: cuDF __array__ exception small fixes
- PR #2139: CumlArray for adjusted_rand_score
- PR #2140: Returning self in fit model functions
- PR #2144: Remove GPU arch < 60 from CMake build
- PR #2153: Added missing namespaces to some Decision Tree files
- PR #2155: C++: fix doxygen build break
- PR #2161: Replacing depreciated bruteForceKnn
- PR #2162: Use stream in transpose prim
- PR #2165: Fit function test correction
- PR #2166: Fix handling of temp file in RF pickling
- PR #2176: C++: fix for adjusted rand index when input array is all zeros
- PR #2179: Fix clang tools version in libcuml recipe
- PR #2183: Fix RAFT in nightly package
- PR #2191: Fix placement of SVM parameter documentation and add examples
- PR #2212: Fix DBScan results (no propagation of labels through border points)
- PR #2215: Fix the printing of forest object
- PR #2217: Fix opg_utils naming to fix singlegpu build
- PR #2223: Fix bug in ARIMA C++ benchmark
- PR #2224: Temporary fix for CI until new Dask version is released
- PR #2228: Update to use __reduce_ex__ in CumlArray to override cudf.Buffer
- PR #2249: Fix bug in UMAP continuous target metrics
- PR #2258: Fix doxygen build break
- PR #2255: Set random_state for train_test_split function in dask RF
- PR #2275: Fix RF fit memory leak
- PR #2274: Fix parameter name verbose to verbosity in mnmg OneHotEncoder
- PR #2277: Updated cub repo path and branch name
- PR #2282: Fix memory leak in Dask RF concatenation
- PR #2301: Scaling KNN dask tests sample size with n GPUs
- PR #2293: Contiguity fixes for input_to_cuml_array and train_test_split
- PR #2295: Fix convert_to_dtype copy even with same dtype
- PR #2305: Fixed race condition in DBScan
- PR #2354: Fix broken links in README
- PR #2619: Explicitly skip raft test folder for pytest 6.0.0
- PR #2788: Set the minimum number of columns that can be sampled to 1 to fix 0 mem allocation error

# cuML 0.13.0 (31 Mar 2020)

## New Features
- PR #1777: Python bindings for entropy
- PR #1742: Mean squared error implementation with cupy
- PR #1817: Confusion matrix implementation with cupy (SNSG and MNMG)
- PR #1766: Mean absolute error implementation with cupy
- PR #1766: Mean squared log error implementation with cupy
- PR #1635: cuML Array shim and configurable output added to cluster methods
- PR #1586: Seasonal ARIMA
- PR #1683: cuml.dask make_regression
- PR #1689: Add framework for cuML Dask serializers
- PR #1709: Add `decision_function()` and `predict_proba()` for LogisticRegression
- PR #1714: Add `print_env.sh` file to gather important environment details
- PR #1750: LinearRegression CumlArray for configurable output
- PR #1814: ROC AUC score implementation with cupy
- PR #1767: Single GPU decomposition models configurable output
- PR #1646: Using FIL to predict in MNMG RF
- PR #1778: Make cuML Handle picklable
- PR #1738: cuml.dask refactor beginning and dask array input option for OLS, Ridge and KMeans
- PR #1874: Add predict_proba function to RF classifier
- PR #1815: Adding KNN parameter to UMAP
- PR #1978: Adding `predict_proba` function to dask RF

## Improvements
- PR #1644: Add `predict_proba()` for FIL binary classifier
- PR #1620: Pickling tests now automatically finds all model classes inheriting from cuml.Base
- PR #1637: Update to newer treelite version with XGBoost 1.0 compatibility
- PR #1632: Fix MBSGD models inheritance, they now inherits from cuml.Base
- PR #1628: Remove submodules from cuML
- PR #1755: Expose the build_treelite function for python
- PR #1649: Add the fil_sparse_format variable option to RF API
- PR #1647: storage_type=AUTO uses SPARSE for large models
- PR #1668: Update the warning statement thrown in RF when the seed is set but n_streams is not 1
- PR #1662: use of direct cusparse calls for coo2csr, instead of depending on nvgraph
- PR #1747: C++: dbscan performance improvements and cleanup
- PR #1697: Making trustworthiness batchable and using proper workspace
- PR #1721: Improving UMAP pytests
- PR #1717: Call `rmm_cupy_allocator` for CuPy allocations
- PR #1718: Import `using_allocator` from `cupy.cuda`
- PR #1723: Update RF Classifier to throw an exception for multi-class pickling
- PR #1726: Decorator to allocate CuPy arrays with RMM
- PR #1719: UMAP random seed reproducibility
- PR #1748: Test serializing `CumlArray` objects
- PR #1776: Refactoring pca/tsvd distributed
- PR #1762: Update CuPy requirement to 7
- PR #1768: C++: Different input and output types for add and subtract prims
- PR #1790: Add support for multiple seeding in k-means++
- PR #1805: Adding new Dask cuda serializers to naive bayes + a trivial perf update
- PR #1812: C++: bench: UMAP benchmark cases added
- PR #1795: Add capability to build CumlArray from bytearray/memoryview objects
- PR #1824: C++: improving the performance of UMAP algo
- PR #1816: Add ARIMA notebook
- PR #1856: Update docs for 0.13
- PR #1827: Add HPO demo Notebook
- PR #1825: `--nvtx` option in `build.sh`
- PR #1847: Update XGBoost version for CI
- PR #1837: Simplify cuML Array construction
- PR #1848: Rely on subclassing for cuML Array serialization
- PR #1866: Minimizing client memory pressure on Naive Bayes
- PR #1788: Removing complexity bottleneck in S-ARIMA
- PR #1873: Remove usage of nvstring and nvcat from LabelEncoder
- PR #1891: Additional improvements to naive bayes tree reduction

## Bug Fixes
- PR #1835 : Fix calling default RF Classification always
- PT #1904: replace cub sort
- PR #1833: Fix depth issue in shallow RF regression estimators
- PR #1770: Warn that KalmanFilter is deprecated
- PR #1775: Allow CumlArray to work with inputs that have no 'strides' in array interface
- PR #1594: Train-test split is now reproducible
- PR #1590: Fix destination directory structure for run-clang-format.py
- PR #1611: Fixing pickling errors for KNN classifier and regressor
- PR #1617: Fixing pickling issues for SVC and SVR
- PR #1634: Fix title in KNN docs
- PR #1627: Adding a check for multi-class data in RF classification
- PR #1654: Skip treelite patch if its already been applied
- PR #1661: Fix nvstring variable name
- PR #1673: Using struct for caching dlsym state in communicator
- PR #1659: TSNE - introduce 'convert_dtype' and refactor class attr 'Y' to 'embedding_'
- PR #1672: Solver 'svd' in Linear and Ridge Regressors when n_cols=1
- PR #1670: Lasso & ElasticNet - cuml Handle added
- PR #1671: Update for accessing cuDF Series pointer
- PR #1652: Support XGBoost 1.0+ models in FIL
- PR #1702: Fix LightGBM-FIL validation test
- PR #1701: test_score kmeans test passing with newer cupy version
- PR #1706: Remove multi-class bug from QuasiNewton
- PR #1699: Limit CuPy to <7.2 temporarily
- PR #1708: Correctly deallocate cuML handles in Cython
- PR #1730: Fixes to KF for test stability (mainly in CUDA 10.2)
- PR #1729: Fixing naive bayes UCX serialization problem in fit()
- PR #1749: bug fix rf classifier/regressor on seg fault in bench
- PR #1751: Updated RF documentation
- PR #1765: Update the checks for using RF GPU predict
- PR #1787: C++: unit-tests to check for RF accuracy. As well as a bug fix to improve RF accuracy
- PR #1793: Updated fil pyx to solve memory leakage issue
- PR #1810: Quickfix - chunkage in dask make_regression
- PR #1842: DistributedDataHandler not properly setting 'multiple'
- PR #1849: Critical fix in ARIMA initial estimate
- PR #1851: Fix for cuDF behavior change for multidimensional arrays
- PR #1852: Remove Thrust warnings
- PR #1868: Turning off IPC caching until it is fixed in UCX-py/UCX
- PR #1876: UMAP exponential decay parameters fix
- PR #1887: Fix hasattr for missing attributes on base models
- PR #1877: Remove resetting index in shuffling in train_test_split
- PR #1893: Updating UCX in comms to match current UCX-py
- PR #1888: Small train_test_split test fix
- PR #1899: Fix dask `extract_partitions()`, remove transformation as instance variable in PCA and TSVD and match sklearn APIs
- PR #1920: Temporarily raising threshold for UMAP reproducibility tests
- PR #1918: Create memleak fixture to skip memleak tests in CI for now
- PR #1926: Update batch matrix test margins
- PR #1925: Fix failing dask tests
- PR #1936: Update DaskRF regression test to xfail
- PR #1932: Isolating cause of make_blobs failure
- PR #1951: Dask Random forest regression CPU predict bug fix
- PR #1948: Adjust BatchedMargin margin and disable tests temporarily
- PR #1950: Fix UMAP test failure


# cuML 0.12.0 (04 Feb 2020)

## New Features
- PR #1483: prims: Fused L2 distance and nearest-neighbor prim
- PR #1494: bench: ml-prims benchmark
- PR #1514: bench: Fused L2 NN prim benchmark
- PR #1411: Cython side of MNMG OLS
- PR #1520: Cython side of MNMG Ridge Regression
- PR #1516: Suppor Vector Regression (epsilon-SVR)

## Improvements
- PR #1638: Update cuml/docs/README.md
- PR #1468: C++: updates to clang format flow to make it more usable among devs
- PR #1473: C++: lazy initialization of "costly" resources inside cumlHandle
- PR #1443: Added a new overloaded GEMM primitive
- PR #1489: Enabling deep trees using Gather tree builder
- PR #1463: Update FAISS submodule to 1.6.1
- PR #1488: Add codeowners
- PR #1432: Row-major (C-style) GPU arrays for benchmarks
- PR #1490: Use dask master instead of conda package for testing
- PR #1375: Naive Bayes & Distributed Naive Bayes
- PR #1377: Add GPU array support for FIL benchmarking
- PR #1493: kmeans: add tiling support for 1-NN computation and use fusedL2-1NN prim for L2 distance metric
- PR #1532: Update CuPy to >= 6.6 and allow 7.0
- PR #1528: Re-enabling KNN using dynamic library loading for UCX in communicator
- PR #1545: Add conda environment version updates to ci script
- PR #1541: Updates for libcudf++ Python refactor
- PR #1555: FIL-SKL, an SKLearn-based benchmark for FIL
- PR #1537: Improve pickling and scoring suppport for many models to support hyperopt
- PR #1551: Change custom kernel to cupy for col/row order transform
- PR #1533: C++: interface header file separation for SVM
- PR #1560: Helper function to allocate all new CuPy arrays with RMM memory management
- PR #1570: Relax nccl in conda recipes to >=2.4 (matching CI)
- PR #1578: Add missing function information to the cuML documenataion
- PR #1584: Add has_scipy utility function for runtime check
- PR #1583: API docs updates for 0.12
- PR #1591: Updated FIL documentation

## Bug Fixes
- PR #1470: Documentation: add make_regression, fix ARIMA section
- PR #1482: Updated the code to remove sklearn from the mbsgd stress test
- PR #1491: Update dev environments for 0.12
- PR #1512: Updating setup_cpu() in SpeedupComparisonRunner
- PR #1498: Add build.sh to code owners
- PR #1505: cmake: added correct dependencies for prims-bench build
- PR #1534: Removed TODO comment in create_ucp_listeners()
- PR #1548: Fixing umap extra unary op in knn graph
- PR #1547: Fixing MNMG kmeans score. Fixing UMAP pickling before fit(). Fixing UMAP test failures.
- PR #1557: Increasing threshold for kmeans score
- PR #1562: Increasing threshold even higher
- PR #1564: Fixed a typo in function cumlMPICommunicator_impl::syncStream
- PR #1569: Remove Scikit-learn exception and depedenncy in SVM
- PR #1575: Add missing dtype parameter in call to strides to order for CuPy 6.6 code path
- PR #1574: Updated the init file to include SVM
- PR #1589: Fixing the default value for RF and updating mnmg predict to accept cudf
- PR #1601: Fixed wrong datatype used in knn voting kernel

# cuML 0.11.0 (11 Dec 2019)

## New Features

- PR #1295: Cython side of MNMG PCA
- PR #1218: prims: histogram prim
- PR #1129: C++: Separate include folder for C++ API distribution
- PR #1282: OPG KNN MNMG Code (disabled for 0.11)
- PR #1242: Initial implementation of FIL sparse forests
- PR #1194: Initial ARIMA time-series modeling support.
- PR #1286: Importing treelite models as FIL sparse forests
- PR #1285: Fea minimum impurity decrease RF param
- PR #1301: Add make_regression to generate regression datasets
- PR #1322: RF pickling using treelite, protobuf and FIL
- PR #1332: Add option to cuml.dask make_blobs to produce dask array
- PR #1307: Add RF regression benchmark
- PR #1327: Update the code to build treelite with protobuf
- PR #1289: Add Python benchmarking support for FIL
- PR #1371: Cython side of MNMG tSVD
- PR #1386: Expose SVC decision function value

## Improvements
- PR #1170: Use git to clone subprojects instead of git submodules
- PR #1239: Updated the treelite version
- PR #1225: setup.py clone dependencies like cmake and correct include paths
- PR #1224: Refactored FIL to prepare for sparse trees
- PR #1249: Include libcuml.so C API in installed targets
- PR #1259: Conda dev environment updates and use libcumlprims current version in CI
- PR #1277: Change dependency order in cmake for better printing at compile time
- PR #1264: Add -s flag to GPU CI pytest for better error printing
- PR #1271: Updated the Ridge regression documentation
- PR #1283: Updated the cuMl docs to include MBSGD and adjusted_rand_score
- PR #1300: Lowercase parameter versions for FIL algorithms
- PR #1312: Update CuPy to version 6.5 and use conda-forge channel
- PR #1336: Import SciKit-Learn models into FIL
- PR #1314: Added options needed for ASVDb output (CUDA ver, etc.), added option
  to select algos
- PR #1335: Options to print available algorithms and datasets
  in the Python benchmark
- PR #1338: Remove BUILD_ABI references in CI scripts
- PR #1340: Updated unit tests to uses larger dataset
- PR #1351: Build treelite temporarily for GPU CI testing of FIL Scikit-learn
  model importing
- PR #1367: --test-split benchmark parameter for train-test split
- PR #1360: Improved tests for importing SciKit-Learn models into FIL
- PR #1368: Add --num-rows benchmark command line argument
- PR #1351: Build treelite temporarily for GPU CI testing of FIL Scikit-learn model importing
- PR #1366: Modify train_test_split to use CuPy and accept device arrays
- PR #1258: Documenting new MPI communicator for multi-node multi-GPU testing
- PR #1345: Removing deprecated should_downcast argument
- PR #1362: device_buffer in UMAP + Sparse prims
- PR #1376: AUTO value for FIL algorithm
- PR #1408: Updated pickle tests to delete the pre-pickled model to prevent pointer leakage
- PR #1357: Run benchmarks multiple times for CI
- PR #1382: ARIMA optimization: move functions to C++ side
- PR #1392: Updated RF code to reduce duplication of the code
- PR #1444: UCX listener running in its own isolated thread
- PR #1445: Improved performance of FIL sparse trees
- PR #1431: Updated API docs
- PR #1441: Remove unused CUDA conda labels
- PR #1439: Match sklearn 0.22 default n_estimators for RF and fix test errors
- PR #1461: Add kneighbors to API docs

## Bug Fixes
- PR #1281: Making rng.h threadsafe
- PR #1212: Fix cmake git cloning always running configure in subprojects
- PR #1261: Fix comms build errors due to cuml++ include folder changes
- PR #1267: Update build.sh for recent change of building comms in main CMakeLists
- PR #1278: Removed incorrect overloaded instance of eigJacobi
- PR #1302: Updates for numba 0.46
- PR #1313: Updated the RF tests to set the seed and n_streams
- PR #1319: Using machineName arg passed in instead of default for ASV reporting
- PR #1326: Fix illegal memory access in make_regression (bounds issue)
- PR #1330: Fix C++ unit test utils for better handling of differences near zero
- PR #1342: Fix to prevent memory leakage in Lasso and ElasticNet
- PR #1337: Fix k-means init from preset cluster centers
- PR #1354: Fix SVM gamma=scale implementation
- PR #1344: Change other solver based methods to create solver object in init
- PR #1373: Fixing a few small bugs in make_blobs and adding asserts to pytests
- PR #1361: Improve SMO error handling
- PR #1384: Lower expectations on batched matrix tests to prevent CI failures
- PR #1380: Fix memory leaks in ARIMA
- PR #1391: Lower expectations on batched matrix tests even more
- PR #1394: Warning added in svd for cuda version 10.1
- PR #1407: Resolved RF predict issues and updated RF docstring
- PR #1401: Patch for lbfgs solver for logistic regression with no l1 penalty
- PR #1416: train_test_split numba and rmm device_array output bugfix
- PR #1419: UMAP pickle tests are using wrong n_neighbors value for trustworthiness
- PR #1438: KNN Classifier to properly return Dataframe with Dataframe input
- PR #1425: Deprecate seed and use random_state similar to Scikit-learn in train_test_split
- PR #1458: Add joblib as an explicit requirement
- PR #1474: Defer knn mnmg to 0.12 nightly builds and disable ucx-py dependency

# cuML 0.10.0 (16 Oct 2019)

## New Features
- PR #1148: C++ benchmark tool for c++/CUDA code inside cuML
- PR #1071: Selective eigen solver of cuSolver
- PR #1073: Updating RF wrappers to use FIL for GPU accelerated prediction
- PR #1104: CUDA 10.1 support
- PR #1113: prims: new batched make-symmetric-matrix primitive
- PR #1112: prims: new batched-gemv primitive
- PR #855: Added benchmark tools
- PR #1149 Add YYMMDD to version tag for nightly conda packages
- PR #892: General Gram matrices prim
- PR #912: Support Vector Machine
- PR #1274: Updated the RF score function to use GPU predict

## Improvements
- PR #961: High Peformance RF; HIST algo
- PR #1028: Dockerfile updates after dir restructure. Conda env yaml to add statsmodels as a dependency
- PR #1047: Consistent OPG interface for kmeans, based on internal libcumlprims update
- PR #763: Add examples to train_test_split documentation
- PR #1093: Unified inference kernels for different FIL algorithms
- PR #1076: Paying off some UMAP / Spectral tech debt.
- PR #1086: Ensure RegressorMixin scorer uses device arrays
- PR #1110: Adding tests to use default values of parameters of the models
- PR #1108: input_to_host_array function in input_utils for input processing to host arrays
- PR #1114: K-means: Exposing useful params, removing unused params, proxying params in Dask
- PR #1138: Implementing ANY_RANK semantics on irecv
- PR #1142: prims: expose separate InType and OutType for unaryOp and binaryOp
- PR #1115: Moving dask_make_blobs to cuml.dask.datasets. Adding conversion to dask.DataFrame
- PR #1136: CUDA 10.1 CI updates
- PR #1135: K-means: add boundary cases for kmeans||, support finer control with convergence
- PR #1163: Some more correctness improvements. Better verbose printing
- PR #1165: Adding except + in all remaining cython
- PR #1186: Using LocalCUDACluster Pytest fixture
- PR #1173: Docs: Barnes Hut TSNE documentation
- PR #1176: Use new RMM API based on Cython
- PR #1219: Adding custom bench_func and verbose logging to cuml.benchmark
- PR #1247: Improved MNMG RF error checking

## Bug Fixes

- PR #1231: RF respect number of cuda streams from cuml handle
- PR #1230: Rf bugfix memleak in regression
- PR #1208: compile dbscan bug
- PR #1016: Use correct libcumlprims version in GPU CI
- PR #1040: Update version of numba in development conda yaml files
- PR #1043: Updates to accommodate cuDF python code reorganization
- PR #1044: Remove nvidia driver installation from ci/cpu/build.sh
- PR #991: Barnes Hut TSNE Memory Issue Fixes
- PR #1075: Pinning Dask version for consistent CI results
- PR #990: Barnes Hut TSNE Memory Issue Fixes
- PR #1066: Using proper set of workers to destroy nccl comms
- PR #1072: Remove pip requirements and setup
- PR #1074: Fix flake8 CI style check
- PR #1087: Accuracy improvement for sqrt/log in RF max_feature
- PR #1088: Change straggling numba python allocations to use RMM
- PR #1106: Pinning Distributed version to match Dask for consistent CI results
- PR #1116: TSNE CUDA 10.1 Bug Fixes
- PR #1132: DBSCAN Batching Bug Fix
- PR #1162: DASK RF random seed bug fix
- PR #1164: Fix check_dtype arg handling for input_to_dev_array
- PR #1171: SVM prediction bug fix
- PR #1177: Update dask and distributed to 2.5
- PR #1204: Fix SVM crash on Turing
- PR #1199: Replaced sprintf() with snprintf() in THROW()
- PR #1205: Update dask-cuda in yml envs
- PR #1211: Fixing Dask k-means transform bug and adding test
- PR #1236: Improve fix for SMO solvers potential crash on Turing
- PR #1251: Disable compiler optimization for CUDA 10.1 for distance prims
- PR #1260: Small bugfix for major conversion in input_utils
- PR #1276: Fix float64 prediction crash in test_random_forest

# cuML 0.9.0 (21 Aug 2019)

## New Features

- PR #894: Convert RF to treelite format
- PR #826: Jones transformation of params for ARIMA models timeSeries ml-prim
- PR #697: Silhouette Score metric ml-prim
- PR #674: KL Divergence metric ml-prim
- PR #787: homogeneity, completeness and v-measure metrics ml-prim
- PR #711: Mutual Information metric ml-prim
- PR #724: Entropy metric ml-prim
- PR #766: Expose score method based on inertia for KMeans
- PR #823: prims: cluster dispersion metric
- PR #816: Added inverse_transform() for LabelEncoder
- PR #789: prims: sampling without replacement
- PR #813: prims: Col major istance prim
- PR #635: Random Forest & Decision Tree Regression (Single-GPU)
- PR #819: Forest Inferencing Library (FIL)
- PR #829: C++: enable nvtx ranges
- PR #835: Holt-Winters algorithm
- PR #837: treelite for decision forest exchange format
- PR #871: Wrapper for FIL
- PR #870: make_blobs python function
- PR #881: wrappers for accuracy_score and adjusted_rand_score functions
- PR #840: Dask RF classification and regression
- PR #870: make_blobs python function
- PR #879: import of treelite models to FIL
- PR #892: General Gram matrices prim
- PR #883: Adding MNMG Kmeans
- PR #930: Dask RF
- PR #882: TSNE - T-Distributed Stochastic Neighbourhood Embedding
- PR #624: Internals API & Graph Based Dimensionality Reductions Callback
- PR #926: Wrapper for FIL
- PR #994: Adding MPI comm impl for testing / benchmarking MNMG CUDA
- PR #960: Enable using libcumlprims for MG algorithms/prims

## Improvements
- PR #822: build: build.sh update to club all make targets together
- PR #807: Added development conda yml files
- PR #840: Require cmake >= 3.14
- PR #832: Stateless Decision Tree and Random Forest API
- PR #857: Small modifications to comms for utilizing IB w/ Dask
- PR #851: Random forest Stateless API wrappers
- PR #865: High Performance RF
- PR #895: Pretty prints arguments!
- PR #920: Add an empty marker kernel for tracing purposes
- PR #915: syncStream added to cumlCommunicator
- PR #922: Random Forest support in FIL
- PR #911: Update headers to credit CannyLabs BH TSNE implementation
- PR #918: Streamline CUDA_REL environment variable
- PR #924: kmeans: updated APIs to be stateless, refactored code for mnmg support
- PR #950: global_bias support in FIL
- PR #773: Significant improvements to input checking of all classes and common input API for Python
- PR #957: Adding docs to RF & KMeans MNMG. Small fixes for release
- PR #965: Making dask-ml a hard dependency
- PR #976: Update api.rst for new 0.9 classes
- PR #973: Use cudaDeviceGetAttribute instead of relying on cudaDeviceProp object being passed
- PR #978: Update README for 0.9
- PR #1009: Fix references to notebooks-contrib
- PR #1015: Ability to control the number of internal streams in cumlHandle_impl via cumlHandle
- PR #1175: Add more modules to docs ToC

## Bug Fixes

- PR #923: Fix misshapen level/trend/season HoltWinters output
- PR #831: Update conda package dependencies to cudf 0.9
- PR #772: Add missing cython headers to SGD and CD
- PR #849: PCA no attribute trans_input_ transform bug fix
- PR #869: Removing incorrect information from KNN Docs
- PR #885: libclang installation fix for GPUCI
- PR #896: Fix typo in comms build instructions
- PR #921: Fix build scripts using incorrect cudf version
- PR #928: TSNE Stability Adjustments
- PR #934: Cache cudaDeviceProp in cumlHandle for perf reasons
- PR #932: Change default param value for RF classifier
- PR #949: Fix dtype conversion tests for unsupported cudf dtypes
- PR #908: Fix local build generated file ownerships
- PR #983: Change RF max_depth default to 16
- PR #987: Change default values for knn
- PR #988: Switch to exact tsne
- PR #991: Cleanup python code in cuml.dask.cluster
- PR #996: ucx_initialized being properly set in CommsContext
- PR #1007: Throws a well defined error when mutigpu is not enabled
- PR #1018: Hint location of nccl in build.sh for CI
- PR #1022: Using random_state to make K-Means MNMG tests deterministic
- PR #1034: Fix typos and formatting issues in RF docs
- PR #1052: Fix the rows_sample dtype to float

# cuML 0.8.0 (27 June 2019)

## New Features

- PR #652: Adjusted Rand Index metric ml-prim
- PR #679: Class label manipulation ml-prim
- PR #636: Rand Index metric ml-prim
- PR #515: Added Random Projection feature
- PR #504: Contingency matrix ml-prim
- PR #644: Add train_test_split utility for cuDF dataframes
- PR #612: Allow Cuda Array Interface, Numba inputs and input code refactor
- PR #641: C: Separate C-wrapper library build to generate libcuml.so
- PR #631: Add nvcategory based ordinal label encoder
- PR #681: Add MBSGDClassifier and MBSGDRegressor classes around SGD
- PR #705: Quasi Newton solver and LogisticRegression Python classes
- PR #670: Add test skipping functionality to build.sh
- PR #678: Random Forest Python class
- PR #684: prims: make_blobs primitive
- PR #673: prims: reduce cols by key primitive
- PR #812: Add cuML Communications API & consolidate Dask cuML

## Improvements

- PR #597: C++ cuML and ml-prims folder refactor
- PR #590: QN Recover from numeric errors
- PR #482: Introduce cumlHandle for pca and tsvd
- PR #573: Remove use of unnecessary cuDF column and series copies
- PR #601: Cython PEP8 cleanup and CI integration
- PR #596: Introduce cumlHandle for ols and ridge
- PR #579: Introduce cumlHandle for cd and sgd, and propagate C++ errors in cython level for cd and sgd
- PR #604: Adding cumlHandle to kNN, spectral methods, and UMAP
- PR #616: Enable clang-format for enforcing coding style
- PR #618: CI: Enable copyright header checks
- PR #622: Updated to use 0.8 dependencies
- PR #626: Added build.sh script, updated CI scripts and documentation
- PR #633: build: Auto-detection of GPU_ARCHS during cmake
- PR #650: Moving brute force kNN to prims. Creating stateless kNN API.
- PR #662: C++: Bulk clang-format updates
- PR #671: Added pickle pytests and correct pickling of Base class
- PR #675: atomicMin/Max(float, double) with integer atomics and bit flipping
- PR #677: build: 'deep-clean' to build.sh to clean faiss build as well
- PR #683: Use stateless c++ API in KNN so that it can be pickled properly
- PR #686: Use stateless c++ API in UMAP so that it can be pickled properly
- PR #695: prims: Refactor pairwise distance
- PR #707: Added stress test and updated documentation for RF
- PR #701: Added emacs temporary file patterns to .gitignore
- PR #606: C++: Added tests for host_buffer and improved device_buffer and host_buffer implementation
- PR #726: Updated RF docs and stress test
- PR #730: Update README and RF docs for 0.8
- PR #744: Random projections generating binomial on device. Fixing tests.
- PR #741: Update API docs for 0.8
- PR #754: Pickling of UMAP/KNN
- PR #753: Made PCA and TSVD picklable
- PR #746: LogisticRegression and QN API docstrings
- PR #820: Updating DEVELOPER GUIDE threading guidelines

## Bug Fixes
- PR #584: Added missing virtual destructor to deviceAllocator and hostAllocator
- PR #620: C++: Removed old unit-test files in ml-prims
- PR #627: C++: Fixed dbscan crash issue filed in 613
- PR #640: Remove setuptools from conda run dependency
- PR #646: Update link in contributing.md
- PR #649: Bug fix to LinAlg::reduce_rows_by_key prim filed in issue #648
- PR #666: fixes to gitutils.py to resolve both string decode and handling of uncommitted files
- PR #676: Fix template parameters in `bernoulli()` implementation.
- PR #685: Make CuPy optional to avoid nccl conda package conflicts
- PR #687: prims: updated tolerance for reduce_cols_by_key unit-tests
- PR #689: Removing extra prints from NearestNeighbors cython
- PR #718: Bug fix for DBSCAN and increasing batch size of sgd
- PR #719: Adding additional checks for dtype of the data
- PR #736: Bug fix for RF wrapper and .cu print function
- PR #547: Fixed issue if C++ compiler is specified via CXX during configure.
- PR #759: Configure Sphinx to render params correctly
- PR #762: Apply threshold to remove flakiness of UMAP tests.
- PR #768: Fixing memory bug from stateless refactor
- PR #782: Nearest neighbors checking properly whether memory should be freed
- PR #783: UMAP was using wrong size for knn computation
- PR #776: Hotfix for self.variables in RF
- PR #777: Fix numpy input bug
- PR #784: Fix jit of shuffle_idx python function
- PR #790: Fix rows_sample input type for RF
- PR #793: Fix for dtype conversion utility for numba arrays without cupy installed
- PR #806: Add a seed for sklearn model in RF test file
- PR #843: Rf quantile fix

# cuML 0.7.0 (10 May 2019)

## New Features

- PR #405: Quasi-Newton GLM Solvers
- PR #277: Add row- and column-wise weighted mean primitive
- PR #424: Add a grid-sync struct for inter-block synchronization
- PR #430: Add R-Squared Score to ml primitives
- PR #463: Add matrix gather to ml primitives
- PR #435: Expose cumlhandle in cython + developer guide
- PR #455: Remove default-stream argument across ml-prims and cuML
- PR #375: cuml cpp shared library renamed to libcuml++.so
- PR #460: Random Forest & Decision Trees (Single-GPU, Classification)
- PR #491: Add doxygen build target for ml-prims
- PR #505: Add R-Squared Score to python interface
- PR #507: Add coordinate descent for lasso and elastic-net
- PR #511: Add a minmax ml-prim
- PR #516: Added Trustworthiness score feature
- PR #520: Add local build script to mimic gpuCI
- PR #503: Add column-wise matrix sort primitive
- PR #525: Add docs build script to cuML
- PR #528: Remove current KMeans and replace it with a new single GPU implementation built using ML primitives

## Improvements

- PR #481: Refactoring Quasi-Newton to use cumlHandle
- PR #467: Added validity check on cumlHandle_t
- PR #461: Rewrote permute and added column major version
- PR #440: README updates
- PR #295: Improve build-time and the interface e.g., enable bool-OutType, for distance()
- PR #390: Update docs version
- PR #272: Add stream parameters to cublas and cusolver wrapper functions
- PR #447: Added building and running mlprims tests to CI
- PR #445: Lower dbscan memory usage by computing adjacency matrix directly
- PR #431: Add support for fancy iterator input types to LinAlg::reduce_rows_by_key
- PR #394: Introducing cumlHandle API to dbscan and add example
- PR #500: Added CI check for black listed CUDA Runtime API calls
- PR #475: exposing cumlHandle for dbscan from python-side
- PR #395: Edited the CONTRIBUTING.md file
- PR #407: Test files to run stress, correctness and unit tests for cuml algos
- PR #512: generic copy method for copying buffers between device/host
- PR #533: Add cudatoolkit conda dependency
- PR #524: Use cmake find blas and find lapack to pass configure options to faiss
- PR #527: Added notes on UMAP differences from reference implementation
- PR #540: Use latest release version in update-version CI script
- PR #552: Re-enable assert in kmeans tests with xfail as needed
- PR #581: Add shared memory fast col major to row major function back with bound checks
- PR #592: More efficient matrix copy/reverse methods
- PR #721: Added pickle tests for DBSCAN and Random Projections

## Bug Fixes

- PR #334: Fixed segfault in `ML::cumlHandle_impl::destroyResources`
- PR #349: Developer guide clarifications for cumlHandle and cumlHandle_impl
- PR #398: Fix CI scripts to allow nightlies to be uploaded
- PR #399: Skip PCA tests to allow CI to run with driver 418
- PR #422: Issue in the PCA tests was solved and CI can run with driver 418
- PR #409: Add entry to gitmodules to ignore build artifacts
- PR #412: Fix for svdQR function in ml-prims
- PR #438: Code that depended on FAISS was building everytime.
- PR #358: Fixed an issue when switching streams on MLCommon::device_buffer and MLCommon::host_buffer
- PR #434: Fixing bug in CSR tests
- PR #443: Remove defaults channel from ci scripts
- PR #384: 64b index arithmetic updates to the kernels inside ml-prims
- PR #459: Fix for runtime library path of pip package
- PR #464: Fix for C++11 destructor warning in qn
- PR #466: Add support for column-major in LinAlg::*Norm methods
- PR #465: Fixing deadlock issue in GridSync due to consecutive sync calls
- PR #468: Fix dbscan example build failure
- PR #470: Fix resource leakage in Kalman filter python wrapper
- PR #473: Fix gather ml-prim test for change in rng uniform API
- PR #477: Fixes default stream initialization in cumlHandle
- PR #480: Replaced qn_fit() declaration with #include of file containing definition to fix linker error
- PR #495: Update cuDF and RMM versions in GPU ci test scripts
- PR #499: DEVELOPER_GUIDE.md: fixed links and clarified ML::detail::streamSyncer example
- PR #506: Re enable ml-prim tests in CI
- PR #508: Fix for an error with default argument in LinAlg::meanSquaredError
- PR #519: README.md Updates and adding BUILD.md back
- PR #526: Fix the issue of wrong results when fit and transform of PCA are called separately
- PR #531: Fixing missing arguments in updateDevice() for RF
- PR #543: Exposing dbscan batch size through cython API and fixing broken batching
- PR #551: Made use of ZLIB_LIBRARIES consistent between ml_test and ml_mg_test
- PR #557: Modified CI script to run cuML tests before building mlprims and removed lapack flag
- PR #578: Updated Readme.md to add lasso and elastic-net
- PR #580: Fixing cython garbage collection bug in KNN
- PR #577: Use find libz in prims cmake
- PR #594: fixed cuda-memcheck mean_center test failures


# cuML 0.6.1 (09 Apr 2019)

## Bug Fixes

- PR #462 Runtime library path fix for cuML pip package


# cuML 0.6.0 (22 Mar 2019)

## New Features

- PR #249: Single GPU Stochastic Gradient Descent for linear regression, logistic regression, and linear svm with L1, L2, and elastic-net penalties.
- PR #247: Added "proper" CUDA API to cuML
- PR #235: NearestNeighbors MG Support
- PR #261: UMAP Algorithm
- PR #290: NearestNeighbors numpy MG Support
- PR #303: Reusable spectral embedding / clustering
- PR #325: Initial support for single process multi-GPU OLS and tSVD
- PR #271: Initial support for hyperparameter optimization with dask for many models

## Improvements

- PR #144: Dockerfile update and docs for LinearRegression and Kalman Filter.
- PR #168: Add /ci/gpu/build.sh file to cuML
- PR #167: Integrating full-n-final ml-prims repo inside cuml
- PR #198: (ml-prims) Removal of *MG calls + fixed a bug in permute method
- PR #194: Added new ml-prims for supporting LASSO regression.
- PR #114: Building faiss C++ api into libcuml
- PR #64: Using FAISS C++ API in cuML and exposing bindings through cython
- PR #208: Issue ml-common-3: Math.h: swap thrust::for_each with binaryOp,unaryOp
- PR #224: Improve doc strings for readable rendering with readthedocs
- PR #209: Simplify README.md, move build instructions to BUILD.md
- PR #218: Fix RNG to use given seed and adjust RNG test tolerances.
- PR #225: Support for generating random integers
- PR #215: Refactored LinAlg::norm to Stats::rowNorm and added Stats::colNorm
- PR #234: Support for custom output type and passing index value to main_op in *Reduction kernels
- PR #230: Refactored the cuda_utils header
- PR #236: Refactored cuml python package structure to be more sklearn like
- PR #232: Added reduce_rows_by_key
- PR #246: Support for 2 vectors in the matrix vector operator
- PR #244: Fix for single GPU OLS and Ridge to support one column training data
- PR #271: Added get_params and set_params functions for linear and ridge regression
- PR #253: Fix for issue #250-reduce_rows_by_key failed memcheck for small nkeys
- PR #269: LinearRegression, Ridge Python docs update and cleaning
- PR #322: set_params updated
- PR #237: Update build instructions
- PR #275: Kmeans use of faster gpu_matrix
- PR #288: Add n_neighbors to NearestNeighbors constructor
- PR #302: Added FutureWarning for deprecation of current kmeans algorithm
- PR #312: Last minute cleanup before release
- PR #315: Documentation updating and enhancements
- PR #330: Added ignored argument to pca.fit_transform to map to sklearn's implemenation
- PR #342: Change default ABI to ON
- PR #572: Pulling DBSCAN components into reusable primitives


## Bug Fixes

- PR #193: Fix AttributeError in PCA and TSVD
- PR #211: Fixing inconsistent use of proper batch size calculation in DBSCAN
- PR #202: Adding back ability for users to define their own BLAS
- PR #201: Pass CMAKE CUDA path to faiss/configure script
- PR #200 Avoid using numpy via cimport in KNN
- PR #228: Bug fix: LinAlg::unaryOp with 0-length input
- PR #279: Removing faiss-gpu references in README
- PR #321: Fix release script typo
- PR #327: Update conda requirements for version 0.6 requirements
- PR #352: Correctly calculating numpy chunk sizing for kNN
- PR #345: Run python import as part of package build to trigger compilation
- PR #347: Lowering memory usage of kNN.
- PR #355: Fixing issues with very large numpy inputs to SPMG OLS and tSVD.
- PR #357: Removing FAISS requirement from README
- PR #362: Fix for matVecOp crashing on large input sizes
- PR #366: Index arithmetic issue fix with TxN_t class
- PR #376: Disabled kmeans tests since they are currently too sensitive (see #71)
- PR #380: Allow arbitrary data size on ingress for numba_utils.row_matrix
- PR #385: Fix for long import cuml time in containers and fix for setup_pip
- PR #630: Fixing a missing kneighbors in nearest neighbors python proxy

# cuML 0.5.1 (05 Feb 2019)

## Bug Fixes

- PR #189 Avoid using numpy via cimport to prevent ABI issues in Cython compilation


# cuML 0.5.0 (28 Jan 2019)

## New Features

- PR #66: OLS Linear Regression
- PR #44: Distance calculation ML primitives
- PR #69: Ridge (L2 Regularized) Linear Regression
- PR #103: Linear Kalman Filter
- PR #117: Pip install support
- PR #64: Device to device support from cuML device pointers into FAISS

## Improvements

- PR #56: Make OpenMP optional for building
- PR #67: Github issue templates
- PR #44: Refactored DBSCAN to use ML primitives
- PR #91: Pytest cleanup and sklearn toyset datasets based pytests for kmeans and dbscan
- PR #75: C++ example to use kmeans
- PR #117: Use cmake extension to find any zlib installed in system
- PR #94: Add cmake flag to set ABI compatibility
- PR #139: Move thirdparty submodules to root and add symlinks to new locations
- PR #151: Replace TravisCI testing and conda pkg builds with gpuCI
- PR #164: Add numba kernel for faster column to row major transform
- PR #114: Adding FAISS to cuml build

## Bug Fixes

- PR #48: CUDA 10 compilation warnings fix
- PR #51: Fixes to Dockerfile and docs for new build system
- PR #72: Fixes for GCC 7
- PR #96: Fix for kmeans stack overflow with high number of clusters
- PR #105: Fix for AttributeError in kmeans fit method
- PR #113: Removed old  glm python/cython files
- PR #118: Fix for AttributeError in kmeans predict method
- PR #125: Remove randomized solver option from PCA python bindings


# cuML 0.4.0 (05 Dec 2018)

## New Features

## Improvements

- PR #42: New build system: separation of libcuml.so and cuml python package
- PR #43: Added changelog.md

## Bug Fixes


# cuML 0.3.0 (30 Nov 2018)

## New Features

- PR #33: Added ability to call cuML algorithms using numpy arrays

## Improvements

- PR #24: Fix references of python package from cuML to cuml and start using versioneer for better versioning
- PR #40: Added support for refactored cuDF 0.3.0, updated Conda files
- PR #33: Major python test cleaning, all tests pass with cuDF 0.2.0 and 0.3.0. Preparation for new build system
- PR #34: Updated batch count calculation logic in DBSCAN
- PR #35: Beginning of DBSCAN refactor to use cuML mlprims and general improvements

## Bug Fixes

- PR #30: Fixed batch size bug in DBSCAN that caused crash. Also fixed various locations for potential integer overflows
- PR #28: Fix readthedocs build documentation
- PR #29: Fix pytests for cuml name change from cuML
- PR #33: Fixed memory bug that would cause segmentation faults due to numba releasing memory before it was used. Also fixed row major/column major bugs for different algorithms
- PR #36: Fix kmeans gtest to use device data
- PR #38: cuda\_free bug removed that caused google tests to sometimes pass and sometimes fail randomly
- PR #39: Updated cmake to correctly link with CUDA libraries, add CUDA runtime linking and include source files in compile target

# cuML 0.2.0 (02 Nov 2018)

## New Features

- PR #11: Kmeans algorithm added
- PR #7: FAISS KNN wrapper added
- PR #21: Added Conda install support

## Improvements

- PR #15: Added compatibility with cuDF (from prior pyGDF)
- PR #13: Added FAISS to Dockerfile
- PR #21: Added TravisCI build system for CI and Conda builds

## Bug Fixes

- PR #4: Fixed explained variance bug in TSVD
- PR #5: Notebook bug fixes and updated results


# cuML 0.1.0

Initial release including PCA, TSVD, DBSCAN, ml-prims and cython wrappers
