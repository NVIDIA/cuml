# cuML Notebooks

## Intro

These notebooks provide examples of how to use cuML. They may be run using a
local install of `cuml`, or via a pre-built docker container. For help
installing `cuml` (either locally or the docker container), see the [install
guide](https://docs.nvidia.com/datascience/install/#install-rapids).

## Getting started notebooks

For a good overview of how cuML works, see [the introductory notebook
on estimators](../docs/source/estimator_intro.ipynb) in the
documentation tree.

## Additional notebooks

Notebook Title | Status | Description
--- | --- | ---
[ARIMA Demo](arima_demo.ipynb) | Working | Forecast using ARIMA on time-series data.
[KMeans Demo](kmeans_demo.ipynb) | Working | Predict using k-means, visualize and compare the results with Scikit-learn's k-means.
[KMeans Multi-Node Multi-GPU Demo](kmeans_mnmg_demo.ipynb) | Working | Predict with MNMG k-means using dask distributed inputs.
[Linear Regression Demo](linear_regression_demo.ipynb) | Working | Demonstrate the use of OLS Linear Regression for prediction.
[Nearest Neighbors Demo](nearest_neighbors_demo.ipynb) | Working | Predict using Nearest Neighbors algorithm.
[Random Forest Demo](random_forest_demo.ipynb) | Working | Use Random Forest for classification, and demonstrate how to pickle the cuML model.
[Random Forest Multi-Node Multi-GPU Demo](random_forest_mnmg_demo.ipynb) | Working | Solve a classification problem using MNMG Random Forest.
[Target Encoder Walkthrough](target_encoder_walkthrough.ipynb) | Working | Understand how to use target encoding and why it is preferred over one-hot and label encoding with the help of criteo dataset for click-through rate modelling.

## For more details

Many more examples can be found in the [Community Contributed Notebooks
Repository](https://github.com/rapidsai/notebooks-contrib).
