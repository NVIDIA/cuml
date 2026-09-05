/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/logger.hpp>
#include <cuml/svm/svm_model.h>

#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>

#include <cuvs/distance/distance.hpp>
#include <cuvs/distance/grammian.hpp>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

namespace ML {
namespace SVM {

/**
 * @brief Solve the quadratic optimization problem using two level decomposition
 * and Sequential Minimal Optimization (SMO).
 *
 * The general decomposition idea by Osuna is to choose q examples from all the
 * training examples, and solve the QP problem for this subset (discussed in
 * section 11.2 by Joachims [1]). SMO is the extreme case where we choose q=2.
 *
 * Here we follow [2] and [3] and use two level decomposition. First we set
 * q_1=1024, and solve the QP sub-problem for that (let's call it QP1). This is
 * the outer iteration, implemented in SmoSolver::Solve.
 *
 * To solve QP1, we use another decomposition, specifically the SMO (q_2 = 2),
 * which is implemented in SmoBlockSolve.
 *
 * References:
 * - [1] Joachims, T. Making large-scale support vector machine learning
 *      practical. In B. Scholkopf, C. Burges, & A. Smola (Eds.), Advances in
 *      kernel methods: Support vector machines. Cambridge, MA: MIT Press (1998)
 * - [2] J. Vanek et al. A GPU-Architecture Optimized Hierarchical Decomposition
 *      Algorithm for Support VectorMachine Training, IEEE Transactions on
 *      Parallel and Distributed Systems, vol 28, no 12, 3330, (2017)
 * - [3] Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs, Journal
 *      of Machine Learning Research, 19, 1-5 (2018)
 */
template <typename math_t>
class SmoSolver {
 public:
  SmoSolver(const raft::handle_t& handle,
            SvmParameter param,
            cuvs::distance::kernels::KernelType kernel_type,
            cuvs::distance::kernels::GramMatrixBase<math_t>* kernel,
            bool is_precomputed = false)
    : handle(handle),
      C(param.C),
      tol(param.tol),
      kernel(kernel),
      kernel_type(kernel_type),
      is_precomputed(is_precomputed),
      cache_size(param.cache_size),
      nochange_steps(param.nochange_steps),
      epsilon(param.epsilon),
      svmType(param.svmType),
      stream(handle.get_stream().get()),
      return_buff(2, stream),
      alpha(0, stream),
      C_vec(0, stream),
      delta_alpha(0, stream),
      f(0, stream),
      y_label(0, stream)
  {
    ML::default_logger().set_level(param.verbosity);
  }

  void GetNonzeroDeltaAlpha(const math_t* vec,
                            int n_ws,
                            const int* idx,
                            math_t* nz_vec,
                            int* n_nz,
                            int* nz_idx,
                            cudaStream_t stream);
  /**
   * @brief Solve the quadratic optimization problem.
   *
   * The output arrays (dual_coefs, support_matrix, idx) will be allocated on the
   * device, they should be unallocated on entry.
   *
   * @param [in] matrix training vectors in matrix format(MLCommon::Matrix::Matrix),
   * size [n_rows x * n_cols]
   * @param [in] n_rows number of rows (training vectors)
   * @param [in] n_cols number of columns (features)
   * @param [in] y labels (values +/-1), size [n_rows]
   * @param [in] sample_weight device array of sample weights (or nullptr if not
   *     applicable)
   * @param [out] dual_coefs size [n_support] on exit
   * @param [out] n_support number of support vectors
   * @param [out] support_matrix support vectors in matrix format, size [n_support, n_cols]
   * @param [out] idx the original training set indices of the support vectors, size [n_support]
   * @param [out] b scalar constant for the decision function
   * @param [in] max_iter: maximum number of total iterations (default -1 for no limit)
   * @param [in] max_outer_iter maximum number of outer iteration (default 100 * n_rows)
   * @param [in] max_inner_iter maximum number of inner iterations (default 10000)
   */
  template <typename MatrixViewType>
  void Solve(MatrixViewType matrix,
             int n_rows,
             int n_cols,
             math_t* y,
             const math_t* sample_weight,
             math_t** dual_coefs,
             int* n_support,
             SupportStorage<math_t>* support_matrix,
             int** idx,
             math_t* b,
             int max_iter       = -1,
             int max_outer_iter = -1,
             int max_inner_iter = 10000);

  /**
   * @brief Update the f vector after a block solve step.
   *
   * \f[ f_i = f_i + \sum_{k\in WS} K_{i,k} * \Delta \alpha_k, \f]
   * where i = [0..n_train-1], WS is the set of workspace indices,
   * and \f$K_{i,k}\f$ is the kernel function evaluated for training vector x_i and workspace vector
   * x_k.
   *
   * @param f size [n_train]
   * @param n_rows
   * @param delta_alpha size [n_ws]
   * @param n_ws
   * @param cacheTile kernel function evaluated for the following set K[X,x_ws],
   *   size [n_rows, n_ws]
   */
  void UpdateF(math_t* f, int n_rows, const math_t* delta_alpha, int n_ws, const math_t* cacheTile);

  /** @brief Initialize the problem to solve.
   *
   * Both SVC and SVR are solved as a classification problem.
   * The optimization target (W) does not appear directly in the SMO
   * formulation, only its derivative through f (optimality indicator vector):
   * \f[ f_i = y_i \frac{\partial W }{\partial \alpha_i}. \f]
   *
   * The f_i values are initialized here, and updated at every solver iteration
   * when alpha changes. The update step is the same for SVC and SVR, only the
   * init step differs.
   *
   * Additionally, we zero init the dual coefficients (alpha), and initialize
   * class labels for SVR.
   *
   * @param[inout] y on entry class labels or target values,
   *    on exit device pointer to class labels
   * @param[in] sample_weight sample weights (can be nullptr, otherwise device
   *    array of size [n_rows])
   * @param[in] n_rows
   * @param[in] n_cols
   */
  void Initialize(math_t** y, const math_t* sample_weight, int n_rows, int n_cols);

  void InitPenalty(math_t* C_vec, const math_t* sample_weight, int n_rows);

  /** @brief Initialize Support Vector Classification
   *
   * We would like to maximize the following quantity
   * \f[ W(\mathbf{\alpha}) = -\mathbf{\alpha}^T \mathbf{1}
   *   + \frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha}, \f]
   *
   * We initialize f as:
   * \f[ f_i = y_i \frac{\partial W(\mathbf{\alpha})}{\partial \alpha_i} =
   *          -y_i +   y_j \alpha_j K(\mathbf{x}_i, \mathbf{x}_j) \f]
   *
   * @param [in] y device pointer of class labels size [n_rows]
   */
  void SvcInit(const math_t* y);

  /**
   * @brief Initializes the solver for epsilon-SVR.
   *
   * For regression we are optimizing the following quantity
   * \f[
   * W(\alpha^+, \alpha^-) =
   * \epsilon \sum_{i=1}^l (\alpha_i^+ + \alpha_i^-)
   * - \sum_{i=1}^l yc_i (\alpha_i^+ - \alpha_i^-)
   * + \frac{1}{2} \sum_{i,j=1}^l
   *   (\alpha_i^+ - \alpha_i^-)(\alpha_j^+ - \alpha_j^-) K(\bm{x}_i, \bm{x}_j)
   * \f]
   *
   * Then \f$ f_i = y_i \frac{\partial W(\alpha}{\partial \alpha_i} \f$
   *      \f$     = yc_i*epsilon - yr_i \f$
   *
   * Additionally we set class labels for the training vectors.
   *
   * References:
   * [1] B. Schölkopf et. al (1998): New support vector algorithms,
   *     NeuroCOLT2 Technical Report Series, NC2-TR-1998-031, Section 6
   * [2] A.J. Smola, B. Schölkopf (2004): A tutorial on support vector
   *     regression, Statistics and Computing 14, 199–222
   * [3] Orchel M. (2011) Support Vector Regression as a Classification Problem
   *     with a Priori Knowledge in the Form of Detractors,
   *     Man-Machine Interactions 2. Advances in Intelligent and Soft Computing,
   *     vol 103
   *
   * @param [in] yr device pointer with values for regression, size [n_rows]
   * @param [in] n_rows
   * @param [out] yc device pointer to classes associated to the dual
   *     coefficients, size [n_rows*2]
   * @param [out] f device pointer f size [n_rows*2]
   */
  void SvrInit(const math_t* yr, int n_rows, math_t* yc, math_t* f);

  int GetNIter() { return n_iter; };

 private:
  const raft::handle_t& handle;
  cudaStream_t stream;

  int n_rows  = 0;  //!< training data number of rows
  int n_cols  = 0;  //!< training data number of columns
  int n_ws    = 0;  //!< size of the working set
  int n_train = 0;  //!< number of training vectors (including duplicates for SVR)

  // Buffers for the domain [n_train]
  rmm::device_uvector<math_t> alpha;    //!< dual coordinates
  rmm::device_uvector<math_t> f;        //!< optimality indicator vector
  rmm::device_uvector<math_t> y_label;  //!< extra label for regression

  rmm::device_uvector<math_t> C_vec;  //!< penalty parameter vector

  // Buffers for the working set [n_ws]
  //! change in alpha parameter during a blocksolve step
  rmm::device_uvector<math_t> delta_alpha;

  // Buffers to return some parameters from the kernel (iteration number, and
  // convergence information)
  rmm::device_uvector<math_t> return_buff;
  math_t host_return_buff[2];

  math_t C;
  math_t tol;      //!< tolerance for stopping condition
  math_t epsilon;  //!< epsilon parameter for epsiolon-SVR

  cuvs::distance::kernels::GramMatrixBase<math_t>* kernel;
  cuvs::distance::kernels::KernelType kernel_type;
  bool is_precomputed;  //!< if true, input is a precomputed kernel matrix
  float cache_size;     //!< size of kernel cache in MiB

  SvmType svmType;  ///!< Type of the SVM problem to solve

  // Variables to track convergence of training
  math_t diff_prev;
  int n_small_diff;
  int nochange_steps;
  int n_increased_diff;
  int n_outer_iter;
  int n_iter;
  bool report_increased_diff;

  bool CheckStoppingCondition(math_t diff)
  {
    if (diff > diff_prev * 1.5 && n_outer_iter > 0) {
      // Ideally, diff should decrease monotonically. In practice we can have
      // small fluctuations (10% increase is not uncommon). Here we consider a
      // 50% increase in the diff value large enough to indicate a problem.
      // The 50% value is an educated guess that triggers the convergence debug
      // message for problematic use cases while avoids false alarms in many
      // other cases.
      n_increased_diff++;
    }
    if (report_increased_diff && n_outer_iter > 100 && n_increased_diff > n_outer_iter * 0.1) {
      CUML_LOG_DEBUG(
        "Solver is not converging monotonically. This might be caused by "
        "insufficient normalization of the feature columns. In that case "
        "MinMaxScaler((0,1)) could help. Alternatively, for nonlinear kernels, "
        "you can try to increase the gamma parameter. To limit execution time, "
        "you can also adjust the number of iterations using the max_iter "
        "parameter.");
      report_increased_diff = false;
    }
    bool keep_going = true;
    if (abs(diff - diff_prev) < 0.001 * tol) {
      n_small_diff++;
    } else {
      diff_prev    = diff;
      n_small_diff = 0;
    }
    if (n_small_diff > nochange_steps) {
      CUML_LOG_ERROR(
        "SMO error: Stopping due to unchanged diff over %d"
        " consecutive steps",
        nochange_steps);
      keep_going = false;
    }
    if (diff < tol) keep_going = false;
    if (isnan(diff)) {
      std::string txt;
      if (std::is_same<float, math_t>::value) {
        txt +=
          " This might be caused by floating point overflow. In such case using"
          " fp64 could help. Alternatively, try gamma='scale' kernel"
          " parameter.";
      }
      THROW("SMO error: NaN found during fitting.%s", txt.c_str());
    }
    return keep_going;
  }

  /// Return the number of maximum iterations.
  int GetDefaultMaxIter(int n_train, int max_outer_iter)
  {
    if (max_outer_iter == -1) {
      max_outer_iter = n_train < std::numeric_limits<int>::max() / 100
                         ? n_train * 100
                         : std::numeric_limits<int>::max();
      max_outer_iter = max(100000, max_outer_iter);
    }
    // else we have user defined iteration count which we do not change
    return max_outer_iter;
  }

  void ResizeBuffers(int n_train, int n_cols)
  {
    // This needs to know n_train, therefore it can be only called during solve
    alpha.resize(n_train, stream);
    C_vec.resize(n_train, stream);
    f.resize(n_train, stream);
    delta_alpha.resize(n_ws, stream);
    if (svmType == EPSILON_SVR) y_label.resize(n_train, stream);
  }

  void ReleaseBuffers()
  {
    alpha.resize(0, stream);
    delta_alpha.resize(0, stream);
    f.resize(0, stream);
    y_label.resize(0, stream);
  }
};

};  // end namespace SVM
};  // end namespace ML
