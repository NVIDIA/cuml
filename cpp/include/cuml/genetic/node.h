/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>

#include <cstdint>
#include <string>

namespace CUML_EXPORT cuml {
namespace genetic {

/**
 * @brief Represents a node in the syntax tree.
 *
 * @code{.cpp}
 * // A non-terminal (aka function) node
 * node func_node{node::type::sub};
 * // A constant node
 * float const_value = 2.f;
 * node const_node{const_value};
 * // A variable (aka feature) node
 * node var_node{20};
 * @endcode
 */
struct node {
  /**
   * @brief All possible types of nodes. For simplicity, all the terminal and
   *        non-terminal types are clubbed together
   */
  enum class type : uint32_t {
    variable = 0,
    constant,

    // note: keep the case statements in alphabetical order under each category
    // of operators.
    functions_begin,
    // different binary function types follow
    binary_begin = functions_begin,
    add          = binary_begin,
    atan2,
    div,
    fdim,
    max,
    min,
    mul,
    pow,
    sub,
    binary_end = sub,  // keep this to be the last binary function in the list
    // different unary function types follow
    unary_begin,
    abs = unary_begin,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cbrt,
    cos,
    cosh,
    cube,
    exp,
    inv,
    log,
    neg,
    rcbrt,
    rsqrt,
    sin,
    sinh,
    sq,
    sqrt,
    tan,
    tanh,
    unary_end     = tanh,  // keep this to be the last unary function in the list
    functions_end = unary_end,
  };  // enum type

  /**
   * @brief Default constructor for node
   */
  explicit node();

  /**
   * @brief Construct a function node
   *
   * @param[in] ft function type
   */
  explicit node(type ft);

  /**
   * @brief Construct a variable node
   *
   * @param[in] fid feature id that represents the variable
   */
  explicit node(int fid);

  /**
   * @brief Construct a constant node
   *
   * @param[in] val constant value
   */
  explicit node(float val);

  /**
   * @param[in] src source node to be copied
   */
  explicit node(const node& src);

  /**
   * @brief assignment operator
   *
   * @param[in] src source node to be copied
   *
   * @return current node reference
   */
  node& operator=(const node& src);

  /** whether the current is either a variable or a constant */
  bool is_terminal() const;

  /** whether the current node is a function */
  bool is_nonterminal() const;

  /** Get the arity of the node. If it is a terminal, then a 0 is returned */
  int arity() const;

  /**
   * @brief Helper method to get node type from input string
   *
   * @param[in] ntype node type in string. Possible strings correlate one-to-one
   *                  with the enum values for `type`
   *
   * @return `type`
   */
  static type from_str(const std::string& ntype);

  /** constant used to represent invalid feature id */
  static const int kInvalidFeatureId;

  /** node type */
  type t;
  union data {
    /**
     * if the node is `variable` type, then this is the column id to be used to
     * fetch its value, from the input dataset
     */
    int fid;
    /** if the node is `constant` type, then this is the value of the node */
    float val;
  } u;
};  // struct node

}  // namespace genetic
}  // namespace CUML_EXPORT cuml
