/*
 * parallel_execution_ff.h
 *
 *  Created on: Aug 3, 2017
 *      Author: fabio
 */

#ifndef GRPPI_FF_PARALLEL_EXECUTION_FF_H_
#define GRPPI_FF_PARALLEL_EXECUTION_FF_H_

#ifdef GRPPI_OMP

#include "../common/mpmc_queue.h"

#include <type_traits>

namespace grppi {

/**
\brief FastFlow (FF) parallel execution policy.

This policy uses FastFlow as implementation back-end.
*/
class parallel_execution_ff {

	  /**
	  \brief Default construct a FF parallel execution policy.

	  Creates a FF parallel execution object.

	  The concurrency degree is determined by the platform.

	  */
	  parallel_execution_ff() noexcept :
	      parallel_execution_ff{default_concurrency_degree}
	  {}

	  /**
	  \brief Constructs a FF parallel execution policy.

	  Creates a FF parallel execution object selecting the concurrency degree.

	  \param concurrency_degree Number of threads used for parallel algorithms.
	  \param order Whether ordered executions is enabled or disabled.
	  */
	  parallel_execution_ff(int concurrency_degree, bool order = true) noexcept :
	      concurrency_degree_{concurrency_degree},
	      ordering_{order}
	  {}

	  /**
	  \brief Set number of grppi threads.
	  */
	  void set_concurrency_degree(int degree) noexcept { concurrency_degree_ = degree; }

	  /**
	  \brief Get number of grppi trheads.
	  */
	  int concurrency_degree() const noexcept { return concurrency_degree_; }

	  /**
	  \brief Enable ordering.
	  */
	  void enable_ordering() noexcept { ordering_= true; }

	  /**
	  \brief Disable ordering.
	  */
	  void disable_ordering() noexcept { ordering_= false; }

	  /**
	  \brief Is execution ordered.
	  */
	  bool is_ordered() const noexcept { return ordering_; }

	  /**
	  \brief Sets the attributes for the queues built through make_queue<T>()
	  */
	  void set_queue_attributes(int size, queue_mode mode, int tokens) noexcept {
	    queue_size_ = size;
	    queue_mode_ = mode;
	    num_tokens_ = tokens;
	  }

	  /**
	  \brief Makes a communication queue for elements of type T.
	  Constructs a queue using the attributes that can be set via
	  set_queue_attributes(). The value is returned via move semantics.
	  */
	  template <typename T>
	  mpmc_queue<T> make_queue() const {
	    return {queue_size_, queue_mode_};
	  }

	  /**
	  \brien Get num of tokens.
	  */
	  int tokens() const noexcept { return num_tokens_; }

	private:

	  constexpr static int default_concurrency_degree = 4;
	  int concurrency_degree_ = default_concurrency_degree;

	  bool ordering_ = true;

	  //constexpr static int default_queue_size = 100;
	  //int queue_size_ = default_queue_size;

	  //constexpr static int default_num_tokens_ = 100;
	  //int num_tokens_ = default_num_tokens_;

	  //queue_mode queue_mode_ = queue_mode::blocking;
	};

	/**
	\brief Metafunction that determines if type E is parallel_execution_ff
	\tparam Execution policy type.
	*/
	template <typename E>
	constexpr bool is_parallel_execution_ff() {
	  return std::is_same<E, parallel_execution_ff>::value;
	}

	/**
	\brief Metafunction that determines if type E is supported in the current build.
	\tparam Execution policy type.
	*/
	template <typename E>
	constexpr bool is_supported();

	/**
	\brief Specialization stating that parallel_execution_ff is supported.
	This metafunction evaluates to false if GRPPI_FF is enabled.
	*/
	template <>
	constexpr bool is_supported<parallel_execution_ff>() {
	  return true;
	}


};

} // end grppi namespace

#else // GRPPI_FF not defined

namespace grppi {


/// Parallel execution policy.
/// Empty type if GRPPI_FF disabled.
struct parallel_execution_ff {};

/**
\brief Metafunction that determines if type E is parallel_execution_ff
This metafunction evaluates to false if GRPPI_FF is disabled.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_parallel_execution_ff() {
  return false;
}

/**
\brief Metafunction that determines if type E is supported in the current build.
\tparam Execution policy type.
*/
template <typename E>
constexpr bool is_supported();

/**
\brief Specialization stating that parallel_execution_native is supported.
*/
template <>
constexpr bool is_supported<parallel_execution_ff>() {
  return false;
}

} // end namespace grppi

#endif



#endif /* GRPPI_FF_PARALLEL_EXECUTION_FF_H_ */
