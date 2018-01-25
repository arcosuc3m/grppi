/**
 * @version		GrPPI v0.3
 * @copyright	Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
 * @license		GNU/GPL, see LICENSE.txt
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You have received a copy of the GNU General Public License in LICENSE.txt
 * also available in <http://www.gnu.org/licenses/gpl.html>.
 *
 * See COPYRIGHT.txt for copyright notices and details.
 */

#ifndef FF_HELPER_CLASSES_HPP
#define FF_HELPER_CLASSES_HPP

#include "../../common/iterator.h"
#include "../../common/execution_traits.h"
#include "../../common/patterns.h"
#include "../../common/reduce_pattern.h"
#include "reduce.h"

#include <numeric>
#include <experimental/optional>
#include <type_traits>

#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/allocator.hpp>


namespace ff {

// helper classes for streaming patterns that require custom emitters and
// custom ordered collectors

// ----- ORDERED STREAM-REDUCE
template <typename TSin, typename Reducer, typename CombineOp>
class ff_OStreamReduce_grPPI : public ff_ofarm {

public:
	ff_OStreamReduce_grPPI(Reducer && red_obj, unsigned int wrks=1) :
		ff_ofarm(false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, wrks),
		conc_degr(wrks) {
		for(int i=0; i<wrks; ++i)
			workers.push_back( new ReduceWorker<TSin,CombineOp>(
					std::forward<CombineOp>(red_obj.get_combiner()))
			);

		em = new ReduceEmitter<TSin,Reducer>(red_obj.get_window_size(), red_obj.get_offset());
		this->add_workers(workers);
		this->setEmitterF(em);
	}

	~ff_OStreamReduce_grPPI() {
		delete em;
	}

private:
	// workers' actual task
	template<typename T>
	struct reduce_task_t {
		reduce_task_t(const std::vector<T>& v) {
			vals.reserve(v.size());
			vals = v;
		}
		~reduce_task_t() {
			vals.clear();
		}
		std::vector<T> vals;
	};

	// -- emitter for stream-reduce pattern
	template<typename InType, typename RedObj>
	struct ReduceEmitter : ff_node {

		ReduceEmitter(int win_size, int offset) :
			_window(win_size), _offset(offset), skip(-1) {
			win_items.reserve(win_size);
		}

		void *svc(void *t) {
			InType *item = (InType*) t;

			if(win_items.size() != _window)
				win_items.push_back( std::forward<InType>(*item) );

			if(win_items.size() == _window) {
				if(_offset < _window) {
					this->ff_send_out( new reduce_task_t<InType>(win_items) );
					win_items.erase(win_items.begin(), std::next(win_items.begin(), _offset));
					ff_free(item);
					return GO_ON;
				}
				if (_offset == _window) {
					this->ff_send_out( new reduce_task_t<InType>(win_items) );
					win_items.erase(win_items.begin(), win_items.end());
					ff_free(item);
					return GO_ON;
				} else {
					if(skip == -1) {
						this->ff_send_out( new reduce_task_t<InType>(win_items) );
						skip++;
					} else if(skip == (_offset-_window)) {
						skip = -1;
						win_items.clear();
						win_items.push_back( std::forward<InType>(*item) );
					} else skip++;
					ff_free(item);
					return GO_ON;
				}
			} else {
				ff_free(item);
				return GO_ON;
			}
		}

	private:
		int _window;
		int _offset;
		int skip;
		std::vector<InType> win_items;
	};

	// -- stream-reduce workers
	template<typename InType, typename Combiner>
	struct ReduceWorker : ff_node {

		ReduceWorker(Combiner&& red_t) :
			_combine_op(std::move(red_t)) { }

		void *svc(void *t) {
			reduce_task_t<InType> * task = (reduce_task_t<InType> *) t;
			grppi::sequential_execution seq{};
			void *outslot = ff_malloc(sizeof(InType));
			InType *result = new (outslot) InType();
			InType identity_{}; // initialize

			*result = grppi::reduce(seq, task->vals.begin(), task->vals.end(),
					identity_, std::forward<Combiner>(_combine_op));

			delete task;
			return outslot;
		}

	private:
		Combiner&& _combine_op;
	};

private:
	size_t conc_degr;
	std::vector<ff_node*> workers;

	ReduceEmitter<TSin,Reducer> *em;
};


// ----- UNORDERED STREAM-REDUCE
template <typename TSin, typename Reducer, typename CombineOp>
class ff_StreamReduce_grPPI : public ff_farm<> {

public:
	ff_StreamReduce_grPPI(Reducer && red_obj, unsigned int wrks=1) :
		ff_farm<>(false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, wrks),
		conc_degr(wrks) {
		for(int i=0; i<wrks; ++i)
			workers.push_back( new ReduceWorker<TSin,CombineOp>(
					std::forward<CombineOp>(red_obj.get_combiner()))
			);

		em = new ReduceEmitter<TSin,Reducer>(red_obj.get_window_size(), red_obj.get_offset());
		cl = new ReduceCollector();
		this->add_workers(workers);
		this->add_emitter(em);
		this->add_collector(cl);
	}

	~ff_StreamReduce_grPPI() {
		delete em; delete cl;
	}

private:
	// workers' actual task
	template<typename T>
	struct reduce_task_t {
		reduce_task_t(const std::vector<T>& v) {
			vals.reserve(v.size());
			vals = v;
		}
		~reduce_task_t() {
			vals.clear();
		}
		std::vector<T> vals;
	};

	// -- emitter for stream-reduce pattern
	template<typename InType, typename RedObj>
	struct ReduceEmitter : ff_node {

		ReduceEmitter(int win_size, int offset) :
			_window(win_size), _offset(offset), nextone(0), skip(-1) {
			win_items.reserve(win_size);
		}

		void *svc(void *t) {
			InType *item = (InType*) t;

			if(win_items.size() != _window)
				win_items.push_back( std::forward<InType>(*item) );

			if(win_items.size() == _window) {
				if(_offset < _window) {
					this->ff_send_out( new reduce_task_t<InType>(win_items) );
					win_items.erase(win_items.begin(), std::next(win_items.begin(), _offset));
					ff_free(item);
					return GO_ON;
				}
				if (_offset == _window) {
					this->ff_send_out( new reduce_task_t<InType>(win_items) );
					win_items.erase(win_items.begin(), win_items.end());
					ff_free(item);
					return GO_ON;
				} else {
					if(skip == -1) {
						this->ff_send_out( new reduce_task_t<InType>(win_items) );
						skip++;
					} else if(skip == (_offset-_window)) {
						skip = -1;
						win_items.clear();
						win_items.push_back( std::forward<InType>(*item) );
					} else skip++;
					ff_free(item);
					return GO_ON;
				}
			} else {
				ff_free(item);
				return GO_ON;
			}
		}

	private:
		int _window;
		int _offset;
		size_t nextone;
		int skip;
		std::vector<InType> win_items;
	};

	// -- stream-reduce workers
	template<typename InType, typename Combiner>
	struct ReduceWorker : ff_node {

		ReduceWorker(Combiner&& red_t) :
			_combine_op(std::move(red_t)) { }

		void *svc(void *t) {
			reduce_task_t<InType> * task = (reduce_task_t<InType> *) t;
			grppi::sequential_execution seq{};

			void *outslot = ff_malloc(sizeof(InType));
			InType *result = new (outslot) InType();
			InType identity_{}; // initialize

			*result = grppi::reduce(seq, task->vals.begin(), task->vals.end(),
					identity_, std::forward<Combiner>(_combine_op));

			delete task;
			return outslot;
		}

	private:
		Combiner&& _combine_op;
	};

	// dummy collector needed by ff_farm. only forwards tasks
	struct ReduceCollector : ff_node {
		ReduceCollector() {}
		void * svc(void *t) {
			return t;
		}
	};

private:
	size_t conc_degr;
	std::vector<ff_node*> workers;

	ReduceEmitter<TSin,Reducer> *em;
	ReduceCollector *cl;
};


// ----- ORDERED STREAM-FILTER
template <typename TSin, typename Filter>
class ff_OStreamFilter_grPPI : public ff_ofarm {

public:
	ff_OStreamFilter_grPPI(Filter&& pre, unsigned int wrks=1):
		ff_ofarm(false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, wrks),
		_predicate(pre), conc_degr(wrks) {
		for(int i=0;i<conc_degr;i++)
			workers.push_back(new FilterWorker<TSin,Filter>( std::forward<Filter>(_predicate)) );

		this->add_workers(workers);
		oc = new FilterCollector<TSin>();
		this->setCollectorF(oc);
	}

	~ff_OStreamFilter_grPPI() {
		delete oc;
	}

private:
	template <typename InType, typename Predicate>
	struct FilterWorker : ff_node_t<InType> {
		FilterWorker(Predicate&& c) : condition(c) { };
		InType * svc(InType *t) {
			if ( condition(*t) ) return t;
			else {
				t->~InType();
				ff_free(t);
				return (InType*)FILTERED;
			}
		}

	private:
		Predicate condition;
	};

	template <typename InType>
	struct FilterCollector : ff_node_t<InType> {
		FilterCollector() { }
		InType *svc(InType * t) {
			if( t == (InType*)FILTERED )
				return this->GO_ON;
			return t;
		}
	};

private:
	Filter _predicate;
	unsigned int conc_degr;
	std::vector<ff_node *> workers;

	FilterCollector<TSin> *oc;
	static constexpr size_t FILTERED = (FF_EOS-0x11);
};


// ----- UNORDERED STREAM-FILTER
template <typename TSin, typename Filter>
class ff_StreamFilter_grPPI : public ff_farm<> {

public:
	ff_StreamFilter_grPPI(Filter&& pre, unsigned int wrks=1):
		ff_farm<>(false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, wrks),
		_predicate(pre), conc_degr(wrks) {
		for(int i=0;i<conc_degr;i++)
			workers.push_back(new FilterWorker<TSin,Filter>( std::forward<Filter>(_predicate)) );

		this->add_workers(workers);
		em = new FilterEmitter<TSin>();
		oc = new FilterCollector<TSin>();
		this->add_emitter(em);
		this->add_collector(oc);
	}

	~ff_StreamFilter_grPPI() {
		delete em; delete oc;
	}

private:
	// dummy emitter needed by ff_farm. only forwards tasks
	template <typename InType>
	struct FilterEmitter : ff_node_t<InType> {
		FilterEmitter() {}
		InType * svc(InType *t) {
			return t;
		}
	};

	template <typename InType, typename Predicate>
	struct FilterWorker : ff_node_t<InType> {
		FilterWorker(Predicate&& c) : condition(c) { };
		InType * svc(InType *t) {
			if ( condition(*t) ) return t;
			else {
				t->~InType();
				ff_free(t);
				return (InType*)FILTERED;
			}
		}

	private:
		Predicate condition;
	};

	template <typename InType>
	struct FilterCollector : ff_node_t<InType> {
		FilterCollector() { }
		InType *svc(InType * t) {
			if( t == (InType*)FILTERED )
				return this->GO_ON;
			return t;
		}
	};

private:
	Filter _predicate;
	unsigned int conc_degr;
	std::vector<ff_node *> workers;

	FilterEmitter<TSin> *em;
	FilterCollector<TSin> *oc;
	static constexpr size_t FILTERED = (FF_EOS-0x11);
};

} // namespace ff


#endif /* FF_HELPER_CLASSES_HPP */

