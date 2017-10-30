/*
 * ff_helper_classes.hpp
 *
 *  Created on: 18 Oct 2017
 *      Author: fabio
 */

#ifndef FF_HELPER_CLASSES_HPP
#define FF_HELPER_CLASSES_HPP

#include <iostream>


#include "../common/iterator.h"
#include "../common/execution_traits.h"
#include "../common/patterns.h"
#include "../common/reduce_pattern.h"

#include <experimental/optional>
#include <type_traits>

#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/allocator.hpp>


namespace ff {

// helper classes for streming patterns that require custom emitters and
// custom ordered collectors

// ----- STREAM-REDUCE

template <typename TSin, typename Reducer>
class ff_StreamReduce_grPPI : public ff_farm<ofarm_lb,ofarm_gt> {

public:
	ff_StreamReduce_grPPI(Reducer && red_obj, unsigned int wrks=1) :
		ff_farm<ofarm_lb,ofarm_gt>(false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, false, wrks),
		conc_degr(wrks) {
		for(int i=0; i<wrks; ++i)
			workers.push_back( new ReduceWorker<TSin,Reducer>(std::forward<Reducer>(red_obj)) );

		em = new ReduceEmitter<TSin,Reducer>(red_obj.getWinSize(), red_obj.getOffset());
		cl = new ReduceCollector<TSin>(this->getgt());
		this->add_workers(workers);
		this->add_emitter(em);
		this->add_collector(cl);
	}

	~ff_StreamReduce_grPPI() {
		delete em;
		delete cl;
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
			_window(win_size), _offset(offset), nextone(0), skip(0) {
			win_items.reserve(win_size);
		}

		void *svc(void *t) {
			std::experimental::optional<InType> check;
			InType *item = (InType*) t;
			check = *item;

			if(!check) return EOS;

			if(win_items.size() != _window)
				win_items.push_back( std::forward<InType>(check.value()) );

			if(win_items.size() == _window) {
				if(_offset <= _window) {
					this->ff_send_out( new reduce_task_t<InType>(win_items) );
					win_items.erase(win_items.begin(), std::next(win_items.begin(), _offset));
					return GO_ON;
				} else {
					if(skip == 0) {
						this->ff_send_out( new reduce_task_t<InType>(win_items) );
						skip++;
					} else if(skip == _offset) {
						skip = 0;
						win_items.clear();
						win_items.push_back( std::forward<InType>(check.value()) );
					} else skip++;
					return GO_ON;
				}
			} else return GO_ON;
		}

	private:
		// Class variables
		int _window;
		int _offset;
		size_t nextone;
		int skip;
		std::vector<InType> win_items;
	};

	// -- stream-reduce workers
	template<typename InType, typename RedObj>
	struct ReduceWorker : ff_node {

		ReduceWorker(RedObj&& red_t) :
			_reduction_obj(std::move(red_t)) { }

		void *svc(void *t) {
			reduce_task_t<InType> * task = (reduce_task_t<InType> *) t;
			grppi::sequential_execution seq{};

			void *outslot = ff_malloc(sizeof(InType));
			InType *result = new (outslot) InType();

			_reduction_obj.add_items(task->vals);
			if(_reduction_obj.reduction_needed())
				*result = _reduction_obj.reduce_window(seq);

			delete task;
			return outslot;
		}

	private:
		// Class variables
		RedObj&& _reduction_obj;

	};

	// simple ordered collector for stream-reduce pattern
	template <typename InType>
	struct ReduceCollector : ff_node_t<InType> {
		ReduceCollector(ofarm_gt * const cl) :
			nextone(0), gt(cl) { }

		void donextone() {
			do nextone = (nextone+1) % gt->getrunning();
			while(!gt->set_victim(nextone));
		}

		int svc_init() {
			assert(gt->getrunning()>0);
			gt->revive();
			gt->set_victim(nextone);
			return 0;
		}

		InType * svc(InType * t) {
			std::experimental::optional<InType> check;
			check = *t;

			if(check)
				this->ff_send_out(t);
			else {
				t->~InType();
				ff_free(t);
			}
			donextone();
			return this->GO_ON;
		}

		void eosnotify(ssize_t id=-1) {
			gt->set_dead(id);
			if (nextone == (size_t)id) {
				nextone = (nextone+1) % gt->getrunning();
				gt->set_victim(nextone);
			}
		}

		void svc_end() { }

	private:
		size_t nextone;
		ofarm_gt *gt;
	};


private:
	size_t conc_degr;
	std::vector<ff_node*> workers;

	ReduceEmitter<TSin,Reducer> *em;
	ReduceCollector<TSin> *cl;
};


// ----- STREAM-FILTER

template <typename TSin, typename Filter>
class ff_StreamFilter_grPPI : public ff_farm<ofarm_lb,ofarm_gt> {

public:
	ff_StreamFilter_grPPI(Filter&& pre, unsigned int wrks=1):
		ff_farm<ofarm_lb,ofarm_gt>(false, DEF_IN_BUFF_ENTRIES, DEF_OUT_BUFF_ENTRIES, true, wrks),
		_predicate(pre), conc_degr(wrks) {
		for(int i=0;i<conc_degr;i++)
			workers.push_back(new FilterWorker<TSin,Filter>( std::forward<Filter>(_predicate)) );

		this->add_workers(workers);
		oc = new FilterCollector<TSin>(this->getgt());
		this->add_collector(oc);
	}

	~ff_StreamFilter_grPPI() {
		delete oc;
	}

private:

	template <typename InType, typename L>
	struct FilterWorker : ff_node_t<InType> {

		FilterWorker(L&& c) : condition(c) { };

		InType * svc(InType *t) {
			std::experimental::optional<InType> check;
			check = *t;

			if(check) {
				if ( condition(check.value()) ) {
					return t;
				} else {
					t->~InType();
					ff_free(t);
				}
			}
			return this->GO_ON;;
		}

	private:
		L condition;
	};

	template <typename InType>
	struct FilterCollector : ff_node_t<InType> {

		FilterCollector(ofarm_gt * const gt) :
			nextone(0), gt(gt) { }

		void donextone() {
			do nextone = (nextone+1) % gt->getrunning();
			while(!gt->set_victim(nextone));
		}

		int svc_init() {
			assert(gt->getrunning()>0);
			gt->revive();
			gt->set_victim(nextone);
			return 0;
		}

		InType *svc(InType * t) {
			std::experimental::optional<InType> check;
			check = *t;

			if(check) {
				if( t != (InType*)FILTERED )
					this->ff_send_out(t);
			}
			donextone();
			return this->GO_ON;
		}

		void eosnotify(ssize_t id=-1) {
			gt->set_dead(id);
			if (nextone == (size_t)id) {
				nextone = (nextone+1) % gt->getrunning();
				gt->set_victim(nextone);
			}
		}

		void svc_end() { }


	private:
		size_t nextone;
		ofarm_gt  * gt;
	};

private:
	Filter _predicate;
	unsigned int conc_degr;
	std::vector<ff_node *> workers;

	FilterCollector<TSin> *oc;
	static constexpr size_t FILTERED = (FF_EOS-0x7);

};



} // namespace ff





#endif /* FF_HELPER_CLASSES_HPP */
