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


#ifndef FF_NODE_WRAP_HPP
#define FF_NODE_WRAP_HPP

#include <experimental/optional>
#include <type_traits>
#include <functional>

#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/allocator.hpp>

namespace ff {

// Middle stage (worker node)
template <typename TSin, typename TSout, typename L>
struct FFNode : ff_node_t<TSin,TSout> {
	L callable;
	FFNode(L&& lf) : callable(lf) {};
	TSout *svc(TSin *t) {
		TSout *out = (TSout*) ff_malloc(sizeof(TSout));
		*out = callable(*t);
		return out;
	}
};

// First stage
template <typename TSout, typename L >
struct FFNode<void,TSout,L> : ff_node {
	L callable;
	FFNode(L&& lf) : callable(lf) {};
	void *svc(void *) {
		std::experimental::optional<TSout> ret;
		void *outslot = ff_malloc(sizeof(TSout));
		TSout *out = new (outslot) TSout();
		ret = callable();
		if(ret) {
			*out = ret.value();
			return outslot;
		} else {
			ff_free(outslot);
			return EOS;
		}
	}
};

// Last stage
template <typename TSin, typename L >
struct FFNode<TSin,void,L> : ff_node_t<TSin,void> {
	L callable;
	FFNode(L&& lf) : callable(lf) {};
	void *svc(TSin *t) {
		callable(*t);
		t->~TSin();
		ff_free(t);
		return GO_ON;
	}
};

# if 0
// Middle stage - Filter for unordered farm
template <typename TSin, typename L>
struct FFNodeFilter : ff_node_t<TSin> {
	L condition;

	FFNodeFilter(L&& c) : condition(c) {};

	TSin *svc(TSin *t) {
		if ( condition(*t) ) {
			return t;
		} else {
			t->~TSin();
			ff_free(t);
		}
		return this->GO_ON;
	}
};
#endif

// ----- STREAM-ITERATION (workers)
template<typename TSin, typename Iterator>
struct IterationWorker : ff_node_t<TSin> {
	IterationWorker(Iterator&& iter_t) :
		_iterator_obj(iter_t) { }
	TSin *svc(TSin *t) {
		TSin *item = (TSin*) t;
		do *item = _iterator_obj.transform(std::forward<TSin>(*item));
		while (!_iterator_obj.predicate(std::forward<TSin>(*item)));
		return item;
	}
	// class variable
	Iterator _iterator_obj;
};

} // namespace ff


#endif // FF_NODE_WRAP_HPP

