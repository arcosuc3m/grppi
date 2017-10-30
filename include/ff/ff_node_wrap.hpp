/*
 * ff_node_wrap.hpp
 *
 * Created on: 24 Aug 2017
 *      Author: fabio
 *
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
struct PMINode : ff_node_t<TSin,TSout> {
    L callable;

    PMINode(L&& lf) : callable(lf) {};

    TSout *svc(TSin *t) {
    	std::experimental::optional<TSin> check;
        TSout *out = (TSout*) ff::ff_malloc(sizeof(TSout));
        check = *t;

        if(check)
        	*out = callable(check.value());

        return out;
    }
};

// First stage
template <typename TSout, typename L >
struct PMINode<void,TSout,L> : ff_node {
    L callable;

    PMINode(L&& lf) : callable(lf) {};

    void *svc(void *) {
    	std::experimental::optional<TSout> ret;
        void *outslot = ff::ff_malloc(sizeof(TSout));
        TSout *out = new (outslot) TSout();

        ret = callable();

        if(ret) {
        	*out = ret.value();
        	return outslot;
        } else {
        	ff::ff_free(outslot);
        	return {};
        }
    }
};

// Last stage
template <typename TSin, typename L >
struct PMINode<TSin,void,L> : ff_node_t<TSin,void> {
	L callable;

    PMINode(L&& lf) : callable(lf) {};

    void *svc(TSin *t) {
    	std::experimental::optional<TSin> check;
        check = *t;

        if(check) {
        	callable(check.value());

        	t->~TSin();
        	ff::ff_free(t);
        }
        return GO_ON;
    }
};

// Middle stage - Filter for unordered farm
template <typename TSin, typename L>
struct PMINodeFilter : ff_node_t<TSin> {
    L condition;

    PMINodeFilter(L&& c) : condition(c) {};

    TSin *svc(TSin *t) {
    	std::experimental::optional<TSin> check;
        check = *t;

        if(check) {
        	if ( condition(check.value()) ) {
        		return t;
        	} else {
        		t->~TSin();
        		ff::ff_free(t);
        	}
        }
        return this->GO_ON;
    }
};


} // namespace ff


#endif // FF_NODE_WRAP_HPP
