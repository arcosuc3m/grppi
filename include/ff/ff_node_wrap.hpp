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
#include <ff/allocator.hpp>


#ifndef FF_ALLOCATOR_IN_USE
//static ff::ff_allocator * ffalloc = 0;
#define FF_MALLOC(size)   (FFAllocator::instance()->malloc(size))
#define FF_FREE(ptr) (FFAllocator::instance()->free(ptr))
#define FF_ALLOCATOR_IN_USE 1
#endif


namespace ff {

/* Class for managing optional contained values, i.e., a value that may or
 * may not be present.
 * std::optional will be available in C++17, this is a sort of custom preview */
template <typename T>
class ff_optional {
    public:
        typedef T type;
        typedef T value_type;
        T elem;
        bool end;
        ff_optional(): end(true) { }
        ff_optional(const T& i): elem(i), end(false) { }

        /* copy assignment operator */
        ff_optional& operator=(const ff_optional& o) {
                 elem = o.elem;
                 end = o.end;
                 return *this;
        }

        T& value(){ return elem; }

        // true if object contains a value
        constexpr explicit operator bool() const {
            return !end;
        }
};

// -------------------------------------------------------------------------

// Middle stage (worker node)
template <typename TSin, typename TSout, typename L>
struct PMINode : ff_node_t<TSin,TSout> {
    L callable;

    PMINode(L&& lf) : callable(lf) {};

    TSout * svc(TSin *t) {
    	std::experimental::optional<TSin> check;
        TSout * out = new TSout();
        check = *t;

        if(check)
        	*out = std::move(callable(check.value()));

        return out;
    }
};


// First stage
template <typename TSout, typename L >
struct PMINode<void,TSout,L> : ff_node {
    L callable;

    PMINode(L&& lf) : callable(lf) {};

    void * svc(void *) {
    	std::experimental::optional<TSout> ret;
        void *outslot = std::malloc(sizeof(TSout));
        TSout *out = new (outslot) TSout();

        ret = std::move(callable());

        if(ret) {
        	*out = ret.value();
        	return outslot;
        } else return {};	// No GO_ON
    }
};


// Last stage
template <typename TSin, typename L >
struct PMINode<TSin,void,L> : ff_node_t<TSin,void> {
	L callable;

    PMINode(L&& lf) : callable(std::move(lf)) {};

    void * svc(TSin *t) {
    	std::experimental::optional<TSin> check;
    	TSin *item = (TSin*) t;
        check = *item;

        if(check)
        	callable(check.value());

        return GO_ON;
    }
};


// Filter - cond is the filtering function
// Note that according to grPPI interface,
// basic filter keeps matching items
template <typename TSin, typename L>
struct PMINodeFilter : ff_node_t<TSin> {
    L cond;

    PMINodeFilter(L&& c) : cond(c) {};

    TSin * svc(TSin *t) {
    	std::experimental::optional<TSin> check;
        check = *t;

        if(check) {
        	if ( cond(check.value()) ) {
        		return t;
        	} else return {};
        } else return {};
    }
};


} // namespace ff


#endif // FF_NODE_WRAP_HPP
