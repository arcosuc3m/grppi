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


//#ifndef FF_ALLOCATOR_IN_USE
////static ff::ff_allocator * ffalloc = 0;
//#define FF_MALLOC(size)   (FFAllocator::instance()->malloc(size))
//#define FF_FREE(ptr) (FFAllocator::instance()->free(ptr))
//#define FF_ALLOCATOR_IN_USE 1
//#endif


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
        TSout * out = (TSout*) ff::ff_malloc(sizeof(TSout));
        check = *t;

        if(check)
        	*out = callable(check.value());

        return out;
    }
};

// First stage
template <typename TSout, typename L >
struct PMINode<void,TSout,L> : ff_node_t<TSout,TSout> {
    L callable;

    PMINode(L&& lf) : callable(lf) {};

    TSout * svc(TSout *) {
    	std::experimental::optional<TSout> ret;
        void *outslot = ff::ff_malloc(sizeof(TSout));
        TSout *out = new (outslot) TSout();

        ret = callable();

        if(ret) {
        	*out = ret.value();
        	return out;
        } else {
        	ff::ff_free(outslot);
        	return {};	// No GO_ON
        }
    }
};

// Last stage
template <typename TSin, typename L >
struct PMINode<TSin,void,L> : ff_node_t<TSin,void> {
	L callable;

    PMINode(L&& lf) : callable(lf) {};

    void * svc(TSin *t) {
    	std::experimental::optional<TSin> check;
    	TSin *item = (TSin*) t;
        check = *item;

        if(check) {
        	callable(check.value());
        	item->~TSin();
        	ff::ff_free(item);
        	return GO_ON;
        }

        return GO_ON; // {} -- problems when returning an empty object at this stage
    }
};

// tag sent by workers when an element is filtered out
//static constexpr size_t FILTERED = (FF_EOS-0x11);

// Middle stage - Filter
template <typename TSin, typename L>
struct PMINodeFilter : ff_node_t<TSin> {
    L condition;

    PMINodeFilter(L&& c) : condition(c) {};

    TSin * svc(TSin *t) {
    	std::experimental::optional<TSin> check;
        check = *t;

        if(check) {
        	if ( condition(check.value()) ) {
        		return t;
        	} else {
        		t->~TSin();
        		ff::ff_free(t);
        		return (TSin*) GO_ON; // GO_ON -- apparently ofarmC skips GO_OUT tags
        	}
        } else return nullptr;
    }
};


} // namespace ff


#endif // FF_NODE_WRAP_HPP
