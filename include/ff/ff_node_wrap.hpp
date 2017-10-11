/*
 * Author: Marco Aldinucci, University of Torino
 * Date: August 4, 2016
 *
 * TODO: change to typed ff_node to ensure a better static type checking at the interface level
 */


#ifndef FF_NODE_WRAP_HPP
#define FF_NODE_WRAP_HPP

#include <experimental/optional>
#include <type_traits>

#include <ff/node.hpp>
#include <ff/allocator.hpp>

//#include "../common/common.hpp"

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
struct PMINode : ff_node {
    L callable;

    PMINode(L const &lf) : callable(lf) {};

    inline void * svc(void *t) {
    	std::experimental::optional<TSin> check;
        void * outslot = FF_MALLOC(sizeof(TSout));
        TSout * out = new (outslot) TSout();
        TSin * input_item = (TSin *) t;
        check = *input_item;

        if(check) {
        	*out = std::move(callable(check.value()));
        	input_item->~TSin();
        	FF_FREE(input_item);
        }

        return outslot;
    }
};


// First stage
template <typename TSout, typename L >
struct PMINode<void,TSout,L> : ff_node {
    L callable;

    PMINode(L const &lf) : callable(lf) {};

    inline void * svc(void *) {
    	std::experimental::optional<TSout> ret;
        void *outslot = FF_MALLOC(sizeof(TSout));
        TSout *out = new (outslot) TSout();

        ret = std::move(callable());

        // if(ret.has_value()) // c++17 only - use bool operator
        if(ret) {
        	*out = ret.value();
        	return outslot;
        } else
        	return EOS;	// No GO_ON
    }
};


// Last stage
template <typename TSin, typename L >
struct PMINode<TSin,void,L> : ff_node {
    L callable;

    PMINode(L const &lf) : callable(lf) {};

    inline void * svc(void *t) {
    	std::experimental::optional<TSin> check;
        TSin * input_item = (TSin *) t;
        check = *input_item;

        if(check) {
        	callable(check.value());
        	input_item->~TSin();
        	FF_FREE(input_item);
        }
        return GO_ON;
    }
};


// Filter - cond is the filtering function
// Note that according to grPPI interface,
// basic filter keeps matching items
template <typename TSin, typename L>
struct PMINodeFilter : ff_node {
    L cond;

    PMINodeFilter (L const &c) : cond(c) {};

    inline void * svc(void *t) {
    	std::experimental::optional<TSin> check;
    	void * outslot = GO_ON;
        TSin * input_item = (TSin *) t;
        check = *input_item;

        if(check) {
        	if ( cond(check.value()) ) {
        		outslot = t;
        	} else {
        		input_item->~TSin();
        		FF_FREE(input_item);
        	}
        }
        return outslot;
    }
};


} // namespace ff


#endif // FF_NODE_WRAP_HPP
