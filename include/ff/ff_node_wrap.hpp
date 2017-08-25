/*
 * Author: Marco Aldinucci, University of Torino
 * Date: August 4, 2016
 *
 * TODO: change to typed ff_node to ensure a better static type checking at the interface level
 */


#ifndef FF_NODE_WRAP_HPP
#define FF_NODE_WRAP_HPP

#include <ff/node.hpp>
#include <ff/allocator.hpp>

#include <experimental/optional>
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
 * TODO: Discuss actual need of this - note that optional is not supported even by c++14 and GNU 6.3
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

        constexpr explicit operator bool() const {
            return !end;
        }
};

// -------------------------------------------------------------------------

static bool outer_ff_pattern = true;

// Here it is assumed that result is an object with value_type public field

// Middle stage (worker node)
template <typename TSin, typename TSout, typename L>
struct PMINode : ff_node {
    L callable;
    PMINode(L const &lf,bool streamitem=true):callable(lf) {};
    inline void * svc(void *t) {
        //typedef typename TSin::value_type in_type;
        void * outslot = FF_MALLOC(sizeof(TSout));
        TSout * out = new (outslot) TSout();
        TSin * input_item = (TSin *) t;
        //std::cout << "Middle stage " << *input_item << "\n";
        *out = std::move(callable(*input_item));
        input_item->~TSin();
        FF_FREE(input_item);
        return(outslot);
    }
};


// First stage
template <typename TSout, typename L >
struct PMINode<void,TSout,L> : ff_node {
    L callable;
    PMINode(L const &lf):callable(lf) {};
    inline void * svc(void *) {
        // grppi::optional<TSout> ret;
    	optional<TSout> ret;
        void * outslot = FF_MALLOC(sizeof(TSout));
        TSout * out = new (outslot) TSout();
        ret = std::move(callable());
        //*out = ret.elem;
        *out = ret;

        // FT: according to previous custom implementation:
        // if ret has a value, end is false but its
        // boolean evaluation returns true
        if( ret.has_value() ) {
            //std::cout << "EOS\n";
            return EOS;
        }
        else {
            //std::cout << "Res " << *out << "\n";
            return(outslot);
        }
        // No GO_ON
    }
};

// Filter
template <typename TSin, typename L>
struct PMINodeFilter : ff_node {
    L cond;
    PMINodeFilter (L const &c):cond(c) {};
    inline void * svc(void *t) {
        void * outslot = GO_ON;
        TSin * input_item = (TSin *) t;
        if (cond(*input_item)) {
            input_item->~TSin();
            FF_FREE(input_item);
            //std::cout << "Filter drop " << *input_item << "\n";
        } else {
            outslot = t;
        }
        return(outslot);
    }
};


// Last stage
template <typename TSin, typename L >
struct PMINode<TSin,void,L> : ff_node {
    L callable;
    PMINode(L const &lf):callable(lf) {};
    inline void * svc(void *t) {
        TSin * input_item = (TSin *) t;
        callable(*input_item);
        input_item->~TSin();
        FF_FREE(input_item);
        //delete input_item;
        return GO_ON;
    }
};


}


#endif // FF_NODE_WRAP_HPP
