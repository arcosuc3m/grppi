/**
* @version		GrPPI v0.1
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
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

#ifndef PPI_OPERATORS
#define PPI_OPERATORS
template <typename T>
class Sum_op{
    public:
       T init(){return 0;}
       void operator()(T& a, T b){ a += b; }
       Sum_op(T t){};
};

template <typename T>
inline auto Sum(T t){
    return Sum_op<T>(t);
}

template <typename T>
class Mult_op{
    public:
       T init(){return 1;}
       void operator()(T& a, T b){ a *= b; }
       Mult_op(T t){};
};

template <typename T>
inline auto Mult(T t){
    return Mult_op<T>(t);
}

template <typename T>
class Sub_op{
    public:
       T init(){return 0;}
       void operator()(T& a, T b){ a -= b; }
       Sub_op(T t){};
};

template <typename T>
inline auto Sub(T t){
    return Sub_op<T>(t);
}

template <typename T>
class Div_op{
    public:
       T init(){return 1;}
       void operator()(T& a, T b){ a /= b; }
       Div_op(T t){};
};

template <typename T>
inline auto Div(T t){
    return Div_op<T>(t);
}


/* FIXME: Adrian comprueba si las operaciones bitwise And y Or se inicializan bien*/
template <typename T>
class BwAnd_op{
    public:
       T init(){return ~0;}
       void operator()(T& a, T b){ a &= b; }
       BwAnd_op(T t){};
};

template <typename T>
inline auto BwAnd(T t){
    return BwAnd_op<T>(t);
}


template <typename T>
class BwOr_op{
    public:
       T init(){return 0;}
       void operator()(T& a, T b){ a |= b; }
       BwOr_op(T t){};
};

template <typename T>
inline auto BwOr(T t){
    return BwOr_op<T>(t);
}

template <typename T>
class BwXor_op{
    public:
       T init(){return 0;}
       void operator()(T& a, T b){ a ^= b; }
       BwXor_op(T t){};
};

template <typename T>
inline auto BwXor(T t){
    return BwXor_op<T>(t);
}

template <typename T>
class And_op{
    public:
       T init(){return true;}
       void operator()(T& a, T b){ a = a && b; }
       And_op(T t){};
};

template <typename T>
inline auto And(T t){
    return And_op<T>(t);
}

template <typename T>
class Or_op{
    public:
       T init(){return false;}
       void operator()(T& a, T b){ a= a || b; }
       Or_op(T t){};
};

template <typename T>
inline auto Or(T t){
    return Or_op<T>(t);
}
#endif
