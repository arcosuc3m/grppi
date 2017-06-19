/**
* @version		GrPPI v0.2
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

#ifndef GRPPI_REDUCE_SEQ_H
#define GRPPI_REDUCE_SEQ_H

using namespace std;
namespace grppi{
//typename std::enable_if<!is_iterator<Output>::value, bool>::type,

template < typename InputIt, typename Output, typename ReduceOperator>
 typename std::enable_if<!is_iterator<Output>::value, void>::type 
reduce(sequential_execution &s, InputIt first, InputIt last, Output & firstOut, ReduceOperator op) {
    typename ReduceOperator::result_type identityVal = !op(false,true);
    while( first != last ) {
       identityVal = op(identityVal, *first);
       first++;
    }
    firstOut = op( firstOut, identityVal);
}

template < typename InputIt, typename ReduceOperator>
 typename ReduceOperator::result_type reduce(sequential_execution &s, InputIt first, InputIt last, ReduceOperator op) {
    typename ReduceOperator::result_type identityVal = !op(false,true);
    auto firstOut = identityVal;
//  first++;
    while( first != last ) {
       firstOut = op( firstOut, *first );
       first++;
    }
    return firstOut;
}


template < typename InputIt, typename OutputIt, typename RedFunc>
 typename  std::enable_if<is_iterator<OutputIt>::value, void>::type 
reduce (sequential_execution &s, InputIt first, InputIt last, OutputIt firstOut, RedFunc && reduce) {

    while( first != last ) {
       reduce(*first, *firstOut);
       first++;
       firstOut++;
    }
}

/*
template < typename InputIt, typename Output, typename RedFunc, typename FinalReduce>
 typename std::enable_if<!is_iterator<Output>::value, void>::type
<<<<<<< HEAD
Reduce(sequential_execution const &s, InputIt first, InputIt last, Output & firstOut, RedFunc const & reduce, FinalReduce const & freduce) {
=======
Reduce(sequential_execution s, InputIt first, InputIt last, Output & firstOut, RedFunc && reduce, FinalReduce && freduce) {
>>>>>>> latest
    while( first != last ) {
       reduce(*first, firstOut);
       first++;
    }
}
*/


/*

template <typename InputIt, typename OutputIt, typename ... MoreIn, typename TaskFunc>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, TaskFunc && taskf, MoreIn ... inputs ) {
    while( first != last ) {
        *firstOut = taskf( *first, *inputs ... );
        NextInputs( inputs... );
        first++;
        firstOut++;
    }
}
*/
}
#endif
