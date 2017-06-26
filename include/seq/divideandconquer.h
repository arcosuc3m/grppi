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

#ifndef GRPPI_DIVIDEANDCONQUER_SEQ_H
#define GRPPI_DIVIDEANDCONQUER_SEQ_H
namespace grppi{

template <typename Input, typename Output, typename DivFunc, typename Operation, typename MergeFunc>
 void divide_and_conquer(sequential_execution &s, Input &problem, Output &output, DivFunc &&divide,
                               Operation &&op, MergeFunc &&merge) {
     
    auto subproblems = divide(problem);
    if(subproblems.size()>1){
        std::vector<Output> partials(subproblems.size());
	int division = 0;
        for(auto i = subproblems.begin(); i != subproblems.end(); i++, division++){
            //THREAD
                divide_and_conquer(s, *i, partials[division], std::forward<DivFunc>(divide), std::forward<Operation>(op), std::forward<MergeFunc>(merge) );
            //END THREAD
        }
        //JOIN
        for(int i = 0; i<partials.size();i++){
              merge(partials[i], output);
        }
    }else{

        op(problem, output);

    }
}


/*
template <typename InputIt, typename OutputIt, typename ... MoreIn, typename Operation>
 void Reduce( InputIt first, InputIt last, OutputIt firstOut, Operation && op, MoreIn ... inputs ) {
    while( first != last ) {
        *firstOut = op( *first, *inputs ... );
        advance_iterators( inputs... );
        first++;
        firstOut++;
    }
}
*/
}
#endif
