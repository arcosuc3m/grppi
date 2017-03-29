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
#include <iostream>
#include <cassert>
#include "ppi/common/operator.hpp"

using namespace std;
using namespace grppi;

int operator_test() {
	int in = 1;

	assert( Sum(in).init() == 0 );
	assert( Mult(in).init() == 1 );
	assert( Sub(in).init() == 0 );
	assert( Div(in).init() == 1 );
	assert( BwAnd(in).init() == ~0 );
	assert( BwOr(in).init() == 0 );
	assert( BwXor(in).init() == 0 );
	assert( And(in).init() == 1 );
	assert( Or(in).init() == 0 );

	cout << '\n';
	printf("0 value: %x\n", 0);
	printf("~0 value: %x\n", ~0);
}

int main() {
	operator_test();
}
