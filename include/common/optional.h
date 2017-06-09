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

#ifndef GRPPI_OPTIONAL_H
#define GRPPI_OPTIONAL_H

namespace grppi{

template <typename T>
class optional {
    public:
        typedef T type;
        typedef T value_type;
        T elem;
        bool end;
        optional(): end(true) { }
        optional(const T& i): elem(i), end(false) { }

        optional& operator=(const optional& o) {
                 elem = o.elem;
                 end = o.end;
                 return *this;
        }

        T& value(){ return elem; }

        constexpr explicit operator bool() const {
            return !end;
        }
};

} // end namespace grppi

#endif
