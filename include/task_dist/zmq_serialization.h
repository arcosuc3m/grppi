/*
 * Copyright 2018 Universidad Carlos III de Madrid
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRPPI_ZMQ_SERIALIZATION_H
#define GRPPI_ZMQ_SERIALIZATION_H

#include <cassert>
#include <exception>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <map>

//#pragma GCC diagnostic warning "-Wparentheses"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>
//#pragma GCC diagnostic pop

#undef COUT
#define COUT if (0) {std::ostringstream foo;foo
#undef ENDL
#define ENDL std::endl;std::cout << foo.str();}

namespace grppi{

namespace internal {

template <typename T>
std::vector<char> serialize(const T &item)
{
  // serialize obj into an std::vector<char>
  std::vector<char> serial_vec;
  boost::iostreams::back_insert_device<std::vector<char> > inserter(serial_vec);
  boost::iostreams::stream<boost::iostreams::back_insert_device<std::vector<char> > > os(inserter);
  boost::archive::binary_oarchive oa(os);

  try {
    oa << item;
    os.flush();
  } catch (...) {
    COUT << "internal::serialize: Type not serializable" << ENDL;
    std::throw_with_nested(std::runtime_error("Type not serializable"));
  }
  COUT << "internal::serialize: &(serial_vec.data())=" << (void *)(serial_vec.data()) << ", serial_vec.size())=" << (void *)(serial_vec.size()) << ENDL;

  return serial_vec;
}

template <typename T>
T deserialize(const char * str_data, long str_size)
{
  // wrap buffer inside a stream and deserialize serial_str into obj
  boost::iostreams::basic_array_source<char> device((char *)str_data, str_size);
  boost::iostreams::stream<boost::iostreams::basic_array_source<char> > is(device);
  boost::archive::binary_iarchive ia(is);

  T item;
  try {
    ia >> item;
  } catch (...) {
    COUT << "internal::deserialize: Type not serializable" << ENDL;
    std::throw_with_nested(std::runtime_error("Type not serializable"));
  }
  return item;
}


template <typename T>
std::vector<char> serialize_any_ptr(const boost::any &aux)
{
    return serialize(*(boost::any_cast<std::shared_ptr<T>>(aux)));
}


} // end namespace internal

} // end grppi internal

#endif
