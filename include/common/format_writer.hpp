#ifndef GRPPI_COMMON_FORMAT_WRITER_H
#define GRPPI_COMMON_FORMAT_WRITER_H

#ifdef GRPPI_DCEX

namespace grppi{

template <typename Container, typename Formatter>
class format_writer
{
   public:
	format_writer(Container & c, Formatter && f) noexcept:
		output_container_{c}, formatter_{f}
	{}

	template <typename I>
	auto operator ()(I && item) const{
            return formatter_(std::forward<I>(item));
	}

	Container get_output(){ return output_container_;}

   private:
       Formatter formatter_;
       Container & output_container_;
};

}
#endif

#endif
