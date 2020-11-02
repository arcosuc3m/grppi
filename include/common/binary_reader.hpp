#ifndef GRPPI_COMMON_BINARY_READER_H
#define GRPPI_COMMON_BINARY_READER_H

#ifdef GRPPI_DCEX

namespace grppi{

long long gcd(long long a, long long b)
{
  if(b==0) return a;
  return gcd(b, a%b);
}

long long lcm(long long a, long long b)
{
	 return (a/gcd(a,b))*b;
}


template <typename ItemType, typename Container, typename Deserializer>
class binary_reader_t
{
   public:
	binary_reader_t(Container & c, Deserializer && deserialize, int item_size) noexcept:
		input_container_{c}, deserialize_{deserialize}, item_size_{item_size}
	{
std::cout<<"BUILDING BINR"<<std::endl;
            //TODO: Implement this functionality
	    // We need a way to ask for the block size
	   block_size_ = 64;
	   long long items_per_block = block_size_/item_size_;
	   //If the number of items in a blocks is an integer then the chunk size is fixed to 1 block.
	   if(block_size_%item_size_==0) {
               blocks_per_chunk_=1;
	   }
	   //Otherwise, we compute the number of blocks for a chunk
	   else{
std::cout<<"Compute chunk size"<<std::endl;
	      long long nbytes_items_block = items_per_block*item_size_;
	      long long nbytes_chunk = lcm(block_size_,nbytes_items_block);
	      blocks_per_chunk_ = nbytes_chunk / block_size_;
	   }
	}

        
	std::vector<ItemType> operator ()(char * buffer) const{
           std::vector<ItemType> data;
	   long current_position = 0;
	   while(current_position != block_size_ * blocks_per_chunk_){
	      ItemType aux = deserialize_(buffer);
	      data.push_back(aux);
	      buffer+= item_size_;
	      current_position += item_size_;
	   }
	   return data;
	}

	Container& get_input(){ return input_container_;}

	long get_chunk_blocks(){ return blocks_per_chunk_;}

	long get_block_size(){ return block_size_;}

   private:
       Container & input_container_;
       Deserializer deserialize_;
       long item_size_;
       long blocks_per_chunk_;
       long block_size_;
};


template <typename ItemType, typename Container, typename Deserializer>
binary_reader_t<ItemType,Container,Deserializer> binary_reader(Container & c, Deserializer && d,int i){
    std::cout<<"Binr -helper"<<std::endl;
       	return binary_reader_t<ItemType, Container, Deserializer>(c,std::forward<Deserializer>(d),i);
}

}
#endif

#endif
