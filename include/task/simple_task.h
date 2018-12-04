#include <set>

namespace grppi{

class simple_task{
  public:

   simple_task(): function_id_{-1}, task_id_{-1} {};

   simple_task(int f_id, long t_id): function_id_{f_id}, task_id_{t_id} {};

   inline bool operator==(const simple_task& rhs) const {
      return function_id_ == rhs.function_id_;
    }

    inline bool operator!=(const simple_task& rhs) const {
      return function_id_ != rhs.function_id_;
    } 

    inline bool operator>(const simple_task& rhs) const {
      return function_id_ > rhs.function_id_;
    }

    inline bool operator<(const simple_task& rhs) const {
      return function_id_ < rhs.function_id_;
    }

    int get_pattern_id() const
    {
      return pattern_;
    }
   
    void set_pattern_id(int p_id)
    {
      pattern_ = p_id;
    }

    int get_id() const
    {
      return function_id_;
    }

    int get_data_location()
    {
      return data_location_;
    } 

    void set_data_location(int loc)
    {
      data_location_ = loc;
    } 

    int get_task_id() const
    {
      return task_id_;
    }
   
    void set_task_dependency(simple_task &t)
    {
      before_dependencies_.insert(t.get_task_id());
      t.after_dependencies_.insert(task_id_);
    }

  public:
    std::vector<int> multiple_input_location;
    std::set<long> before_dependencies_; 
    std::set<long> after_dependencies_;
  private:
    int function_id_;
    long task_id_;
    int data_location_;
    int pattern_ = -1;
};

}
