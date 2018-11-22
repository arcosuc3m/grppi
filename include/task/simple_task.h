namespace grppi{

class simple_task{
  public:

   simple_task(): function_id_{-1} {};

   simple_task(int f_id): function_id_{f_id} {};

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

    int get_id()
    {
      return function_id_;
    }
  private:
    int function_id_;
};

}
