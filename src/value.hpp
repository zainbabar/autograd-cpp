#include <memory>
#include <vector>
using namespace std;

class Node {
  private:
     vector<shared_ptr<Node>> backward;  // the inputs that led to this current node
   public:
    double data; // data
    double grad; // gradient

    Node(double data): data{data}, grad{0} {}
    
   

};
