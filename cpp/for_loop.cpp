#include <iostream>
#include <cstdlib>

using namespace std;

int main() {

  for (int i = 1; i <= 10; i += 2) {
    cout << "Iteration " << i << endl;
    if (i == 3) {
      i = 6;
    }
  }

  return 0;
}
