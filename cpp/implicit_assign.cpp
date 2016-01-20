#include <iostream>
#include <cstdlib>

using namespace std;

void foo() {
    printf( "Hello world\n" );
}

int main() {
    // int *x = malloc(sizeof(int) * 10);
    foo();

    bool isGood = true;

    cout << isGood << endl;
        

    return 0;
}

