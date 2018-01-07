#include <iostream>
#include <cstdlib>

using namespace std;

// Check if x is a prime number
bool isPrime(int x) {
    if ( x < 2 ) {
        return false;
    }
    
    for (int i = 2; i <= x/2; i++) {
        if ( x % i == 0 ) {
            return false;
        }
    }
    return true;
}

int main() {
    int A = 0;
    int B = 0;
    while ( B < 30 ) {
        if ( isPrime(B) ) {
            A += 61;
            cout << "+ A = " << A << " B = " << B << endl;
        }
        B++;
    }
    cout << A << endl;
}
