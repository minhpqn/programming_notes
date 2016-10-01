#include <stdio.h>

int main() {

  struct {
    double n;
    char c;
  } A;

  printf("size of A = %lu\n", sizeof(A));
  printf("total size of elements = %lu\n", sizeof(double) + sizeof(char));

  return 0;
}
