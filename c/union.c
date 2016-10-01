#include <stdio.h>

int main() {

  union {
    int i;
    float f;
    char c;
  } A;

  A.i = 2;

  printf("%d\n", A.i);

  printf("%f\n", A.f);

  A.c = 'h';
  A.f = 3.0;
  A.i = 2;

  printf("%d\n", A.i);
  printf("%f\n", A.f);
  printf("%c\n", A.c);

  printf("%lu\n", sizeof(A));

  return 0;
}
