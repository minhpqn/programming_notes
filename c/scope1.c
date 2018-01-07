#include <stdio.h>

int main()
{
  int a;
  {
    int b;
  }
  
  {
    int c;
  }

  a = 1;
  b = 1;
  c = 2;

  printf("%d\n", a);

  return 0;
}
