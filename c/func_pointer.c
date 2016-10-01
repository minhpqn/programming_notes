#include <stdio.h>

double add(double x, double y) {
  return (x + y);
}

double sub(double x, double y) {
  return (x - y);
}

int main() {
  int i;
  double (*c[10])();

  for (i = 0; i < 10; i++) {
    if ( i % 2 == 0 ) {
      c[i] = &add;
    }
    else {
      c[i] = &sub;
    }
  }

  double xarray[10] = { 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 2.0};
  double yarray[10] = { 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 2.0};

  for (i = 0; i < 10; i++) {
    printf("%lf\n", c[i](xarray[i], yarray[i]));
  }

  return 0;
}
