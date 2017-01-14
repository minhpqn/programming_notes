use strict;
use warnings;
use utf8;

# Find the minimum natural number that ends with digit 4 such that moving its
# last digit to the first position (i.e. 1234 => 4123) increased it exactly
# four fold?

my $x = 1;
my $num;
while (1) {
    $num = join('', $x, '4');
    my $new_num = join('', '4', $x);
    if ( $num * 4 == $new_num ) {
        last;
    }
    $x++;
}

print "$num\n";
