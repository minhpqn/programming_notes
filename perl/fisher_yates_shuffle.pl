package FisherYatesShuffle;
use strict;
use warnings;

# Date: 2015/10/08
# randomly shuffle an array using Fisher-Yates shuffle algorithm
# https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle

__PACKAGE__->main() unless caller;

sub fisher_yates_shuffle {
    my $array = shift;
    my $i;
    for( $i = @$array; --$i; ) {
        my $j = int rand($i+1);
        next if $i == $j;
        @$array[$i, $j] = @$array[$j, $i];
    }
}

sub main {
    my @array = (0..10);
    printf( "-- Original array: %s\n", join(' ', @array) );
    my @temp = @array;
    fisher_yates_shuffle(\@temp);
    printf( "-- Shuffled array: %s\n", join(' ', @temp) );
    print "# Set random seed\n";
    for my $i (1..3) {
        my @temp = @array;
        srand 99999;
        fisher_yates_shuffle(\@temp);
        printf( "-- iterate $i, shuffled array: %s\n", join(' ', @temp) );
    }
}
