package MyBinomialLib;
use strict;
use warnings;

# Calculate expected wealth
# In a coin toss, you get $1 for heads, and lose $10 for tails. Prob(head)=90%. What is the expected wealth after 10 tosses. We start with $10

# Reading: binomial distribution (page 145 OpenIntro Statistics)
# How to solve the question in R?

__PACKAGE__->main() unless caller;

# calculate the choose k of n
# return n!/(k! * (n-k)!)
sub choose {
    my ( $n, $k ) = @_;

    return 0 if $k > $n || $k < 0;

    # choose(n, k) = choose(n, n-k)
    my $t = $k;
    if ( $n - $k < $k ) {
        $t = $n - $k;
    }
    
    my $comb = 1;
    for my $i (1..$t) {
        $comb *= ($n - $t + $i);
        $comb /= $i;
    }

    return $comb;
}

# Calculate probability of obtaining k heads in n experiments
# given that P(head) = p
sub prob_k_head {
    my ( $n, $k, $p ) = @_;

    return choose($n, $k) * ($p ** $k) * ( (1-$p) ** ($n - $k) );
}

sub test_prob_k_head {
    my $n = 10;
    my $p = 0.9;
    for my $i (1..10) {
        my $prob = prob_k_head($n, $i, $p);
        my $comb = choose($n, $i);
        print "choose($n, $i) = $comb, prob($n, $i, $p) = $prob\n";
    }
}

sub main {
    # test_prob_k_head();
    my $n = 10;
    my $exp = 10;
    my $p = 0.9;
    for my $k (1..10) {
        $exp += prob_k_head($n, $k, $p) * (11 * $k - 100);
    }

    print "Expected wealth = $exp\n";
}
