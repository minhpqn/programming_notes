#!/usr/bin/perl
use strict;
use warnings;

# retrieve index of matching location using Perl regular expressions
my $sequence;
while (my $line = <DATA>) {
    if ($line=~ /(HDWFLSFKD)/g) {
        print "its found index location: ",
            pos($line)-length($1), "-",  pos($line), "\n";
    } else {
        $sequence .= $line;
        print "came in else\n";
    }
}
close DATA

__DATA__
MLTSHQKKF*HDWFLSFKD*SNNYNSKQNHSIKDIFNRFNHYIYNDLGIRTIA
MLTSHQKKFSNNYNSKQNHSIKDIFNRFNHYIYNDLGIRTIA
MLTSHQKKFSNNYNSK*HDWFLSFKD*QNHSIKDIFNRFNHYIYNDLGIRTIA
