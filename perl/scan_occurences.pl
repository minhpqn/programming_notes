use strict;
use warnings;

chomp(my $line = <DATA>);
print "'$line'\n";
my $count = 0;
while ( $line =~ /((\w+)\s+fish\b)/gi ) {
    ++$count;

    my $l = pos($line) - length($1);

    print "The $count occurence is '$1' at the position $l\n";
}

close DATA;

__DATA__
One fish two fish red fish blue fish
