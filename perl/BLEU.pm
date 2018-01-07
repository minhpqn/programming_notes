package BLEU;
use List::Util qw/min/;
use strict;
use warnings;

our (@ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS, $VERSION);
use Exporter;
$VERSION     = 1.0.0;
@ISA = qw(Exporter);
@EXPORT      = qw();
@EXPORT_OK   = qw/bleu baseline_bleu bleu_ngram baseline_bleu_ngram/;
%EXPORT_TAGS = ( );

# Implementation of BLEU scores
# References:
# [1] http://www.aclweb.org/anthology/P02-1040.pdf
# [2] https://en.wikipedia.org/wiki/BLEU

__PACKAGE__->main() unless caller;

# Return bleu score (modified precision)
# Given output and reference text (ARRAY of strings)
sub bleu {
    my ($output, $ref, $n) = @_;

    my $output_ngrams = get_ngrams($output, $n);
    my $ref_ngrams = get_ngrams($ref, $n);

    return bleu_ngram($output_ngrams, $ref_ngrams);
}

sub bleu_ngram {
    my ($output_ngrams, $ref_ngrams) = @_;

    my %ref_count;
    my %count;

    for my $ngr ( @$output_ngrams ) {
        $count{$ngr}++;
        $ref_count{$ngr} = 0;
    }

    for my $ngr ( @$ref_ngrams ) {
        $ref_count{$ngr}++;
    }

    my $count_sum = 0;
    for my $ngr ( keys %count ) {
        my $count_clip = min($count{$ngr}, $ref_count{$ngr});
        $count_sum += $count_clip;
    }

    my $size = @$output_ngrams;
    return $size == 0 ? 0 : $count_sum / $size;
}

sub baseline_bleu {
    my ($output, $ref, $n) = @_;

    my $output_ngrams = get_ngrams($output, $n);
    my $ref_ngrams = get_ngrams($ref, $n);

    return baseline_bleu_ngram($output_ngrams, $ref_ngrams);
}

sub baseline_bleu_ngram {
    my ($output_ngrams, $ref_ngrams) = @_;

    my %ref_ngram;
    @ref_ngram{@$ref_ngrams} = ( );

    my $num_matched = 0;
    for my $ngr ( @$output_ngrams ) {
        if (exists $ref_ngram{$ngr}) {
            $num_matched++;
        }
    }

    my $size = @$output_ngrams;
    return $size == 0 ? 0 : $num_matched / $size;
}

sub get_ngrams {
    my ($array, $n) = @_;

    my $size = @$array;
    my $ngrams = [];
  INDEX:
    for my $i (0..$size-1) {
        my $min = $i;
        my $max = $i + $n - 1;
        last INDEX if ($size <= $max);
        my $ngr = join(' ', @{$array}[$min..$max]);
        push @$ngrams, $ngr;
    }

    return $ngrams;
}

sub test {
    my ($got, $expected) = @_;

    if ( $got eq $expected || $got - $expected < 0.0001) {
        print " OK  Got: $got  Expected: $expected\n";
    }
    else {
        print "  X  Got: $got  Expected: $expected\n";
    }
}

sub main {
    my $output1 = [ qw(the the the the the the the) ];
    my $output2 = [ qw(the cat) ];
    my $ref1 = [qw(the cat is on the mat)];
    my $ref2 = [qw(there is a cat on the mat)];
    test( baseline_bleu_ngram($output1, $ref1), 7/7 );
    test( baseline_bleu($output1, $ref1, 1), 7/7 );
    test( bleu($output1, $ref1, 1), 2/7 );
    test( baseline_bleu($output2, $ref1, 2), 1/1 );
    test( bleu($output2, $ref1, 2), 1/1 );

    my $outputA = [ qw(Israeli officials responsibility of airport safety) ];
    my $outputB = [ qw(airport security Israeli officials are responsible) ];
    my $ref = [ qw(Israeli officials are responsible for airport security) ];

    test( baseline_bleu_ngram($outputB, $ref, 1), 6/6 );
    test( baseline_bleu($outputB, $ref, 2), 4/5);
    test( baseline_bleu($outputB, $ref, 3), 2/4);
    test( baseline_bleu($outputB, $ref, 4), 1/3);

    test( baseline_bleu_ngram($outputA, $ref, 1), 3/6 );
    test( baseline_bleu($outputA, $ref, 2), 1/5);
    test( baseline_bleu($outputA, $ref, 3), 0/4);
    test( baseline_bleu($outputA, $ref, 4), 0/3);
}
