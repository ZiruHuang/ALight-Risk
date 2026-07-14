#!/usr/bin/perl
use strict;
use warnings;

my $fasta_file = shift or die "Usage: $0 fasta_file\n";

open my $fh, '<', $fasta_file or die "Cannot open $fasta_file: $!\n";

# 输出列名
print "ID\tSeq\n";

while (<$fh>) {
    chomp;
    # 捕获整个描述行（">"后面的所有字符）
         if (/^>(.+)/) {
                 print "$1\t";
                     } else {
                             print "$_\n";
                                 }
                                 }
    
                                 close $fh;  
