#!/usr/bin/env python3

import sys
from Bio import SeqIO

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_fasta> <output_fasta>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

    def is_valid_sequence(seq):
        return set(seq).issubset(valid_aas)

    clean_records = []
    for record in SeqIO.parse(input_file, "fasta"):
        if is_valid_sequence(str(record.seq)):
            clean_records.append(record)

    SeqIO.write(clean_records, output_file, "fasta")
    print(f"Wrote {len(clean_records)} valid sequences to {output_file}")

if __name__ == "__main__":
    main()
