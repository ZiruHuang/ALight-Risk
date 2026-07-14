import sys

def fasta_to_oneline(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    result = []
    current_header = None
    current_sequence = []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if current_header:
                result.append(current_header)
                result.append("".join(current_sequence))
            current_header = line
            current_sequence = []
        else:
            current_sequence.append(line)

    if current_header:
        result.append(current_header)
        result.append("".join(current_sequence))

    for line in result:
        print(line)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fasta_to_oneline.py <input_fasta_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    fasta_to_oneline(input_file)
