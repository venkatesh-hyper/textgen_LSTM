import re

def remove_punctuation_and_empty_lines(input_file, output_file):

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # Remove punctuation using regular expression
            line = re.sub(r'[^\w\s]', '', line)

            # Remove leading/trailing whitespace and check if line is not empty
            if line.strip():
                f_out.write(line)

# Example usage:
input_file = 'data.txt'
output_file = 'p_data.txt'
remove_punctuation_and_empty_lines(input_file, output_file)