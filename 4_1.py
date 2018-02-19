#! /usr/bin/python
from collections import defaultdict

def make_rare(input_file = "ner_train.dat", output_file = "ner_train_rare.dat", threshold = 5):
    cnt = defaultdict(int)
    # count for words
    with open(input_file,'r') as input:
        for line in input.read().split('\n'):
            l = line.strip()
            if l:
                ls = l.split(" ")
                word = " ".join(ls[:-1])
                cnt[word] += 1
    # append to new file
    line_output = []
    with open(input_file, 'r') as input:
            for line in input.read().split('\n'):
                l = line.strip()
                if l:
                    ls = l.split(" ")
                    word = ' '.join(ls[:-1])
                    if cnt[word] >= threshold:
                        line_output.append(line)
                    else:
                        line_output.append(' '.join(["_RARE_", ls[-1]]))
                else:
                    line_output.append('')

    with open(output_file, 'w') as output:
        output.write("\n".join(line_output))

if __name__ == "__main__":
    make_rare()





