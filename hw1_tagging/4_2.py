#! /usr/bin/python
import os
from collections import defaultdict
from math import log

def get_emission_prob(input_file = "ner_rare.counts"):
    lines = []
    count = defaultdict(int)

    with open(input_file,'r') as input:
        for line in input.readlines():
            if line.strip():
                lines.append(line.split())

    for line in lines:
        if line[1] == "WORDTAG":
            tag = line[2]
            count[tag] += float(line[0])

    emission = defaultdict(lambda:defaultdict(lambda: float("-inf")))
    for line in lines:
        if line[1] == "WORDTAG":
            word = " ".join(line[3:])
            tag = line[2]
            emission[word][tag] = log(float(line[0])) - log(count[tag])

    return emission

def naive_tagger(output_file = "4_2.txt"):
    emission = get_emission_prob()
    tag_dict = defaultdict(tuple)

    for word in emission:
        max_prob = -float("inf")
        max_tag = ""
        for tag in emission[word]:
            if emission[word][tag] > max_prob:
                max_prob = emission[word][tag]
                max_tag = tag
        tag_dict[word] = (max_tag, max_prob)

    cache = []
    rare = '_RARE_'
    with open("ner_dev.dat", 'r') as input:
        for line in input.readlines():
            word = line.strip()
            if not word:
                cache.append('')
            elif word in tag_dict:
                cache.append('{} {} {}'.format(word, tag_dict[word][0], tag_dict[word][1]))
            else:
                cache.append('{} {} {}'.format(word, tag_dict[rare][0], tag_dict[rare][1]))

    with open(output_file,'w') as output:
        output.write("\n".join(cache))
        output.write("\n")

if __name__ == "__main__":
    os.system("python count_freqs.py ner_train_rare.dat > ner_rare.counts")
    naive_tagger()
