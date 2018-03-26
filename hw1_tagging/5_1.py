#! /usr/bin/python
from math import log
import os
def get_transition_prob(input_file = "ner_rare.counts"):
    cache = []
    transition_prob = {}
    with open(input_file, "r") as input:
        for line in input.readlines():
            cache.append(line.split())
    bi_gram = {(line[2], line[3]) : float(line[0]) for line in cache if line[1] == "2-GRAM"}
    tri_gram = {(line[2], line[3], line[4]) : float(line[0]) for line in cache if line[1] == "3-GRAM"}

    for (k, v) in tri_gram.items():
        transition_prob[k] = log(v) - log(bi_gram[(k[0], k[1])])

    return transition_prob


if __name__ == "__main__":
    os.system("python count_freqs.py ner_train_rare.dat > ner_rare.counts")
    transition_prob = get_transition_prob()
    with open("trigrams.txt", "r") as input, open("5_1.txt", "w") as output:
        for line in input.readlines():
            l = line.strip()
            if l:
                words = l.split()
                output.write("{} {} {} {}\n".format(words[0], words[1], words[2],
                                                    transition_prob[(words[0], words[1], words[2])]))

