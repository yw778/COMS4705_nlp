#! /usr/bin/python
from math import log
from collections import defaultdict
import time

def get_count(input_file = "ner_train_rare.dat"):
    cnt = defaultdict(int)
    # count for words
    with open(input_file,'r') as input:
        for line in input.read().split('\n'):
            l = line.strip()
            if l:
                ls = l.split(" ")
                word = " ".join(ls[:-1])
                cnt[word] += 1
    return cnt

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

def get_tags(intput_file = "ner_rare.counts"):
    tags = {line.split(" ")[2] for line in open(intput_file, 'r')
            if line.strip() and line.split(" ")[1] == "WORDTAG"}
    return tags

def viterbi_algo(x):
    def get_tag(i):
        return tags if i >= 0 else {"*"}

    pi = defaultdict(lambda: float("-inf"))
    bp = defaultdict(str)
    pi[(-1, "*", "*")] = 0
    n = len(x)

    for k in range(n):
        for u in get_tag(k - 1):
            for v in get_tag(k):
                max_val = float("-inf")
                arg_max = ""
                for w in get_tag(k - 2):
                    if e[x[k]][v] != 0 and (w,u,v) in q:
                        new_val = pi[(k - 1, w, u)] + q[(w, u, v)] + e[x[k]][v]
                        if new_val > max_val:
                            max_val = new_val
                            arg_max = w
                pi[(k, u, v)] = max_val
                bp[(k, u, v)] = arg_max

    u_val = v_val = ""
    max_val = float("-inf")
    for u in get_tag(n - 2):
        for v in get_tag(n - 1):
            if (u, v, "STOP") in q:
                new_val = pi[(n - 1, u, v)] + q[(u, v, "STOP")]
                if new_val > max_val:
                    max_val = new_val
                    u_val = u
                    v_val = v

    result = [''] * n
    result[-1] = v_val
    if n >= 2:
        result[-2] = u_val

    for k in range(n - 3, -1, -1):
        result[k] = bp[(k + 2, result[k+1], result[k+2])]

    return result, [pi[(k, result[k-1], result[k])] if k >= 1 else pi[(0, "*", result[k])] for k in range(n)]

def viterbi_tagger(input_file = "ner_dev.dat", output_file = "5_2.txt", threshold = 5):
    sentences = []
    with open(input_file, "r") as input:
        raw_words = []
        for line in input:
            word = line.strip()
            if not word:
                sentences.append(raw_words)
                raw_words = []
            else:
                raw_words.append(word)
        if len(raw_words) != 0:
            sentences.append(raw_words)

    count = get_count()
    cache = []
    for sentence in sentences:
        x = []
        for word in sentence:
            word_rare = "_RARE_" if word not in count else word
            x.append(word_rare)
        tags, prob = viterbi_algo(x)
        for i in range(len(x)):
            cache.append(" ".join([sentence[i], tags[i], str(prob[i])]))
        cache.append("")

    with open(output_file, "w") as output:
        output.write("\n".join(cache))
        output.write("\n")

if __name__ == "__main__":
    running_time = time.time()
    tags = get_tags()
    q = get_transition_prob()
    e = get_emission_prob()
    viterbi_tagger()
    running_time = time.time() - running_time
    print "running_time: %.8f" %(running_time)