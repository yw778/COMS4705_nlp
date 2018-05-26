#!/usr/local/bin/python2.7
from collections import defaultdict
import os
from math import log
import json
import time

class CKY(object):

    COUNT_FILE = "cfg.rare.counts"
    def __init__(self, input_train_file, input_dev_file, output_file):
        self.input_train_file = input_train_file
        self.input_dev_file = input_dev_file
        self.output_file = output_file
        self.terminal = set()
        self.nonterminal = defaultdict(int)
        self.unary = defaultdict(int)
        self.binary = defaultdict(int)
        self.q_unary= defaultdict(lambda: float("-inf"))
        self.q_binary = defaultdict(lambda: defaultdict(lambda: float("-inf")))

    def get_parameters(self):
        os.system("python count_cfg_freq.py %s > %s" % (self.input_train_file, CKY.COUNT_FILE))

        for line in open(CKY.COUNT_FILE, "r"):
            words = line.strip().split(" ")
            if words[1] == "NONTERMINAL":
                self.nonterminal[words[2]] += int(words[0])
            elif words[1] == "BINARYRULE":
                self.binary[tuple(words[-3:])] += int(words[0])
            elif words[1] == "UNARYRULE":
                self.unary[tuple(words[-2:])] += int(words[0])
                self.terminal.add(words[3])

        for k, v in self.binary.items():
            # self.q_binary[k] = log(float(v)) - log(float(self.nonterminal[k[0]]))
            self.q_binary[k[0]][(k[1], k[2])] = log(float(v)) - log(float(self.nonterminal[k[0]]))
        for k, v in self.unary.items():
            self.q_unary[k] = log(float(v)) - log(float(self.nonterminal[k[0]]))



    def dp_and_write_output(self):

        def data_generator():
            for line in open(self.input_dev_file):
                x = []
                words_ori = line.strip().split(" ")
                for word in words_ori:
                    if word not in self.terminal:
                        x.append("_RARE_")
                    else:
                        x.append(word)
                yield x, words_ori


        def build_tree(lo, hi, bp, X, x_ori):
            if lo != hi:
                rules, s = bp[(lo, hi, X)]
                return [X, build_tree(lo, s, bp, rules[0], x_ori),
                        build_tree(s + 1, hi, bp, rules[1], x_ori)]
            else:
                return [X, x_ori[lo]]

        cache = []
        # t = time.clock()
        for x, x_ori in data_generator():
            n = len(x)
            pi = defaultdict(lambda: float("-inf"))
            bp = defaultdict(tuple)
            # initialization
            for i in xrange(n):
                for X in self.nonterminal:
                    if (X, x[i]) in self.q_unary:
                        pi[(i, i, X)] = self.q_unary[(X, x[i])]

            for l in xrange(1, n):
                for i in xrange(n - l):
                    j = i + l
                    for X in self.nonterminal:
                        prob_tmp = float("-inf")
                        rule_tmp = ()
                        s_tmp = 0
                        for rule in self.q_binary[X]:
                            Y1 = rule[0]
                            Y2 = rule[1]
                            for s in xrange(i, j):
                                prob = self.q_binary[X][rule] + pi[(i, s, Y1)] + pi[(s + 1, j, Y2)]
                                if prob > prob_tmp:
                                    prob_tmp = prob
                                    rule_tmp = rule
                                    s_tmp = s
                        pi[(i, j, X)] = prob_tmp
                        bp[(i, j, X)] = (rule_tmp, s_tmp)

            if pi[(0, n - 1, "S")] != -float("inf"):
                tree = build_tree(0, n - 1, bp, "S", x_ori)
                cache.append(json.dumps(tree))
            else:
                X_tmp = ""
                prob_tmp = -float("inf")
                for X in self.nonterminal:
                    if pi[(0, n - 1, X)] > prob_tmp:
                        X_tmp = X
                        prob_tmp = pi[(0, n - 1, X)]
                tree = build_tree(0, n-1, bp, X_tmp, x_ori)
                cache.append(json.dumps(tree))

        with open(self.output_file, "w") as output:
            output.write("\n".join(cache))












