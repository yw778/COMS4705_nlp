#!/usr/local/bin/python2.7
import json
import os
from collections import defaultdict
class Rare_Maker(object):
    MIN_FRE = 5
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.counter = defaultdict(int)

    def getCount(self):
        """
        get count of the terminal words
        """
        os.system("python count_cfg_freq.py " + self.input_file + " > cfg.counts")
        for line in open("cfg.counts", 'r'):
            lines = line.strip().split(" ")
            if lines[1] == "UNARYRULE":
                self.counter[lines[3]] += int(lines[0])

    def replace(self):
        """
        replace rare words with rare and output in new file
        """
        cache = []
        for line in open(self.input_file):
            tree = json.loads(line)
            self.replace_helper(tree)
            line_ori = json.dumps(tree)
            cache.append(line_ori)
        with open(self.output_file, 'w') as output:
            output.write("\n".join(cache))

    def replace_helper(self, tree):
        """
        recursive helper function
        """
        if len(tree) == 2 and self.counter[tree[1]] < Rare_Maker.MIN_FRE:
                tree[1] = "_RARE_"
        elif len(tree) == 3:
            self.replace_helper(tree[1])
            self.replace_helper(tree[2])



