from collections import defaultdict


class NetProperties:
    def __init__(self, word_embed_dim, pos_embed_dim, label_embed_dim, hidden_dim1, hidden_dim2, minibatch_size):
        self.word_embed_dim = word_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.label_embed_dim = label_embed_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.minibatch_size = minibatch_size


class Vocab:
    def __init__(self, word_path, pos_path, label_path, action_path):
        # word
        self.word2id_dict = {}
        for l in open(word_path, 'r').read().strip().split('\n'):
            line = l.strip().split(" ")
            self.word2id_dict[line[0]] = int(line[1])

        # pos
        self.pos2id_dict = {}
        for l in open(pos_path, 'r').read().strip().split('\n'):
            line = l.strip().split(" ")
            self.pos2id_dict[line[0]] = int(line[1])

        # label
        self.label2id_dict = {}
        for l in open(label_path, 'r').read().strip().split('\n'):
            line = l.strip().split(" ")
            self.label2id_dict[line[0]] = int(line[1])

        # action
        self.action2id_dict = {}
        self.id2action_dict = {}
        for l in open(action_path, 'r').read().strip().split('\n'):
            line = l.strip().split(" ")
            self.action2id_dict[line[0]] = int(line[1])
            self.id2action_dict[int(line[1])] = line[0]

    def word2id(self, word):
        return self.word2id_dict[word] if word in self.word2id_dict else self.word2id_dict['<unk>']

    def pos2id(self, pos_feat):
        return self.pos2id_dict[pos_feat] if pos_feat in self.pos2id_dict else self.pos2id_dict['<null>']

    def label2id(self, label_feat):
        return self.label2id_dict[label_feat]

    def action2id(self, action_feat):
        return self.action2id_dict[action_feat]

    def id2action(self, id):
        return self.id2action_dict[id]

    def num_words(self):
        return len(self.word2id_dict)

    def num_pos_feats(self):
        return len(self.pos2id_dict)

    def num_labels(self):
        return len(self.label2id_dict)

    def num_actions(self):
        return len(self.action2id_dict)

