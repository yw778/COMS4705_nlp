from optparse import OptionParser
from network_util import NetProperties
from network_util import Vocab
import pickle
# from network import Network
from network_3 import Network

if __name__=='__main__':
    parser = OptionParser()
    parser.add_option("--pre_trained", dest="pre_trained_weights", metavar="FILE", default=None)
    parser.add_option("--train", dest="train_file", metavar="FILE", default="./data/train.data")
    parser.add_option("--model", dest="model_path", metavar="FILE", default="./model/model")
    parser.add_option("--vocab", dest="vocab_path", metavar="FILE", default="./data/vocabs.data")
    parser.add_option("--word", dest="word_path", metavar="FILE", default="./data/vocabs.word")
    parser.add_option("--pos", dest="pos_path", metavar="FILE", default="./data/vocabs.pos")
    parser.add_option("--label", dest="label_path", metavar="FILE", default="./data/vocabs.labels")
    parser.add_option("--action", dest="action_path", metavar="FILE", default="./data/vocabs.actions")
    parser.add_option("--we", type="int", dest="we", default=64)
    parser.add_option("--pe", type="int", dest="pe", default=32)
    parser.add_option("--le", type="int", dest="le", default=32)
    parser.add_option("--hidden1", type="int", dest="hidden1", default=200)
    parser.add_option("--hidden2", type="int", dest="hidden2", default=200)
    parser.add_option("--minibatch", type="int", dest="minibatch", default=1000)
    parser.add_option("--epochs", type="int", dest="epochs", default=7)

    (options, args) = parser.parse_args()

    net_properties = NetProperties(options.we, options.pe, options.le, options.hidden1, options.hidden2, options.minibatch)

    # creating vocabulary file
    vocab = Vocab(options.word_path, options.pos_path, options.label_path, options.action_path)

    # writing properties and vocabulary file into pickle
    pickle.dump((vocab, net_properties), open(options.vocab_path, 'w'))

    # constructing network
    if options.pre_trained_weights:
        network = Network(vocab, net_properties, options.pre_trained_weights)
    else:
        network = Network(vocab, net_properties)

    # training
    network.train(options.train_file, options.epochs)

    # saving network
    network.save(options.model_path)
