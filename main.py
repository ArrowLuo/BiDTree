import os
from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset, DepsDataset, \
    get_processing_relation
from general_utils import get_logger
from model import DepsModel
from config import Config
from itertools import chain
from build_data import build_data
import argparse

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_preprocess', default=False, action='store_true')
    parser.add_argument('--do_train', default=False, action='store_true')
    parser.add_argument('--do_evaluate', default=False, action='store_true')
    parser.add_argument('--current_path', type=str, default=".")
    parser.add_argument('--dim', type=int, default=300, help='dimension size of embedding.')
    parser.add_argument('--lr', type=float, default=0.0010, help='the initial learning rate for optimizer.')
    parser.add_argument('--batch_size', type=int, default=20, help='the training batch size.')
    parser.add_argument('--nepochs', type=int, default=100, help='the max training epoch.')
    parser.add_argument('--nepoch_no_imprv', type=int, default=10, help='the coefficient of stop early.')
    parser.add_argument('--dropout', type=float, default=0.4, help='the dropout coefficient.')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='the coefficient of decayed learning rate.')
    parser.add_argument('--data_sets', type=str, default="laptops_2014", help="dataset for train, dev, and test.")
    parser.add_argument('--show_test_results', default=False, action='store_true')

    args, _ = parser.parse_known_args()
    return args

def config_from_args(args):
    config = Config()
    for key, value in vars(args).items():
        config.__dict__[key] = value
    config.auto_config()
    logger = get_logger(config.log_path)
    return config, logger

if __name__ == "__main__":
    args = parse_parameters()
    config, logger = config_from_args(args)

    if args.do_preprocess:
        build_data(config, logger)

    # load vocabs
    vocab_words = load_vocab(config.words_filename)
    vocab_tags = load_vocab(config.tags_filename)
    vocab_chars = load_vocab(config.chars_filename)
    vocab_relations = load_vocab(config.relations_filename)

    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars, lowercase=config.lowercase, chars=config.chars)
    processing_tag = get_processing_word(vocab_tags, lowercase=False)
    processing_relation = get_processing_relation(vocab_relations)

    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)

    # create dataset
    dev = CoNLLDataset(config.dev_filename, processing_word, processing_tag=processing_tag)
    test = CoNLLDataset(config.test_filename, processing_word, processing_tag=processing_tag)
    train = CoNLLDataset(config.train_filename, processing_word, processing_tag=processing_tag)

    data = [dev, test, train]
    _ = map(len, chain.from_iterable(w for w in (s for s in data)))
    max_sentence_size = max(train.max_words_len, test.max_words_len, dev.max_words_len)
    max_word_size = max(train.max_chars_len, test.max_chars_len, dev.max_chars_len)

    processing_word = get_processing_word(vocab_words, lowercase=config.lowercase)
    dev_deps = DepsDataset(config.dev_deps_filename, processing_word, processing_relation)
    test_deps = DepsDataset(config.test_deps_filename, processing_word, processing_relation)
    train_deps = DepsDataset(config.train_deps_filename, processing_word, processing_relation)

    data = [dev_deps, test_deps, train_deps]
    _ = map(len, chain.from_iterable(w for w in (s for s in data)))
    max_btup_deps_len = max(dev_deps.max_btup_deps_len, test_deps.max_btup_deps_len, train_deps.max_btup_deps_len)
    max_upbt_deps_len = max(dev_deps.max_upbt_deps_len, test_deps.max_upbt_deps_len, train_deps.max_upbt_deps_len)

    # build model
    config.ntags = len(vocab_tags)
    config.nwords = len(vocab_words)
    config.nchars = len(vocab_chars)
    config.nrels = len(vocab_relations)
    config.max_sentence_size = max_sentence_size
    config.max_word_size = max_word_size
    config.max_btup_deps_len = max_btup_deps_len
    config.max_upbt_deps_len = max_upbt_deps_len
    model = DepsModel(config, embeddings, logger=logger)
    model.build()

    if args.do_train:
        model.train(train, train_deps, dev, dev_deps, vocab_words, vocab_tags)
    if args.do_evaluate:
        model.evaluate(test, test_deps, vocab_words, vocab_tags)
