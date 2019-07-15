from data_utils import CoNLLDataset, DepsDataset, get_vocabs, UNK, NUM, \
    write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word, \
    get_processing_relation, get_relations_vocabs

def build_data(config, logger):
    """
    Procedure to build data
    """

    # Generators
    processing_word = get_processing_word(lowercase=config.lowercase)
    test = CoNLLDataset(config.test_filename, processing_word)
    dev = CoNLLDataset(config.dev_filename, processing_word)
    train = CoNLLDataset(config.train_filename, processing_word)

    # Build Word and Tag vocab
    logger.info("Build Word and Tag vocab...")
    vocab_words, vocab_poss, vocab_chunks, vocab_tags = get_vocabs([train, dev, test])
    vocab = vocab_words
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.words_filename)
    vocab_tags = [tags for tags in vocab_tags]
    vocab_tags.remove("O")
    vocab_tags.insert(0, "O")
    write_vocab(vocab_tags, config.tags_filename)

    # Trim GloVe Vectors
    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename, config.trimmed_filename, config.dim)

    # Build and save char vocab
    logger.info("Build chars vocab...")
    train = CoNLLDataset(config.train_filename)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.chars_filename)

    # Build and save Depstree
    processing_relation = get_processing_relation()
    dev_deps = DepsDataset(config.dev_deps_filename, processing_word, processing_relation)
    train_deps = DepsDataset(config.train_deps_filename, processing_word, processing_relation)

    logger.info("Build relations vocab...")
    vocab_relations = get_relations_vocabs([train_deps, dev_deps])
    vocab_relations.add(UNK)
    write_vocab(vocab_relations, config.relations_filename)
