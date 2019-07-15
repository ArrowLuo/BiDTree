import numpy as np
import cPickle as pickle
import math

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

class CoNLLDataset(object):
    """
    Class that iterates over CoNLL Dataset
    """

    def __init__(self, filename, processing_word=None, processing_pos=None, processing_chunk=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_word: (optional) function that takes a word as input
            processing_pos: (optional) function that takes a pos as input
            processing_chunk: (optional) function that takes a chunk as input
            processing_tag: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_pos = processing_pos
        self.processing_chunk = processing_chunk
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None

        self.max_words_len = 0
        self.max_chars_len = 0

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, poss, chunks, tags = [], [], [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, poss, chunks, tags
                        self.max_words_len = self.max_words_len if self.max_words_len > len(words) else len(words)
                        words, poss, chunks, tags = [], [], [], []
                else:
                    word, pos, chunk, tag = line.split(' ')
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                        self.max_chars_len = len(word[0]) if len(word[0]) > self.max_chars_len else self.max_chars_len
                    if self.processing_pos is not None:
                        pos = self.processing_pos(pos)
                    if self.processing_chunk is not None:
                        chunk = self.processing_chunk(chunk)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    poss += [pos]
                    chunks += [chunk]
                    tags += [tag]

    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

class DepsDataset(object):

    def __init__(self, filename, processing_word=None, processing_relation=None, max_iter=None):
        self.filename = filename
        self.processing_word = processing_word
        self.max_iter = max_iter
        self.processing_relation = processing_relation
        self.length = None

        self.max_btup_deps_len = 0
        self.max_upbt_deps_len = 0

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            btup_idx_list, btup_words_list, btup_depwords_list, \
            btup_deprels_list, btup_depwords_length_list, \
            upbt_idx_list, upbt_words_list, upbt_depwords_list, \
            upbt_deprels_list, upbt_depwords_length_list, \
            btup_formidx_list, upbt_formidx_list= pickle.load(f)

            for btup_idx, btup_words, btup_depwords, btup_deprels, btup_depwords_length, \
                upbt_idx, upbt_words, upbt_depwords, upbt_deprels, upbt_depwords_length, \
                btup_formidx, upbt_formidx \
                    in zip(btup_idx_list, btup_words_list, btup_depwords_list,
                           btup_deprels_list, btup_depwords_length_list,
                           upbt_idx_list, upbt_words_list, upbt_depwords_list,
                           upbt_deprels_list, upbt_depwords_length_list,
                           btup_formidx_list, upbt_formidx_list):

                niter += 1
                if self.max_iter is not None and niter > self.max_iter:
                    break

                if self.processing_word is not None:
                    btup_words = [self.processing_word(word) for word in btup_words]
                    upbt_words = [self.processing_word(word) for word in upbt_words]

                if self.processing_relation is not None:
                    btup_deprels = [[self.processing_relation(rels) for rels in deps_rels] for deps_rels in btup_deprels]
                    upbt_deprels = [[self.processing_relation(rels) for rels in deps_rels] for deps_rels in upbt_deprels]

                self.max_btup_deps_len = self.max_btup_deps_len \
                    if self.max_btup_deps_len > max(btup_depwords_length) else max(btup_depwords_length)
                self.max_upbt_deps_len = self.max_upbt_deps_len \
                    if self.max_upbt_deps_len > max(upbt_depwords_length) else max(upbt_depwords_length)

                yield btup_idx, btup_words, btup_depwords, btup_deprels, btup_depwords_length, \
                      upbt_idx, upbt_words, upbt_depwords, upbt_deprels, upbt_depwords_length, \
                      btup_formidx, upbt_formidx

    def __len__(self):
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length

##################################
def get_vocabs(datasets):
    vocab_words = set()
    vocab_poss = set()
    vocab_chunks = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, poss, chunks, tags in dataset:
            vocab_words.update(words)
            vocab_poss.update(poss)
            vocab_chunks.update(chunks)
            vocab_tags.update(tags)
    return vocab_words, vocab_poss, vocab_chunks, vocab_tags

def get_relations_vocabs(datasets):
    vocab_relations = set()
    for dataset in datasets:
        for _, _, _, btup_deprels, _, \
            _, _, _, upbt_deprels, _, \
            _, _ in dataset:
            vocab_relations.update([rels for dep_rels in btup_deprels for rels in dep_rels])
            vocab_relations.update([rels for dep_rels in upbt_deprels for rels in dep_rels])
    return vocab_relations

def get_char_vocab(dataset):
    vocab_char = set()
    for words, _, _, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename, lowercase=False):
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            if lowercase:
                word = word.lower()
            vocab.add(word)
    return vocab


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx

    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array
    
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    stdv_ = 1. / math.sqrt(dim)
    embeddings = np.random.uniform(low=-stdv_, high=stdv_, size=(len(vocab), dim))

    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = map(float, line[1:])
            if word in vocab and len(embedding) == dim:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with open(filename) as f:
        return np.load(f)["embeddings"]

def get_processing_relation(vocab_relations=None, lowercase=False):
    """
    Args:
        vocab_relations: dict[relation] = idx
    """
    def f(relation):
        # 1. preprocess relation
        if lowercase:
            relation = relation.lower()
        # 2. get id of word
        if vocab_relations is not None:
            if relation in vocab_relations:
                relation = vocab_relations[relation]
            else:
                relation = vocab_relations[UNK]
        return relation

    return f

def get_processing_word(vocab_words=None, vocab_chars=None, lowercase=False, chars=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns: 
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                word = vocab_words[UNK]

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, fixed_sentence_length=None, fixd_words_length=None, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = fixed_sentence_length if fixed_sentence_length != None else max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
        
    elif nlevels == 2:
        max_length_word = fixd_words_length if fixd_words_length != None else max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = fixed_sentence_length if fixed_sentence_length != None else max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError("`nlevels` must be 1 or 2.")

    return sequence_padded, sequence_length


def minibatches(seq_data, deps_data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns: 
        list of tuples
    """
    x_batch, y_batch, z_batch, v_batch = [], [], [], []
    btup_idx_list, btup_words_list, btup_depwords_list, \
    btup_deprels_list, btup_depwords_length_list, \
    upbt_idx_list, upbt_words_list, upbt_depwords_list, \
    upbt_deprels_list, upbt_depwords_length_list, \
    btup_formidx_list, upbt_formidx_list = [], [], [], [], [], [], [], [], [], [], [], []
    for (x, y, z, v), \
        (btup_idx, btup_words, btup_depwords, btup_deprels, btup_depwords_length,
         upbt_idx, upbt_words, upbt_depwords, upbt_deprels, upbt_depwords_length,
         btup_formidx, upbt_formidx) in zip(seq_data, deps_data):
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch, z_batch, v_batch, \
                  btup_idx_list, btup_words_list, btup_depwords_list, \
                  btup_deprels_list, btup_depwords_length_list, \
                  upbt_idx_list, upbt_words_list, upbt_depwords_list, \
                  upbt_deprels_list, upbt_depwords_length_list, \
                  btup_formidx_list, upbt_formidx_list
            x_batch, y_batch, z_batch, v_batch = [], [], [], []
            btup_idx_list, btup_words_list, btup_depwords_list, \
            btup_deprels_list, btup_depwords_length_list, \
            upbt_idx_list, upbt_words_list, upbt_depwords_list, \
            upbt_deprels_list, upbt_depwords_length_list, \
            btup_formidx_list, upbt_formidx_list = [], [], [], [], [], [], [], [], [], [], [], []
        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]
        z_batch += [z]
        v_batch += [v]
        btup_idx_list += [btup_idx]
        btup_words_list += [btup_words]
        btup_depwords_list += [btup_depwords]
        btup_deprels_list += [btup_deprels]
        btup_depwords_length_list += [btup_depwords_length]

        upbt_idx_list += [upbt_idx]
        upbt_words_list += [upbt_words]
        upbt_depwords_list += [upbt_depwords]
        upbt_deprels_list += [upbt_deprels]
        upbt_depwords_length_list += [upbt_depwords_length]

        btup_formidx_list += [btup_formidx]
        upbt_formidx_list += [upbt_formidx]

    if len(x_batch) != 0:
        yield x_batch, y_batch, z_batch, v_batch, \
              btup_idx_list, btup_words_list, btup_depwords_list, \
              btup_deprels_list, btup_depwords_length_list, \
              upbt_idx_list, upbt_words_list, upbt_depwords_list, \
              upbt_deprels_list, upbt_depwords_length_list, \
              btup_formidx_list, upbt_formidx_list


def get_chunk_type(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('_')[-1]

def get_chunk_alpha(tok, idx_to_tag):
    tag_name = idx_to_tag[tok]
    return tag_name.split('_')[0]

def get_chunks(seq, vocab_tags):
    """
    Args:
        seq: [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1] sequence of labels
        vocab_tags: {'O': 0, 'B_AP': 1, 'I_AP': 2}
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [1, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1]
        vocab_tags = {'O': 0, 'B_AP': 1, 'I_AP': 2}
        result = [('AP', 0, 1), ('AP', 2, 3), ('AP', 3, 5), ('AP', 6, 9), ('AP', 10, 11)]
    """
    default = vocab_tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in vocab_tags.iteritems()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            tok_chunk_alpha = get_chunk_alpha(tok, idx_to_tag)
            if chunk_type is None and tok_chunk_alpha == "B":
                chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type is not None and tok_chunk_type != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None
                if tok_chunk_alpha == "B":
                    chunk_type, chunk_start = tok_chunk_type, i
            elif chunk_type is not None and tok_chunk_type == chunk_type:
                if tok_chunk_alpha == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks
