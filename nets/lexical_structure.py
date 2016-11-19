from tpl.machine_learning.aux_data_structure import Alphabet

import codecs 
import numpy as np
from sklearn.covariance import GraphLassoCV, EmpiricalCovariance
from scipy.stats import multivariate_normal

SKIPGRAM_WORD_EMBEDDING_FILE = '~/nlp/lib/lexicon/google_word_vector/GoogleNews-vectors-negative300.txt'

class DiscourseMatrix(object):
    """Skipgram discourse matrix model

    We want to store all of the discourse vector in the same matrix.
    Later, we want to fit multivariate gaussian to the data.

    The discourse matrix cannot be expanded.
    """
    def __init__(self, num_rows, word_embedding_dict):
        self.wed = word_embedding_dict 
        self.matrix = np.zeros((num_rows, self.wed.num_units * 2))
        self.num_vectors_so_far = 0
        self.cov_matrix = np.array([])
        self.means = np.array([])

    def add_discourse_text(self, sentence1, sentence2):
        """Add a discourse vector to the matrix

        This is different than discourse vector in the NDMM
        because we want to be skimpy on the number of parameters.
        Next step we want to use graphical lasso to reduce the number of entries
        on the covariance matrix.
        """
        vec = np.zeros(self.wed.num_units)
        for word in sentence1.split(' '):
            if word in self.wed:
                vec += self.wed[word]
        self.matrix[self.num_vectors_so_far,:self.wed.num_units] = vec
        for i, word in enumerate(sentence2.split(' ')):
            if word in self.wed and i >= 3:
                vec += self.wed[word]
        self.matrix[self.num_vectors_so_far,self.wed.num_units:] = vec
        self.num_vectors_so_far += 1

    def fit_mvn(self):
        print "fitting cov"
        model = GraphLassoCV(n_jobs=8)
        model = EmpiricalCovariance()
        model.fit(self.matrix[:self.num_vectors_so_far])
        print "done"
        self.cov_matrix = model.covariance_
        self.means = np.mean(self.matrix[:self.num_vectors_so_far], 0)
        return (self.means, self.cov_matrix)

    def compute_ll(self, discourse_vector):
        return multivariate_normal.logpdf(discourse_vector, self.means, self.cov_matrix)

    def to_dict(self):
        return {
                'matrix' : self.matrix.tolist(),
                'num_vectors_so_far' : self.num_vectors_so_far,
                'cov_matrix' : self.cov_matrix.tolist(),
                'means' : self.means.tolist()
                }

    @classmethod
    def from_dict(self, mdict):
        m = DiscourseMatrix(mdict['num_vectors_so_far'], None)
        m.matrix = np.array(mdict['matrix'])
        m.cov_matrix = np.array(mdict['cov_matrix'])
        m.means = np.array(mdict['means'])
        return m

class WordEmbeddingMatrix(object):

    def __init__(self, word_embedding_npy_file, word_list_file):
        self.wm = np.load(word_embedding_npy_file)
        self.word2index = {}
        index = 0 
        for x in codecs.open(word_list_file, encoding='utf8'):
            self.word2index[x.strip()] = index
            index += 1
        assert(index == self.wm.shape[0])

    def index_tokens(self, token_list, ignore_OOV=True):
        indices = []
        for token in token_list:
            if token in self.word2index:
                indices.append(self.word2index[token])
            elif token.lower() in self.word2index:
                indices.append(self.word2index[token.lower()])
            elif not ignore_OOV:
                indices.append(0)
        return indices

    def get_embedding(self, token):
        if token in self.word2index:
            return self.wm[self.word2index[token], :]
        else:
            return np.zeros(self.wm.shape[1])

    @property
    def num_units(self):
        return self.wm.shape[1]

class WordEmbeddingDict(object):
    """A simple wrapper around Dict with lazy loading """

    def __init__(self, word_embedding_file):
        #f = codecs.open(word_embedding_file, encoding='utf8')
        f = open(word_embedding_file)
        self.vocab_size, self.num_units = f.readline().strip().split(' ')
        self.num_units = int(self.num_units)
        self.vocab_size = int(self.vocab_size)
        self.word_to_vector = {}
        print 'Start reading in the dictionary'
        line_number = 1
        line_skipped = 0 
        for line in f:
            line_number += 1
            try:
                word, vector = line.split(' ', 1)
                self.word_to_vector[word.decode('utf8')] = vector.strip()
            except:
                line_skipped += 1
        print 'Skipped %s out of %s lines' % (line_skipped, line_number)

    def __getitem__(self, key):
        vector = self.word_to_vector[key]
        # to save some time and space, we process a vector entry lazily
        if isinstance(vector, str) or isinstance(vector, unicode):
            self.word_to_vector[key] = np.array([float(x) for x in vector.split(' ')])
        return self.word_to_vector[key]     

    def __contains__(self, key):
        return key in self.word_to_vector

class LexicalOccurrenceVector(object):
    """Lexical Occurrence Vector that automatically grows, backed by numpy.array
    
    >>> lv = LexicalOccurrenceVector(initial_capacity=2)
    >>> lv['word1'] = 10
    >>> lv['word2'] = 100
    >>> lv['word1']
    10.0
    >>> lv[1]
    100.0
    >>> lv.growth_factor = 3
    >>> lv['word3'] = 1000
    >>> len(lv) 
    3
    """

    def __init__(self, alphabet=Alphabet(), initial_capacity=1000000):
        self.alphabet = alphabet
        self.growth_factor = 0.5
        self.count_vector = np.zeros(max(initial_capacity, len(alphabet)))
        self.pdf_ = None
        self.logpdf_ = None

    @property
    def vector(self):
        return self.count_vector[:len(self.alphabet)]

    @property
    def pdf(self):
        if self.pdf_ is None:
            self.pdf_ = self.vector 
            self.pdf_ = self.pdf_ / np.sum(self.pdf_)
        return self.pdf_

    @property
    def logpdf(self):
        if self.logpdf is None:
            self.logpdf_ = np.log(self.pdf)
        return self.logpdf_

    def __getitem__(self, key):
        self.grow_to_alphabet()
        if isinstance(key, int):
            return self.count_vector[key]
        elif isinstance(key, str) or isinstance(key, unicode):
            index = self.alphabet[key]
            if index >= len(self.count_vector):
                self.grow()
            return self.count_vector[index]
        else:
            raise TypeError("Key must be either int (index) or string (word). Got %s" % key)

    def __setitem__(self, key, value):
        self.grow_to_alphabet()
        if isinstance(key, int):
            self.count_vector[key] = value
        elif isinstance(key, str) or isinstance(key, unicode):
            index = self.alphabet[key]
            if index >= len(self.count_vector):
                self.grow()
            self.count_vector[index] = value
        else:
            raise TypeError("Key must be either int (index) or string (word). Got %s" % key)

    def __add__(self, other):
        assert(self.alphabet == other.alphabet)
        self.count_vector[:len(self.alphabet)] += other.count_vector[:len(self.alphabet)]
    
    def __len__(self):
        return len(self.alphabet)

    @classmethod
    def reindex(cls, lv, new_alphabet):
        """Create a new LV with the new alphabet 
        
        Reindex the word indices and return a new lv
        """
        new_lv = LexicalOccurrenceVector(alphabet=new_alphabet)
        for i in range(len(lv)):
            word_i = lv.alphabet[i]
            new_lv[word_i] = lv[i]
        return new_lv

    def grow_to_alphabet(self):
        """Grow the count vector to catch up with the alphabet

        We need this function because the alphabet can be shared by
        many vectors and the alphabet can keep growing
        """
        if len(self.alphabet) > len(self.count_vector):
            self.grow()

    def fit_to_alphabet(self):
        """Shrink or grow the count vector to match the alphabet exactly
        """
        if len(self.count_vector) > len(self.alphabet):
            self.count_vector = self.count_vector[:len(self.alphabet)]
        elif len(self.count_vector) < len(self.alphabet):
            new_vector = np.zeros(len(self.alphabet))
            new_vector[:len(self.count_vector)] = self.count_vector
            self.count_vector = new_vector

    def grow(self):
        """Grow the count vector by an appropriate amount"""
        additional_length = max(int(round(len(self.count_vector) * self.growth_factor)),
                len(self.count_vector) - len(self.alphabet))
        new_vector = np.concatenate((self.count_vector, np.zeros(additional_length)))
        self.count_vector = new_vector

    def kl_div(self, lv):
        p = self.vector + 1e-6
        p = p / np.sum(p)
        q = lv.vector + 1e-6
        q = q / np.sum(q)
        return np.sum(np.where(p != 0, p * np.log(p/q), 0))

    def jensen_shannon(self, lv):
        p = self.pdf
        q = lv.pdf
        avg = (p + q) / 2
        with np.errstate(all='ignore'):
            div_p = np.sum(np.where(p != 0.0, p * (np.log(p) - np.log(avg)), 0))
            div_q = np.sum(np.where(q != 0.0, q * (np.log(q) - np.log(avg)), 0))
        return (div_p + div_q) / 2

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    pass
    
