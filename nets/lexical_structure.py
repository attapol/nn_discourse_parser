import numpy as np

SKIPGRAM_WORD_EMBEDDING_FILE = 'google_word_vector/GoogleNews-vectors-negative300.txt'


class WordEmbeddingDict(object):
    """A simple wrapper around Dict with lazy loading """

    def __init__(self, word_embedding_file):
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

