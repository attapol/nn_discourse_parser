from collections import Counter
import sys
import numpy as np
import scipy as sp
from tpl.language.lexical_structure import WordEmbeddingMatrix
import cognitive_disco.dense_feature_functions as df

def _get_word2vec_ff(num_units, projection):
    if num_units == 50:
        dict_file = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/wsj-skipgram50.txt'
    elif num_units == 100:
        dict_file = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/wsj-skipgram100.txt'
    elif num_units == 300:
        dict_file = '/home/j/llc/tet/nlp/lib/lexicon/google_word_vector/GoogleNews-vectors-negative300-wsj_vocab.txt'
    else:
        raise ValueError('num units must be {50, 100, 300}. Got %s ' % num_units)

    word2vec = df.EmbeddingFeaturizer(dict_file)
    if projection == 'mean_pool':
        return word2vec.mean_args
    elif projection == 'sum_pool':
        return word2vec.additive_args
    elif projection == 'max_pool':
        return word2vec.max_args
    elif projection == 'top':
        return word2vec.top_args
    else:
        raise ValueError('projection must be one of {mean_pool, sum_pool, max_pool, top}. Got %s ' % projection)

def _get_zh_word2vec_ff(num_units, vec_type, projection, cdtb):
    prefix = 'zh_gigaword3'
    if cdtb:
        file_name = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/%s-%s%s-cdtb_vocab.txt' \
                % (prefix, vec_type, num_units)
    else: 
        file_name = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/%s-%s%s.txt' \
                % (prefix, vec_type, num_units)
    word2vec = df.EmbeddingFeaturizer(file_name)
    if projection == 'mean_pool':
        return word2vec.mean_args
    elif projection == 'sum_pool':
        return word2vec.additive_args
    elif projection == 'max_pool':
        return word2vec.max_args
    elif projection == 'top':
        return word2vec.top_args
    else:
        raise ValueError('projection must be one of {mean_pool, sum_pool, max_pool, top}. Got %s ' % projection)

def _sparse_featurize_relation_list(relation_list, ff_list, alphabet=None):
    if alphabet is None:
        alphabet = {}
        grow_alphabet = True
    else:
        grow_alphabet = False
    feature_vectors = []
    print 'Applying feature functions...'
    for relation in relation_list:
        feature_vector_indices = []
        for ff in ff_list:
            feature_vector = ff(relation)
            for f in feature_vector:
                if grow_alphabet and f not in alphabet:
                    alphabet[f] = len(alphabet)
                if f in alphabet:
                    feature_vector_indices.append(alphabet[f])
        feature_vectors.append(feature_vector_indices)

    print 'Creating feature sparse matrix...'
    feature_matrix = sp.sparse.lil_matrix((len(relation_list), len(alphabet)))
    for i, fv in enumerate(feature_vectors):
        feature_matrix[i, fv] = 1    
    return feature_matrix.tocsr(), alphabet    

def sparse_featurize(relation_list_list, ff_list):
    print 'Featurizing...'
    data_list = []
    alphabet = None
    for relation_list in relation_list_list:
        data, alphabet = _sparse_featurize_relation_list(relation_list, ff_list, alphabet)
        data_list.append(data)
    return (data_list, alphabet)    

def convert_seconds_to_hours(num_seconds):
    m, s = divmod(num_seconds, 60)
    h, m = divmod(m, 60)
    return (h, m, s)

def compute_mi(feature_matrix, label_vector):
    """Compute mutual information of each feature 

    """
    num_labels = np.max(label_vector) + 1
    num_features = feature_matrix.shape[1]
    num_rows = feature_matrix.shape[0]
    total = num_rows + num_labels

    c_y = np.zeros(num_labels)
    for l in label_vector:
        c_y[l] += 1.0
    c_y += 1.0

    c_x_y = np.zeros((num_features, num_labels)) 
    c_x = np.zeros(num_features)
    for i in range(num_rows):
        c_x_y[:, label_vector[i]] += feature_matrix[i, :]
        c_x += feature_matrix[i, :]
    c_x_y += 1.0
    c_x += 1.0

    c_x_c_y = np.outer(c_x, c_y)
    c_not_x_c_y = np.outer((total - c_x), c_y)
    c_not_x_y = c_y - c_x_y

    inner = c_x_y / total * np.log(c_x_y * total / c_x_c_y) + \
            c_not_x_y / total * np.log(c_not_x_y * total / c_not_x_c_y) 
    mi_x = inner.sum(1)    
    return mi_x

def prune_feature_matrices(feature_matrices, mi, num_features):
    sorted_indices = mi.argsort()[-num_features:]
    return [x[:, sorted_indices] for x in feature_matrices]

class BrownDictionary(object):

    def __init__(self):
        self.word_to_brown_mapping = {}
        self.num_clusters = 0
        brown_cluster_file_name  = 'brown-rcv1.clean.tokenized-CoNLL03.txt-c3200-freq1.txt'
        #brown_cluster_file_name  = 'brown-rcv1.clean.tokenized-CoNLL03.txt-c320-freq1.txt'
        #brown_cluster_file_name  = 'brown-rcv1.clean.tokenized-CoNLL03.txt-c100-freq1.txt'
        self._load_brown_clusters('resources/%s' % brown_cluster_file_name)

    def _load_brown_clusters(self, path):
        try:
            lexicon_file = open(path)
        except:
            print 'fail to load brown cluster data'
        cluster_set = set()
        for line in lexicon_file:
            cluster_assn, word, _ = line.split('\t')
            if cluster_assn not in cluster_set:
                cluster_set.add(cluster_assn)
            self.word_to_brown_mapping[word] = len(cluster_set) - 1
        self.num_clusters = len(cluster_set)

    def _get_brown_cluster_bag(self, tokens):
        bag = set()
        for token in tokens:
            if token in self.word_to_brown_mapping:
                cluster_assn = self.word_to_brown_mapping[token]
                if cluster_assn not in bag:
                    bag.add(cluster_assn)
        return bag

    def get_brown_sparse_matrices_relations(self, relations):
        X1 = sp.sparse.csr_matrix((len(relations), self.num_clusters),dtype=float)    
        X2 = sp.sparse.csr_matrix((len(relations), self.num_clusters),dtype=float)    
        for i, relation in enumerate(relations):
            bag1 = self._get_brown_cluster_bag(relation.arg_tokens(1))
            for cluster in bag1:
                X1[i, cluster] = 1.0
            bag2 = self._get_brown_cluster_bag(relation.arg_tokens(2))
            for cluster in bag2:
                X2[i, cluster] = 1.0
        return (X1, X2)

    def get_brown_matrices_data(self, relation_list_list, use_sparse):
        """Extract sparse 

        For each directory, returns
            (X1, X2, Y) 
            X1 and X2 are sparse matrices from arg1 and arg2 respectively.
            Y is an integer vector of type int32
        """
        data = []
        alphabet = None
        # load the data
        for relation_list in relation_list_list:
            # turn them into a data matrix
            print 'Making matrices' 
            X1, X2 = self.get_brown_sparse_matrices_relations(relation_list)    
            if not use_sparse:
                X1 = X1.toarray()
                X2 = X2.toarray()
            Y, alphabet = level2_labels(relation_list, alphabet)
            data.append((X1, X2, Y))
        return (data, alphabet)

def label_vectorize(relation_list_list, lf):
    alphabet = {}
    for i, valid_label in enumerate(lf.valid_labels()):
        alphabet[valid_label] = i 

    label_vectors = []
    for relation_list in relation_list_list:
        label_vector = [alphabet[lf.label(x)] for x in relation_list]
        label_vectors.append(np.array(label_vector, np.int64))
    return label_vectors, alphabet

def compute_baseline_acc(label_vector):
    label_counter = Counter()
    for label in label_vector:
        label_counter[label] += 1.0
    _, freq = label_counter.most_common(1)[0]
    return round(freq / len(label_vector), 4)

def convert_level2_labels(relations):
    # TODO: this is not enough because we have to exclude some tinay classes
    new_relation_list = []
    for relation in relations:
        split_sense = relation.senses[0].split('.')
        if len(split_sense) >= 2:
            relation.relation_dict['Sense']= ['.'.join(split_sense[0:2])]
            new_relation_list.append(relation)
    return new_relation_list

def level2_labels(relations, alphabet=None):
    if alphabet is None:
        alphabet = {}
        label_set = set()
        for relation in relations:
            label_set.add(relation.senses[0])
        print label_set
        sorted_label = sorted(list(label_set))
        for i, label in enumerate(sorted_label):
            alphabet[label] = i
    label_vector = []
    for relation in relations:
        if relation.senses[0] not in alphabet:
            alphabet[relation.senses[0]] = len(alphabet)
        label_vector.append(alphabet[relation.senses[0]])
    return np.array(label_vector, np.int64), alphabet

def get_wbm(num_units):
    if num_units == 50:
        dict_file = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/wsj-skipgram50.npy'
        vocab_file = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/wsj-skipgram50_vocab.txt'
    elif num_units == 100:
        dict_file = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/wsj-skipgram100.npy'
        vocab_file = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/wsj-skipgram100_vocab.txt'
    elif num_units == 300:
        #dict_file = '/home/j/llc/tet/nlp/lib/lexicon/google_word_vector/GoogleNews-vectors-negative300.npy'
        dict_file = \
                '/home/j/llc/tet/nlp/lib/lexicon/google_word_vector/GoogleNews-vectors-negative300-wsj_vocab.npy'
        vocab_file = \
                '/home/j/llc/tet/nlp/lib/lexicon/google_word_vector/GoogleNews-vectors.negative300-wsj_vocab-vocab.txt'
    else:
        # this will crash the next step and te's too lazy to make it throw an exception.
        dict_file = None
        vocab_file = None
    wbm = WordEmbeddingMatrix(dict_file, vocab_file)
    return wbm

def get_zh_wbm(num_units):
    dict_file = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/zh_gigaword3-skipgram%s-cdtb_vocab.npy' % num_units
    vocab_file = '/home/j/llc/tet/nlp/lib/lexicon/homemade_word_vector/zh_gigaword3-skipgram%s-cdtb_vocab-vocab.txt' % num_units
    return WordEmbeddingMatrix(dict_file, vocab_file)

def set_logger(file_name, dry_mode=False):
    if not dry_mode:
        sys.stdout = open('%s.log' % file_name, 'w', 1)
    json_file = open('%s.json' % file_name, 'w', 1)
    return json_file

import cognitive_disco.base_label_functions as l
from cognitive_disco.nets.learning import DataTriplet
from cognitive_disco.data_reader import extract_implicit_relations
from cognitive_disco.nets.lstm import prep_serrated_matrix_relations

def get_data_srm(dir_list, wbm, max_length=75):
    sense_lf = l.SecondLevelLabel()
    relation_list_list = [extract_implicit_relations(dir, sense_lf) 
            for dir in dir_list]
    data_list = []
    for relation_list in relation_list_list:
        data = prep_serrated_matrix_relations(relation_list, wbm, max_length)
        data_list.append(data)
    label_vectors, label_alphabet = \
            label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(
            data_list, [[x] for x in label_vectors], [label_alphabet])
    return data_triplet

def make_givens_srm(givens, input_vec, T_training_data, 
            output_vec, T_training_data_label, start_idx, end_idx):
    """Make the 'given' dict for SGD training for discourse

    the input vecs should be [X1 mask1 X2 mask2]
    X1 and X2 are TxNxd serrated matrices.
    """
    # first arg embedding and mask
    givens[input_vec[0]] = T_training_data[0][:,start_idx:end_idx, :]
    givens[input_vec[1]] = T_training_data[1][:,start_idx:end_idx]

    # second arg embedding and mask
    givens[input_vec[2]] = T_training_data[2][:,start_idx:end_idx, :]
    givens[input_vec[3]] = T_training_data[3][:,start_idx:end_idx]

    for i, output_var in enumerate(output_vec):
        givens[output_var] = T_training_data_label[i][start_idx:end_idx]

def make_givens(givens, input_vec, T_training_data, 
            output_vec, T_training_data_label, start_idx, end_idx):
    """Make the 'given' dict for SGD training for discourse

    the input vecs should be [X1 X2]
    X1 and X2 are Nxd matrices.
    """
    # first arg embedding and mask
    givens[input_vec[0]] = T_training_data[0][start_idx:end_idx]
    givens[input_vec[1]] = T_training_data[1][start_idx:end_idx]
    for i, output_var in enumerate(output_vec):
        givens[output_var] = T_training_data_label[i][start_idx:end_idx]

if __name__ == '__main__':
    fm = np.array([ [1, 0, 1],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 1, 0],
                    [0, 0, 0]])
    lv = np.array([0,0,0,1,1,1])
    compute_mi(fm, lv)

