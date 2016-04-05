import codecs
import sys
import json

from nets.bilinear_layer import NeuralNet
from nets.data_reader import DRelation
import nets.dense_feature_functions as df

import theano

class DiscourseParser(object):

    def __init__(self, model_file, dict_file):
        self.net = NeuralNet.load(model_file)
        self.word2vec_ff = get_word2vec_ff(dict_file, 'sum_pool')
        self.index_to_label = {}
        for label, label_index in self.net.label_alphabet.items():
            self.index_to_label[label_index] = label

    def classify_sense(self, data_dir, output_dir):
        relation_file = '%s/relations-no-senses.json' % data_dir
        parse_file = '%s/parses.json' % data_dir
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        relation_dicts = [json.loads(x) for x in open(relation_file)]
        relation_list = [DRelation(x, parse) for x in relation_dicts ]
        data_matrices = self.word2vec_ff(relation_list)        

        classify = theano.function(self.net.input, self.net.predict[0])
        predicted_labels = classify(*data_matrices)
        output = codecs.open('%s/output.json' % output_dir, 'wb', encoding ='utf8')
        for i, relation_dict in enumerate(relation_dicts):
            relation_dict['Sense'] = [self.index_to_label[predicted_labels[i]]]
            relation_dict['Arg1']['TokenList'] = [x[2] for x in relation_dict['Arg1']['TokenList']]
            relation_dict['Arg2']['TokenList'] = [x[2] for x in relation_dict['Arg2']['TokenList']]
            relation_dict['Connective']['TokenList'] = \
                    [x[2] for x in relation_dict['Connective']['TokenList']]
            if len(relation_dict['Connective']['TokenList']) > 0:
                relation_dict['Type'] = 'Explicit'
            else:
                relation_dict['Type'] = 'Implicit'
            output.write(json.dumps(relation_dict) + '\n')

def get_word2vec_ff(dic_tfile, projection):
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
        raise ValueError('projection must be one of {mean_pool, sum_pool, max_pool, top}. Got %s '\
                % projection)

if __name__ == '__main__':
    language = sys.argv[1]
    input_dataset = sys.argv[2]
    input_run = sys.argv[3]
    output_dir = sys.argv[4]
    if language == 'en':
        model_file = 'conll_ff_test.pkl'
        dict_file = 'GoogleNews-vectors-negative300.txt'
    elif language == 'zh':
        model_file = 'conll_zh_ff_test.pkl'
        dict_file = 'zh_gigaword3-skipgram250.txt'
    parser = DiscourseParser(model_file, dict_file)
    parser.classify_sense(input_dataset, output_dir)

