import codecs
import json
import sys
import timeit

import numpy as np
import theano

import nets.base_label_functions as l
import nets.nets.util as util
from nets.data_reader import DRelation
from nets.learning import DataTriplet
from nets.templates import build_ff_network

def extract_non_explicit_relations(data_folder, label_function=None):
    parse_file = '%s/parses.json' % data_folder
    parse = json.load(codecs.open(parse_file, encoding='utf8'))

    relation_file = '%s/relations.json' % data_folder
    relation_dicts = [json.loads(x) for x in open(relation_file)]
    relations = [DRelation(x, parse) for x in relation_dicts if x['Type'] != 'Explicit']
    if label_function is not None:
        relations = [x for x in relations if label_function.label(x) is not None]
    return relations

def load_data(dir_list, relation_to_matrices_fn, sense_lf=None):
    if sense_lf is None:
        sense_lf = l.OriginalLabel()
    relation_list_list = [extract_non_explicit_relations(dir, sense_lf) for dir in dir_list]
    data_list = [relation_to_matrices_fn(relation_list) for relation_list in relation_list_list]
    label_vectors, label_alphabet = util.label_vectorize(relation_list_list, sense_lf)
    data_triplet = DataTriplet(data_list, [[x] for x in label_vectors], [label_alphabet])
    return data_triplet

def implicit_second_level_ff_train(args):
    dir_list = ['conll15-st-03-04-15-train', 'conll15-st-03-04-15-dev', 'conll15-st-03-04-15-test']
    num_units = 300
    projection = 'sum_pool'
    word2vec_ff = util._get_word2vec_ff(num_units, projection)
    data_triplet = load_data(dir_list, word2vec_ff, l.SecondLevelLabel())
    train(num_hidden_layers=1, num_hidden_units=400, model_name='second_level_ff',
            data_triplet=data_triplet, minibatch_size=53)

def implicit_conll_zh_ff_train(args):
    dir_list = ['../cognitive_disco/conll16st-zh-01-08-2016-train',
            '../cognitive_disco/conll16st-zh-01-08-2016-dev',
            '../cognitive_disco/conll16st-zh-01-08-2016-test']
    num_units = 250
    projection = 'sum_pool'
    vec_type = 'skipgram'
    num_hidden_layers = 2
    num_hidden_units = 250
    word2vec_ff = util._get_zh_word2vec_ff(num_units, vec_type, projection, cdtb=True)
    data_triplet = load_data(dir_list, word2vec_ff)

    model_name ='conll_zh_ff'
    net, best_dev_model, best_test_model = \
            train(num_hidden_layers=num_hidden_layers, num_hidden_units=num_hidden_units,
                    model_name=model_name, data_triplet=data_triplet,
                    minibatch_size=None)
    net.label_alphabet = data_triplet.label_alphabet_list[0]

    eval_model(net, data_triplet.dev_data, data_triplet.dev_data_label[0], best_dev_model)
    net.save('%s_dev.pkl' % model_name, best_dev_model)
    eval_model(net, data_triplet.test_data, data_triplet.test_data_label[0], best_test_model)
    net.save('%s_test.pkl' % model_name, best_test_model)


def implicit_conll_ff_train(args):
    dir_list = ['../cognitive_disco/conll15-st-03-04-15-train', 
            '../cognitive_disco/conll15-st-03-04-15-dev', 
            '../cognitive_disco/conll15-st-03-04-15-test']
    num_units = 300
    projection = 'sum_pool'
    num_hidden_layers = 2
    num_hidden_units = 300

    word2vec_ff = util._get_word2vec_ff(num_units, projection)
    data_triplet = load_data(dir_list, word2vec_ff)
    model_name ='conll_ff'
    net, best_dev_model, best_test_model = \
            train(num_hidden_layers=num_hidden_layers, num_hidden_units=num_hidden_units,
                    model_name=model_name, data_triplet=data_triplet,
                    minibatch_size=None)
    net.label_alphabet = data_triplet.label_alphabet_list[0]

    eval_model(net, data_triplet.dev_data, data_triplet.dev_data_label[0], best_dev_model)
    net.save('%s_dev.pkl' % model_name, best_dev_model)
    eval_model(net, data_triplet.test_data, data_triplet.test_data_label[0], best_test_model)
    net.save('%s_test.pkl' % model_name, best_test_model)


def eval_model(net, data, label, params=None):
    if params is not None:
        for param, best_param in zip(net.params, params):
            param.set_value(best_param)
    classify = theano.function(net.input, net.predict[0])
    predicted_labels = classify(*data)
    accuracy = float(np.sum(predicted_labels == label)) / len(label)
    print accuracy


def train(num_hidden_layers, num_hidden_units, model_name, data_triplet, 
        minibatch_size=None, dry=False):
    if dry:
        num_reps = 2
        n_epochs = 2
        model_name = '_model_trainer_dry_run'
    else:
        num_reps = 50
        n_epochs = 5

    json_file = util.set_logger(model_name, dry)

    baseline_dev_acc = util.compute_baseline_acc(data_triplet.dev_data_label[0])
    baseline_test_acc = util.compute_baseline_acc(data_triplet.test_data_label[0])

    best_dev_so_far = 0.0
    best_test_so_far = 0.0
    best_dev_model = None
    best_test_model = None
    random_batch_size = minibatch_size == None

    net, trainer = build_ff_network(data_triplet, num_hidden_layers, num_hidden_units)
    for rep in xrange(num_reps):
        random_seed = rep + 10 
        rng = np.random.RandomState(random_seed)
        net.reset(rng)
        trainer.reset()
        if random_batch_size:
            minibatch_size = np.random.randint(20, 60)
        start_time = timeit.default_timer()
        best_iter, best_dev_acc, best_test_acc, best_parameters = \
                trainer.train_minibatch_triplet(minibatch_size, n_epochs)
        best_dev_acc = round(best_dev_acc, 4)
        best_test_acc = round(best_test_acc, 4)
        end_time = timeit.default_timer()
        rep_elapsed = end_time - start_time
        print '== Rep %s : Training process takes %s seconds' % (rep, rep_elapsed)
        print '== The training process will require %s hour %s minutes %s seconds more.' % \
                util.convert_seconds_to_hours(rep_elapsed * (num_reps - rep - 1))
        print '== Best iteration is %s; ' % best_iter + \
                'Test accuracy = %s; ' % best_test_acc + \
                'Baseline test accuracy = %s; ' % baseline_test_acc + \
                'Best dev accuracy = %s; ' % best_dev_acc + \
                'Baseline dev accuracy = %s' % baseline_dev_acc 

        if best_test_acc > best_test_so_far:
            best_test_model = best_parameters
            best_test_so_far = best_test_acc
        if best_dev_acc > best_dev_so_far:
            best_dev_model = best_parameters
            best_dev_so_far = best_dev_acc

        result_dict = {
                'test accuracy': best_test_acc,
                'baseline test accuracy': baseline_test_acc,
                'best dev accuracy': best_dev_acc,
                'baseline dev accuracy': baseline_dev_acc,
                'best iter': best_iter,
                'random seed': random_seed,
                'num rep': rep,
                'minibatch size': minibatch_size,
                'learning rate': trainer.learning_rate,
                'lr smoother': trainer.lr_smoother,
                'experiment name': model_name,
                'num hidden units': num_hidden_units,
                'num hidden layers': num_hidden_layers,
                'cost function': 'crossentropy',
                'dropout' : False
                }
        json_file.write('%s\n' % json.dumps(result_dict, sort_keys=True))
    return net, best_dev_model, best_test_model

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    globals()[experiment_name](sys.argv[3:])

