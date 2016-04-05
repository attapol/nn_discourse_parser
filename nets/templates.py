import numpy as np
import theano.tensor as T

from cognitive_disco.nets.bilinear_layer import \
        InputLayer, make_multilayer_net_from_layers, MaskedInputLayer, NeuralNet
from cognitive_disco.nets.learning import AdagradTrainer
from cognitive_disco.nets.lstm import SerialLSTM

def build_ff_network(data_triplet, num_hidden_layers, num_hidden_units, 
        learning_rate=0.001, lr_smoother = 0.01, dropout=False):
    rng = np.random.RandomState(100)
    X_list = [T.matrix(), T.matrix()]
    input_layers = [InputLayer(rng, data_triplet.input_dimensions()[0], False, X_list[0]),
            InputLayer(rng, data_triplet.input_dimensions()[1], False, X_list[1])]
    net, layers = make_multilayer_net_from_layers(
            input_layers, Y=T.lvector(), use_sparse=False,
            num_hidden_layers=num_hidden_layers, 
            num_hidden_units=num_hidden_units, 
            num_output_units=data_triplet.output_dimensions()[0],
            output_activation_fn=T.nnet.softmax,
            dropout=dropout)
    trainer = AdagradTrainer(net, net.crossentropy, 
            learning_rate, lr_smoother, data_triplet, ff_make_givens)
    return net, trainer

def ff_make_givens(givens, input_vec, T_training_data, 
            output_vec, T_training_data_label, start_idx, end_idx):
    for i, input_var in enumerate(input_vec):
        givens[input_var] = T_training_data[i][start_idx:end_idx]

    for i, output_var in enumerate(output_vec):
        givens[output_var] = T_training_data_label[i][start_idx:end_idx]

def build_lstm_network(data_triplet, num_hidden_layers, num_hidden_units, proj_type,
        learning_rate=0.001, lr_smoother = 0.01):
    rng = np.random.RandomState(100)
    num_units = data_triplet.training_data[0].shape[2]
    arg1_model = SerialLSTM(rng, num_units, proj_type)
    arg2_model = SerialLSTM(rng, num_units, proj_type)
    arg1_pooled = MaskedInputLayer(rng, num_units, proj_type,
            arg1_model.activation_train, arg1_model.mask, arg1_model.c_mask)
    arg2_pooled = MaskedInputLayer(rng, num_units, proj_type,
            arg2_model.activation_train, arg2_model.mask, arg2_model.c_mask)
    _, pred_layers = make_multilayer_net_from_layers(
            input_layers=[arg1_pooled, arg2_pooled],
            Y=T.lvector(), use_sparse=False,
            num_hidden_layers=num_hidden_layers,
            num_hidden_units=num_hidden_units,
            num_output_units=data_triplet.output_dimensions()[0],
            output_activation_fn=T.nnet.softmax,
            dropout=False)
    net = NeuralNet([arg1_model, arg2_model] + pred_layers)
    net.input = arg1_model.input + arg2_model.input
    trainer = AdagradTrainer(net, net.crossentropy, 
            learning_rate, lr_smoother, data_triplet, SerialLSTM.make_givens)
    return net, trainer

