# -*- mode: python; tab-width: 4;

# Copyright (C) 2016 Andres Hernandez
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the license for more details.
from __future__ import print_function
from six import string_types

from functools import partial
from os.path import isfile
from copy import deepcopy
from joblib import Parallel, delayed
import dill
import data_utils as du
import instruments as inst
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization, Input
from keras.layers.merge import add, concatenate
from keras import backend as K
from keras.models import load_model
from copy import copy

seed = 1027
n_jobs = 2

def proper_name(name):
    name = name.replace(" ", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(",", "_")
    name = name.replace("-", "_")
    name = name.replace("+", "p")
    return name
            
def flatten_name(name, node='Models', risk_factor='IR'):
    name = proper_name(name)
    return node + '/' + risk_factor + '/' + name
            
class NeuralNetwork(object):
    def __init__(self, model_dict, model_callback=None, preprocessing=None,
                 lr=0.001, loss='mean_squared_error', prefix='', postfix='',
                 method=Nadam, train_file=None, do_transform=True, **kwargs):
        self._model_dict = model_dict
        self.model_name = model_dict['name']
        if 'file_name' in model_dict:
            self._file_name = model_dict['file_name']
        else:
            self._file_name = self.model_name
        if 'transformation' in model_dict:
            self._func = model_dict['transformation']
        else:            
            self._func = None
        
        if 'inverse_transformation' in model_dict:
            self._inv_func = model_dict['inverse_transformation']
        else:            
            self._inv_func = None  
        
        self.name = prefix + self.model_name
        self.postfix = postfix
        
        if train_file is not None:
            self.train_file_name = train_file
        else:
            self.train_file_name = flatten_name(self.name)
            self.train_file_name = self.train_file_name.lower().replace('/', '_')
            self.train_file_name = du.data_dir + self.train_file_name
		
        self.do_transform = do_transform
        
        #Get training data if required
        if 'valid_size' in kwargs:
            self.valid_size = kwargs['valid_size']
        else:
            self.valid_size = 0.2
            
        if 'test_size' in kwargs:
            self.test_size = kwargs['test_size']
        else:
            self.test_size = 0.2
            
        if 'total_size' in kwargs:
            self.total_size = kwargs['total_size']
        else:
            self.total_size = 1.0
        self.__get_data()
        
        self.model = None
        self.history = None
        self.method = method
        self._model_callback = model_callback
        self.lr = lr
        self.loss = loss
        self._preprocessing = preprocessing


    def __get_data(self):
        self.x_train = None
        self.x_valid = None
        self.x_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        if self.train_file_name is None:
            self._data = None
            self._transform = None
        else:
            # File name is h5_model_node + _ + risk factor + '_' + self.name
            self._data = inst.retrieve_swo_train_set(self.train_file_name, 
                                                     self.do_transform, 
                                                     self._func, 
                                                     self._inv_func,
                                                     valid_size=self.valid_size,
                                                     test_size=self.test_size,
                                                     total_size=self.total_size)
            self._transform = self._data['transform']

    def file_name(self):
        # File name is self.name + _nn
        file_name = proper_name(self._file_name) + '_nn' + self.postfix
        file_name = file_name.lower().replace('/', '_')
        return du.data_dir + file_name

    def __tofile(self):
        if self.model:
            file_name = self.file_name() + '.h5'
            self.model.save(file_name)

    def __fromfile(self):
        file_name = self.file_name() + '.h5'
        if isfile(file_name):
            self.model = load_model(file_name)
        else:
            self.model = None

    def __getstate__(self):
        self.__tofile()
        # keras model should not be saved by dill, but rather use its own 
        # method, however deepcopy encounters recursion if left in there
        model = self.model
        del self.__dict__['model']
        
        d = deepcopy(self.__dict__)
        self.model = model
        del d['_data']
        del d['x_train']
        del d['x_valid']
        del d['x_test']
        del d['y_train']
        del d['y_valid']
        del d['y_test']
        return d


    def __setstate__(self, d):
        self.__dict__ = d
        self._data = self.__get_data()
        history = self.history
        self.train(nb_epochs=0)
        self.history = history
        self.__fromfile()


    def train(self, nb_epochs):
        self.y_train = self._data['y_train']
        self.y_valid = self._data['y_valid']
        self.y_test = self._data['y_test']
        method = self.method(lr=self.lr)
        self.x_train, self.x_valid, self.x_test, self.model, self.history = \
            self._model_callback(self._data, method, self.loss, 
                                 nb_epochs=nb_epochs)


    def fit(self, nb_epochs):
        if self.model is None:
            raise RuntimeError('Model not yet instantiated')
        batch_size = self.history['params']['batch_size']
        history2 = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, 
                          nb_epoch=nb_epochs, verbose=2, 
                          validation_data=(self.x_valid, self.y_valid))
        self.history = {'history': history2.history,
                        'params': history2.params}


    def test(self, batch_size=16):
        return self.model.evaluate(self.x_test, self.y_test, batch_size=batch_size)


    def predict(self, data):
        if self.model is None:
            raise RuntimeError('Model not yet instantiated')
        if self._preprocessing is not None:
            data = self._preprocessing(data)
        y = self.model.predict(data)
        if self._transform is not None:
            y = self._transform.inverse_transform(y)
        return y


def logarithmic_mean_squared_error(y_true, y_pred):
    return -K.mean(K.log(1.-K.clip(K.square(y_pred-y_true),0., 1.-K.epsilon())))

#_paper
def hullwhite_fnn_model(data, method, loss, exponent=6, nb_epochs=0, 
                        batch_size=16, activation='tanh', layers=4, 
                        init='he_normal', dropout=0.5, dropout_first=None, 
                        dropout_middle=None, dropout_last=None,
                        early_stop=125, lr_patience=40,
                        reduce_lr=0.5, reduce_lr_min=0.000009,
                        residual_cells=1, **kwargs):
    assert(isinstance(activation, string_types))
    if activation == "elu":
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 1.0
        activation = ELU(alpha)
    elif activation == "rbf":
        activation = Activation(rbf)
    else:
        activation = Activation(activation)
    
    x_train = data['x_train']
    x_valid = data['x_valid']
    x_test = data['x_test']
    y_train = data['y_train']
    y_valid = data['y_valid']
    
    if dropout_first is None:
        dropout_first = dropout
    if dropout_middle is None:
        dropout_middle = dropout_first
    if dropout_last is None:
        dropout_last = dropout_middle
        
    assert residual_cells >= 0
    
    if residual_cells == 0:
        print('Simple with no BN or residual')
    else:
        print('Residual with BN (ex Out) - Activation before Dense - with %s residual cells' % residual_cells)
    print(' - Early Stop: Patience %s; Reduce LR Patience %s, Factor: %s, Min: %s' % \
            (early_stop, lr_patience, reduce_lr, reduce_lr_min))
    print(' - Exp:%s, Layer:%s, df:%s, dm:%s, dl:%s' % \
            (exponent, layers, dropout_first, dropout_middle, dropout_last))
    print(' - Loss:%s' % loss)
    #A copy of the activation layer needs to be used, instead of the layer
    #directly because otherwise keras will not be able to load a saved configuration
    #from a json file
    act_idx = 1
    inp = Input(shape=(x_train.shape[1],))
    ly = BatchNormalization()(inp)
    ly = Dense(2**exponent, kernel_initializer=init)(ly)
    act = copy(activation)
    act.name = act.name + "_" + str(act_idx)
    act_idx = act_idx + 1
    ly = act(ly)
    ly = Dropout(dropout_first)(ly)
    if residual_cells > 0:
        for i in range(layers-1):
            middle = BatchNormalization()(ly)
            act = copy(activation)
            act.name = act.name + "_" + str(act_idx)
            act_idx = act_idx + 1
            middle = act(middle)
            middle = Dense(2**exponent, kernel_initializer=init)(middle)
            middle = Dropout(dropout_middle)(middle)
            for j in range(residual_cells-1):
                act = copy(activation)
                act.name = act.name + "_" + str(act_idx)
                act_idx = act_idx + 1
                middle = act(middle)
                middle = Dense(2**exponent, kernel_initializer=init)(middle)
                middle = Dropout(dropout_middle)(middle)
            ly = add([ly, middle])
        ly = Dropout(dropout_last)(ly)
    else:
        for i in range(layers-1):
            ly = Dense(2**exponent, kernel_initializer=init)(ly)
            act = copy(activation)
            act.name = act.name + "_" + str(act_idx)
            act_idx = act_idx + 1
            ly = act(ly)
            ly = Dropout(dropout_middle)(ly)
    ly = Dense(y_train.shape[1], kernel_initializer=init)(ly)
    nn = Model(inputs=inp, outputs=ly)
    nn.compile(method, loss=loss)
    
    if nb_epochs > 0:
        callbacks = []
        if early_stop is not None:
            earlyStopping = EarlyStopping(monitor='val_loss', patience=early_stop)
            callbacks.append(earlyStopping)
        if reduce_lr is not None:
            reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr,
                                         patience=lr_patience, min_lr=reduce_lr_min, 
                                         verbose=1)
            callbacks.append(reduceLR)
        history2 = nn.fit(x_train, y_train, batch_size=batch_size, 
                          epochs=nb_epochs, verbose=2, callbacks=callbacks,
                          validation_data=(x_valid, y_valid))
        history = {'history': history2.history,
                   'params': history2.params}
    else:
        history = {'history': [],
                   'params': []}
    return (x_train, x_valid, x_test, nn, history)
		

def hullwhite_cnn_model(data, method, loss, exponent=8, dropout_conv=0.2, 
                        dropout_dense=0.5, nb_epochs=0, batch_size=16, 
                        nb_filters_swo=64, nb_filters_ir=32, nb_pool=2, 
                        nb_conv_swo=3, nb_conv_ir=3, nb_opts=13, nb_swaps=12,
                        alpha=1.0):
    #TODO: Interface changed, fix data calls
    x_swo_train = data['x_swo_train']
    x_swo_valid = data['x_swo_valid']
    x_swo_test = data['x_swo_test']
    x_ir_train = data['x_ir_train']
    x_ir_valid = data['x_ir_valid']
    x_ir_test = data['x_ir_test']
    y_train = data['y_train']
    y_valid = data['y_valid']
    
    x_swo_train = x_swo_train.reshape(x_swo_train.shape[0], 1, nb_swaps, nb_opts)
    x_swo_valid = x_swo_valid.reshape(x_swo_valid.shape[0], 1, nb_swaps, nb_opts)
    if x_swo_test is not None:
        x_swo_test = x_swo_test.reshape(x_swo_test.shape[0], 1, nb_swaps, nb_opts)
    
    x_ir_train = x_ir_train.reshape(x_ir_train.shape[0], 1, 1, x_ir_train.shape[1])
    x_ir_valid = x_ir_valid.reshape(x_ir_valid.shape[0], 1, 1, x_ir_valid.shape[1])
    if x_ir_test is not None:
        x_ir_test = x_ir_test.reshape(x_ir_test.shape[0], 1, 1, x_ir_test.shape[1])
    
    x_train = [x_swo_train, x_ir_train]
    x_valid = [x_swo_valid, x_ir_valid]
    if x_swo_test is not None:
        x_test = [x_swo_test, x_ir_test]
    else:
        x_test = None
    nn2D = Sequential()
    nn2D.add(Convolution2D(nb_filters_swo, nb_conv_swo, nb_conv_swo,
                           input_shape=(1, nb_swaps, nb_opts)))
    nn2D.add(Activation(ELU(alpha)))
    nn2D.add(Convolution2D(nb_filters_swo, nb_conv_swo, nb_conv_swo))
    nn2D.add(Activation(ELU(alpha)))
    nn2D.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    nn2D.add(Dropout(dropout_conv))
    nn2D.add(Flatten())
    
    nn1D = Sequential()
    nn1D.add(Convolution2D(nb_filters_ir, 1, nb_conv_ir,
                           border_mode='valid',
                           input_shape=(1, 1, x_ir_train.shape[-1])))
    nn1D.add(Activation(ELU(alpha)))
    nn1D.add(Convolution2D(nb_filters_ir, 1, nb_conv_ir))
    nn1D.add(Activation(ELU(alpha)))
    nn1D.add(MaxPooling2D(pool_size=(1, nb_pool)))
    nn1D.add(Dropout(dropout_conv))
    nn1D.add(Flatten())
    
    merged = Sequential()
    merged.add(concatenate([nn2D, nn1D], axis=-1))
    
    merged.add(Dense(2**exponent))
    merged.add(Activation(ELU(alpha)))
    merged.add(Dense(2**exponent))
    merged.add(Activation(ELU(alpha)))
    merged.add(Dropout(dropout_dense))
    merged.add(Dense(y_train.shape[1]))
    merged.add(Activation('linear'))
    merged.add(Dropout(dropout_dense))
    merged.compile(method, loss=loss)
    
    if nb_epochs > 0:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=50)
        history2 = merged.fit(x_train, y_train, batch_size=batch_size, 
                              nb_epoch=nb_epochs, verbose=2, callbacks=[earlyStopping],
                              validation_data=(x_valid, y_valid))
        history = {'history': history2.history,
                   'params': history2.params}
    else:
        history = None
    return (x_train, x_valid, x_test, merged, history)


def preprocessing_fnn(x):
        if len(x[0].shape) == 1:
            p = np.concatenate(x)
            p.shape = (1, p.shape[0])
        else:
            p = np.concatenate(x, axis=1)
        return p


def rbf(x):
    #This is not really a radial basis function, but it is similar
    return K.exp(-K.square(x))
    
'''
Helper functions to instantiate neural networks with particular activations,
hyper-parameters, or topologies
'''
def hullwhite_fnn(exponent=6, batch_size=16, lr=0.001, layers=3, 
                  loss='mean_squared_error', activation='tanh',  prefix='', 
                  postfix='', dropout=0.5, dropout_first=None, 
                  dropout_middle=None, dropout_last=None, early_stop=125, 
                  lr_patience=40, reduce_lr=0.5, reduce_lr_min=0.000009,
                  model_dict=inst.g2, residual_cells=1, train_file=None,
                  do_transform=True, **kwargs):
    hwfnn = partial(hullwhite_fnn_model, exponent=exponent, batch_size=batch_size, 
                    activation=activation, layers=layers, dropout=dropout, 
                    dropout_first=dropout_first, dropout_middle=dropout_middle,
                    dropout_last=dropout_last, early_stop=early_stop,
                    lr_patience=lr_patience, reduce_lr=reduce_lr, 
                    reduce_lr_min=reduce_lr_min, residual_cells=residual_cells,
                    **kwargs)
    model = NeuralNetwork(model_dict, hwfnn, lr=lr, loss=loss, 
                          preprocessing=preprocessing_fnn, 
                          prefix=prefix, postfix=postfix, train_file=train_file,
                          do_transform=do_transform, **kwargs)
    return model

def hullwhite_cnn(lr=0.001, exponent=8, dropout_conv=0.2, dropout_dense=0.5, 
                  batch_size=16, nb_filters_swo=64, nb_filters_ir=32,
                  nb_pool=2, nb_conv_swo=3, nb_conv_ir=3, nb_opts=13, 
                  nb_swaps=12, loss='mean_squared_error', prefix='', postfix='',
                  early_stop=125, lr_patience=40, reduce_lr=0.5, 
                  reduce_lr_min=0.000009, model_dict=inst.g2):
    hwcnn = partial(hullwhite_cnn_model, exponent = exponent, dropout_conv=dropout_conv, 
                    dropout_dense=dropout_dense, batch_size=batch_size, 
                    nb_filters_swo=nb_filters_swo, nb_filters_ir=nb_filters_ir, 
                    nb_pool=nb_pool, nb_conv_swo=nb_conv_swo, nb_conv_ir=nb_conv_ir,
                    nb_opts=nb_opts, nb_swaps=nb_swaps)
    return NeuralNetwork(model_dict, hwcnn,
                         lr=lr, loss=loss, preprocessing=lambda x: x,
                         prefix=prefix, postfix=postfix)

'''
Function to save a model to file or load it back
'''
def write_model(model):
    file_name = model.file_name() +'.p'
    print('Saving model to file: %s' % file_name)
    dill.dump(model, open(file_name, 'wb'))


def read_model(file_name):
    print('Reading model from file: %s' % file_name)
    model = dill.load(open(file_name, 'rb'))
    return model


'''
The following testing functions were intended to help with hyper-parameter
optimization.
'''
def test_helper(func, exponent, layer, lr, dropout_first, dropout_middle, 
                dropout_last, alpha, prefix='SWO GBP ', postfix='',
                with_comparison=False):
    print('Test %s, %s, %s, %s, %s %s %s' % (exponent, layer, lr, dropout_first,
                                       dropout_middle, dropout_last, alpha))
    model = func(exponent=exponent, lr=lr, layers=layer, 
                 dropout_first=dropout_first, dropout_middle=dropout_middle,
                 dropout_last=dropout_last, prefix=prefix, postfix=postfix, 
                 alpha=alpha)
    model.train(200)
    val_loss = np.mean(model.history['history']['val_loss'][-5:])
    
#    if with_comparison:
#        swo = inst.get_swaptiongen(inst.hullwhite_analytic)
#        _, values = swo.compare_history(model, dates=dates)
#        
    
    return (val_loss, layer, exponent, lr, dropout_first, dropout_middle, 
            dropout_last, alpha)


def test_fnn(func):
    parameters = [(6, 4, 0.25, 0.25, 0.25, 0.001, 1.0)]
    results = Parallel(n_jobs=n_jobs)(delayed(test_helper)(func, exp, layer, lr, 
                       dof, dom, dol, alpha)
                              for exp, layer, dof, dom, dol, lr, alpha in parameters)

    results = sorted(results, key = lambda x: x[0], reverse=True)
    for result in results:
        print(result)


def test_helper_cnn(func, dropout, lr, exponent, exp_filter_ir,
                    exp_filter_swo, nb_conv_ir, nb_conv_swo,
                    prefix='SWO GBP ', postfix=''):
    print('Test %s, %s, %s, %s, %s, %s, %s' % (dropout, lr, exponent, exp_filter_ir, 
                                   exp_filter_swo, nb_conv_ir, nb_conv_swo))
    model = func(lr=lr, exponent=exponent, dropout_conv=dropout, 
                 dropout_dense=dropout, nb_filters_swo=2**exp_filter_swo, 
                 nb_filters_ir=2**exp_filter_ir, nb_conv_swo=nb_conv_swo, 
                 nb_conv_ir=nb_conv_ir, prefix=prefix, postfix=postfix)
    model.train(500)
    loss = np.mean(model.history['history']['val_loss'][-5:])
    return (loss, dropout, lr, exponent, exp_filter_ir, exp_filter_swo,
            nb_conv_ir, nb_conv_swo)


def test_cnn(func):
    # CNN parameters
    parameters = [(0.2, 0.001, 6, 5, 6, 3, 3)]

    results = Parallel(n_jobs=n_jobs)(delayed(test_helper_cnn)(func, dropout,
                       lr, exponent, exp_filter_ir, exp_filter_swo, conv_ir,
                       conv_swo)
      for dropout,lr,exponent,exp_filter_ir,exp_filter_swo,conv_ir,conv_swo in parameters)
    results = sorted(results, key=lambda x: x[0], reverse=True)
    for result in results:
        print(result)