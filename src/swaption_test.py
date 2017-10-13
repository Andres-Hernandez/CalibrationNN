# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:18:10 2016

@author: hernandeza
"""
from __future__ import print_function

import sys
import getopt
import traceback

traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

import neural_network as nn
import instruments as inst
import data_utils as du
import numpy as np

def run(x, total = 499000., compare=False, epochs=500, prefix='SWO GBP ', 
        postfix='', dropout_first=None, dropout_middle=None, 
        dropout_last=None, dropout=0.2, earlyStopPatience=125, reduceLRPatience=40, 
        reduceLRFactor=0.5, reduceLRMinLR=0.000009, save=True, layers=4, 
        lr=0.001, exponent=6, load=False, file_name=None, model_dict=inst.g2, 
        residual_cells=1, train_file=None, do_transform=True, loss='mean_squared_error'):

    assert residual_cells >= 0
    lossid = "".join(map(lambda x: x[0], loss.split('_')))
    postfix += "_" + lossid + '_lr_%.1e_ex%s_lay%s_d%s' % (lr, exponent, layers, int(dropout*100))
    if residual_cells > 0:
        postfix = postfix + '_bn_res_%s' % residual_cells
    else:
        postfix = postfix + '_simple'
    postfix = postfix + '_rlr_%.1e_rlrmin_%.1e_rlrpat_%s_estop_%s' % (reduceLRFactor, reduceLRMinLR, reduceLRPatience, earlyStopPatience)
    print('run  ' + str(x) + ' ' + postfix)
    if x < 1.0:
        nn.total_size = x
    else:
        nn.total_size = x/total
    nn.valid_size = nn.total_size*0.2
    nn.test_size = 0.0
    
    if load:
        assert(file_name is not None)
        model = nn.read_model(file_name)
    else:
        model = nn.hullwhite_fnn(exponent=exponent, layers=layers, lr=lr,
                                 prefix=prefix, postfix=postfix,
                                 dropout=dropout,
                                 dropout_first=dropout_first, 
                                 dropout_middle=dropout_middle, 
                                 dropout_last=dropout_last, 
                                 earlyStopPatience=earlyStopPatience,
                                 reduceLRPatience=reduceLRPatience, 
                                 reduceLRFactor=reduceLRFactor, 
                                 reduceLRMinLR=reduceLRMinLR,
                                 model_dict=model_dict, 
                                 residual_cells=residual_cells,
                                 train_file=train_file,
                                 do_transform=do_transform,
                                 activation="elu")
        model.train(epochs)
        if save:
            nn.write_model(model)
    
    if compare:
        swo = inst.get_swaptiongen(model_dict)
        (dates, values) = swo.compare_history(model)
        
        file_name = du.data_dir + 'test' + postfix + '_'
        np.save(file_name+'values', values)
        np.save(file_name+'dates', dates)
        np.save(file_name+'val_hist', model.history['history']['val_loss'])
        np.save(file_name+'train_hist', model.history['history']['loss'])
            
    return 0

def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    argl = ['training-data=', 'seed=', 'calibrate-history', 'hull-white-fnn',
            'hull-white-rbf', 'hull-white-cnn', 'exponent=', 'lr=', 
            'layers=', 'dropout=', 'save', 'compare-history', 'test-fnn',
            'test-rbf', 'test-elu', 'test-cnn', 'test-cnn', 'run-fnn=', 
            'gbp-to-h5', 'prefix=', 'postfix=', 'dropout-first=',
            'dropout-middle=', 'dropout-last=', 'hull-white-relu',
            'hull-white-elu', 'epochs=', 'file-name=', 'error-adjusted',
            'model=', 'earlyStopPatience=', 'reduceLRPatience=', 
            'reduceLRFactor=', 'reduceLRMinLR=', 'residual-cells=',
            'history-start=', 'history-end=', 'history-part=',
            'sample-size=', 'no-transform', 'loss=', 'append']
    try:
        try:
            opts, args = getopt.getopt(argv[1:], '', argl)
        except getopt.GetoptError as err:
            print(str(err))
            return 2

        prepTraining = False
        size = 0
        seed = 0
        calibrate = False
        hw = -1
        exponent = 6
        lr = 0.001
        dropout = 0.5
        dropout_first = None
        dropout_middle = None
        dropout_last = None
        earlyStopPatience=None
        reduceLRPatience=None
        reduceLRFactor=None 
        reduceLRMinLR=None
        layers = 4
        save = False
        compare = False
        epochs = 200
        prefix = 'SWO GBP '
        postfix = None
        run_fnn = False
        load = False
        file_name = None #"../data/swo_gbp_g2pp_nn_adj_err_s0.99_0-264_mse_lr_5.0e-03_ex6_lay9_d30_bn_res_1_rlr_5.0e-01_rlrmin_5.0e-06_rlrpat_25_estop_76.p"
        with_error = False
        residual_cells=1
        history_start=None
        history_end=None
        history_part=0.4
        model_dict = inst.g2
        sample_size = 0
        do_transform = True
        append = False
        loss = 'mean_squared_error'
        for o, a in opts:
            if o == '--training-data':
                print('Prepare training data')
                prepTraining = True
                sample_size = int(a)
            elif o == '--seed':
                seed = int(a)
            elif o == '--sample-size':
                sample_size = int(a)
            elif o == '--calibrate-history':
                print('Calibrate history')
                calibrate = True
            elif o == '--hull-white-fnn':
                hw = 0
            elif o == '--hull-white-rbf':
                hw = 1
            elif o == '--hull-white-relu':
                hw = 2
            elif o == '--hull-white-elu':
                hw = 3
            elif o == '--hull-white-cnn':
                hw = 4
            elif o == '--test-fnn':
                print('Test fnn')
                nn.test_fnn(nn.hullwhite_fnn)
                return 0
            elif o == '--test-rbf':
                print('Test rbf')
                nn.test_fnn(nn.hullwhite_rbf)
                return 0
            elif o == '--test-relu':
                print('Test relu')
                nn.test_fnn(nn.hullwhite_relu)
                return 0
            elif o == '--test-elu':
                print('Test ELU')
                nn.test_fnn(nn.hullwhite_elu)
                return 0
            elif o == '--test-cnn':
                print('Test cnn')
                nn.test_cnn(nn.hullwhite_cnn)
                return 0
            elif o == '--gbp-to-h5':
                du.gbp_to_hdf5()
                return 0
            elif o == '--exponent':
                exponent = int(a)
            elif o == '--lr':
                lr = float(a)
            elif o == '--dropout':
                dropout = float(a)
            elif o == '--dropout-first':
                dropout_first = float(a)
            elif o == '--dropout-middle':
                dropout_middle = float(a)
            elif o == '--dropout-last':
                dropout_last = float(a)
            elif o == '--earlyStopPatience':
                earlyStopPatience = int(a)
            elif o == '--reduceLRPatience':
                reduceLRPatience = int(a)
            elif o == '--reduceLRFactor':
                reduceLRFactor = float(a)
            elif o == '--reduceLRMinLR':
                reduceLRMinLR = float(a)
            elif o == '--residual-cells':
                residual_cells = int(a)
            elif o == '--layers':
                layers = int(a)
            elif o == '--save':
                save = True
            elif o == '--compare-history':
                compare = True
            elif o == '--epochs':
                epochs = int(a)
            elif o == '--run-fnn':
                run_fnn = True
                size = float(a)
            elif o == '--prefix':
                prefix = a
            elif o == '--postfix':
                postfix = a
            elif o == '--file-name':
                file_name = a
                load = True
            elif o == '--error-adjusted':
                with_error = True
            elif o == '--history-start':
                history_start = int(a)
            elif o == '--history-end':
                history_end = int(a)
            elif o == '--history-part':
                history_part = float(a)
            elif o == '--no-transform':
                do_transform = False
            elif o == '--loss':
                loss = a
            elif o == '--append':
                append = True
            elif o == '--model':
                a = a.lower()
                if a == 'hullwhite' or a == 'hullwhite_analytic':
                    model_dict = inst.hullwhite_analytic
                elif a == 'g2' or a == 'g2++':
                    model_dict = inst.g2
                elif a == 'g2++_local':
                    model_dict = inst.g2_local
                elif a == 'g2++_vae':
                    model_dict = inst.g2_vae
                else:
                    raise RuntimeError('Unkown model')

        if prepTraining:
            swo = inst.get_swaptiongen(model_dict)
            file_name = inst.sample_file_name(swo, sample_size, with_error, history_start, 
                                              history_end, history_part)
            swo.training_data(sample_size, plot=False, threshold=10, save=True, 
                              append=append, seed=seed, with_error=with_error,
                              history_start=history_start, history_end=history_end,
                              history_part=history_part, file_name=file_name)
        elif calibrate:
            swo = inst.get_swaptiongen(model_dict)
            if history_start is not None and history_end is not None:
                swo.calibrate_history(history_start, history_end)
            else:
                swo.calibrate_history()
        elif run_fnn:
            swo = inst.get_swaptiongen(model_dict)
            train_file = inst.sample_file_name(swo, sample_size, with_error, 
                                              history_start, history_end, 
                                              history_part)
            if postfix is None:
                postfix = inst.postfix(size, with_error, history_start, 
                                       history_end, history_part)
                
            run(size, compare=compare, epochs=epochs, prefix=prefix, 
                postfix=postfix, dropout=dropout, dropout_first=dropout_first, 
                dropout_middle=dropout_middle, dropout_last=dropout_last,
                earlyStopPatience=earlyStopPatience, reduceLRPatience=reduceLRPatience, 
                reduceLRFactor=reduceLRFactor, reduceLRMinLR=reduceLRMinLR,
                save=save, layers=layers, lr=lr, exponent=exponent, 
                load=load, file_name=file_name, model_dict=model_dict,
                residual_cells=residual_cells, train_file=train_file,
                do_transform=do_transform, loss=loss)
        else:
            if hw == 0:
                model = nn.hullwhite_fnn(exponent=exponent, lr=lr, 
                                         layers=layers,
                                         prefix=prefix, postfix=postfix,
                                         dropout_first=dropout_first,
                                         dropout_middle=dropout_middle,
                                         dropout_last=dropout_last)
            elif hw == 1:
                model = nn.hullwhite_rbf(exponent=exponent, lr=lr, 
                                         layers=layers,
                                         prefix=prefix, postfix=postfix,
                                         dropout_first=dropout_first,
                                         dropout_middle=dropout_middle,
                                         dropout_last=dropout_last)
            elif hw == 2:
                model = nn.hullwhite_relu(exponent=exponent, lr=lr, 
                                          layers=layers,
                                          prefix=prefix, postfix=postfix,
                                          dropout_first=dropout_first,
                                          dropout_middle=dropout_middle,
                                          dropout_last=dropout_last)
            elif hw == 3:
                model = nn.hullwhite_elu(exponent=exponent, lr=lr, 
                                         layers=layers,
                                         prefix=prefix, postfix=postfix,
                                         dropout_first=dropout_first,
                                         dropout_middle=dropout_middle,
                                         dropout_last=dropout_last)
            elif hw == 4:
                model = nn.hullwhite_cnn(exponent=exponent, lr=lr, 
                                         reg=dropout, prefix=prefix, 
                                         postfix=postfix)
            else:
                print('Unknown option')
                return 1
            model.train(epochs)
            if save:
                print('Saving Model')
                nn.write_model(model)
            if compare:
                swo.compare_history(model)
          

    except Exception as err:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = {
                         'filename': exc_traceback.tb_frame.f_code.co_filename,
                         'lineno'  : exc_traceback.tb_lineno,
                         'name'    : exc_traceback.tb_frame.f_code.co_name,
                         'type'    : exc_type.__name__,
                         'message' : exc_value, # or see traceback._some_str()
                        }
        del(exc_type, exc_value, exc_traceback)
        print('')
        print(traceback.format_exc())
        print('')
        print(traceback_template % traceback_details)
        print('')
        print(str(err))
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())