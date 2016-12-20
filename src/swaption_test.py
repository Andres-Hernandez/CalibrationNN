# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:18:10 2016

@author: hernandeza
"""
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

def run(x, total = 239575., compare=False, epochs=500, prefix='SWO GBP ', 
        postfix='', dropout_first=0.2, dropout_middle=0.2, 
        dropout_last=0.2, save=True, layers=4, lr=0.001, exponent=6,
        load=False, file_name=None, model_dict=inst.hullwhite_analytic):
    print 'run  ' + str(x) + ' ' + postfix
    nn.total_size = x/total
    nn.valid_size = nn.total_size*0.2
    nn.test_size = 0.0
    
    if load:
        assert(file_name is not None)
        model = nn.read_model(file_name)
    else:
        model = nn.hullwhite_elu(exponent=exponent, layers=layers, lr=lr,
                                 prefix=prefix, postfix=postfix, 
                                 dropout_first=dropout_first, 
                                 dropout_middle=dropout_middle, 
                                 dropout_last=dropout_last,
                                 model_dict=model_dict)
        model.train(epochs)
        if save:
            nn.write_model(model)
    
    if compare:
        swo = inst.get_swaptiongen(model_dict)
        (dates, values) = swo.compare_history(model)
        
        file_name = du.data_dir + 'test' + postfix + '_fnn_l' \
            + str(layers) + '_e' + str(exponent) + '_epoch' + str(epochs) + '_'
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
            'model=']
    try:
        try:
            opts, args = getopt.getopt(argv[1:], '', argl)
        except getopt.GetoptError as err:
            print str(err)
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
        layers = 4
        save = False
        compare = False
        epochs = 200
        prefix = 'SWO GBP '
        postfix = ''
        run_fnn = False
        load = False
        file_name = None
        with_error = True
        model_dict = inst.hullwhite_analytic
        for o, a in opts:
            if o == '--training-data':
                print 'Prepare training data'
                prepTraining = True
                size = int(a)
            elif o == '--seed':
                seed = int(a)
            elif o == '--calibrate-history':
                print 'Calibrate history'
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
                print 'Test fnn'
                nn.test_fnn(nn.hullwhite_fnn)
                return 0
            elif o == '--test-rbf':
                print 'Test rbf'
                nn.test_fnn(nn.hullwhite_rbf)
                return 0
            elif o == '--test-relu':
                print 'Test relu'
                nn.test_fnn(nn.hullwhite_relu)
                return 0
            elif o == '--test-elu':
                print 'Test ELU'
                nn.test_fnn(nn.hullwhite_elu)
                return 0
            elif o == '--test-cnn':
                print 'Test cnn'
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
                size = int(a)
            elif o == '--prefix':
                prefix = a
            elif o == '--postfix':
                postfix = a
            elif o == '--file-name':
                file_name = a
                load = True
            elif o == '--error-adjusted':
                with_error = True
            elif o == '--model':
                a = a.lower()
                if a == 'hullwhite' or a == 'hullwhite_analytic':
                    model_dict = inst.hullwhite_analytic
                elif a == 'g2' or a == 'g2++':
                    model_dict = inst.g2
                else:
                    raise RuntimeError('Unkown model')

        if prepTraining:
            swo = inst.get_swaptiongen(model_dict)
            swo.training_data(size, plot=False, threshold=10, save=True, 
                              append=True, seed=seed, with_error=with_error)
        elif calibrate:
            swo = inst.get_swaptiongen(model_dict)
            swo.calibrate_history()
        elif run_fnn:
            run(size, compare=compare, epochs=epochs, prefix=prefix, 
                postfix=postfix, dropout_first=dropout_first, 
                dropout_middle=dropout_middle, dropout_last=dropout_last, 
                save=save, layers=layers, lr=lr, exponent=exponent, 
                load=load, file_name=file_name, model_dict=model_dict)
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
                print 'Unknown option'
                return 1
            model.train(epochs)
            if save:
                print 'Saving Model'
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
                         'message' : exc_value.message, # or see traceback._some_str()
                        }
        del(exc_type, exc_value, exc_traceback)
        print
        print traceback.format_exc()
        print
        print traceback_template % traceback_details
        print
        print str(err)
        print >>sys.stderr, err
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())