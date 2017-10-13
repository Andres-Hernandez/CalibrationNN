# -*- mode: python; tab-width: 4;

# Copyright (C) 2016 Andres Hernandez
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the license for more details.
from __future__ import print_function

import string
import data_utils as du
import pandas as pd
import numpy as np
import QuantLib as ql
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import empirical_covariance
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import variational_autoencoder as vae

seed = 1027

h5_model_node = 'Models'
h5_error_node = 'Errors'
# Interest Rate Curve hdf5 node
h5_irc_node = 'IRC' 

# This controls whether functionEvaluation is available in QuantLib library
has_function_evals = True
float_type = np.float32

class IRCurve (du.TimeSeriesData):
    '''Class for accessing IR curve data and instantianting QuantLib instances
    The curves are instantiated as interpolated zero curves using the monotonic
    cubic interpolation. 
    
    # Arguments
        ccy: String identifying the currency. Should contain only 
            alphanumerical characters
        tenor: String identifying the tenor. Should contain only 
            alphanumerical characters
            
        
    '''
    def __init__(self, ccy, tenor):
        if tenor in string.digits:
            tenor = '_' + tenor
        self.ccy = ccy.upper()
        self.tenor = tenor
        self.name = h5_irc_node + '_' + ccy + '_' + tenor
        self.name = self.name.lower()
        self.key_ts = du.h5_ts_node + '/' + h5_irc_node + '/' + self.ccy \
            + '/' + tenor.upper()
        self._daycounter = ql.ActualActual()
        self.values = None
        super(IRCurve, self).__init__(self.key_ts)


    def __getitem__(self, date):
        data = super(IRCurve, self).__getitem__(date)
        ts = pd.Timestamp(date)
        refdate = ql.Date(ts.day, ts.month, ts.year)
        return self.__curveimpl(refdate, data)


    def rebuild(self, date, vals):
        if self._pipeline is None:
            self.pca()
        refdate = ql.Date(date.day, date.month, date.year)  
        data = self._pipeline.inverse_transform(vals)[0]
        return (data, self.__curveimpl(refdate, data))


    def build(self, date, data):
        return self.__curveimpl(date, data)


    def __curveimpl(self, refdate, values):
        self.values = values
        ql.Settings.instance().evaluationDate = refdate
        dates = [refdate+int(d) for d in self.axis(0)]
        return ql.MonotonicCubicZeroCurve(dates, values, self._daycounter)


def castToTenor(index):
    frequency = index.tenor().frequency()
    if frequency == 365:
        return 'OIS'
    return 'L%dM' % int(12/frequency)

def formatVol(v, digits = 2):
    format = '%%.%df %%%%' % digits
    return format % (v * 100)

def formatPrice(p, digits = 2):
    format = '%%.%df' % digits
    return format % p

def proper_name(name):
    name = name.replace(" ", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(",", "_")
    name = name.replace("-", "_")
    name = name.replace("+", "p")
    return name

def flatten_name(name, node=h5_model_node, risk_factor='IR'):
    name = proper_name(name)
    return node + '/' + risk_factor + '/' + name

def postfix(size, with_error, history_start, history_end, history_part):
    if with_error:
        file_name = '_adj_err'
    else:
        file_name = '_unadj_err'
    file_name += '_s' + str(size)
    if history_start is not None:
        file_name += '_' + str(history_start) + '-' + str(history_end)
    else:
        file_name += '_' + str(history_part)
	
    return file_name
    
def sample_file_name(swo, size, with_error, history_start, history_end, history_part):
    file_name = flatten_name(swo.name).lower().replace('/', '_')
    file_name = du.data_dir + file_name
                
    if with_error:
        file_name += '_adj_err'
    else:
        file_name += '_unadj_err'
    file_name += '_s' + str(size)
    if history_start is not None:
        file_name += '_' + str(history_start) + '-' + str(history_end)
    else:
        file_name += '_' + str(history_part)

    return file_name
	
class SwaptionGen (du.TimeSeriesData):
    '''
    SwaptionGen provides different functionality for working with QuantLib 
    swaptions. 
    Input parameters:
    index - IR index from which many conventions will be taken. A clone of the 
            original index will be kept.
    modelF - Function to instantiate the model. The function should take only
             one parameter: a term structure handle
    engineF - Function to instantiate the swaption pricing engine. The function
              should take two parameters: a model and a term structure handle
        Methods:
    SwaptionGen['<date, format yyyy-mm-dd>'] returns the surface data for the day
    SwaptionGen.set_date(date) adjusts the swaption helpers to use the quotes 
    from that date SwaptionGen.updateQuotes(volas) adjusts the swaption helpers 
    to use the given quotes
    SwaptionGen.calibrate_history(name) calibrates the model on each day and 
    saves the results to the HDF5 file. The table has node Models/IR/<name>
    The index of the table is the dates
    The columns are:
        OrigEvals - the number of evaluations needed to start from default params
        OptimEvals - the number of evaluations needed if starting from optimized solution
        OrigObjective - objective value starting from default parameters
        OrigMeanError - mean error starting from default parameters
        HistEvals - the number of evaluations needed if starting from yesterday's params
        HistObjective - objective value starting from yesterday's params
        HistMeanError - mean error starting from yesterday's params
        OrigParams - optimized parameters starting from default parameters
        HistParams - optimized parameters starting from yesterday's params
    '''
    def __init__(self, index, model_dict):
        self._model_dict = model_dict
        if not 'model' in self._model_dict \
        or not 'engine' in self._model_dict \
        or not 'name' in self._model_dict:
            raise RuntimeError('Missing parameters')
        self.ccy = index.currency().code()
        self.model_name = self._model_dict['name'].replace("/", "")
        self.name = 'SWO ' + self.ccy 
        if 'file_name' in self._model_dict:
            self.name += ' ' + self._model_dict['file_name']
        else:
            self.name += + ' ' + self.model_name
        
        self.key_ts = du.h5_ts_node + '/SWO/' + self.ccy
        self.key_model = flatten_name('SWO/' + self.ccy + '/' + self.model_name)
        self.key_error = flatten_name('SWO/' + self.ccy + '/' + self.model_name, 
                                      node=h5_error_node)
        super(SwaptionGen, self).__init__(self.key_ts)
        
        #Yield Curve and dates
        self._ircurve = IRCurve(self.ccy, castToTenor(index))
        self._dates = self.intersection(self._ircurve)
        self.refdate = ql.Date(self._dates[0].day, self._dates[0].month, self._dates[0].year)
        ql.Settings.instance().evaluationDate = self.refdate
        self._termStructure = ql.RelinkableYieldTermStructureHandle()        
        self._termStructure.linkTo(self._ircurve[self._dates[0]])
        
        #Index, swaption model and pricing engine
        self.index = index.clone(self._termStructure)
        self.model = self._model_dict['model'](self._termStructure)
        self.engine = self._model_dict['engine'](self.model, self._termStructure)
        
        #Quotes & calibration helpers
        volas = self.__getitem__(self._dates[0])
        volas.shape = (volas.shape[0]*volas.shape[1],)
        self._quotes = [ql.SimpleQuote(vola) for vola in volas]
        self.__create_helpers()

        #Standard calibration nick-nacks
        if 'method' in self._model_dict:
            self.method = self._model_dict['method'][0]
            self.end_criteria = self._model_dict['method'][1]
            self.constraint = self._model_dict['method'][2]
        else:
            self.method = ql.LevenbergMarquardt()
            self.end_criteria = ql.EndCriteria(1000, 250, 1e-7, 1e-7, 1e-7)
            self.constraint = ql.PositiveConstraint()
            
        self._defaultParams = self.model.params()        
        self._sampler = self._model_dict['sampler']

        #Output transformations
        if 'transformation' in self._model_dict:
            self._transformation = self._model_dict['transformation']
        else:            
            self._transformation = None
        
        if 'inverse_transformation' in self._model_dict:
            self._inverseTrans = self._model_dict['inverse_transformation']
        else:            
            self._inverseTrans = None
        
        #Misc
        self.okFormat = '%12s |%12s |%12s |%12s |%12s'
        self.errFormat = '%2s |%12s |'
        self.values = None

        
    def set_date(self, date):
        #Set Reference Date
        ts = pd.Timestamp(date)
        dt = ql.Date(ts.day, ts.month, ts.year)
        if dt != self.refdate or self.values is None:
            self.refdate = dt
            #Update term structure
            self._termStructure.linkTo(self._ircurve[date])
            
            #Update quotes
            volas = self.__getitem__(date)
            volas.shape = (volas.shape[0]*volas.shape[1],)
            self.updateQuotes(volas)


    def updateQuotes(self, volas):
        self.values = volas
        [quote.setValue(vola) for vola, quote in zip(volas, self._quotes)]


    def __create_helpers(self):
        mg = np.meshgrid(self.axis(0).values, self.axis(1).values)
        mg[0].shape = (mg[0].shape[0]*mg[0].shape[1],)
        mg[1].shape = (mg[1].shape[0]*mg[1].shape[1],)
        self._maturities = mg[0]
        self._lengths = mg[1]
        
        #Create swaption helpers
        self.helpers = [ql.SwaptionHelper(ql.Period(int(maturity), ql.Years), 
                                          ql.Period(int(length), ql.Years),
                                          ql.QuoteHandle(quote),
                                          self.index, self.index.tenor(), 
                                          self.index.dayCounter(), 
                                          self.index.dayCounter(), 
                                          self._termStructure,
                                          ql.CalibrationHelper.ImpliedVolError)
                     for maturity, length, quote in zip(mg[0], mg[1], self._quotes) ]
        #Set pricing engine
        for swaption in self.helpers:
            swaption.setPricingEngine(self.engine)

    def __history(self, history_start, history_end, history_part, with_error):
        if history_start is not None and history_end is not None:
            assert(history_start >= 0)
            assert(history_start < len(self._dates))
            assert(history_end > history_start)
            assert(history_end <= len(self._dates))
            dates = self._dates[history_start:history_end]
        else:
            assert(history_part > 0)
            assert(history_part <= 1.0)
            dates = self._dates[:len(self._dates)*history_part]
                                
        #Get history of parameters
        nb_swo_params = len(self._defaultParams)
        columns_orig = ['OrigParam%d' % x for x in range(nb_swo_params)]
        columns_hist = ['HistParam%d' % x for x in range(nb_swo_params)]
        df_model = pd.get_store(du.h5file)[self.key_model]
        #Pick the best of the two
        swo_param_history_orig = df_model.loc[dates][columns_orig]
        swo_param_history_hist = df_model.loc[dates][columns_hist]
        orig_vs_hist = df_model.loc[dates]['OrigObjective'].values < df_model.loc[dates]['HistObjective'].values
        swo_param_history = swo_param_history_orig.copy(deep=True)
        swo_param_history[~orig_vs_hist] = swo_param_history_hist[~orig_vs_hist]
        
        #Get history of errors
        if with_error:
            df_error = pd.get_store(du.h5file)[self.key_error]
            swo_error_history = df_error.loc[dates]
        else:
            swo_error_history = None
        return (dates, swo_param_history, swo_error_history)
        
    def train_history(self, *kwargs):
        #Retrieves training data from history 
        if 'history_start' in kwargs:
            history_start = kwargs['history_start']
            history_end = kwargs['history_end']
            history_part = None
        else:
            history_start = None
            history_end = None
            if 'history_part' in kwargs:
                history_part = kwargs['history_part']
            else:
                history_part = 0.4
        
        if 'save' in kwargs and kwargs['save']:
            if 'file_name' in kwargs:
                file_name = kwargs['file_name']
            else:
                file_name = sample_file_name(self, 0, True, history_start, 
                                             history_end, history_part)
            print('Saving to file %s' % file_name)
        
        (dates, y, swo_error) = \
            self.__history(history_start, history_end, history_part, True)
        x_ir = self._ircurve.to_matrix(dates)
            
        #Calculate volatilities according to different conditions
        nb_instruments = len(self.helpers)
        nb_dates = len(dates)
        x_swo = np.zeros((nb_dates, nb_instruments), float_type)
        
        for row in range(nb_dates):
            #Set term structure
            self.set_date(dates[row])
            self.model.setParams(ql.Array(y[row, :].tolist()))
            for swaption in range(nb_instruments):
                try:
                    NPV = self.helpers[swaption].modelValue()
                    vola = self.helpers[swaption].impliedVolatility(NPV, 1.0e-6, 1000, 0.0001, 2.50)
                    x_swo[row, swaption] = np.clip(vola - swo_error[row, swaption], 0., np.inf)
                except RuntimeError as e:
                    print('Exception (%s) for (sample, maturity, length): (%s, %s, %s)' % (e, row, self._maturities[swaption], self._lengths[swaption]))

        if 'save' in kwargs and kwargs['save']:            
            if 'append' in kwargs and kwargs['append']:
                try:
                    x_swo_l = np.load(file_name + '_x_swo.npy')
                    x_ir_l = np.load(file_name + '_x_ir.npy')
                    y_l = np.load(file_name + '_y.npy')
                    x_swo = np.concatenate((x_swo_l, x_swo), axis=0)
                    x_ir = np.concatenate((x_ir_l, x_ir), axis=0)                
                    y = np.concatenate((y_l, y), axis=0)
                except Exception as e:
                    print(e)
            
            np.save(file_name + '_x_swo', x_swo)
            np.save(file_name + '_x_ir', x_ir)
            np.save(file_name + '_y', y)
        return (x_swo, x_ir, y)
            
            
    def calibrate_history(self, *args):
        clean = True
        start = 0
        end = len(self._dates)
        if len(args) > 0:
            if len(args) != 2:
                raise RuntimeError("Invalid input")
            clean = False
            start = args[0]
            if args[1] > 0:
                end = args[1]                

        prev_params = self._defaultParams
        nb_params = len(prev_params)
        nb_instruments = len(self.helpers)
        columns = ['OrigEvals', 'OptimEvals', 'OrigObjective', 'OrigMeanError', 
                   'HistEvals', 'HistObjective', 'HistMeanError', 
                   'HistObjectivePrior', 'HistMeanErrorPrior'] 
        size_cols = len(columns)
        columns = columns + map(lambda x: 'OrigParam' + str(x), range(nb_params))
        columns = columns + map(lambda x: 'HistParam' + str(x), range(nb_params))
        if clean:
            rows_model = np.empty((len(self._dates), len(columns)))
            rows_model.fill(np.nan)
            rows_error = np.zeros((len(self._dates), nb_instruments))
        else:
            with pd.get_store(du.h5file) as store:
                df_model = store[self.key_model]
                if len(df_model.columns) != len(columns) or \
                        len(df_model.index) != len(self._dates):
                    raise RuntimeError("Incompatible file")
                rows_model = df_model.values
                
                df_error = store[self.key_error]
                if len(df_error.columns) != nb_instruments or \
                        len(df_error.index) != len(self._dates):
                    raise RuntimeError("Incompatible file")
                rows_error = df_error.values

        header = self.okFormat % ('maturity','length','volatility','implied','error')
        dblrule = '=' * len(header)        
        
        for iDate in range(start, end):
            self.model.setParams(self._defaultParams)
            #Return tuple (date, origEvals, optimEvals, histEvals, 
            #origObjective, origMeanError, histObjective, histMeanError, 
            #original params, hist parameters, errors)            
            try:
                res = self.calibrate(self._dates[iDate], prev_params)
            except RuntimeError as e:
                print('')
                print(dblrule)
                print(self.name + " " + str(self._dates[iDate]))
                print("Error: %s" % e)
                print(dblrule)
                res = (self._dates[iDate], -1, -1, -1, -1, -1, -1, -1, -1, -1, 
                       self._defaultParams, self._defaultParams, np.zeros((1, nb_instruments)))
            rows_model[iDate, 0:size_cols] = res[1:size_cols+1]
            rows_model[iDate, size_cols:size_cols+nb_params] = res[size_cols+1]
            rows_model[iDate, size_cols+nb_params:size_cols+nb_params*2] = res[size_cols+2]
            rows_error[iDate, :] = res[-1]
            prev_params = self.model.params()

        df_model = pd.DataFrame(rows_model, index=self._dates, columns = columns)
        df_error = pd.DataFrame(rows_error, index=self._dates)
        
        with pd.get_store(du.h5file) as store:
            store[self.key_model] = df_model
            store[self.key_error] = df_error


    def __errors(self):
        totalError = 0.0
        withException = 0
        errors = np.zeros((1, len(self.helpers)))
        for swaption in range(len(self.helpers)):
            vol = self._quotes[swaption].value()
            NPV = self.helpers[swaption].modelValue()
            try:
                implied = self.helpers[swaption].impliedVolatility(NPV, 1.0e-4, 1000, 0.001, 1.80)
                errors[0, swaption] = vol - implied
                totalError += abs(errors[0, swaption])
            except RuntimeError:
                withException = withException + 1
        denom = len(self.helpers) - withException
        if denom == 0:
            average_error = float('inf')
        else:
            average_error = totalError/denom
            
        return average_error, errors


    def calibrate(self, date, *args):
        name = self.model_name
        header = self.okFormat % ('maturity','length','volatility','implied','error')
        rule = '-' * len(header)
        dblrule = '=' * len(header)
    
        print('')
        print(dblrule)
        print(name + " " + str(date))
        print(rule)

        self.set_date(date)
        try:
            self.model.calibrate(self.helpers, self.method, 
                                 self.end_criteria, self.constraint)
                                 
            params = self.model.params()
            origObjective = self.model.value(params, self.helpers)
            print('Parameters   : %s ' % self.model.params() )
            print('Objective    : %s ' % origObjective)

            origMeanError, errors = self.__errors()
            print('Average error: %s ' % formatVol(origMeanError,4))
            print(dblrule)
        except RuntimeError as e:
            print("Error: %s" % e)
            print(dblrule)
            origObjective = float("inf")
            origMeanError = float("inf")
            errors = np.zeros((1, len(self.helpers)))            

        origEvals = self.model.functionEvaluation()
        origParams = np.array([ v for v in self.model.params() ])
        if len(args) > 0:
            #Recalibrate using optimized parameters
            try:
                self.model.calibrate(self.helpers, self.method, 
                                     self.end_criteria, self.constraint)
                optimEvals = self.model.functionEvaluation()
            except RuntimeError as e:
                optimEvals = -1
            
            #Recalibrate using previous day's parameters
            self.model.setParams(args[0])
            try:
                histObjectivePrior = self.model.value(self.model.params(), self.helpers)
                histMeanErrorPrior, _ = self.__errors()
                self.model.calibrate(self.helpers, self.method, 
                                     self.end_criteria, self.constraint)
                histObjective = self.model.value(self.model.params(), self.helpers)
                histMeanError, errors_hist = self.__errors()
                if histObjective < origObjective:
                    errors = errors_hist
            except RuntimeError:
                histObjectivePrior = float("inf")
                histMeanErrorPrior = float("inf")
                histObjective = float("inf")
                histMeanError = float("inf")
                
            histEvals = self.model.functionEvaluation()
            histParams = np.array([ v for v in self.model.params() ])
                
            #Return tuple (date, origEvals, optimEvals, histEvals, 
            #origObjective, origMeanError, histObjective, histMeanError, 
            #histObjectivePrior, histMeanErrorPrior,
            #original params, hist parameters, errors)
            return (date, origEvals, optimEvals, origObjective, origMeanError, 
                    histEvals, histObjective, histMeanError, histObjectivePrior,
                    histMeanErrorPrior, origParams, histParams, errors)
                
        return (date, origEvals, origObjective, origMeanError, origParams, errors)

    def __random_draw(self, nb_samples, with_error=True, history_start=None,
                      history_end=None, history_part=0.4, ir_pca=True):
        #Correlated IR, Model parameters, and errors
        nb_swo_params = len(self._defaultParams)
        nb_instruments = len(self.helpers)
        (dates, swo_param_history, swo_error_history) = \
            self.__history(history_start, history_end, history_part, with_error)
                
        if ir_pca:
            #Get PCA pipeline & PCA value matrix for IR Curves
            ir_pca = self._ircurve.pca(dates=dates)
            nb_ir_params  = ir_pca.named_steps['pca'].n_components_
            ir_param_history = ir_pca.transform(self._ircurve.to_matrix(dates))
        else:
            ir_param_history = self._ircurve.to_matrix(dates)
        
        #Scale and draw random samples
        if self._transformation is not None:
            swo_param_history = self._transformation(swo_param_history)
        if with_error:
            history = np.concatenate((swo_param_history, ir_param_history, swo_error_history), axis=1)
        else:
            history = np.concatenate((swo_param_history, ir_param_history), axis=1)
        
        
        draws = self._sampler(history, nb_samples)
        
        #Separate
        if self._inverseTrans is not None:
            y = self._inverseTrans(draws[:, 0:nb_swo_params])
        else:
            y = draws[:, 0:nb_swo_params]
        ir_draw = draws[:, nb_swo_params:nb_swo_params + nb_ir_params]

        if with_error:
            error_draw = draws[:, nb_swo_params + nb_ir_params:]
            #error_draw[-5:, :] = merr
        else:
            error_draw = np.zeros((nb_samples, nb_instruments))
        return (y, ir_draw, error_draw, dates)


    def training_data(self, nb_samples, with_error=True, **kwargs):
        #Prepares nb_samples by sampling from the model parameters and generating
        #a term structure with the use of PCA, and then evaluates the set of
        #swaptions to produce volatilities
        #The sample is of the form (x_swo, x_ir,y), where x_swo and x_ir are the 
        #future input for the supervised machine learning algorithm, and y is 
        #the desired output
        #Draw random model parameters and IR curves
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
        else:
            np.random.seed(0)

        if 'history_start' in kwargs:
            history_start = kwargs['history_start']
            history_end = kwargs['history_end']
            history_part = None
        else:
            history_start = None
            history_end = None
            if 'history_part' in kwargs:
                history_part = kwargs['history_part']
            else:
                history_part = 0.4
        
        if 'save' in kwargs and kwargs['save']:
            if 'file_name' in kwargs:
                file_name = kwargs['file_name']
            else:
                file_name = sample_file_name(self, nb_samples, with_error, 
                                             history_start, history_end, 
                                             history_part)
            print('Saving to file %s' % file_name)
            
        (y, ir_draw, error_draw, dates) = self.__random_draw(nb_samples, 
                                                        with_error=with_error,
                                                        history_start=history_start,
                                                        history_end=history_end,
                                                        history_part=history_part)
        
        #Draw random dates
        date_index = np.random.randint(0, len(dates), nb_samples)
        dates = dates[date_index]
                                
        #Calculate volatilities according to different conditions
        nb_instruments = len(self.helpers)
        x_swo = np.zeros((nb_samples, nb_instruments), float_type)
        x_ir  = np.zeros((nb_samples, len(self._ircurve.axis(0))), float_type)
        if 'plot' in kwargs and kwargs['plot']:
            plot_ir = True
        else:
            plot_ir = False
            
        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = nb_instruments + 1
        indices = np.ones((nb_samples, ), dtype=bool)
        for row in range(nb_samples):
            if row % 1000 == 0:
                print('Processing sample %s' % row)
            #Set term structure
            try:
                (x_ir[row, :], curve) = self._ircurve.rebuild(dates[row], ir_draw[row, :])
                if plot_ir:
                    du.plot_data(self._ircurve.axis(0).values, x_ir[row, :])
                self._termStructure.linkTo(curve)
                self.model.setParams(ql.Array(y[row, :].tolist()))
                nb_nan_swo = 0
                if row == nb_samples-1:
                    NPV = self.helpers[0].modelValue()
                    vola = self.helpers[0].impliedVolatility(NPV, 1.0e-6, 1000, 0.0001, 2.50)
                    print("%s, %s" % (NPV, vola))
                for swaption in range(nb_instruments):
                    try:
                        NPV = self.helpers[swaption].modelValue()
                        vola = self.helpers[swaption].impliedVolatility(NPV, 1.0e-6, 1000, 0.0001, 2.50)
                        x_swo[row, swaption] = np.clip(vola - error_draw[row, swaption], 0., np.inf)
                    except RuntimeError as e:
                        print('Exception (%s) for (sample, maturity, length): (%s, %s, %s)' % (e, row, self._maturities[swaption], self._lengths[swaption]))
                        nb_nan_swo = nb_nan_swo + 1
                        if nb_nan_swo > threshold:
                            print('Throwing out sample %s' % row)
                            indices[row] = False
                            break;
            except RuntimeError as e:
                print('Throwing out sample %s. Exception: %s' % (row, e))

        if ~np.any(indices):
            raise RuntimeError('All samples were thrown out')
        
        if np.any(~indices):
            #Remove rows with too many nans
            x_swo = x_swo[indices, :]
            x_ir = x_ir[indices, :]
            y = y[indices, :]
            print('%s samples had too many nans' % np.sum(~indices))
        
        if 'save' in kwargs and kwargs['save']:
            if 'append' in kwargs and kwargs['append']:
                try:
                    x_swo_l = np.load(file_name + '_x_swo.npy')
                    x_ir_l = np.load(file_name + '_x_ir.npy')
                    y_l = np.load(file_name + '_y.npy')
                    x_swo = np.concatenate((x_swo_l, x_swo), axis=0)
                    x_ir = np.concatenate((x_ir_l, x_ir), axis=0)                
                    y = np.concatenate((y_l, y), axis=0)
                except Exception as e:
                    print(e)
            
            np.save(file_name + '_x_swo', x_swo)
            np.save(file_name + '_x_ir', x_ir)
            np.save(file_name + '_y', y)
        return (x_swo, x_ir, y)


    def evaluate(self, params, irValues, date):
        self.refdate = ql.Date(date.day, date.month, date.year)
        _, curve = self._ircurve.build(self.refdate, irValues)
        self._termStructure.linkTo(curve)
        qlParams = ql.Array(params.tolist())
        self.model.setParams(qlParams)
        return self.__errors()


    def errors(self, predictive_model, date):
        with pd.get_store(du.h5file) as store:
            df_error = store[self.key_error]
            orig_errors = df_error.loc[date]
            store.close()
        
        self.refdate = ql.Date(1, 1, 1901)
        self.set_date(date)
        params = predictive_model.predict((self.values, self._ircurve.values))
        self.model.setParams(ql.Array(params.tolist()[0]))
        _, errors = self.__errors()
        return (orig_errors, errors)


    def history_heatmap(self, predictive_model, dates=None):
        self.refdate = ql.Date(1, 1, 1901)
        if dates is None:
            dates = self._dates
        
        errors_mat = np.zeros((len(dates), len(self.helpers)))
        for i, date in enumerate(dates):
            date = self._dates[i]
            self.set_date(date)
            params = predictive_model.predict((self.values, self._ircurve.values))
            self.model.setParams(ql.Array(params.tolist()[0]))
            _, errors_mat[i, :] = self.__errors()
    
        return errors_mat
    
        
    def objective_values(self, predictive_model, date_start, date_end):
        dates = self._dates[date_start:date_end]
        objective_predict = np.empty((len(dates), ))
        volas_predict = np.empty((len(dates), ))
        for i, date in enumerate(dates):
            self.set_date(date)
            params = predictive_model.predict((self.values, self._ircurve.values))
            self.model.setParams(ql.Array(params.tolist()[0]))
            volas_predict[i], _ = self.__errors()
            try:
                objective_predict[i] = self.model.value(self.model.params(), self.helpers)
            except RuntimeError:
                objective_predict[i] = np.nan

        return (objective_predict, volas_predict)
                
        
    def objective_shape(self, predictive_model, date, nb_samples_x=100, nb_samples_y=100):
        store = pd.get_store(du.h5file)
        df = store[self.key_model]
        store.close()
        self.set_date(date)
        
        params_predict = predictive_model.predict((self.values, self._ircurve.values))
        params_predict = params_predict.reshape((params_predict.shape[1],))
        self.model.setParams(ql.Array(params_predict.tolist()))
        print("Predict value = %f" % self.model.value(self.model.params(), self.helpers))
        orig_objective = df.ix[date, 'OrigObjective']
        hist_objective = df.ix[date, 'HistObjective']
        if orig_objective < hist_objective:
            name = 'OrigParam'
        else:
            name = 'HistParam'
                    
        params_calib = np.array([df.ix[date, name + '0'],
                                 df.ix[date, name + '1'],
                                 df.ix[date, name + '2'],
                                 df.ix[date, name + '3'],
                                 df.ix[date, name + '4']])
        self.model.setParams(ql.Array(params_calib.tolist()))
        print("Calib value = %f" % self.model.value(self.model.params(), self.helpers))
        
        params_optim = np.array(self._defaultParams)
        self.model.setParams(self._defaultParams)
        print("Optim value = %f" % self.model.value(self.model.params(), self.helpers))
        
        #The intention is to sample the plane that joins the three points:
        #params_predict, params_calib, and params_optim
        #A point on that plane can be described by Q(alpha, beta)
        #Q(alpha, beta) = params_predict+(alpha-beta(A*B)/(A*A))A+beta B
        #with 
        A = params_calib - params_predict
        # and 
        B = params_optim - params_predict
        
        Aa = np.sqrt(np.dot(A, A))
        Ba = np.dot(A, B)/Aa
        lim_alpha = np.array([np.min((Ba, 0)), np.max((Ba/Aa, 1))])
        da = lim_alpha[1] - lim_alpha[0]
        lim_alpha += np.array([-1.0, 1.0])*da/10
        lim_beta = np.array([-0.1, 1.1])
        
        ls_alpha = np.linspace(lim_alpha[0], lim_alpha[1], nb_samples_x)
        ls_beta = np.linspace(lim_beta[0], lim_beta[1], nb_samples_y)
        xv, xy = np.meshgrid(ls_alpha, ls_beta)
        sh = xv.shape
        samples = [params_predict + (alpha-beta*Ba/Aa)*A+beta*B 
                   for alpha, beta in zip(xv.reshape((-1,)), xy.reshape((-1,)))]
        
        objectives = np.empty((len(samples), ))
        for i, params in enumerate(samples):
            try:
                self.model.setParams(ql.Array(params.tolist()))
                objectives[i] = self.model.value(self.model.params(), self.helpers)
            except RuntimeError:
                objectives[i] = np.nan

        return (objectives.reshape(sh), lim_alpha, lim_beta)
        
        
        
    def compare_history(self, predictive_model, dates=None, plot_results=False):
        store = pd.get_store(du.h5file)
        df = store[self.key_model]
        store.close()
        self.refdate = ql.Date(1, 1, 1901)
        vals = np.zeros((len(df.index),4))
        values = np.zeros((len(df.index),13))
        if dates is None:
            dates = self._dates
        
        
        method = ql.LevenbergMarquardt()
        end_criteria = ql.EndCriteria(250, 200, 1e-7, 1e-7, 1e-7)
        lower = ql.Array(5, 1e-9)
        upper = ql.Array(5, 1.0)
        lower[4] = -1.0
        constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)
        
        for i, date in enumerate(dates):
            self.set_date(date)
            params = predictive_model.predict((self.values, self._ircurve.values))
            self.model.setParams(ql.Array(params.tolist()[0]))
            meanErrorPrior, _ = self.__errors()
            try:
                objectivePrior = self.model.value(self.model.params(), self.helpers)
            except RuntimeError:
                objectivePrior = np.nan

            self.model.calibrate(self.helpers, method, 
                                 end_criteria, constraint)  
            meanErrorAfter, _ = self.__errors()
            paramsC = self.model.params()
            try:
                objectiveAfter = self.model.value(self.model.params(), self.helpers)
            except RuntimeError:
                objectiveAfter = np.nan

            origMeanError = df.ix[date, 'OrigMeanError']
            histMeanError = df.ix[date, 'HistMeanError']
            origObjective = df.ix[date, 'OrigObjective']
            histObjective = df.ix[date, 'HistObjective']
                

            values[i, 0] = origMeanError
            values[i, 1] = histMeanError
            values[i, 2] = meanErrorPrior
            values[i, 3] = origObjective
            values[i, 4] = histObjective
            values[i, 5] = objectivePrior
            values[i, 6] = meanErrorAfter
            values[i, 7] = objectiveAfter
            if origObjective < histObjective:
                values[i, 8] = df.ix[date, 'OrigParam0']
                values[i, 9] = df.ix[date, 'OrigParam1']
                values[i, 10] = df.ix[date, 'OrigParam2']
                values[i, 11] = df.ix[date, 'OrigParam3']
                values[i, 12] = df.ix[date, 'OrigParam4']
            else:
                values[i, 8] = df.ix[date, 'HistParam0']
                values[i, 9] = df.ix[date, 'HistParam1']
                values[i, 10] = df.ix[date, 'HistParam2']
                values[i, 11] = df.ix[date, 'HistParam3']
                values[i, 12] = df.ix[date, 'HistParam4']

            print('Date=%s' % date)
            print('Vola: Orig=%s Hist=%s ModelPrior=%s ModelAfter=%s' % (origMeanError, histMeanError, meanErrorPrior, meanErrorAfter))
            print('NPV:  Orig=%s Hist=%s Model=%s ModelAfter=%s' % (origObjective, histObjective, objectivePrior, objectiveAfter))
            print('Param0: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 8], params[0][0], paramsC[0]))
            print('Param1: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 9], params[0][1], paramsC[1]))
            print('Param2: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 10], params[0][2], paramsC[2]))
            print('Param3: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 11], params[0][3], paramsC[3]))
            print('Param4: Cal:%s , Model:%s, Cal-Mod:%s' % (values[i, 12], params[0][4], paramsC[4]))
            
            vals[i, 0] = (meanErrorPrior - origMeanError)/origMeanError*100.0
            vals[i, 1] = (meanErrorPrior - histMeanError)/histMeanError*100.0
            vals[i, 2] = (meanErrorAfter - origMeanError)/origMeanError*100.0
            vals[i, 3] = (meanErrorAfter - histMeanError)/histMeanError*100.0
            
            print('      impO=%s impH=%s impAfterO=%s impAfterH=%s' % (vals[i, 0], vals[i, 1], vals[i,2], vals[i, 3]))
        
        if plot_results:
            r = range(vals.shape[0])
            fig = plt.figure(figsize=(16, 16))
            f1 = fig.add_subplot(211)
            f1.plot(r, vals[:, 0])
            f2 = fig.add_subplot(212)
            f2.plot(r, vals[:, 1])
        return (dates, values)


'''
Sampling functions
'''

def random_normal_draw(history, nb_samples, **kwargs):
    """Random normal distributed draws
    
    Arguments:
        history: numpy 2D array, with history along axis=0 and parameters 
            along axis=1
        nb_samples: number of samples to draw
        
    Returns:
        numpy 2D array, with samples along axis=0 and parameters along axis=1
    """
    scaler = StandardScaler()
    scaler.fit(history)
    scaled = scaler.transform(history)
    sqrt_cov = sqrtm(empirical_covariance(scaled)).real
    
    #Draw correlated random variables
    #draws are generated transposed for convenience of the dot operation
    draws = np.random.standard_normal((history.shape(-1), nb_samples))
    draws = np.dot(sqrt_cov, draws)
    draws = np.transpose(draws)
    return scaler.inverse_transform(draws)

'''
Dictionary defining model
It requires 4 parameters:
    name
    model: a function that creates a model. It takes as parameter a 
            yield curve handle
    engine: a function that creates an engine. It takes as paramters a
            calibration model and a yield curve handle
    sampler: a sampling function taking a history and a number of samples
            to produce
            
Optional parameters:
    transformation: a preprocessing function
    inverse_transformation: a postprocessing function
    method: an optimization object
    
'''
hullwhite_analytic = {'name' : 'Hull-White (analytic formulae)',
                     'model' : ql.HullWhite, 
                     'engine' : ql.JamshidianSwaptionEngine,
                     'transformation' : np.log, 
                     'inverse_transformation' : np.exp,
                     'sampler': random_normal_draw}

def g2_transformation(x):
    if isinstance(x, pd.DataFrame):
        x = x.values
    y = np.zeros_like(x)
    y[:, :-1] = np.log(x[:, :-1])
    y[:, -1] = x[:, -1]
    return y

def g2_inverse_transformation(x):
    y = np.zeros_like(x)
    y[:, :-1] = np.exp(x[:, :-1])
    y[:, -1] = np.clip(x[:, -1], -1.0, +1.0)
    return y

def g2_method():
    n = 5
    lower = ql.Array(n, 1e-9);
    upper = ql.Array(n, 1.0);
    lower[n-1] = -1.0;
    upper[n-1] = 1.0;
    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper);

    maxSteps = 5000;
    staticSteps = 600;
    initialTemp = 50.0;
    finalTemp = 0.001;
    sampler = ql.SamplerMirrorGaussian(lower, upper, seed);
    probability = ql.ProbabilityBoltzmannDownhill(seed);
    temperature = ql.TemperatureExponential(initialTemp, n);
    method = ql.MirrorGaussianSimulatedAnnealing(sampler, probability, temperature, 
                                                 ql.ReannealingTrivial(),
                                                 initialTemp, finalTemp)
    criteria = ql.EndCriteria(maxSteps, staticSteps, 1.0e-8, 1.0e-8, 1.0e-8);
    return (method, criteria, constraint)

def g2_method_local():
    method = ql.LevenbergMarquardt()
    criteria = ql.EndCriteria(250, 200, 1e-7, 1e-7, 1e-7)
    lower = ql.Array(5, 1e-9)
    upper = ql.Array(5, 1.0)
    lower[4] = -1.0
    constraint = ql.NonhomogeneousBoundaryConstraint(lower, upper)
    return (method, criteria, constraint)

g2 = {'name' : 'G2++',
      'model' : ql.G2, 
      'engine' : lambda model, _: ql.G2SwaptionEngine(model, 6.0, 16),
      'transformation' : g2_transformation,
      'inverse_transformation' : g2_inverse_transformation,
      'method': g2_method(),
      'sampler': random_normal_draw}

g2_local = {'name' : 'G2++_local',
      'model' : ql.G2, 
      'engine' : lambda model, _: ql.G2SwaptionEngine(model, 6.0, 16),
      'transformation' : g2_transformation,
      'inverse_transformation' : g2_inverse_transformation,
      'method': g2_method_local(),
      'sampler': random_normal_draw}

g2_vae = {'name' : 'G2++',
      'model' : ql.G2, 
      'engine' : lambda model, _: ql.G2SwaptionEngine(model, 6.0, 16),
      'transformation' : g2_transformation,
      'inverse_transformation' : g2_inverse_transformation,
      'method': g2_method(),
      'sampler': vae.sample_from_generator,
      'file_name': 'g2pp_vae'}

def get_swaptiongen(modelMap):
    index = ql.GBPLibor(ql.Period(6, ql.Months))
    swo = SwaptionGen(index, modelMap)
    return swo


def calibrate_history(model_dict, *args):
    swo = get_swaptiongen(model_dict)
    swo.calibrate_history(*args)    
    return swo


def local_hw_map(swo, date, pointA, pointB, off_x=0.1, off_y=0.1, 
                 low_x=1e-8, low_y=1e-8, nb_points=20):
    assert(len(pointA) == 2 and len(pointB) == 2)
    if pointA[0] > pointB[0]:
        max_x = pointA[0]
        min_x = pointB[0]
    else:
        min_x = pointA[0]
        max_x = pointB[0]
    if pointA[1] > pointB[1]:
        max_y = pointA[1]
        min_y = pointB[1]
    else:
        min_y = pointA[1]
        max_y = pointB[1]
    off_x = (max_x - min_x)*off_x
    off_y = (max_y - min_y)*off_y
    max_x += off_x
    min_x -= off_x
    max_y += off_y
    min_y -= off_y
    if min_x <= low_x:
        min_x = low_x
    if min_y <= low_y:
        min_y = low_y
        
    assert(min_x <= max_x and min_y <= max_y)
    rx = np.linspace(min_x, max_x, nb_points)
    ry = np.linspace(min_y, max_y, nb_points)
    xx, yy = np.meshgrid(rx, ry)
    
    result = np.empty(xx.shape)
    result.fill(np.nan)    
    swo.set_date(date)
    for i, x in enumerate(rx):
        for j, y in enumerate(ry):
            swo.model.setParams(ql.Array([x, y]))
            result[i, j] = swo.model.value(swo.model.params(), swo.helpers)
            
    return (xx, yy, result)