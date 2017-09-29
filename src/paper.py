# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:07:34 2016

@author: hernandeza
"""

import data_utils as du
import numpy as np
#import prettyplotlib as ppl
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import brewer2mpl
import neural_network as nn
import instruments as inst
import QuantLib as ql
import pandas as pd
#import plotly.plotly as py
#import plotly.graph_objs as go

def get_fnn(middle='_adj_error_fnn_l4_e6_epoch200'):
    dates = np.load(du.data_dir + 'test_' + middle + '_dates.npy')
    values = np.load(du.data_dir + 'test_' + middle + '_values.npy')
    val_hist = np.load(du.data_dir + 'test_' + middle + '_val_hist.npy')
    train_hist = np.load(du.data_dir + 'test_' + middle + '_train_hist.npy')
    #start = dates.shape[0]*0.4
    #dates = dates[start:]
    #values = values[start:]
    return (dates, values, val_hist, train_hist)

#            values[i, 0] = origMeanError
#            values[i, 1] = histMeanError
#            values[i, 2] = meanErrorPrior
#            values[i, 3] = origObjective
#            values[i, 4] = histObjective
#            values[i, 5] = objectivePrior

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def plot():
    #du.data_dir = "../data/"
    mark_read = 'adj_error_s'
    mark_write = 'adj_error_insample40_s'
    data_labels = ('Default Starting Point', 'Historical Starting Point', 'Feed-forward Neural Net')
    labels = ('225k_lr5em5_ex6_lay5_d20',)
    av_hist = np.empty((500, len(labels)))
    av_hist.fill(np.nan)
    max_len = 0
    for rank, label in enumerate(labels):
        dates, values, _, _ = get_fnn(mark_read + label)
        size = len(dates)
        origMeanError = values[:, 0].reshape((size, 1))*100
        histMeanError = values[:, 1].reshape((size, 1))*100
        origObjective = values[:, 3].reshape((size, 1))
        histObjective = values[:, 4].reshape((size, 1))
        meanErrorPrior = values[:, 2].reshape((size, 1))*100    
        objective = values[:, 5].reshape((size, 1))
        mean_data = np.concatenate((origMeanError, histMeanError, meanErrorPrior), axis=1)
        obje_data = np.concatenate((origObjective, histObjective, objective), axis=1)
    
        colors = ('#66c2a5', '#fc8d62', '#8da0cb')
        du.plot_data(dates, mean_data, figsize=(21, 12), labels=data_labels, 
                     save=du.data_dir + mark_write + label + '_vola_error_fnn.eps', colors=colors,
                     legend_fontsize=22, legend_color='black', 
                     xlabel_fontsize=22, xlabel_color='black', 
                     ylabel_fontsize=22, ylabel_color='black',
                     xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f} %', 
                     ytick_fontsize=18, ytick_color='black',
                     title='Average Volatility Error', title_fontsize=26)
        du.plot_data(dates, obje_data, figsize=(21, 12), labels=data_labels, 
                     save=du.data_dir + mark_write + label + '_npv_error_fnn.eps', colors=colors,
                     legend_fontsize=22, legend_color='black', 
                     xlabel_fontsize=22, xlabel_color='black', 
                     ylabel_fontsize=22, ylabel_color='black',
                     xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f}', 
                     ytick_fontsize=18, ytick_color='black',
                     title='NPV Mean Square Error', title_fontsize=26)
        
        _, _, val_hist, train_hist = get_fnn(mark_read + label)
        av_val = running_mean(val_hist, 30)
        av_hist[:av_val.shape[0], rank] = av_val
        if av_val.shape[0] > max_len:
            max_len = av_val.shape[0]
            
    av_hist = av_hist[:max_len, :]
    du.plot_data(None, av_hist, figsize=(22, 11), labels=labels, 
                 save=du.data_dir + mark_write + '_cross_validation_fnn.eps', xlabel='Epoch', 
                 legend_fontsize=22, legend_color='black', 
                 xlabel_fontsize=22, xlabel_color='black', 
                 ylabel_fontsize=22, ylabel_color='black',
                 xtick_fontsize=18, xtick_color='black', yticks_format='{:.3f}', 
                 ytick_fontsize=18, ytick_color='black')
      
def plot2():
    data_labels = ('Default Starting Point', 'FNN With Error Adjustment .15', 'FNN With Error Adjustement .2')
    dates, ad_values, ad_val, _ = get_fnn(middle='adj_error_s150k_d15')
    _, un_values, un_val, _ = get_fnn(middle='adj_error_s150k_d20')
    size = len(dates)
    origMeanError = ad_values[:, 0].reshape((size, 1))*100
    origObjective = ad_values[:, 3].reshape((size, 1))
    ad_mean_prior = ad_values[:, 2].reshape((size, 1))*100
    un_mean_prior = un_values[:, 2].reshape((size, 1))*100
    ad_obje_prior = ad_values[:, 5].reshape((size, 1))
    un_obje_prior = un_values[:, 5].reshape((size, 1))
    mean_data = np.concatenate((origMeanError, un_mean_prior, ad_mean_prior), axis=1)    
    obje_data = np.concatenate((origObjective, un_obje_prior, ad_obje_prior), axis=1)

    colors = ('#66c2a5', '#fc8d62', '#8da0cb')
    du.plot_data(dates, mean_data, figsize=(22, 12), labels=data_labels, 
                 save=du.data_dir + 'vola_error_fnn_unadj_vs_adj_error.eps', 
                 legend_fontsize=22, legend_color='black', colors=colors,
                 xlabel_fontsize=22, xlabel_color='black', 
                 ylabel_fontsize=22, ylabel_color='black',
                 xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f} %', 
                 ytick_fontsize=18, ytick_color='black')
    du.plot_data(dates, obje_data, figsize=(22, 12), labels=data_labels, 
                 save=du.data_dir + 'npv_error_fnn_unadj_vs_adj_error.eps', 
                 legend_fontsize=22, legend_color='black', colors=colors,
                 xlabel_fontsize=22, xlabel_color='black', 
                 ylabel_fontsize=22, ylabel_color='black',
                 xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f}', 
                 ytick_fontsize=18, ytick_color='black')

    if ad_val.shape[0] > un_val.shape[0]:
        max_len = ad_val.shape[0]
    else:
        max_len = un_val.shape[0]
    av_hist = np.empty((max_len, 2))
    av_hist.fill(np.nan)
    av_val = running_mean(ad_val, 10)
    av_hist[:av_val.shape[0], 0] = av_val
    av_val = running_mean(un_val, 10)
    av_hist[:av_val.shape[0], 1] = av_val
    data_labels = ('With Error Adjustement', 'Without Error Adjustment')
    du.plot_data(None, av_hist, figsize=(22, 11), labels=data_labels, 
                 save=du.data_dir + 'cross_validation_fnn_unadj_vs_adj_error.eps', xlabel='Epoch', 
                 legend_fontsize=22, legend_color='black', 
                 xlabel_fontsize=22, xlabel_color='black', 
                 ylabel_fontsize=22, ylabel_color='black',
                 xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f}', 
                 ytick_fontsize=18, ytick_color='black')


def plot4():
    swo = inst.get_swaptiongen(inst.hullwhite_analytic)
    df = pd.get_store(du.h5file)[swo.key_model]
    d1 = df.loc['2015-06-01']
    d2 = df.loc['2015-06-02']
    x1_orig = d1.loc['OrigParam0']
    x1_hist = d1.loc['HistParam0']
    x2_orig = d2.loc['OrigParam0']
    x2_hist = d2.loc['HistParam0']
    y1_orig = d1.loc['OrigParam1']
    y1_hist = d1.loc['HistParam1']
    y2_orig = d2.loc['OrigParam1']
    y2_hist = d2.loc['HistParam1']
    x_default = 0.1
    y_default = 0.01
    xmin = min(x_default, x1_orig, x2_orig, x1_hist, x2_hist)
    xmax = max(x_default, x1_orig, x2_orig, x1_hist, x2_hist)
    ymin = min(y_default, y1_orig, y2_orig, y1_hist, y2_hist)
    ymax = max(y_default, y1_orig, y2_orig, y1_hist, y2_hist)
    (X, Y, Z) = inst.local_hw_map(swo, '2015-06-02', [xmin, ymin], [xmax, ymax], nb_points=20)
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.3, cstride=1, rstride=1, linewidth=0, antialiased=True)
    
    xpoints = np.array([x1_orig, x2_orig, x2_hist, x_default])
    ypoints = np.array([y1_orig, y2_orig, y2_hist, y_default])
    zpoints = np.zeros(xpoints.shape)
    for x,y,rank in zip(xpoints, ypoints, range(len(xpoints))):
        swo.model.setParams(ql.Array([x, y]))
        zpoints[rank] = swo.model.value(swo.model.params(), swo.helpers)
        
    ax.scatter(xpoints, ypoints, zpoints)
    plt.savefig('../data/surf.eps', bbox_inches="tight")  
    xpoints.shape = (4, 1)
    ypoints.shape = (4, 1)
    zpoints.shape = (4, 1)
    print np.concatenate((xpoints, ypoints, zpoints), axis=1)
    return (X, Y, Z)


def plot_history(file_name):
    file_name = "../data_corr_mid_2014/" + file_name
    model = nn.read_model(file_name)
    data = np.array(model.history['history']['val_loss'])
    data.shape = (data.shape[0], 1)
    du.plot_data(None, data)
    return data

    
def g2_plot_all():
    #du.data_dir = "../data/"
    mark_read_1 = 'adj_err_s'
    mark_read_2 = '_mse_lr_1.0e-04_ex6_lay9_d20_bn_res_1_rlr_5.0e-01_rlrmin_5.0e-06_rlrpat_10_estop_41'
    mark_write = 'history_adj_err'
    data_labels = ('Simulated Annealing', 'Neural Network')
    labels = ('0.5_0-264', 
              '0.5_44-308', 
              '0.5_88-352', 
              '0.5_132-396', 
              '0.5_176-440', 
              '0.5_220-484',
              '0.5_264-528',
              '0.5_308-572',
              '0.5_352-616',
              '0.99_396-660', 
              '0.99_440-704', 
              '0.99_484-748', 
              '0.99_528-792', 
              '0.99_572-836', 
              '0.99_616-880')
    #labels = ('0.5_0-264', 
    #          '0.5_132-396', 
    #          '0.5_308-572', 
    #          '0.99_440-704', 
    #          '0.99_572-836')
    #labels = ('0.5_0-264',)
    
    #              '0.5_264-528',
    npv = None
    vola = None
    out_of_sample=264
    for rank, label in enumerate(labels):
        dates, values, _, _ = get_fnn(mark_read_1 + label + mark_read_2)
        if npv is None:
            npv = np.empty((dates.shape[0], len(data_labels)))
            npv.fill(np.nan)
            vola = np.empty((dates.shape[0], len(data_labels)))
            vola.fill(np.nan)
        
        lims = [int(x) for x in label.split('_')[1].split('-')]
        npv[lims[0]:, 1] = values[lims[0]:, 5] #Objective prior
        temp = values[lims[0]:, 4] #History
        temp3 = values[lims[0]:, 3] #Default starting point
        filt = temp3 < temp
        temp[filt] = temp3[filt]
        npv[lims[0]:, 0] = temp

        vola[lims[0]:, 1] = values[lims[0]:, 2]
        temp_v = values[lims[0]:, 1] #History
        temp3_v = values[lims[0]:, 0] #Default starting point
        filt = temp3_v < temp_v
        temp_v[filt] = temp3_v[filt]
        vola[lims[0]:, 0] = temp_v
    
    vola *= 100
    #colors = ('#66c2a5', '#fc8d62', '#8da0cb')
    colors = ('#fc8d62', '#8da0cb')
    du.plot_data(dates, npv, figsize=(21, 12), labels=data_labels, 
                     save=du.data_dir + mark_write + '_npv_error_fnn.eps', colors=colors,
                     legend_fontsize=22, legend_color='black', 
                     xlabel_fontsize=22, xlabel_color='black', 
                     ylabel_fontsize=22, ylabel_color='black',
                     xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f}', 
                     ytick_fontsize=18, ytick_color='black',
                     title='NPV Mean Square Error', title_fontsize=26,
                     out_of_sample=out_of_sample)
    du.plot_data(dates, vola, figsize=(21, 12), labels=data_labels, 
                     save=du.data_dir + mark_write + '_vola_error_fnn.eps', colors=colors,
                     legend_fontsize=22, legend_color='black', 
                     xlabel_fontsize=22, xlabel_color='black', 
                     ylabel_fontsize=22, ylabel_color='black',
                     xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f} %', 
                     ytick_fontsize=18, ytick_color='black',
                     title='Average Volatility Error', title_fontsize=26,
                     out_of_sample=out_of_sample)
    temp = vola[:, 1] - vola[:, 0]
    temp = temp.reshape((temp.shape[0], 1))
    du.plot_data(dates, temp, figsize=(21, 12), labels=None, 
                     save=du.data_dir + mark_write + '_vola_diff_error_fnn.eps', colors=colors,
                     legend_fontsize=22, legend_color='black', 
                     xlabel_fontsize=22, xlabel_color='black', 
                     ylabel_fontsize=22, ylabel_color='black',
                     xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f} %', 
                     ytick_fontsize=18, ytick_color='black',
                     title='Difference in Average Volatility Error', title_fontsize=26,
                     out_of_sample=out_of_sample)
    
    
    
def g2_vola_heat_map():
    file_name = du.data_dir + 'swo_gbp_g2pp_nn_adj_err_s0.5_0-264_mse_lr_1.0e-04_ex6_lay9_d20_bn_res_1_rlr_5.0e-01_rlrmin_5.0e-06_rlrpat_10_estop_41.p'
    model = nn.read_model(file_name)
    
    model_dict = inst.g2
    swo = inst.get_swaptiongen(model_dict)
    errs = swo.history_heatmap(model, dates=swo._dates[264:308])

def g2_objective_graph():
    mark_read_1 = 'adj_err_s'
    mark_read_2 = '_mse_lr_1.0e-04_ex6_lay9_d20_bn_res_1_rlr_5.0e-01_rlrmin_5.0e-06_rlrpat_10_estop_41'
    mark_write = 'history_adj_err_4m'
    data_labels = ('Simulated Annealing', 'Neural Network')
    labels = ('0.5_0-264', 
              '0.5_44-308', 
              '0.5_88-352', 
              '0.5_132-396', 
              '0.5_176-440', 
              '0.5_220-484',
              '0.5_264-528',
              '0.5_308-572',
              '0.5_352-616',
              '0.99_396-660', 
              '0.99_440-704', 
              '0.99_484-748', 
              '0.99_528-792', 
              '0.99_572-836', 
              '0.99_616-880')
    
    labels = ('0.5_0-264',  
              '0.5_88-352',
              '0.5_176-440', 
              '0.5_264-528',
              '0.5_352-616', 
              '0.99_440-704', 
              '0.99_528-792', 
              '0.99_616-880')
    
    model_dict = inst.g2
    swo = inst.get_swaptiongen(model_dict)
    max_rank = len(labels)-1
    prev = 0
    
    npv = None
    vola = None    
    for rank, label in enumerate(labels):
        dates, values, _, _ = get_fnn(mark_read_1 + label + mark_read_2)
        if npv is None:
            npv = np.empty((dates.shape[0], len(data_labels)))
            npv.fill(np.nan)
            vola = np.empty((dates.shape[0], len(data_labels)))
            vola.fill(np.nan)
            out_of_sample = int(label.split('_')[1].split('-')[1])
            
        file_name = du.data_dir + 'swo_gbp_g2pp_nn_' + mark_read_1 + label + mark_read_2 + '.p'
        model = nn.read_model(file_name)
        if rank < max_rank:
            max_date = int(labels[rank+1].split('_')[1].split('-')[1])
        else:
            max_date = -1
        
        #Objective prior
        npv[prev:max_date, 1], vola[prev:max_date, 1] = swo.objective_values(model, prev, max_date)

        temp = values[prev:max_date, 4] #History
        temp3 = values[prev:max_date, 3] #Default starting point
        filt = temp3 < temp
        temp[filt] = temp3[filt]
        npv[prev:max_date, 0] = temp

        
        temp_v = values[prev:max_date, 1] #History
        temp3_v = values[prev:max_date, 0] #Default starting point
        filt = temp3_v < temp_v
        temp_v[filt] = temp3_v[filt]
        vola[prev:max_date, 0] = temp_v
        
        prev = max_date
        
    
    vola *= 100
    #colors = ('#66c2a5', '#fc8d62', '#8da0cb')
    colors = ('#fc8d62', '#8da0cb')
    du.plot_data(dates, npv, figsize=(21, 12), labels=data_labels, 
                     save=du.data_dir + mark_write + '_npv_error_fnn.eps', colors=colors,
                     legend_fontsize=22, legend_color='black', 
                     xlabel_fontsize=22, xlabel_color='black', 
                     ylabel_fontsize=22, ylabel_color='black',
                     xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f}', 
                     ytick_fontsize=18, ytick_color='black',
                     title='NPV Mean Square Error', title_fontsize=26,
                     out_of_sample=out_of_sample)
    du.plot_data(dates, vola, figsize=(21, 12), labels=data_labels, 
                     save=du.data_dir + mark_write + '_vola_error_fnn.eps', colors=colors,
                     legend_fontsize=22, legend_color='black', 
                     xlabel_fontsize=22, xlabel_color='black', 
                     ylabel_fontsize=22, ylabel_color='black',
                     xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f} %', 
                     ytick_fontsize=18, ytick_color='black',
                     title='Average Volatility Error', title_fontsize=26,
                     out_of_sample=out_of_sample)
    temp = vola[:, 1] - vola[:, 0]
    temp = temp.reshape((temp.shape[0], 1))
    du.plot_data(dates, temp, figsize=(21, 12), labels=None, 
                     save=du.data_dir + mark_write + '_vola_diff_error_fnn.eps', colors=colors,
                     legend_fontsize=22, legend_color='black', 
                     xlabel_fontsize=22, xlabel_color='black', 
                     ylabel_fontsize=22, ylabel_color='black',
                     xtick_fontsize=18, xtick_color='black', yticks_format='{:.2f} %', 
                     ytick_fontsize=18, ytick_color='black',
                     title='Difference in Average Volatility Error', title_fontsize=26,
                     out_of_sample=out_of_sample)
    
    
    return (npv, vola)
    
#model = nn.read_model('../data/swo_gbp_hull_white_analytic_formulae_nn_s140000.p')
#swo = inst.get_swaptiongen(inst.hullwhite_analytic)
#orig_errors, errors = swo.errors(model, '2013-01-01')
#    
#orig_errors = orig_errors.reshape(12, 13)
#errors = errors.reshape(12, 13)
#    
#red_blue = red_purple = brewer2mpl.get_map('RdBu', 'Diverging', 7).mpl_colormap
#fig, ax = ppl.subplots(1)
#ppl.pcolormesh(fig, ax, orig_errors,
#               xticklabels=swo.axis(0).values, 
#               yticklabels=swo.axis(1).values,
#               cmap=red_blue)
#fig, ax = ppl.subplots(1)
#ppl.pcolormesh(fig, ax, errors,
#               xticklabels=swo.axis(0).values, 
#               yticklabels=swo.axis(1).values,
#               cmap=red_blue)

#X, Y, Z = plot4()
#g2_plot_all()
#objectives, lim_alpha, lim_beta = g2_objective_graph()
(npv, vola) = g2_objective_graph()