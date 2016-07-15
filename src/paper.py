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
#import neural_network as nn
import instruments as inst
import QuantLib as ql
import pandas as pd
#import plotly.plotly as py
#import plotly.graph_objs as go

def get_fnn(middle='_adj_error'):
    dates = np.load(du.data_dir + 'test_' + middle + '_fnn_l4_e6_epoch200_dates.npy')
    values = np.load(du.data_dir + 'test_' + middle + '_fnn_l4_e6_epoch200_values.npy')
    val_hist = np.load(du.data_dir + 'test_' + middle + '_fnn_l4_e6_epoch200_val_hist.npy')
    train_hist = np.load(du.data_dir + 'test_' + middle + '_fnn_l4_e6_epoch200_train_hist.npy')
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
    du.data_dir = '../data_corr_mid_2014/'
    mark_read = 'adj_error_s'
    mark_write = 'adj_error_insample40_s'
    data_labels = ('Default Starting Point', 'Historical Starting Point', 'Feed-forward Neural Net')
    labels = ('50k_d20', '100k_d20', '150k_d20')
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
    swo = inst.getSwaptionGen(inst.hullwhite_analytic)
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

#model = nn.read_model('../data/swo_gbp_hull_white_analytic_formulae_nn_s140000.p')
#swo = inst.getSwaptionGen(inst.hullwhite_analytic)
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
plot()