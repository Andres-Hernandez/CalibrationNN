# -*- mode: python; tab-width: 4;

# Copyright (C) 2016 Andres Hernandez
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the license for more details.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

data_dir = '../data/'
h5file = data_dir + 'data.h5'
h5_ts_node = 'TS'
swo_gbp_tskey = h5_ts_node + '/SWO/GBP'
ois_gbp_tskey = h5_ts_node + '/IRC/GBP/OIS'
l6m_gbp_tskey = h5_ts_node + '/IRC/GBP/L6M'

# http://tableaufriction.blogspot.co.uk/2012/11/finally-you-can-use-tableau-data-colors.html
tableau20 = [
(0.121568627451, 0.466666666667, 0.705882352941),
(0.682352941176, 0.780392156863, 0.909803921569),
(1.0, 0.498039215686, 0.0549019607843),
(1.0, 0.733333333333, 0.470588235294),
(0.172549019608, 0.627450980392, 0.172549019608),
(0.596078431373, 0.874509803922, 0.541176470588),
(0.839215686275, 0.152941176471, 0.156862745098),
(1.0, 0.596078431373, 0.588235294118),
(0.580392156863, 0.403921568627, 0.741176470588),
(0.772549019608, 0.690196078431, 0.835294117647),
(0.549019607843, 0.337254901961, 0.294117647059),
(0.76862745098, 0.611764705882, 0.580392156863),
(0.890196078431, 0.466666666667, 0.760784313725),
(0.96862745098, 0.713725490196, 0.823529411765),
(0.498039215686, 0.498039215686, 0.498039215686),
(0.780392156863, 0.780392156863, 0.780392156863),
(0.737254901961, 0.741176470588, 0.133333333333),
(0.858823529412, 0.858823529412, 0.552941176471),
(0.0901960784314, 0.745098039216, 0.811764705882),
(0.619607843137, 0.854901960784, 0.898039215686)]

tableau10mid = [
(1.0, 0.619607843137, 0.290196078431),
(0.929411764706, 0.4, 0.364705882353),
(0.678431372549, 0.545098039216, 0.788235294118),
(0.447058823529, 0.619607843137, 0.807843137255),
(0.403921568627, 0.749019607843, 0.360784313725),
(0.929411764706, 0.592156862745, 0.792156862745),
(0.803921568627, 0.8, 0.364705882353),
(0.658823529412, 0.470588235294, 0.43137254902),
(0.635294117647, 0.635294117647, 0.635294117647),
(0.427450980392, 0.8, 0.854901960784)]

almost_black = '#262626'
light_grey = np.array([float(248)/float(255)]*3)


def read_csv(file_name):
    #For swaptions Data is assumed to come in the form
    #<Date, format YYYY-mm-dd>,<Option Term>,<Swaption Term>, <Value>
    #For term structures Data is assumed to come in the form
    #<Date>, <Term>, <Value>
    cols = pd.read_csv(file_name, nrows=1).columns
    return pd.read_csv(file_name, parse_dates=[0], infer_datetime_format=True, 
                     index_col=cols.tolist()[:-1])


def store_hdf5(file_name, key, val):
    with pd.HDFStore(file_name) as store:
        store[key] = val
        store.close()


def csv_to_hdf5(file_name, key, hdf5file_name):
    res = read_csv(file_name)
    store_hdf5(hdf5file_name, key, res)


def from_hdf5(key, file_name=h5file):
    with pd.HDFStore(file_name) as store:
        data =  store[key]
        store.close()
    return data

        
def tofile(file_name, model):
    joblib.dump(model, file_name) 

    
def fromfile(file_name):
    return joblib.load(file_name) 


class TimeSeriesData(object):
    def __init__(self, key, file_name=h5file):
        self._data = from_hdf5(key, file_name)
        fr = self._data.iloc[0]
        fd = self._data.loc[fr.name[0]].index
        if hasattr(fd, 'levels'):
            self._levels = fd.levels
            self._axis_shape = tuple( [len(x) for x in self._levels] )
        else:
            self._levels = [fd]
            self._axis_shape = ( len(fd), )
        self._dates = self._data.index.levels[0]
        self._pipeline = None
        
    def __getitem__(self, date):
        return self.__getimpl(date)

    def __getimpl(self, date):
        data = self._data.loc[date].as_matrix()
        data.shape = self._axis_shape
        return data
    
    def axis(self, i):
        return self._levels[i];
    
    def dates(self):
        return self._dates
        
    def intersection(self, other):
        return self._dates.intersection(other._dates)
        
    def to_matrix(self, *args):
        if len(args) > 0:
            dates = args[0]
        else:
            dates = self._data.index.levels[0]
        nbrDates = len(dates)
        mat = np.zeros((nbrDates,) + self._axis_shape)
        for iDate in range(nbrDates):
            mat[iDate] = self.__getimpl(dates[iDate])
        
        return mat

    def pca(self, **kwargs):
        if 'n_components' in kwargs:
            nComp = kwargs['n_components']
        else:
            nComp = 0.995

        if 'dates' in kwargs:
            mat = self.to_matrix(kwargs['dates'])
        else:
            mat = self.to_matrix()
        scaler = StandardScaler()
        pca = PCA(n_components=nComp)
        self._pipeline = Pipeline([('scaler', scaler), ('pca', pca)])
        self._pipeline.fit(mat)
        
        if 'file' in kwargs:
            tofile(kwargs['file'], self._pipeline)
        
        return self._pipeline


def plot_data(times, data, labels=None, figsize=(10, 7.5), frame_lines=False,
              yticks_format=None, save=None, min_x_ticks=5, max_x_ticks=8,
              interval_multiples=True, colors=None, title=None,
              legend_fontsize=None, legend_color=almost_black,
              title_fontsize=None, title_color=almost_black,
              xlabel=None, ylabel=None, 
              xlabel_color=almost_black, ylabel_color=almost_black,
              xlabel_fontsize=14, ylabel_fontsize=14, 
              xtick_fontsize=14, ytick_fontsize=14,
              xtick_color=almost_black, ytick_color=almost_black,
              out_of_sample=None):
    # Taken from 
    # http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
    # and
    # http://blog.olgabotvinnik.com/blog/2013/08/21/2013-08-21-prettyplotlib-painlessly-create-beautiful-matplotlib/
    almost_black = '#262626'
    light_grey = np.array([float(248)/float(255)]*3)
    
    if times is None:
        times = np.arange(data.shape[0])
    
    #Validate input
    if labels is not None:
        assert data.shape[1] == len(labels)
    assert len(times) == data.shape[0]    
    
    if colors is None:
        if data.shape[1] <= 2:
            colors = ('#ef8a62', '#67a9cf')
        if data.shape[1] <= 10:
            colors = tableau10mid
        else:
            colors = tableau20
    nb_colors = len(colors)
    
    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    
    # Remove the plot frame lines
    if not frame_lines:
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()

    #Set y ticks and limits
    data[np.isinf(data)] = np.nan
    yindex = ~np.isnan(data)
    ymin = np.min(data[yindex])
    ymin -= np.abs(ymin)*0.05
    ymax = np.max(data[yindex])
    ymax += np.abs(ymax)*0.05
    ystep = (ymax-ymin)/10.
    ytick_range = np.arange(ymin+ystep, ymax, ystep)
    plt.ylim(ymin, ymax)
    if yticks_format is None:
        plt.yticks(ytick_range, fontsize=ytick_fontsize)
    else:
        plt.yticks(ytick_range, [yticks_format.format(x) for x in ytick_range], 
                   fontsize=ytick_fontsize)

    ax.tick_params(axis='y', colors=ytick_color)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=ylabel_fontsize)
        ax.yaxis.label.set_color(ylabel_color)
        
    #Set x ticks and limits
    if isinstance(times, np.ndarray) and times.dtype.type == np.datetime64:
        locator = AutoDateLocator(minticks=min_x_ticks, maxticks=max_x_ticks,
                                  interval_multiples=interval_multiples)
        formatter = DateFormatter('%d-%m-%Y')        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.autoscale_view()
        xtick_range = times
        xmin = times[0]
        xmax = times[-1]
        delta = xmax - xmin
        xmin = xmin - delta*0.02
        xmax = xmax + delta*0.02
        
    else:
        xmin = 0
        xmax = data.shape[0]+1
        xtick_range = range(xmin, xmax)
        
    plt.xlim(xmin, xmax)        
    plt.xticks(fontsize=xtick_fontsize)
    ax.tick_params(axis='x', colors=xtick_color)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=xlabel_fontsize)
        ax.xaxis.label.set_color(xlabel_color)
            
    # Provide tick lines across the plot
    for y in ytick_range:
        plt.plot(xtick_range, [y] * len(xtick_range), "--", lw=0.5, color="black", alpha=0.3)

    #   The following was only for a few graphs in the presentation
    if out_of_sample is not None:
        xbreak = times[out_of_sample]
        ax.axvline(xbreak, lw=2.0, color="black", alpha=0.3)
        # place a text box in upper left in axes coords
        ax.text(xbreak, ymax*0.8, r'$\mathbf{\rightarrow}$ Out of sample', 
                fontsize=ylabel_fontsize, verticalalignment='center', 
                horizontalalignment='left')
        xbreak = times[out_of_sample-4]
        ax.text(xbreak, ymax*0.8, r'In sample $\mathbf{\leftarrow}$', 
                fontsize=ylabel_fontsize, verticalalignment='center', 
                horizontalalignment='right')
            
    # Remove the tick marks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",    
                    labelbottom="on", left="off", right="off", labelleft="on") 
    
    #Plot the data
    for i in range(data.shape[1]):
        if labels is not None:
            plt.plot(times, data[:, i], lw=2.5, color=colors[i%nb_colors], label=labels[i])
        else:
            plt.plot(times, data[:, i], lw=2.5, color=colors[i%nb_colors])
    
    
    if labels is not None:
        #Modify legend
        legend = ax.legend(frameon=True, scatterpoints=1)
        rect = legend.get_frame()
        rect.set_facecolor(light_grey)
        rect.set_linewidth(0.0)
        
        # Change the legend label colors to almost black, too
        texts = legend.texts
        for t in texts:
            if legend_color is not None:
                t.set_color(legend_color)
            else:
                t.set_color(almost_black)
            if legend_fontsize is not None:
                t.set_fontsize(legend_fontsize)

    if title is not None:
        if title_color is not None:
            ax.title.set_color(title_color)
        else:
            ax.title.set_color(almost_black)
        if title_fontsize is not None:
            ax.set_title(title, fontsize=title_fontsize)
        else:
            ax.set_title(title)
        

    if save is not None:
        plt.savefig(save, bbox_inches="tight")  


def gbp_to_hdf5():
    swo_csvfile = data_dir + 'swaption_gbp_20130101_20160601.csv'
    ois_csvfile = data_dir + 'ois_gbp_20130101_20160601.csv'
    l6m_csvfile = data_dir + 'libor_6m_gbp_20130101_20160601.csv'
    csv_to_hdf5(swo_csvfile, swo_gbp_tskey, h5file)
    csv_to_hdf5(ois_csvfile, ois_gbp_tskey, h5file)
    csv_to_hdf5(l6m_csvfile, l6m_gbp_tskey, h5file)