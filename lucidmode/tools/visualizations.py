
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- Project: lucidmode                                                                                  -- #
# -- Description: A Lightweight Framework with Transparent and Interpretable Machine Learning Models.    -- #
# -- visualizations.py: python script with visualization functions                                       -- #
# -- Author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- Repository: https://github.com/lucidmode/lucidmode                                                  -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# -- Load libraries for script
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------------------------- WEIGHTS ON LAYER -- #
# --------------------------------------------------------------------------------------------------------- #

# - Weight values per layer (Colored bar for each neuron, separation of all layers).

# ------------------------------------------------------------------------------ COST FUNCTION EVOLUTION -- #
# --------------------------------------------------------------------------------------------------------- #

# - CostFunction (train-val) evolution (two lines plot with two y-axis).

# plot cost evolution
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')
# plt.figure(figsize=(16, 4))
# plt.plot(list(J.keys()), list(J.values()), color='r', linewidth=3)
# plt.title('Cost over epochs')
# plt.xlabel('epochs')
# plt.ylabel('cost');
# plt.show()

# -------------------------------------------------------------------------------- CONVOLUTION OPERATION -- #
# --------------------------------------------------------------------------------------------------------- #

# - Convolution operation between layers.

# ---------------------------------------------------------------------------------------- IMAGE CATALOG -- #
# --------------------------------------------------------------------------------------------------------- #

# - A matrix of nxm randomly seleceted images for visual exploration

# cols = 10
# rows = 4
# fig, axs = plt.subplots(rows, cols, figsize=(16, 5))
# for i in range(rows):
    #img = 0
    #l = np.nonzero(labels == i)
    #for j in np.random.choice(l[0], cols):
        #axs[i, img].axis('off')
        #hm = images[j, :].reshape(28, 28)
        #axs[i, img].imshow(hm.astype(np.uint8))
        #img += 1

# -- -------------------------------------------- PLOT: OHLC Candlesticks + Colored Classificator Result -- #
# -- --------------------------------------------------------------------------------------------------- -- #

def plot_ohlc_class(p_ohlc, p_theme, p_data_class, p_vlines):
    """

    """

    # default value for lables to use in main title, and both x and y axisp_fonts
    if p_theme['p_labels'] is not None:
        p_labels = p_theme['p_labels']
    else:
        p_labels = {'title': 'Main title', 'x_title': 'x axis title', 'y_title': 'y axis title'}

    # tick values calculation for simetry in y axes
    y0_ticks_vals = np.arange(min(p_ohlc['low']), max(p_ohlc['high']),
                             (max(p_ohlc['high']) - min(p_ohlc['low'])) / 5)
    y0_ticks_vals = np.append(y0_ticks_vals, max(p_ohlc['high']))
    y0_ticks_vals = np.round(y0_ticks_vals, 4)

    # reset the index of the input data
    p_ohlc.reset_index(inplace=True, drop=True)
    
    # auxiliar lists
    train_val_error = []
    train_val_success = []

    if 'train_y' in list(p_data_class.keys())[0]:

        # p_ohlc has all the prices and p_data_class has the prediction classes
        # since p_data_class is smaller than p_ohlc, a lagged shift is needed
        feature_lag = int(np.where(p_ohlc['timestamp'] == p_data_class['train_y'].index[0])[0]) 
        ohlc_lag = list(np.arange(0, feature_lag, 1))
        
        # add vertical line to indicate where ends the ohlc lag for feature engineering
        p_vlines.append(p_ohlc['timestamp'][feature_lag])
        p_vlines = sorted(p_vlines)

        # error and success in train
        for row_e in np.arange(0, len(p_data_class['train_y'].index.to_list()), 1):
            if p_data_class['train_y'][row_e] != p_data_class['train_y_pred'][row_e]:
                train_val_error.append(feature_lag + row_e)
            else:
                train_val_success.append(feature_lag + row_e)

        # accuracy in train data set
        train_val_acc = round(len(train_val_success) / (len(train_val_error) + len(train_val_success)), 2)

    # ------------------------------------------------------------------------------ In case of val set -- #
    
    if 'val_y' in list(p_data_class.keys())[0]:
        
        # p_ohlc has all the prices and p_data_class has the prediction classes
        # since p_data_class is smaller than p_ohlc, a lagged shift is needed
        feature_lag = 4# int(np.where(p_ohlc['timestamp'] == p_data_class['val_y'].index[0])[0]) 
        ohlc_lag = list(np.arange(0, feature_lag, 1))
        
        # add vertical line to indicate where ends the ohlc lag for feature engineering
        p_vlines.append(p_ohlc['timestamp'][feature_lag])
        p_vlines = sorted(p_vlines)

        # error and success in val
        for row_s in np.arange(0, len(p_data_class['val_y'].index.to_list()), 1):
            if p_data_class['val_y'].iloc[row_s, 1] != p_data_class['val_y_pred'].iloc[row_s,1]:
                train_val_error.append(feature_lag + row_s)
            else:
                train_val_success.append(feature_lag + row_s)

        # overall accuracy
        train_val_acc = round(len(train_val_success) / (len(train_val_error) + len(train_val_success)), 2)


    # Instantiate a figure object
    fig_g_ohlc = go.Figure()    

    # Layout for margin, and both x and y axes
    fig_g_ohlc.update_layout(margin=go.layout.Margin(l=50, r=50, b=20, t=60, pad=20),
                             xaxis=dict(title_text=p_labels['x_title']),
                             yaxis=dict(title_text=p_labels['y_title']))

    # Add layer for coloring in gray the non predicted candles in OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(
        x=[p_ohlc['timestamp'].iloc[i] for i in ohlc_lag],
        open=[p_ohlc['open'].iloc[i] for i in ohlc_lag],
        high=[p_ohlc['high'].iloc[i] for i in ohlc_lag],
        low=[p_ohlc['low'].iloc[i] for i in ohlc_lag],
        close=[p_ohlc['close'].iloc[i] for i in ohlc_lag],
        increasing={'line': {'color': 'grey'}},
        decreasing={'line': {'color': 'grey'}},
        name='Subset/Feature-Lag'))

    # Add layer for the success based color of candles in OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(
        x=[p_ohlc['timestamp'].iloc[i] for i in train_val_success],
        open=[p_ohlc['open'].iloc[i] for i in train_val_success],
        high=[p_ohlc['high'].iloc[i] for i in train_val_success],
        low=[p_ohlc['low'].iloc[i] for i in train_val_success],
        close=[p_ohlc['close'].iloc[i] for i in train_val_success],
        increasing={'line': {'color': 'skyblue'}},
        decreasing={'line': {'color': 'skyblue'}},
        name='Prediction Success'))

    # Add layer for the error based color of candles in OHLC candlestick chart
    fig_g_ohlc.add_trace(go.Candlestick(
        x=[p_ohlc['timestamp'].iloc[i] for i in train_val_error],
        open=[p_ohlc['open'].iloc[i] for i in train_val_error],
        high=[p_ohlc['high'].iloc[i] for i in train_val_error],
        low=[p_ohlc['low'].iloc[i] for i in train_val_error],
        close=[p_ohlc['close'].iloc[i] for i in train_val_error],
        increasing={'line': {'color': 'red'}},
        decreasing={'line': {'color': 'red'}},
        name='Prediction Error'))

    # Update layout for the background
    fig_g_ohlc.update_layout(yaxis=dict(tickfont=dict(color='grey',
     size=p_theme['p_fonts']['font_axis']), tickvals=y0_ticks_vals),
     xaxis=dict(tickfont=dict(color='grey',
     size=p_theme['p_fonts']['font_axis'])))

    # Update layout for the y axis
    fig_g_ohlc.update_xaxes(rangebreaks=[dict(pattern="day of week", bounds=['sat', 'sun'])])

    # If parameter vlines is used
    if p_vlines is not None:
        # Dynamically add vertical lines according to the provided list of x dates.
        shapes_list = list()
        for i in p_vlines:
            shapes_list.append({'type': 'line', 'fillcolor': p_theme['p_colors']['color_1'],
                                'line': {'color': p_theme['p_colors']['color_1'],
                                         'dash': 'dashdot', 'width': 2},
                                'x0': i, 'x1': i, 'xref': 'x',
                                'y0': min(p_ohlc['low']), 'y1': max(p_ohlc['high']), 'yref': 'y'})

        # add v_lines to the layout
        fig_g_ohlc.update_layout(shapes=shapes_list)

    # Legend format
    fig_g_ohlc.update_layout(legend=go.layout.Legend(x=.2, y=-.3, orientation='h',
                                                     bordercolor='dark grey',
                                                     borderwidth=1,
                                                     font=dict(size=p_theme['p_fonts']['font_axis'])))

    # Update layout for the background
    fig_g_ohlc.update_layout(title_font_size=p_theme['p_fonts']['font_title'],
                             title=dict(x=0.5, text=p_labels['title'] + ' | acc: ' + 
                                                    str(train_val_acc)),
                             yaxis=dict(title=p_labels['y_title'],
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)),
                             xaxis=dict(title=p_labels['x_title'], rangeslider=dict(visible=False),
                                        titlefont=dict(size=p_theme['p_fonts']['font_axis'] + 4)))

    # Final plot dimensions
    fig_g_ohlc.layout.autosize = True
    fig_g_ohlc.layout.width = p_theme['p_dims']['width']
    fig_g_ohlc.layout.height = p_theme['p_dims']['height']

    return fig_g_ohlc
