# Archivo que compila funciones de estadistica y ciencias de datos
# obtenidas del curso de Data Science de JHU
# Ricardo Alvarez


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

#######################################################################################
# Esta función calcula el tamaño de los bins de un histograma usando la ecuación
# de free_diaconis 
def FreedmanDiaconis( data):
    quartiles = stats.mstats.mquantiles( data, [0.25, 0.5, 0.75])
    iqr = quartiles[2] - quartiles[ 0]
    n = len(data)
    h = 2.0 * (iqr/n**(1.0/3.0))
    return int(h)

#######################################################################################
# Esta función clasifica que tanta correlación lineal tienen dos variables en función 
# del valor de rho (R)
def ClassifyCorrelation(r):
    r = abs(r)
    if r < 0.16:
        return "correlación lineal casi nula"
    if r < 0.29:
        return "correlación lineal leve"
    if r < 0.49:
        return "correlación lineal baja"
    if r < 0.69:
        return "correlación lineal moderada"
    if r < 0.89:
        return "correlación lineal alta"
    return "very strong"

#######################################################################################
# Esta función calcula el valor de correlaciones entre dos variables
# Calcula el Pearson - correlación lineal
# Calcula el spearman - correlación monotonica

def Correlation(data, x, y):
    print("Correlation coefficients:")
    r = stats.pearsonr(data[x], data[y])[0]
    print( "r =", r, f"({ClassifyCorrelation(r)})")
    rho = stats.spearmanr(data[x], data[y])[0]
    print( "rho =", rho, f"({ClassifyCorrelation(rho)})")
    
    
#######################################################################################
# Multibox plot creates a chart between a categorical and a numerical value
def multiboxplot(data, numeric, categorical, skip_data_points=True):
    figure = plt.figure(figsize=(10, 6))
    axes = figure.add_subplot(1, 1, 1)
    grouped = data.groupby(categorical)
    labels = pd.unique(data[categorical].values)
    labels.sort()
    grouped_data = [grouped[numeric].get_group( k) for k in labels]
    patch = axes.boxplot(grouped_data, labels=labels, patch_artist=True, zorder=1)
    if not skip_data_points:
        for i, k in enumerate(labels):
            subdata = grouped[numeric].get_group( k)
            x = np.random.normal(i + 1, 0.01, size=len(subdata))
            axes.plot(x, subdata, 'o', alpha=0.4, color="DimGray", zorder=2)}
    axes.set_xlabel(categorical)
    axes.set_ylabel(numeric)
    axes.set_title("Distribution of {0} by {1}".format(numeric, categorical))
    plt.show()
    plt.close()

#######################################################################################
# This a lowess_scatter
def lowess_scatter(data, x, y, jitter=0.0, skip_lowess=False):
    if skip_lowess:
        fit = np.polyfit(data[x], data[y], 1)
        line_x = np.linspace(data[x].min(), data[x].max(), 10)
        line = np.poly1d(fit)
        line_y = list(map(line, line_x))
    else:
        lowess = sm.nonparametric.lowess(data[y], data[x], frac=.3)
        line_x = list(zip(*lowess))[0]
        line_y = list(zip(*lowess))[1]
    figure = plt.figure(figsize=(10, 6))
    axes = figure.add_subplot(1, 1, 1)
    xs = data[x]
    if jitter > 0.0:
        xs = data[x] + stats.norm.rvs( 0, 0.5, data[x].size)
    axes.scatter(xs, data[y], marker="o", color="DimGray", alpha=0.5)
    axes.plot(line_x, line_y, color="DarkRed")
    title = "Plot of {0} v. {1}".format(x, y)
    if not skip_lowess:
        title += " with LOWESS"
    axes.set_title(title)
    axes.set_xlabel(x)
    axes.set_ylabel(y)
    plt.show()
    plt.close()
    
#######################################################################################
# Esta función evalua una variable categorica y da la cantidad y la frecuencia de cada una 
def summarize_category(series):
    res_regu = value_counts(series)
    res_norm = value_counts(series, normalize=True)
    result = concat([res_regu, res_norm], axis=1, keys=['Cantidad', 'Frecuencia'])
    result = result.sort_index()
    return result
