from __future__ import print_function
import pandas as pd

# from pandas_ods_reader import read_ods
from scipy import stats
# Non-linear least squares fitting
from scipy.optimize import curve_fit

import numpy as np
import matplotlib
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates

import statsmodels

import datetime
import random
import os
import tempfile
import config

# Avoid out of memory
# matplotlib.use('agg')
# Interact
matplotlib.use('QtAgg')


# PLOT STYLES
plt.style.use('ggplot')
#plt.style.use('fivethirtyeight')
#plt.style.use('seaborn-dark')
#plt.style.use('seaborn-whitegrid')
#plt.style.use('bmh')

# Get temporary directory
temp_dir = tempfile.gettempdir()

# Colors in plots
colors = [
    'cornflowerblue',
    'crimson',
    'darkolivegreen',
    'steelblue',
    'mediumpurple',
    'yellowgreen'
]


def weibull2P_pdf(x, sh, sc):
    """Weibull generic function

    Args:
        x (np.array): Data         
        sh (float): Shape > 0 
        sc (float): Scale > 0

    Returns:
        np.array: Array with probability
    """    
    return (sh / sc) * (x / sc)**(sh - 1) * np.exp(-(x / sc)**sh)


def weibull2P_cdf(x, sh, sc):
    """Weibull generic function

    Args:
        x (np.array): Data         
        sh (float): Shape > 0 
        sc (float): Scale > 0

    Returns:
        np.array: Array with probability
    """    
    return 1 - np.exp(-(x / sc)**sh)


def weibull3P_pdf(x, sh, sc, lo):
    """Weibull generic function

    Args:
        x (np.array): Data         
        sh (float): Shape > 0 
        sc (float): Scale > 0
        lo (float): Location > 0

    Returns:
        np.array: Array with probability
    """    
    return (sh / sc) * ((x - lo) / sc)**(sh - 1) * np.exp(-((x - lo) / sc)**sh)

def weibull3P_cdf(x, sh, sc, lo):
    """Weibull generic function

    Args:
        x (np.array): Data         
        sh (float): Shape > 0 
        sc (float): Scale > 0

    Returns:
        np.array: Array with probability
    """    
    return 1 - np.exp(-((x-lo) / sc)**sh)


def read_excel_climatic(file_path):
    """Read clamtic excel file 

    Args:
        file_path (string): The file path

    Returns:
        panda.Dataframe: The readead data
    """    
    df = pd.read_excel (file_path)
    df['dt'] = pd.to_datetime(df['Fecha'], dayfirst=True)
    df['da'] = df['dt'].dt.date
    df['rain'] = df['Precipitación Acumulada mm']
    # Drop not necessary columns
    return df


def read_excel_flow(file_path, warea):
    """Read flow excel file 

    Args:
        file_path (string): The file path
        warea (float): The watershed area

    Returns:
        panda.Dataframe: The readead data
    """    
    df = pd.read_excel (file_path)
    df['dt'] = pd.to_datetime(df['Fecha'], dayfirst=True)
    df['da'] = df['dt'].dt.date
    df['year'] = pd.DatetimeIndex(df['dt']).year
    df['month'] = pd.DatetimeIndex(df['dt']).month
    
    df['temp'] = df['Valor'].str.replace(',', '.')
    df['flow'] = pd.to_numeric(df['temp'])
    # L/s/km2
    df['sflow'] = df['flow'] / warea * 1000
    # Drop not necessary columns
    df = df.drop(
        columns=['Fecha', 'Valor', 'Índice de calidad', 'Índice de revisión', 'temp']
    )
    return df


def plot_flow(df, st, ax):
    plot = df.plot(x="dt", y=["flow"], grid=True, title=st, ax=ax)
    plot.xaxis.set_label_text('Date')
    plot.yaxis.set_label_text('Flow (m3/s)')
    return plot

# Dataframes
dfs = dict()
dcs = dict()
# Plots
plots = dict()

####################################################
# Flow data
rnames = ['Florida', 'Paso_Pache', 'Picada_de_Varela', 'Santa_Lucia']
for n in range(0,len(rnames)):
    st = rnames[n]
    dfs[st] = read_excel_flow(
        os.path.join(config.DATA_PATH, config.FLOW_STAT[st][-1]),
        config.FLOW_STAT[st][1]
    )
    # Average 7 days
    dfs[st]['flowa7'] = dfs[st]['flow'].rolling(7).mean()
    dfs[st]['sflowa7'] = dfs[st]['sflow'].rolling(7).mean()
    # Average 30 days
    dfs[st]['flowa30'] = dfs[st]['flow'].rolling(30).mean()
    dfs[st]['sflowa30'] = dfs[st]['sflow'].rolling(30).mean()    
####################################################


####################################################
# Quantiles
dfq = dict()
tl = range(0,1001)
ql = [x / 1000 for x in tl]
sts = ['Florida', 'Paso_Pache', 'Picada_de_Varela', 'Santa_Lucia']
for n in range(0,len(sts)):
    st = sts[n]
    dfq[st] = dfs[st].quantile(ql)
    dfq[st] = dfq[st].rename_axis('q').reset_index()
    dfq[st]['p'] = (1 - dfq[st]['q'])*100
    sflowa7_max = 5
    dfres = dfq[st][ dfq[st]['sflowa7'] < sflowa7_max ] 
    xx = dfres['sflowa7'].to_numpy()
    yy = dfres['q'].to_numpy()
    plt.plot(xx, yy, 'b-', label='{} data'.format(st))
    popt, pcov = curve_fit(weibull2P_cdf, xx, yy, bounds=([0,0], [5., 100.]))
    plt.plot(xx, weibull2P_cdf(xx, *popt), 'r-', label='fit: shape=%5.3f, scale=%5.3f' % tuple(popt))
    # popt, pcov = curve_fit(weibull3P_cdf, xx, yy, bounds=([0, 0, -2.], [5., 100., 2.]))
    # plt.plot(xx, weibull3P_cdf(xx, *popt), 'r-', label='fit: a=%5.3f, s=%5.3f, l=%5.3f' % tuple(popt))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xscale('log')
    plt.legend()
    plt.show()


fig, ax = plt.subplots(1, 1, sharex=False, sharey=False)
fig.set_size_inches(config.FIG_SIZE)
fig.set_dpi(config.FIG_DPI)
tl = range(0,1001)
ql = [x / 1000 for x in tl]
cont = 0
for st in sts:
    quan = dfs[st]['sflow'].quantile(ql)
    quan = quan.rename_axis('q').reset_index()
    quan['p'] = (1 - quan['q'])*100
    ax.plot(quan['p'], quan['sflow'], label="{}".format(st), c=colors[cont])
    cont = cont + 1

ax.set_xlim(0,100)
ax.set_xticks(np.arange(0, 105, step=5))
ax.set_yscale('log')
ax.set_xlabel('Probability (%)')
ax.set_ylabel('SFlow (L/s/km2)')
ax.set_title("Curva permanencia")
plt.legend()
plt.tight_layout()
plt.legend()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_pemanencia.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)


for st in sts:
    dfo = dfs[st].sort_values('sflowa7', ascending=False)
    qq = dfo['sflowa7'].to_numpy()
    plt.plot(qq, label="{}".format(st))
    plt.xlabel('Day')
    plt.ylabel('SFlow(L/s/km2)')
    plt.legend()
    plt.yscale('log')
    plt.show()

flows = dict()
fitted = dict()
for st in sts:
    flows[st] = dfs[st]['sflowa7'].to_numpy()
    flows[st] = flows[st][~np.isnan(flows[st])]
    # flows[st] = flows[st][flows[st] < 100]
    data = flows[st]
    shape, loc, scale = stats.weibull_min.fit(data) 
    print("{}: shape={}, loc={}, scale={}".format(st, shape, loc, scale))
    fitted[st] = stats.weibull_min(shape, loc, scale)

fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)
fig.set_size_inches(config.FIG_SIZE)
fig.set_dpi(config.FIG_DPI)
for st in sts:
    data = flows[st]
    ecdf= statsmodels.distributions.ECDF(data)
    x = np.linspace(data.min(), data.max(), 10000)
    stats.probplot(data, dist=fitted[st], plot=axs[0])    
    axs[1].plot(x, fitted[st].cdf(x), label=st)
    axs[1].scatter(x, ecdf(x), s=2, label="{}({})".format("ECDF ", st))
    # axs[1].scatter(dfq[st]['sflowa7'], dfq[st]['q'], label="{}({})".format(st,"Cuantiles"))
    tr10 = fitted[st].ppf(0.1)
    tr5 = fitted[st].ppf(0.2)
    print(st, tr10, tr5)

axs[1].set_xscale('log')
axs[1].set_yticks(np.arange(0, 1.05, step=0.05))

plt.tight_layout()
plt.legend()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_weibull_min.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
####################################################

####################################################
# Weibull fit fot 7Q10 or 7q10
# 1 - Encontrar el mínimo para cada año de datos, es decir, caudal mínimo de 7 días de duración;
# 2 - Ajustar esa serie a una distribución de Weibull;
# 3 - Seleccionar, de esa distribución, el dato que tenga 10 años de período de retorno
# 4 - Listo.
dfq = dict()
# Min 7 day for each year for each station
sfa7min = dict()
tl = range(0, 101)
ql = [x / 100 for x in tl]
sts =  ['Florida', 'Paso_Pache', 'Picada_de_Varela', 'Santa_Lucia']
for st in sts:
    # Years in this data
    years = dfs[st]['year'].unique()
    # Get min values for each year
    dmin = list()
    for y in years:
        dd = dfs[st].loc[ dfs[st]['year'] == y ]['sflowa7'].to_numpy()
        # x = x[~numpy.isnan(x)]
        dd = dd[~np.isnan(dd)]
        nn = len(dd)
        nnmin = 330
        if nn < nnmin:
            msg = "Warning: Year {} on station {} have {} (less than {})!!!!!!!!!!".format(
                y, st, nn, nnmin
            )
            print(msg)
        miny = dd.min()
        dmin.append(miny)
    # Append data to dict
    sfa7min[st] = dmin

# Now adjust min serie to Weibull distribution
flows = dict()
fitted = dict()
for st in sts:
    flows[st] = np.array( sfa7min[st] )
    data = flows[st]
    shape, loc, scale = stats.weibull_min.fit(data)
    print(st, shape, loc, scale)
    fitted[st] = stats.weibull_min(shape, loc, scale)

fig, axs = plt.subplots(1, 2, sharex=False, sharey=False)
fig.set_size_inches(config.FIG_SIZE)
fig.set_dpi(config.FIG_DPI)
cont = 0
for st in sts:
    data = flows[st]
    nn = len(data)
    # Sort
    data.sort()
    pp = np.zeros(nn)
    for i in range(0,len(data)):
        pp[i] = (i + 1) / (nn + 1)
    x = np.linspace(data.min(), data.max(), 10000)
    stats.probplot(data, dist=fitted[st], plot=axs[0])
    axs[1].plot(x, fitted[st].cdf(x), label=st, c=colors[cont])
    axs[1].scatter(data, pp, label="{} ({} datos)".format(st,len(data)), c=colors[cont])
    tr15 = fitted[st].ppf(1/15.)
    tr10 = fitted[st].ppf(0.1)
    tr5 = fitted[st].ppf(0.2)
    strp = "{}: 7q10 = {}, 7q5 = {}".format(st, tr10, tr5)
    print(strp)
    cont = cont + 1

for i in range(0, len(sts)):
    axs[0].get_lines()[i*2].set_marker('+')
    axs[0].get_lines()[i*2].set_markeredgecolor(colors[i])
    axs[0].get_lines()[i*2].set_markerfacecolor(colors[i])
    axs[0].get_lines()[i*2].set_markersize(3.0)
    axs[0].get_lines()[i*2].set_color(colors[i])
    axs[0].get_lines()[i*2+1].set_linewidth(2.0)
    axs[0].get_lines()[i*2+1].set_color(colors[i])

axs[1].set_xlabel('SFlow (L/s/km2)')
axs[1].set_ylabel('Probability')
axs[1].set_yticks(np.arange(0, 1.05, step=0.05))
axs[1].set_title("Weibull fit")

#axs[1].set_xticks(np.arange(0, 2.00, step=0.1))
axs[1].set_xscale('log')

plt.tight_layout()
plt.legend()
plt.savefig(
     os.path.join(
        temp_dir,
        '{}_weibull.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )
    )
)
plt.close(fig)
####################################################

####################################################
plt.hist(data, density=True, alpha=0.5)
x = np.linspace(data.min(), data.max(), 100)
plt.plot(x, fitted.pdf(x))
plt.title("Weibull min fit sflowa7")
plt.xlabel("sflowa7 (L/s/km2)")
stats.probplot(data, dist=fitted, plot=plt.figure().add_subplot(111))
plt.title("Weibull probability plot of slowa7")
plt.show()


x = np.linspace(data.min(), data.max(), 500)
plt.plot(x, fitted.cdf(x))
plt.xscale('log')
plt.yticks(np.arange(0, 1, step=0.05))
plt.show()

# Basic general plot
x = np.linspace(data.min(), data.max(), 500)
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
fig.set_size_inches(config.FIG_SIZE)
fig.set_dpi(config.FIG_DPI)
axs.plot(x, fitted.cdf(x))
axs.set_xscale('log')
axs.set_yticks(np.arange(0, 1, step=0.05))
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_weibull.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
####################################################

####################################################
# Climatic 
cnames = ['Las_Brujas']
for n in range(0,len(cnames)):
    st = cnames[n]
    dcs[st] = read_excel_climatic(
        os.path.join(config.DATA_PATH, config.CLIMATIC_STAT[st][-1])
    )
    # Dont do ir ... there are other data
    # dcs[st].drop(dcs[st][dcs[st]['rain'] <= 0].index, inplace=True)
####################################################

####################################################
# Basic general plot
fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
fig.set_size_inches(config.FIG_SIZE)
fig.set_dpi(config.FIG_DPI)
# To plot
pnames = ['Florida', 'Santa_Lucia', 'Paso_Pache', 'Picada_de_Varela']
for n in range(0,len(pnames)):
    st = pnames[n]
    plots[st] = plot_flow(dfs[st], st, ax=axs[n])
# hide
for ax in axs:
    ax.label_outer()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_flows.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
####################################################

####################################################
# Specific flow plot
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
fig.set_size_inches(config.FIG_SIZE)
fig.set_dpi(config.FIG_DPI)
# To plot
pnames = ['Florida', 'Santa_Lucia', 'Paso_Pache', 'Picada_de_Varela']
for n in range(0,len(pnames)):
    st = pnames[n]
    plot = dfs[st].plot(x="dt", y=["sflow"], label=[st], grid=True, ax=axs)
plot.xaxis.set_label_text('Date')
plot.yaxis.set_label_text('SFlow (L/s/km2)')
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_sflows.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
####################################################

####################################################
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)
# To plot
pnames = ['Florida', 'Santa_Lucia', 'Paso_Pache', 'Picada_de_Varela']
for n in range(0,len(pnames)):
    st = pnames[n]
    plot = dfq[st].plot(x="p", y=["sflow"], label=[st], grid=True, ax=axs, logy=True)
plot.xaxis.set_label_text('Probability (%)')
plot.yaxis.set_label_text('SFlow (L/s/km2)')

axs.xaxis.set_major_locator(MultipleLocator(10))
axs.xaxis.set_minor_locator(MultipleLocator(1))
axs.set_xlim([0,100])

plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_pcurves.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
####################################################

####################################################
# WD data
dfwq = pd.read_csv("data/wq_santa_lucia.csv")
dfwq['dt'] = pd.to_datetime(dfwq['Fecha'])
dfwq['da'] = dfwq['dt'].dt.date

# Sorted unique names
snames = dfwq['Estación'].unique().sort()

# New data
# Field
dfwq['TEMP'] = dfwq['Temperatura (ºC)']
dfwq['PH'] = dfwq['Potencial de hidrogeno (pH) (sin unid)']
dfwq['OD'] = dfwq['Oxigeno disuelto (mg/L)']
dfwq['CE'] = dfwq['Conductividad (µS/cm)']
dfwq['TURB'] = dfwq['Turbidez (NTU)']

# Lab
dfwq['SSSSTT'] = dfwq['Sólidos suspendidos totales (mg/L)']
dfwq['SSTT'] = dfwq['Sólidos totales (mg/L)']
dfwq['CF'] = pd.to_numeric(
    dfwq['Coliformes Termotolerantes (Fecales) (Membrana Filtrante) (ufc/100ml)'],
    errors='coerce'
)

dfwq['PT'] = pd.to_numeric(dfwq['Fósforo total (µg P/L)'], errors='coerce')/1000
dfwq['PO4'] = pd.to_numeric(dfwq['Fosfato (ortofosfato) (µg PO4-P/L)'], errors='coerce')/1000

dfwq['NT'] = dfwq['Nitrógeno total (mg N/L)']
dfwq['NO3'] = pd.to_numeric(dfwq['Nitrato (mg NO3-N/L)'], errors='coerce')
dfwq['NO2'] = pd.to_numeric(dfwq['Ion Nitrito (mg NO2-N/L)'], errors='coerce')
dfwq['NH4'] = pd.to_numeric(dfwq['Nitrógeno Amoniacal (amonio) (mg NH4-N/L)'], errors='coerce')
# Check this calcs
dfwq['NH3NI'] = dfwq['NH4'] / (1 + 10**-dfwq['PH']/10**-(0.09018+2729.92/(273.15 + dfwq['TEMP']))) 
# Clorofila A
dfwq['CHLA'] = pd.to_numeric(dfwq['Clorofila_a_(lab_µg) (µg/L)'], errors='coerce')/1000
# 
dfwq['NT_D_PT'] = dfwq['NT']/dfwq['PT']
dfwq['NO3_D_PO4'] = dfwq['NO3']/dfwq['PO4']

###############
# RSJ
lst = ["SJ01", "SJ02", "SJ03", "SJ04", "SJ05", "SJ06"]
flow_st = 'Picada_de_Varela'
dfwqRSJ = dfwq.query("Estación in @lst")
# Find min and max
dtmin = dfwqRSJ['dt'].min()
dtmax = dfwqRSJ['dt'].max()

dfwqRSJ = pd.merge(dfwqRSJ, dfs['Picada_de_Varela'], how="left", on="da")
# Change name on merge
dfwqRSJ.columns = dfwqRSJ.columns.str.replace('dt_x', 'dt')
# get uniques
fdates = dfwq['da'].unique()
fdlist = list(fdates)

# Filter flows
temp = dfs[flow_st]
fflows = temp[temp['dt'].ge(dtmin) & temp['dt'].le(dtmax)]
# fflows = temp.query("da in @fdlist")
# fflows = fflows.dropna()

# Filter rain
ttemp = dcs['Las_Brujas']
rrains = ttemp[ttemp['dt'].ge(dtmin) & ttemp['dt'].le(dtmax) & ttemp['rain'].gt(0)]
# Order
rrains = rrains.sort_values(by="dt")

# Reindex
dfwqRSJ.set_index('dt', inplace=True)

###
# Field
fig, axs = plt.subplots(5, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)

# rrains.plot.scatter(x = 'dt', y = 'rain', title='Rain (Las Brujas)', ylabel='Rain (mm)', legend=True, logy=True, ax=axs[0])
# rrains.plot(x = 'dt', y = 'rain', title='Rain', kind='bar', legend=True, logy=False, ax=axs[0])

fflows.plot(x = 'dt', y = 'flow', title='Flow (Picada de Varela, m3/s)', legend=True, grid=True, logy=True, ax=axs[0])
# fflows.plot(x = 'dt', y = 'flowa30', title='7 day flow (Picada de Varela, m3/s)', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSJ.plot(x = 'dt_y', y = 'flow', title='Flow (Picada de Varela, m3/s)', marker='o', legend=True, grid=True, logy=True, ax=axs[0])

dfwqRSJ.groupby('Estación')['TEMP'].plot(title='Temperature (C)', legend=True, grid=True, logy=False, ax=axs[1])
dfwqRSJ.groupby('Estación')['PH'].plot(title='PH', legend=True, grid=True, logy=False, ax=axs[2])
dfwqRSJ.groupby('Estación')['CE'].plot(title='CE (uS/cm)', legend=True, grid=True, logy=True, ax=axs[3])
dfwqRSJ.groupby('Estación')['OD'].plot(title='OD (mg/L)', legend=True, grid=True, logy=False, ax=axs[4])

axs[4].set_xlim(dtmin, dtmax)
xlims = axs[4].get_xlim()
axs[4].hlines(y = 5.0, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')

axs[4].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[4].xaxis.set_minor_locator(mdates.MonthLocator())
# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSJ_FIELD.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)


###
# Nitrogen
fig, axs = plt.subplots(6, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)

fflows.plot(x = 'dt', y = 'flow', title='Flow (Paso Pache, m3/s)', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSJ.plot(x = 'dt_y', y = 'flow', title='Flow (Paso Pache, m3/s)', marker='o', legend=True, grid=True, logy=True, ax=axs[0])

dfwqRSJ.groupby('Estación')['TEMP'].plot(title='Temperatura (C)', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSJ.groupby('Estación')['PH'].plot(title='pH (-)', legend=True, grid=True, logy=True, ax=axs[2])

dfwqRSJ.groupby('Estación')['NH3NI'].plot(title='Amoníaco libre (mgN/L)', legend=True, grid=True, logy=True, ax=axs[3])
dfwqRSJ.groupby('Estación')['NH4'].plot(title='Amonio (mgN/L)', legend=True, grid=True, logy=True, ax=axs[4])
dfwqRSJ.groupby('Estación')['NO3'].plot(title='Nitratos (mgN/L)', legend=True, grid=True, logy=True, ax=axs[5])

axs[5].set_xlim(dtmin, dtmax)
xlims = axs[5].get_xlim()
axs[3].hlines(y = 0.02, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')
axs[5].hlines(y = 10.00, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')

axs[5].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[5].xaxis.set_minor_locator(mdates.MonthLocator())

# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSJ_NITROGEN.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

###
# NT, PT, NT/PT
fig, axs = plt.subplots(5, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)

rrains.plot.scatter(x = 'dt', y = 'rain', title='Rain (Las Brujas)', ylabel='Rain (mm)', legend=True, logy=True, ax=axs[0])
# rrains.plot(x = 'dt', y = 'rain', title='Rain', kind='bar', legend=True, logy=False, ax=axs[0])

fflows.plot(x = 'dt', y = 'flow', title='Flow (Picada de Varela, m3/s)', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSJ.plot(x = 'dt_y', y = 'flow', title='Flow (Picada de Varela, m3/s)', marker='o', legend=True, grid=True, logy=True, ax=axs[1])

dfwqRSJ.groupby('Estación')['NT'].plot(title='NT', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSJ.groupby('Estación')['PT'].plot(title='PT', legend=True, grid=True, logy=True, ax=axs[3])
dfwqRSJ.groupby('Estación')['NT_D_PT'].plot(title='NT/PT', legend=True, grid=True, logy=True, ax=axs[4])

axs[4].set_xlim(dtmin, dtmax)
xlims = axs[4].get_xlim()
axs[4].hlines(y = 7.2, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')

axs[4].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[4].xaxis.set_minor_locator(mdates.MonthLocator())
# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSJ_NP.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

###
# NT, PT, Turbiedad
fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)

fflows.plot(x = 'dt', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSJ.plot(x = 'dt_y', y = 'flow', title='Flow (Picada de Varela)', marker='o', legend=True, grid=True, logy=True, ax=axs[0])

dfwqRSJ.groupby('Estación')['NT'].plot(title='NT', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSJ.groupby('Estación')['PT'].plot(title='PT', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSJ.groupby('Estación')['Turbidez (NTU)'].plot(title='Turbidez (NTU)', legend=True, grid=True, logy=True, ax=axs[3])

# xlims = axs[3].get_xlim()
# axs[3].hlines(y = 7.2, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')

axs[3].set_xlim(dtmin, dtmax)
axs[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[3].xaxis.set_minor_locator(mdates.MonthLocator())
# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSJ_NPT.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###



###
# NO3, PO4, NO3/PO4
fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)
fflows.plot(x = 'dt', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSJ.plot(x = 'dt_y', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSJ.groupby('Estación')['NO3'].plot(title='NO3', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSJ.groupby('Estación')['PO4'].plot(title='PO4', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSJ.groupby('Estación')['NO3_D_PO4'].plot(title='NO3/PO4', legend=True, grid=True, logy=True, ax=axs[3])

axs[3].set_xlim(dtmin, dtmax)
axs[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[3].xaxis.set_minor_locator(mdates.MonthLocator())
# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSJ_NO3PO4.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

###
# FLOW -> NT PT PO4 NO3
params = ['NT', 'PT', 'PO4', 'NO3']
pst = [["SJ01", "SJ02", "SJ03"], ["SJ04", "SJ05", "SJ06"]]
for pn in params:
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.set_size_inches(config.FIG_XXLSIZE)
    fig.set_dpi(config.FIG_DPI)
    for row in range(0, len(pst)):
        for col in range(0, len(pst[0])):
            st = pst[row][col]
            dftemp = dfwqRSJ.query("Estación == @st")
            if not dftemp.empty:
                dftemp.plot.scatter(
                    x='flow', y=pn, title=st,
                    ax=axs[row, col], legend=True, grid=True, logx=True, logy=False
                )
                axs[row, col].set_xlim(dtmin, dtmax)
    # plt.show()
    plt.tight_layout()
    plt.savefig(
         os.path.join(
             temp_dir, 
            '{}_RSJ_{}F.pdf'.format(
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                pn
            )     
        )
    )
    plt.close(fig)
# FIN RSJ
###############

###############
# SL
lst = ["SL01", "SL02", "SL03", "SL04", "SL05", "SL06"]
flow_st = 'Paso_Pache'
dfwqRSL= dfwq.query("Estación in @lst")
# Find min and max
dtmin = dfwqRSL['dt'].min()
dtmax = dfwqRSL['dt'].max()

dfwqRSL= pd.merge(dfwqRSL, dfs[flow_st], how="left", on="da")
# Change name on merge
dfwqRSL.columns = dfwqRSL.columns.str.replace('dt_x', 'dt')
# get uniques
fdates = dfwq['da'].unique()
fdlist = list(fdates)

# Filter flows
temp = dfs[flow_st]
fflows = temp[temp['dt'].ge(dtmin) & temp['dt'].le(dtmax)]
# fflows = temp.query("da in @fdlist")

# Reindex
dfwqRSL.set_index('dt', inplace=True)

###
# Field
fig, axs = plt.subplots(5, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)

# rrains.plot.scatter(x = 'dt', y = 'rain', title='Rain (Las Brujas)', ylabel='Rain (mm)', legend=True, logy=True, ax=axs[0])
# rrains.plot(x = 'dt', y = 'rain', title='Rain', kind='bar', legend=True, logy=False, ax=axs[0])

fflows.plot(x = 'dt', y = 'flow', title='Flow (Picada de Varela, m3/s)', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSL.plot(x = 'dt_y', y = 'flow', title='Flow (Picada de Varela, m3/s)', marker='o', legend=True, grid=True, logy=True, ax=axs[0])

dfwqRSL.groupby('Estación')['TEMP'].plot(title='Temperature (C)', legend=True, grid=True, logy=False, ax=axs[1])
dfwqRSL.groupby('Estación')['PH'].plot(title='PH', legend=True, grid=True, logy=False, ax=axs[2])
dfwqRSL.groupby('Estación')['CE'].plot(title='CE (uS/cm)', legend=True, grid=True, logy=True, ax=axs[3])
dfwqRSL.groupby('Estación')['OD'].plot(title='OD (mg/L)', legend=True, grid=True, logy=False, ax=axs[4])

axs[4].set_xlim(dtmin, dtmax)
xlims = axs[4].get_xlim()
axs[4].hlines(y = 5.0, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')

axs[4].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[4].xaxis.set_minor_locator(mdates.MonthLocator())
# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSL_FIELD.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

###
# Nitrogen
fig, axs = plt.subplots(6, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)

fflows.plot(x = 'dt', y = 'flow', title='Flow (Paso Pache, m3/s)', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSL.plot(x = 'dt_y', y = 'flow', title='Flow (Paso Pache, m3/s)', marker='o', legend=True, grid=True, logy=True, ax=axs[0])

dfwqRSL.groupby('Estación')['TEMP'].plot(title='Temperatura (C)', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSL.groupby('Estación')['PH'].plot(title='pH (-)', legend=True, grid=True, logy=True, ax=axs[2])

dfwqRSL.groupby('Estación')['NH3NI'].plot(title='Amoníaco libre (mgN/L)', legend=True, grid=True, logy=True, ax=axs[3])
dfwqRSL.groupby('Estación')['NH4'].plot(title='Amonio (mgN/L)', legend=True, grid=True, logy=True, ax=axs[4])
dfwqRSL.groupby('Estación')['NO3'].plot(title='Nitratos (mgN/L)', legend=True, grid=True, logy=True, ax=axs[5])

axs[5].set_xlim(dtmin, dtmax)
xlims = axs[5].get_xlim()
axs[3].hlines(y = 0.02, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')
axs[5].hlines(y = 10.00, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')

axs[5].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[5].xaxis.set_minor_locator(mdates.MonthLocator())

# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSL_NITROGEN.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

###
# NT, PT, NT/PT
fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)
fflows.plot(x = 'dt', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSL.plot(x = 'dt_y', y = 'flow', title='Flow',  marker="o", legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSL.groupby('Estación')['NT'].plot(title='NT', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSL.groupby('Estación')['PT'].plot(title='PT', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSL.groupby('Estación')['NT_D_PT'].plot(title='NT/PT', legend=True, grid=True, logy=True, ax=axs[3])
xlims = axs[3].get_xlim()
axs[3].hlines(y = 7.2, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')

axs[3].set_xlim(dtmin, dtmax)
axs[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[3].xaxis.set_minor_locator(mdates.MonthLocator())
# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSL_NP.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

###
# NT, PT, Turbiedad
fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)

fflows.plot(x = 'dt', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSL.plot(x = 'dt_y', y = 'flow', title='Flow (Picada de Varela)', marker='o', legend=True, grid=True, logy=True, ax=axs[0])

dfwqRSL.groupby('Estación')['NT'].plot(title='NT', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSL.groupby('Estación')['PT'].plot(title='PT', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSL.groupby('Estación')['Turbidez (NTU)'].plot(title='Turbidez (NTU)', legend=True, grid=True, logy=True, ax=axs[3])

# xlims = axs[3].get_xlim()
# axs[3].hlines(y = 7.2, xmin=xlims[0], xmax=xlims[1], color = 'green', linestyle = 'dashed')

axs[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[3].xaxis.set_minor_locator(mdates.MonthLocator())
# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSL_NPT.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

###
# NO3, PO4, NO3/PO4
fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)
fflows.plot(x = 'dt', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSL.plot(x = 'dt_y', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSL.groupby('Estación')['NO3'].plot(title='NO3', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSL.groupby('Estación')['PO4'].plot(title='PO4', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSL.groupby('Estación')['NO3_D_PO4'].plot(title='NO3/PO4', legend=True, grid=True, logy=True, ax=axs[3])

axs[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[3].xaxis.set_minor_locator(mdates.MonthLocator())
# plt.show()
plt.tight_layout()
plt.savefig(
     os.path.join(
         temp_dir, 
        '{}_RSL_NO3PO4.pdf'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        )     
    )
)
plt.close(fig)
###

###
# FLOW -> NT PT PO4 NO3
params = ['NT', 'PT', 'PO4', 'NO3']
pst = [["SL01", "SL02", "SL03"], ["SL04", "SL05", "SL06"]]
for pn in params:
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.set_size_inches(config.FIG_XXLSIZE)
    fig.set_dpi(config.FIG_DPI)
    for row in range(0, len(pst)):
        for col in range(0, len(pst[0])):
            st = pst[row][col]
            dftemp = dfwqRSL.query("Estación == @st")
            if not dftemp.empty:
                dftemp.plot.scatter(
                    x='flow', y=pn, title=st,
                    ax=axs[row, col], legend=True, grid=True, logx=True, logy=False
                )
    # plt.show()
    plt.tight_layout()
    plt.savefig(
         os.path.join(
             temp_dir, 
            '{}_RSL_{}F.pdf'.format(
                datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                pn
            )     
        )
    )
    plt.close(fig)
# FIN RSL
###############



