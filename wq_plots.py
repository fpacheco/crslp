from __future__ import print_function
import pandas as pd
# from pandas_ods_reader import read_ods

import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates

import datetime
import random
import os
import tempfile
import config


# PLOT STYLES
plt.style.use('ggplot')
#plt.style.use('fivethirtyeight')
#plt.style.use('seaborn-dark')
#plt.style.use('bmh')

# Get temporary directory
temp_dir = tempfile.gettempdir()


def read_excel_climatic(file_path):
    df = pd.read_excel (file_path)
    df['dt'] = pd.to_datetime(df['Fecha'], dayfirst=True)
    df['da'] = df['dt'].dt.date
    df['rain'] = df['Precipitación Acumulada mm']
    # Drop not necessary columns
    return df


def read_excel_flow(file_path, warea):
    df = pd.read_excel (file_path)
    df['dt'] = pd.to_datetime(df['Fecha'])
    df['da'] = df['dt'].dt.date
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
# Quantiles
dfq = dict()
tl = range(0,101)
ql = [x / 100 for x in tl]
qnames = ['Florida', 'Santa_Lucia', 'Paso_Pache', 'Picada_de_Varela']
for n in range(0,len(qnames)):
    st = qnames[n]
    dfq[st] = dfs[st].quantile(ql)
    dfq[st] = dfq[st].rename_axis('q').reset_index()
    dfq[st]['p'] = (1 - dfq[st]['q'])*100

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
dfwq['NT'] = dfwq['Nitrógeno total (mg N/L)']
dfwq['PT'] = pd.to_numeric(dfwq['Fósforo total (µg P/L)'], errors='coerce')/1000
dfwq['PO4'] = pd.to_numeric(dfwq['Fosfato (ortofosfato) (µg PO4-P/L)'], errors='coerce')/1000
dfwq['NO3'] = pd.to_numeric(dfwq['Nitrato (mg NO3-N/L)'], errors='coerce')
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

# Filter rain
ttemp = dcs['Las_Brujas']
rrains = ttemp[ttemp['dt'].ge(dtmin) & ttemp['dt'].le(dtmax) & ttemp['rain'].gt(0)]

# Reindex
dfwqRSJ.set_index('dt', inplace=True)

###
# NT, PT, NT/PT
fig, axs = plt.subplots(5, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)

rrains.plot.scatter(x = 'dt', y = 'rain', title='Rain (Las Brujas)', ylabel='Rain (mm)', legend=True, logy=False, ax=axs[0])
# rrains.plot(x = 'dt', y = 'rain', title='Rain', kind='bar', legend=True, logy=False, ax=axs[0])

fflows.plot(x = 'dt', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSJ.plot(x = 'dt_y', y = 'flow', title='Flow', marker='o', legend=True, grid=True, logy=True, ax=axs[1])

dfwqRSJ.groupby('Estación')['NT'].plot(title='NT', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSJ.groupby('Estación')['PT'].plot(title='PT', legend=True, grid=True, logy=True, ax=axs[3])
dfwqRSJ.groupby('Estación')['NT_D_PT'].plot(title='NT/PT', legend=True, grid=True, logy=True, ax=axs[4])

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
# NO3, PO4, NO3/PO4
fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)
fflows.plot(x = 'dt', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSJ.plot(x = 'dt_y', y = 'flow', title='Flow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSJ.groupby('Estación')['NO3'].plot(title='NO3', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSJ.groupby('Estación')['PO4'].plot(title='PO4', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSJ.groupby('Estación')['NO3_D_PO4'].plot(title='NO3/PO4', legend=True, grid=True, logy=True, ax=axs[3])

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



