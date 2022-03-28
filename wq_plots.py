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
import config


# PLOT STYLES
plt.style.use('ggplot')
#plt.style.use('fivethirtyeight')
#plt.style.use('seaborn-dark')
#plt.style.use('bmh')


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
# Plots
plots = dict()

####################################################
# To read 
rnames = ['Florida', 'Paso_Pache', 'Picada_de_Varela', 'Santa_Lucia']
for n in range(0,len(rnames)):
    st = rnames[n]
    dfs[st] = read_excel_flow(
        os.path.join(config.DATA_PATH, config.FLOW_STAT[st][-1]),
        config.FLOW_STAT[st][1]
    )
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
plt.savefig('/tmp/{}_flows.pdf'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
    plot = dfs[st].plot(x="dt", y=["sflow"], grid=True, ax=axs)
plot.xaxis.set_label_text('Date')
plot.yaxis.set_label_text('SFlow (L/s/km2)')
plt.tight_layout()
plt.savefig('/tmp/{}_sflows.pdf'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
plt.savefig('/tmp/{}_pcurves.pdf'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
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
dfwq['NT_D_PT'] = dfwq['NT']/dfwq['PT']

# RSJ
lst = ["SJ01", "SJ02", "SJ03", "SJ04", "SJ05", "SJ06"]
dfwqRSJ = dfwq.query("Estación in @lst")
dfwqRSJ = pd.merge(dfwqRSJ, dfs['Picada_de_Varela'], how="left", on="da")
fdates = dfwq['da'].unique()
fdlist = list(fdates)

# Find min and max
dtmin = dfwqRSJ['dt'].min()
dtmax = dfwqRSJ['dt'].max()
# Filter flows
temp = dfs['Picada_de_Varela']
fflows = temp[temp['dt'].ge(dtmin) & temp['dt'].le(dtmax)] 
# fflows = temp.query("da in @fdlist")

# Reindex
dfwqRSJ.set_index('dt', inplace=True)

fig, axs = plt.subplots(4, 1, sharex=True, sharey=False)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)
fflows.plot(x = 'dt', y = 'sflow', title='SFlow', legend=True, grid=True, logy=True, ax=axs[0])
dfwqRSJ.groupby('Estación')['NT'].plot(title='NT', legend=True, grid=True, logy=True, ax=axs[1])
dfwqRSJ.groupby('Estación')['PT'].plot(title='PT', legend=True, grid=True, logy=True, ax=axs[2])
dfwqRSJ.groupby('Estación')['NT_D_PT'].plot(title='NT/PT', legend=True, grid=True, logy=True, ax=axs[3])

axs[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(2, 4, 6, 8, 10, 12)))
axs[3].xaxis.set_minor_locator(mdates.MonthLocator())
#axs.xaxis.set_major_locator(MultipleLocator(120))
#axs.xaxis.set_minor_locator(MultipleLocator(30))

# plt.show()
plt.tight_layout()
plt.savefig('/tmp/{}_RSJ_NP.pdf'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    )
)
plt.close(fig)

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
fig.set_size_inches(config.FIG_XXLSIZE)
fig.set_dpi(config.FIG_DPI)
pst = ["SJ01", "SJ02", "SJ03", "SJ04", "SJ05", "SJ06"]
for n in range(0,len(pst)):
    st = pst[n]
    dftemp = dfwqRSJ.query("Estación == @st")
    dftemp.plot.scatter(
        x='sflow', y='NT', title = st,
        ax=axs[n], legend=True, grid=True, logy=False
    )
# plt.show()
plt.tight_layout()
plt.savefig('/tmp/{}_RSJ_NTF.pdf'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    )
)
plt.close(fig)












"""
pd.pivot_table(
    dfwqRSJ.reset_index(),
    index='dt', 
    columns='Estación', 
    values='NT_D_PT'
).plot(subplots=True, layout=(2, 3))
"""


def simiq(x, coef=0.9):
    if isinstance(x, str):
        z = x.replace('-', '')
        z = z.replace('*', '')
        z = z.replace(',', '.')
        if z.find('<')==0:
            z = z.replace('<','')
            f = float(z)
            f = f*coef
            return f
    return x


# Logo
LOGO_PATH = "/ingesur/publico_001/Empresa/papeleria y Web/logos/finales/logo_solo_IGS_200x93.png"
im = image.imread(LOGO_PATH)
datafile = cbook.get_sample_data(LOGO_PATH, asfileobj=False)
print('loading %s' % datafile)
im = image.imread(datafile)
#im[:, :, -1] = 0.5  # set the alpha channel

## Tamaños de figuras
FIG_SIZE = (11.69,6) # A4H in inches - texto de la leyenda
FIG_DPI = 300
## Archivo ODS con toda la información
ODS_PATH='/ingesur/publico_001/Trabajos/Conaprole/Villa_Rodriguez/20-PM2/2-Ejecucion/3-Analisis_de_resultados/TODOS_LOS_DATOS_v4.ods'
## Las fechas de las campañas que quiero agregar a los gráficos
# El formato es dd/mm/AAAA, es decir 25/09/2019.
FCAMP=['14/01/2021','20/04/2021']
FFCAMP = list()
FCAMP_STR = ''
for f in FCAMP:
    FFCAMP.append( datetime.datetime.strptime(f, '%d/%m/%Y') )
    FCAMP_STR += FFCAMP[-1].strftime("%Y%m%d") + '_'
# Sheet names
CHEM_DATA_SHEET_NAME="Todos"
ACUM_DATA_SHEET_NAME="Acum"
# Lo leo
df = read_ods(ODS_PATH, CHEM_DATA_SHEET_NAME)
dfacum = read_ods(ODS_PATH, ACUM_DATA_SHEET_NAME)
ACUM_PM2=[
    float(dfacum[ dfacum['ID']=='P1']['Acum']),
    float(dfacum[ dfacum['ID']=='P2']['Acum']),
    float(dfacum[ dfacum['ID']=='P6']['Acum']),
    float(dfacum[ dfacum['ID']=='P8']['Acum']),
]

PLOT_TEXT = [
    ('Descarga de Conaprole',48408),
    ('Arroyo Castellanos',38574),
    ('Arroyo Gregorio',37932),
    ('Arroyo Sauce',32365),
    ('Arroyo Valdes',27780),
    ('Arroyo Cagancha',25763),
    ('Arroyo Flores',12527),
]

#####################################################################
## Conversiones varias
# Fecha
df['dt'] = pd.to_datetime(df['Fecha'])
# Temperatura
df['T']= df['Temp'].replace('-', '')
df['T']= df['T'].replace('s/d', '')
df['T']= df['T'].replace('*', '')
df['T'] = pd.to_numeric(df['T'])
# pH
df['pH']= df['pH'].replace('-', '')
df['pH']= df['pH'].replace('s/d', '')
df['pH']= df['pH'].replace('*', '')
df['pH'] = pd.to_numeric(df['pH'])
# Oxígeno disuelto
df['OD']= df['OD (mg/l)'].replace('-', '')
df['OD']= df['OD'].replace('s/d', '')
df['OD']= df['OD'].replace('*', '')
df['OD'] = pd.to_numeric(df['OD'])
# Conductividad eléctrica
df['CE']= df['CE (uS/cm)'].replace('-', '')
df['CE']= df['CE'].replace('s/d', '')
df['CE']= df['CE'].replace('*', '')
df['CE'] = pd.to_numeric(df['CE'])

##
# DQO
df['DQO']= df['DQO (mgO2/L)'].apply(simiq)
# NH4
df['NH4']= df['NH4  (mgN/L)'].apply(simiq)
# No3
df['NO3']= df['NO3 (mgN/L)'].apply(simiq)
# PT
df['PT']= df['PT (mg/L)'].apply(simiq)
# CF
df['CF']= df['CF (u.f.c/100ml)'].apply(simiq)


##
#####################################################################

#####################################################################
## Creo una dataframes
mffc = max(FFCAMP)
# Una para cada cada punto relevante
dfP1 = df[ (df['ID']=='P1') & (df['dt']<=mffc) ]
dfP2 = df[ (df['ID']=='P2') & (df['dt']<=mffc) ]
dfP6 = df[ (df['ID']=='P6') & (df['dt']<=mffc) ]
dfP7 = df[ (df['ID']=='P7') & (df['dt']<=mffc) ]
dfP8 = df[ (df['ID']=='P8') & (df['dt']<=mffc) ]
dfP9 = df[ (df['ID']=='P9') & (df['dt']<=mffc) ]

# Solo los datos de PM2
# dfPM2 = df[ (df['ID']=='P1') | (df['ID']=='P2') | (df['ID']=='P6') | (df['ID']=='P7') | (df['ID']=='P8') | (df['ID']=='P9') & (df['dt']<=mffc) ]
dfPM2 = df[ (df['ID']=='P1') | (df['ID']=='P2') | (df['ID']=='P6') | (df['ID']=='P8') ]
# Solo las campañas específicas
# dfPM2L = dfPM2[ dfPM2['dt']==FFCAMP ]
PM2L = list()
for f in FFCAMP:
    PM2L.append( dfPM2[ dfPM2['dt']==f ] )

## Gráficos de parámetros de campo
xmin = 10000
xmax = 60000
xdelta = 5000
PARAMS = {
    'T': ['Temperatura (C)'],
    'pH': ['pH'],
    'OD': ['Oxígeno disuelto (mg/L)',],
    'CE': ['Conductividad Eléctrica (uS/cm)'],
    ##
    'DQO': ['DQO (mgO/L)'],
    'NH4': ['Amonio (mgN/L)',],
    'NO3': ['Nitrato (mgN/L)',],
    'PT': ['Fósforo total (mg/L',],
    'CF': ['Coliformes fecales (UFC/100 mL)',]
}

## Select color an markers for each date
mm = list()
cc = list()
markers = ['^','*','o','P', 'X','D']
colors = ['darkorange','olivedrab','dodgerblue','red','darkviolet','teal']
plabel = list()
m = c = None
cont = 0
for i in range(0, len(PM2L)):
    # Select markers
    mtemp = random.choice(markers)
    while (m==mtemp):
        mtemp = random.choice(markers)
    m = mtemp
    mm.append(m)
    # Select colors
    ctemp = random.choice(colors)
    while (c==ctemp):
        ctemp = random.choice(colors)
    c = ctemp
    cc.append(c)
    # The label
    plabel.append( FCAMP[cont] )
    cont = cont + 1

## Do the plots
for k,v in PARAMS.items():
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    plt.subplots_adjust(left=0.06, bottom=0.1, right=0.99, top=0.93, wspace=0.01, hspace=0.01)
    ax = plt.gca()
    ax2 = ax.twiny()
    # Boxplot
    #bdata = [list(dfP1[k].dropna()),list(dfP2[k].dropna()), list(dfP6[k].dropna()), list(dfP7[k].dropna()), list(dfP8[k].dropna()), list(dfP9[k].dropna())]
    bdata = [list(dfP1[k].dropna()),list(dfP2[k].dropna()), list(dfP6[k].dropna()), list(dfP8[k].dropna())]
    # widths = ancho de la caja
    ax.boxplot(bdata, widths=1000, positions=ACUM_PM2)
    # Lines plots
    cont = 0
    for dfPM2L in PM2L:
        # ldata = [float(dfPM2L[ dfPM2L['ID']=='P1'][k]), float(dfPM2L[ dfPM2L['ID']=='P2'][k]), float(dfPM2L[ dfPM2L['ID']=='P6'][k]), float(dfPM2L[ dfPM2L['ID']=='P7'][k]), float(dfPM2L[ dfPM2L['ID']=='P8'][k]), float(dfPM2L[ dfPM2L['ID']=='P9'][k])]
        ldata = [float(dfPM2L[ dfPM2L['ID']=='P1'][k]), float(dfPM2L[ dfPM2L['ID']=='P2'][k]), float(dfPM2L[ dfPM2L['ID']=='P6'][k]), float(dfPM2L[ dfPM2L['ID']=='P8'][k])]
        ax.plot(ACUM_PM2, ldata, label="{}".format(plabel[cont]) , marker=mm[cont], linestyle=':', color=cc[cont], linewidth=1)
        cont = cont + 1
    #plt.title("Temperatura", y=1)
    #plt.subtitle("Temperatura", y=1)
    ax.set_xlabel("Puntos de monitoreo")
    ax.set_ylabel(v[0])
    # ax.set_xticklabels(['P1', 'P2', 'P6', 'P7', 'P8', 'P9'])
    ax.set_xticklabels(['P1', 'P2', 'P6', 'P8'])
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xlim([xmin,xmax])
    print("k=", k)
    print("v=", v)
    if k == 'CF':
        ax.set_yscale('log')
    # X2 ax2
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(xmin,xmax,xdelta))
    ax2.set_xlabel("Distancia acumulada (m)")
    # Lines
    deltay = (ax.get_ylim()[1]-ax.get_ylim()[0])/100
    for e in PLOT_TEXT:
        ax2.annotate(e[0],
            xy=(e[1], ax.get_ylim()[1]),
            xycoords='data',
            xytext=(-15,-40),
            textcoords='offset points',
            fontsize=10,
            color='#A04000',
            va="top",
            ha="center",
            rotation=90,
            bbox=dict(boxstyle="round", fc="white", ec="#A04000", lw=1),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="angle,angleA=0,angleB=90,rad=10",
                color='#A04000',
                lw=1
            )
        )
    # finals
    ax.grid(False)
    ax.yaxis.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.05), shadow=True, ncol=2)
    ax2.grid(True)
    fig.figimage(im, 3270, 40, zorder=100)
    # plt.show()
    plt.savefig('/tmp/{}{}.png'.format(
            FCAMP_STR,
            k
        )
    )
    plt.close(fig)

## Gráficos de parámetros de laboratorio