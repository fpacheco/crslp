# Data directory
DATA_PATH = "data"

# Logo
LOGO_PATH = "data/logo_200x93.png"

# Water Quality Stations
# {'ID': [COTA_CERO_SOBRE_CER_WHARTON, AREA_KM_2, DISTANCE, RIVER NAME], ...}
WQ_STAT = {
    'SLC01': [-999.99, -999.99, -999.99, 'Río Santa Lucía Chico'],
    'SLC02': [-999.99, -999.99, -999.99, 'Río Santa Lucía Chico'],
    'SLC03': [-999.99, -999.99, -999.99, 'Río Santa Lucía Chico'],
    'SLC04': [-999.99, -999.99, -999.99, 'Río Santa Lucía Chico'],
    'SLC05': [-999.99, -999.99, -999.99, 'Río Santa Lucía Chico'],
    'SL01': [-999.99, -999.99, -999.99, 'Río Santa Lucía'],
    'SL02': [-999.99, -999.99, -999.99, 'Río Santa Lucía'],
    'SL03': [-999.99, -999.99, -999.99, 'Río Santa Lucía'],
    'SL04': [-999.99, -999.99, -999.99, 'Río Santa Lucía'],
    'SL05': [-999.99, -999.99, -999.99, 'Río Santa Lucía'],
    'SJ01': [-999.99, -999.99, -999.99, 'Río San José'],
    'SJ02': [-999.99, -999.99, -999.99, 'Río San José'],
    'SJ03': [-999.99, -999.99, -999.99, 'Río San José'],
    'SJ04': [-999.99, -999.99, -999.99, 'Río San José'],
    'SJ05': [-999.99, -999.99, -999.99, 'Río San José'],
}

# Flow Stations
# {'ID': [COTA_CERO_SOBRE_CER_WHARTON, AREA_KM_2, DISTANCE, NAME, FILENAME], ...}
FLOW_STAT = {
    'Florida': [44.34, 1747.60, -999.99, 'Río Santa Lucía Chico', 'flow_Florida_Puente_R.5.xls'],
    'Paso_Pache': [14.76, 4916.00, -999.99, 'Río Santa Lucía', 'flow_Paso_Pache_R.5_Nueva.xls'],
    'Santa_Lucia': [2.27, 8374.00, -999.99, 'Río Santa Lucía', 'flow_Santa_Lucia_R-11.xls'],
    'Picada_de_Varela': [19.79, 2346.00, -999.99, 'Río San José', 'flow_Picada_de_Varela.xls'],
}

# Plot text
PLOT_TEXT = [
    ('Descarga de Conaprole', 48408),
    ('Arroyo Castellanos', 38574),
    ('Arroyo Gregorio', 37932),
    ('Arroyo Sauce', 32365),
    ('Arroyo Valdes', 27780),
    ('Arroyo Cagancha', 25763),
    ('Arroyo Flores', 12527),
]

# Sheet name for flow data
FLOW_SHEET = 'Sheet0'

# Figure size
FIG_SIZE = (12, 23.38) # A3H in inches
FIG_DPI = 300
