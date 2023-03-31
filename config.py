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
    'Santa_Lucia_IC': [2.27, 8374.00, -999.99, 'Río Santa Lucía', 'wq_santa_lucia_IC.csv'],
}

CLIMATIC_STAT = {
    'Las_Brujas': [44.34, 1747.60, -999.99, 'Las Brujas', 'climatic_Las_Brujas.xls']
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
FIG_XXLSIZE = (46, 24) # A3H in inches
FIG_XLSIZE = (23, 12) # A3H in inches
FIG_LSIZE = (12, 6) # A3H in inches
FIG_SIZE = (12, 6) # A3H in inches
FIG_DPI = 300


headers = ['Fecha',
 'Estación',
 'Aceites y grasas (mg/L)',
 'Alacloro (µg/L)',
 'Aldrin (µg/L)',
 'Alfa cipermetrina (µg/L)',
 'AMPA (µg/L)',
 'Atrazina (µg/L)',
 'Atrazina desetil (µg/L)',
 'Atrazina desisopropil (µg/L)',
 'Azoxiestrobina (µg/L)',
 'Clordano Cis (µg/L)',
 'Clordano Trans (µg/L)',
 'Clorofila_a_(lab_µg) (µg/L)',
 'Clorpirifos (µg/L)',
 'Clorpirifos Metil (µg/L)',
 'Coliformes Termotolerantes (Fecales) (Membrana Filtrante) (ufc/100ml)',
 'Color (unidad de color)',
 'Compuestos Halogenados adsorbibles (AOX) (µg/L)',
 'Conductividad (µS/cm)',
 'Cromo VI (mg/L)',
 'DBO5 (mg O2/L)',
 'Demanda química de oxigeno (DQO) (mg O2/L)',
 'Diazinon (µg/L)',
 'Dieldrin (µg/L)',
 'Diuron (µg/L)',
 'Endosulfan alfa (µg/L)',
 'Endosulfan beta (µg/L)',
 'Endosulfan sulfato (µg/L)',
 'Endrin (µg/L)',
 'Escherichia coli (Memabrana Filtrante) (ufc/100ml)',
 'Escherichia coli (Sustrato Definido - Colilert®) (NMP/100mL)',
 'Etil Paration (µg/L)',
 'Etion (µg/L)',
 'Feofitina a (µg/L)',
 'Fipronil (µg/L)',
 'Floración Algal (sin unid)',
 'Fluroxipir meptil (µg/L)',
 'Fosfato (ortofosfato) (µg PO4-P/L)',
 'Fósforo total (µg P/L)',
 'Glifosato (µg/L)',
 'Heptacloro (µg/L)',
 'Heptacloro Epoxido (µg/L)',
 'Hexaclorobenceno (µg/L)',
 'Ion Nitrito (mg NO2-N/L)',
 'Lindano (µg/L)',
 'Malathion (µg/L)',
 'Mercurio total (mg/L)',
 'Metil Paration (µg/L)',
 'Metoxicloro (µg/L)',
 'Mirex (µg/L)',
 'Nitrato (mg NO3-N/L)',
 'Nitrógeno Amoniacal (amonio) (mg NH4-N/L)',
 'Nitrógeno total (mg N/L)',
 "o,p' DDD (µg/L)",
 "o,p' DDE (µg/L)",
 "o,p' DDT (µg/L)",
 'Oxigeno disuelto (mg/L)',
 'Potencial de hidrogeno (pH) (sin unid)',
 "p,p' DDD (µg/L)",
 "p,p' DDE (µg/L)",
 "p,p' DDT (µg/L)",
 'Profundidad_Sitio (m)',
 'Saturación de oxígeno (%)',
 'Simazina (µg/L)',
 'Sólidos suspendidos totales (mg/L)',
 'Sólidos totales (mg/L)',
 'Temperatura (ºC)',
 'Transparencia (cm)',
 'Trifloxiestrobina (µg/L)',
 'Trifluralina (µg/L)',
 'Turbidez (NTU)',
 'dt'
]
