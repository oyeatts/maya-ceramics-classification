import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df_pmcdp = pd.read_excel("postclassic_maya_ceramic_database_pottery.xlsx", sheet_name="Sheet1")
list_ = []
for i in range(198,1505):
    list_.append(i)
df_pmcdp = df_pmcdp.drop(list_, axis = 0)
df_locs_paxcaman = df_pmcdp['Ceramic Type'] == 'Paxcamán Red'
df_locs_ixpop = df_pmcdp['Ceramic Type'] == 'Ixpop Polychrome'
df_locs_saca = df_pmcdp['Ceramic Type'] == 'Sacá Polychrome'
df_locs_fulano = df_pmcdp['Ceramic Type'] == 'Fulano Black'
df_locs_augustine = df_pmcdp['Ceramic Type'] == 'Augustine Red'
df_locs_trapeche = df_pmcdp['Ceramic Type'] == 'Trapeche Pink'
df_locs_mul = df_pmcdp['Ceramic Type'] == 'Mul Polychrome'

df_locs = df_locs_augustine | df_locs_fulano | df_locs_ixpop | df_locs_mul | df_locs_paxcaman | df_locs_saca | df_locs_trapeche
df_locs = np.asarray(df_locs)

tipu_df = df_pmcdp.iloc[df_locs]
tipu_df = tipu_df.drop(columns = ["Ware", "Country", "Province", "Site_Name", "Ceramic Type"])
tipu_df["Investigator"] = "tipu"

df_northern_lowlands = pd.read_excel("Smyth_1998_pottery.xlsx")
df_locs_sayil = df_northern_lowlands['Site_Name'] == 'Sayil'

df_locs_sayil = np.asarray(df_locs_sayil)
sayil_locs = df_northern_lowlands.iloc[df_locs_sayil]
sayil_df = sayil_locs.drop(columns=["State", "Subregion", "Material", "Site_Name", "Type", 'Comp_Group'])
sayil_df["Investigator"] = 'sayil'

df_belize_valley = pd.read_excel("Douglas_etal_2021.xlsx")
df_locs_cahal_pech = df_belize_valley['Site Name'] == 'Cahal Pech'

df_locs_cahal_pech = np.asarray(df_locs_cahal_pech)
cahal_pech_locs = df_belize_valley.iloc[df_locs_cahal_pech]
cahal_pech_df = cahal_pech_locs.drop(columns=['Alternate ID', 'Analytical History', 'Investigator Name', 'Institution', 'Source/Excavator/Museum', 'Region', 'Country', 
                                              'State/Province', 'County/District', 'Local Subregion', 'Site Name', 'Site Number', 'Latitude',
                                              'Longitude', 'UTM Zone', 'Northing', 'Easting', 'Material', 'Ware', 'Ceramic Type', 'Vessel Form', 'Exterior Decoration',
                                              'Interior Decoration', 'Paste Color', 'Major Temper', 'Minor Temper', 'Culture', 'Context', 'Provenience', 'Period', 'Date', 'Picture',
                                              'Comments', 'Notes'])
cahal_pech_df["Investigator"] = 'cahal pech'


df_kaxob_formative = pd.read_excel("Angelini_1998.xlsx")
df_locs_kaxob = df_kaxob_formative['material'] == 'POTTERY'

df_locs_kaxob = np.asarray(df_locs_kaxob)
kaxob_formative_locs = df_kaxob_formative.iloc[df_locs_kaxob]
kaxob_formative_df = kaxob_formative_locs.drop(columns=['sour+A1+B1:B1:O1', 'ang_id', 'year', 'material', 'inclusions',	
                                                        'p_consiste', 'p_inclusio', 'rock', 'fs', 'op', 'zone', 'square', 
                                                        'complex', 'certype', 'context', 'shape', 'chem1'])
kaxob_formative_df['Investigator'] = 'kaxob_formative'
kaxob_labels = np.asarray(kaxob_formative_df.columns)
new_labels = []
for i in range(len(kaxob_labels)):
    l = kaxob_labels[i]
    new = l.capitalize()
    new_labels.append(new)
    
kaxob_labels = new_labels
kaxob_labels[0] = kaxob_labels[0].upper()
kaxob_formative_df = np.array(kaxob_formative_df)
kaxob_formative_df = pd.DataFrame(kaxob_formative_df, columns=kaxob_labels)

df_paris_etal = pd.read_excel("Paris_etal_2021.xlsx")
df_locs_moxviquil = df_paris_etal['Site_Name'] == 'Moxviquil'

df_locs_moxviquil = np.asarray(df_locs_moxviquil)
moxviquil_locs = df_paris_etal.iloc[df_locs_moxviquil]
moxviquil_df = moxviquil_locs.drop(columns=['Chemgrp', 'Country', 'Province', 'Site_Name', 'Ceramic Type'])
#moxviquil_df['Investigator'] = 'moxviquil'

df_locs_chichen = df_paris_etal['Site_Name'] == 'Chichen Itza'

df_locs_chichen = np.asarray(df_locs_chichen)
chichen_locs = df_paris_etal.iloc[df_locs_chichen]
chichen_df = chichen_locs.drop(columns=['Chemgrp', 'Country', 'Province', 'Site_Name', 'Ceramic Type'])

#moxviquil_df['Investigator'] = 'moxviquil'
df_locs_tierra = df_paris_etal['Site_Name'] == 'Tierra Colorada'

df_locs_tierra = np.asarray(df_locs_tierra)
tierra_locs = df_paris_etal.iloc[df_locs_tierra]
tierra_locs = tierra_locs.drop(40, axis = 0)
tierra_df = tierra_locs.drop(columns=['Chemgrp', 'Country', 'Province', 'Site_Name', 'Ceramic Type'])

df_locs_calakmul = df_paris_etal['Site_Name'] == 'Calakmul'
df_locs_calakmul = np.asarray(df_locs_calakmul)
calakmul_locs = df_paris_etal.iloc[df_locs_calakmul]
calakmul_df = calakmul_locs.drop(columns=['Chemgrp', 'Country', 'Province', 'Site_Name', 'Ceramic Type'])

df_locs_altar = df_paris_etal['Site_Name'] == 'Altar de Sacrificios'
df_locs_altar = np.asarray(df_locs_altar)
altar_locs = df_paris_etal.iloc[df_locs_altar]
altar_df = altar_locs.drop(columns=['Chemgrp', 'Country', 'Province', 'Site_Name', 'Ceramic Type'])