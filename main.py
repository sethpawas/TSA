import streamlit as st 
import pickle
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor

st.set_page_config( 
    layout='wide')
st.title("Predicted HF")
st.sidebar.title('Features')

#----------------------------------------------------------------------------------------------------
first_kpi, second_kpi = st.columns([10000, 1])

def get_input_data(MIN_df, MAX_df):

	f1 = st.sidebar.slider("Release_Dose_Calc_mg_m", MIN_df.loc["Release_Dose_Calc_mg_m"][0], MAX_df.loc["Release_Dose_Calc_mg_m"][0], float((MIN_df.loc["Release_Dose_Calc_mg_m"][0] + MAX_df.loc["Release_Dose_Calc_mg_m"][0])/2), step = 0.01)
	f2 = st.sidebar.slider("HEADBOX_PRES_psig", MIN_df.loc["HEADBOX_PRES_psig"][0], MAX_df.loc["HEADBOX_PRES_psig"][0], float((MIN_df.loc["HEADBOX_PRES_psig"][0] + MAX_df.loc["HEADBOX_PRES_psig"][0])/2), step = 0.01)
	f3 = st.sidebar.slider("Steam_Box_Pressure_PSIG", MIN_df.loc["Steam_Box_Pressure_PSIG"][0], MAX_df.loc["Steam_Box_Pressure_PSIG"][0], float((MIN_df.loc["Steam_Box_Pressure_PSIG"][0] + MAX_df.loc["Steam_Box_Pressure_PSIG"][0])/2), step = 0.01)
	f4 = st.sidebar.slider("PM15_Winder_Speed_FPM", MIN_df.MIN[3], MAX_df.MAX[3], float((MIN_df.MIN[3] + MAX_df.MAX[3])/2), step = 0.01)
	f5 = st.sidebar.slider("Refiner_1_HPD_T", MIN_df.MIN[4], MAX_df.MAX[4], float((MIN_df.MIN[4] + MAX_df.MAX[4])/2), step = 0.01)
	f6 = st.sidebar.slider("UHLEBOX1_VAC_Inhg", MIN_df.MIN[5], MAX_df.MAX[5], float((MIN_df.MIN[5] + MAX_df.MAX[5])/2), step = 0.01)
	f7 = st.sidebar.slider("Hi_Bright_PM15_Ratio", MIN_df.MIN[6], MAX_df.MAX[6], float((MIN_df.MIN[6] + MAX_df.MAX[6])/2), step = 0.01)
	f8 = st.sidebar.slider("EosYMoiPv_pct", MIN_df.MIN[7], MAX_df.MAX[7], float((MIN_df.MIN[7] + MAX_df.MAX[7])/2), step = 0.01)
	f9 = st.sidebar.slider("Coating_Dose_Calc_mg_m", MIN_df.MIN[8], MAX_df.MAX[8], float((MIN_df.MIN[8] + MAX_df.MAX[8])/2), step = 0.01)	
	f10 = st.sidebar.slider("Refiner_3_HP_Control", MIN_df.MIN[9], MAX_df.MAX[9], float((MIN_df.MIN[9] + MAX_df.MAX[9])/2), step = 0.01)
	f11 = st.sidebar.slider("Metso_Rush_Drag_to_DV", MIN_df.MIN[10], MAX_df.MAX[10], float((MIN_df.MIN[10] + MAX_df.MAX[10])/2), step = 0.01)
	f12 = st.sidebar.slider("EUC_PM15_Ratio", MIN_df.MIN[11], MAX_df.MAX[11], float((MIN_df.MIN[11] + MAX_df.MAX[11])/2), step = 0.01)
	f13 = st.sidebar.slider("VirginStock_PM15_Ratio", MIN_df.MIN[12], MAX_df.MAX[12], float((MIN_df.MIN[12] + MAX_df.MAX[12])/2), step = 0.01)
	f14 = st.sidebar.slider("RPW_Pressure_PSI", MIN_df.MIN[13], MAX_df.MAX[13], float((MIN_df.MIN[13] + MAX_df.MAX[13])/2), step = 0.01)
	f15 = st.sidebar.slider("PM_BWT_lbs_3000sqft", MIN_df.MIN[14], MAX_df.MAX[14], float((MIN_df.MIN[14] + MAX_df.MAX[14])/2), step = 0.01)
	f16 = st.sidebar.slider("PM15_Fan_Pump_ORP_pH", MIN_df.MIN[15], MAX_df.MAX[15], float((MIN_df.MIN[15] + MAX_df.MAX[15])/2), step = 0.01)
	f17 = st.sidebar.slider("Refiner_2_HPD_T", MIN_df.MIN[16], MAX_df.MAX[16], float((MIN_df.MIN[16] + MAX_df.MAX[16])/2), step = 0.01)
	f18 = st.sidebar.slider("PM_15_TWS_Flow__ton", MIN_df.MIN[17], MAX_df.MAX[17], float((MIN_df.MIN[17] + MAX_df.MAX[17])/2), step = 0.01)

	input_data = pd.DataFrame.from_dict({'Release_Dose_Calc_mg_m': [f1], 'HEADBOX_PRES_psig': [f2],
		'Steam_Box_Pressure_PSIG': [f3], 'PM15_Winder_Speed_FPM': [f4], 'Refiner_1_HPD_T': [f5],
		'UHLEBOX1_VAC_Inhg': [f6], 'Hi_Bright_PM15_Ratio': [f7], 'EosYMoiPv_pct': [f8],
		'Coating_Dose_Calc_mg_m': [f9], 'Refiner_3_HP_Control': [f10], 'Metso_Rush_Drag_to_DV': [f11],
		'EUC_PM15_Ratio': [f12], 'VirginStock_PM15_Ratio': [f13], 'RPW_Pressure_PSI': [f14], 
		'PM_BWT_lbs_3000sqft': [f15], 'PM15_Fan_Pump_ORP_pH': [f16], 'Refiner_2_HPD_T': [f17], 
		'PM_15_TWS_Flow__ton': [f18]})

	input_data= pd.DataFrame(input_data)
	return input_data

import time									
										
@st.cache								
def load_model(input_data):

	pickle_in = open(r'HF_py_1011.pkl', 'rb') 
	xgb = pickle.load(pickle_in)
	pickle_in.close()

	df_1 = input_data
	pred_value = xgb.predict(df_1)
	pred = pred_value.tolist()
	return pred

def main():
	TSA_data = pd.read_csv(r"TSA_TS7.csv")

	list_24 = ['Release_Dose_Calc_mg_m' ,'HEADBOX_PRES_psig' ,'Steam_Box_Pressure_PSIG' ,'PM15_Winder_Speed_FPM' ,'Refiner_1_HPD_T' ,'UHLEBOX1_VAC_Inhg'
		,'Hi_Bright_PM15_Ratio' ,'EosYMoiPv_pct' ,'Coating_Dose_Calc_mg_m' ,'Refiner_3_HP_Control' ,'Metso_Rush_Drag_to_DV' ,'EUC_PM15_Ratio' 
		,'VirginStock_PM15_Ratio' ,'RPW_Pressure_PSI' ,'PM_BWT_lbs_3000sqft' ,'PM15_Fan_Pump_ORP_pH' ,'Refiner_2_HPD_T' ,'PM_15_TWS_Flow__ton']

	TSA_18_min = TSA_data[list_24]
	TSA_18_min = TSA_18_min.min()
	MIN_df = TSA_18_min.to_frame(name = 'MIN')

	TSA_18_max = TSA_data[list_24]
	TSA_18_max = TSA_18_max.max()
	MAX_df = TSA_18_max.to_frame(name = 'MAX')

	input_data = get_input_data(MIN_df, MAX_df)
	pred_value = load_model(input_data)
	with first_kpi:

    		number1 = round(pred_value[0], 2)
    		st.markdown(f"<h1 style='text-align: center;background : rgb(0,165,173); color: white;'>{number1}</h1>", unsafe_allow_html=True)

	st.markdown("<hr/>", unsafe_allow_html = True)

if __name__=='__main__': 
    main()
