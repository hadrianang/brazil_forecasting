#%%
# 9_1_ClimateBaselines.py
# The goal of this script 


# 1. Imports ----
import pandas as pd
import numpy as np
import torch 
import os
import json

from scipy.stats import gmean
from darts import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.metrics import mape, smape, mse, rmse, mae, mql, ql, mic, ic, mcrps
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from darts.models import (
    LinearRegressionModel,
    Prophet
)
from darts.utils.likelihood_models import GaussianLikelihood, PoissonLikelihood, QuantileRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pytorch_lightning.callbacks import Callback, EarlyStopping
import pickle

from IPython import get_ipython
ip = get_ipython()
if ip:
    print("Running in iPython")
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "1")
    ip.run_line_magic("aimport", "helpers")
    ip.run_line_magic("aimport", "new_helpers")

import helpers as hp
import new_helpers as nhp

torch.set_float32_matmul_precision("medium")
include_stringency_index = False
accel = "gpu"

quantiles = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

gpu_id = 0
set_name = "Climate Tuning_Round2"
tuning_name = "Notebook_24_07_2025" #Name of the tuning scenario to use
model_name = "RandomForest"
random_seed = 20

#%%
#2. Setup ----
base_dir = os.getcwd()

climate_tuning_dir = os.path.join(base_dir, "Climate Tuning", "Notebook_24_07_2025")
model_input_dir = os.path.join(base_dir, "ModelInput_Round2")
output_dir = os.path.join(base_dir, "ClimBaselines")

fc_dir = os.path.join(base_dir, set_name, tuning_name, model_name, "Forecasts")
fc_rd2_dir = os.path.join(base_dir, set_name, tuning_name, model_name, "Forecasts_Round2")

fc_dict = hp.load_pickle(os.path.join(fc_dir, "forecast_dict.pkl"))
fc_rd2_dict = hp.load_pickle(os.path.join(fc_rd2_dir, "forecast_dict_round1.pkl"))
tuning_res_dir = os.path.join(climate_tuning_dir, model_name)
forecast_res_dir = fc_dir
os.makedirs(forecast_res_dir, exist_ok = True)
model_best_params = hp.read_json(os.path.join(tuning_res_dir, "best_params.json"))
tuning_config = hp.read_json(os.path.join(climate_tuning_dir, "tuning_details.json"))

clim_var_names = tuning_config["clim_var_names"]
global_scale = tuning_config["global_scale"]
retrain_per_fold = tuning_config["retrain_per_fold"] 
approach = tuning_config["approach"]
use_orig_scaler = False #If we use the original scaler during tuning to rescale values
use_orig_normer = True #Use the original clim_train1 series (same as tuning) for the normalisation of error metrics - makes test loss comparable with validation loss

multi_model_predict = True

weekly_cal = pd.read_csv(os.path.join(model_input_dir, "Calendar.csv"))
ext_weekly_cal = pd.read_csv(os.path.join(model_input_dir, "CalendarExt.csv"))
climate_df = pd.read_csv(os.path.join(model_input_dir, "TimeVaryingCovs.csv"))

#Remove uf = "ES" since there is no dengue data for Espirito Santo
climate_df = climate_df[climate_df["uf"] != "ES"]

#Dictionary of forecasts to be evaluated
eval_dict = fc_dict



# %%
# 3. Set Info and Series Creation ----
#Define set info used to split the data into different sets. This section should ideally align with what was used during hype
set_info = weekly_cal[["epiweek", "Year", "Week", "WeekStart", "WeekMid", "WeekEnd"]]
set_info = set_info.iloc[1:]


#TIME SERIES USED IN CLIMATE TUNING - NO UNDERSCORE BEFORE NUMBER
#Define the data sets using a column of booleans in the calendar. 
#all epiweek numbers are in YYYYWW integer format
set_info["clim_train1"] = set_info["epiweek"] <= 201510
set_info["clim_val1"] = (set_info["epiweek"] >= 201526) & (set_info["epiweek"] <= 201625)
set_info["clim_train2"] = set_info["epiweek"] <= 201610
set_info["clim_val2"] = (set_info["epiweek"] >= 201626) & (set_info["epiweek"] <= 201725)
set_info["clim_train3"] = set_info["epiweek"] <= 201710
set_info["clim_val3"] = (set_info["epiweek"] >= 201726) & (set_info["epiweek"] <= 201825)

set_info["clim_test"] = (set_info["epiweek"] >= 201841) & (set_info["epiweek"] <= 201940)
set_info["clim_test_input"] = set_info["epiweek"] <= 201825


#TIME SERIES FOR DENGUE
input_set_info = weekly_cal[["epiweek", "Year", "Week", "WeekStart", "WeekMid", "WeekEnd"]]
input_set_info = input_set_info.iloc[1:]

#Sets for input series to generate forecasts to be used during hyperparameter tuning
input_set_info["clim_input_1a"] = input_set_info["epiweek"] <= 201825
input_set_info["clim_input_1b"] = input_set_info["epiweek"] <= 201925
input_set_info["clim_input_1c"] = input_set_info["epiweek"] <= 202025

input_set_info["clim_gap_1a"] = (input_set_info["epiweek"] >= 201826) & (input_set_info["epiweek"] <= 201840)
input_set_info["clim_gap_1b"] = (input_set_info["epiweek"] >= 201926) & (input_set_info["epiweek"] <= 201940)
input_set_info["clim_gap_1c"] = (input_set_info["epiweek"] >= 202026) & (input_set_info["epiweek"] <= 202040)

#Validation sets, which we can also use to see if our forecasts have similar loss to the test and validation sets
input_set_info["clim_val_1a"] = (input_set_info["epiweek"] >= 201841) & (input_set_info["epiweek"] <= 201940)
input_set_info["clim_val_1b"] = (input_set_info["epiweek"] >= 201941) & (input_set_info["epiweek"] <= 202040)
input_set_info["clim_val_1c"] = (input_set_info["epiweek"] >= 202041) & (input_set_info["epiweek"] <= 202140)

#Input sets for generating forecasts for early stopping and validation loss during dengue model training
input_set_info["clim_input_1"] = input_set_info["epiweek"] <= 202110
input_set_info["clim_input_2"] = input_set_info["epiweek"] <= 202210
input_set_info["clim_input_3"] = input_set_info["epiweek"] <= 202310

input_set_info["clim_gap_1"] = (input_set_info["epiweek"] >= 202111) & (input_set_info["epiweek"] <= 202125)
input_set_info["clim_gap_2"] = (input_set_info["epiweek"] >= 202211) & (input_set_info["epiweek"] <= 202225)
input_set_info["clim_gap_3"] = (input_set_info["epiweek"] >= 202311) & (input_set_info["epiweek"] <= 202325)

input_set_info["clim_val_1"] = (input_set_info["epiweek"] >= 202126) & (input_set_info["epiweek"] <= 202225)
input_set_info["clim_val_2"] = (input_set_info["epiweek"] >= 202226) & (input_set_info["epiweek"] <= 202325)
input_set_info["clim_val_3"] = (input_set_info["epiweek"] >= 202326) & (input_set_info["epiweek"] <= 202425)

#Hold sets - which are also used as inputs to generate forecasts over the test sets
input_set_info["clim_hold_1"] = input_set_info["epiweek"] <= 202225
input_set_info["clim_hold_2"] = input_set_info["epiweek"] <= 202325
input_set_info["clim_hold_3"] = input_set_info["epiweek"] <= 202425

#Targets - to evaluate the model output when hold sets are inputs. Forecasts over targets used in evaluating the model
input_set_info["clim_target_gap_1"] = (input_set_info["epiweek"] >= 202226) & (input_set_info["epiweek"] <= 202240)
input_set_info["clim_target_gap_2"] = (input_set_info["epiweek"] >= 202326) & (input_set_info["epiweek"] <= 202340)
input_set_info["clim_target_gap_3"] = (input_set_info["epiweek"] >= 202426) & (input_set_info["epiweek"] <= 202440)

#Targets - to evaluate the model output when hold sets are inputs. Forecasts over targets used in evaluating the model
input_set_info["clim_target_1"] = (input_set_info["epiweek"] >= 202241) & (input_set_info["epiweek"] <= 202340)
input_set_info["clim_target_2"] = (input_set_info["epiweek"] >= 202341) & (input_set_info["epiweek"] <= 202440)
input_set_info["clim_target_3"] = (input_set_info["epiweek"] >= 202441) & (input_set_info["epiweek"] <= 202540)

uf_list = climate_df["uf"].unique()
uf_mapper = climate_df[["uf", "uf_name"]].copy().drop_duplicates()
uf_dict = dict(zip(uf_mapper["uf"], uf_mapper["uf_name"]))
uf_name_list = [uf_dict[curr_uf] for curr_uf in uf_list]
admin_level_list = ["State" for curr in uf_name_list]

#Get a version of the climate DataFrame where the precipiation related variables have been log transformed
log_trans_cols = ["precip_min", "precip_med", "precip_max"]
proc_climate_df = hp.transform_df_cols(climate_df, log_trans_cols, np.log1p)


group_col = "uf"
## 3.1 Series Creation ----
#SERIES USED IN CLIMATE TUNING
#Training set without the pre-processing, mainly for normalising error metrics
train1 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_train1")
train2 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_train2")
train3 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_train3")

#Training sets with pre-processing for actual model training
train1_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_train1")
train2_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_train2")
train3_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_train3")

#Note that since the validation and test sets are only used for evaluation / error calculation, we use the non pre-processed version (no log-transforms) 
val1 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_val1")
val2 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_val2")
val3 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_val3")

eval_train = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_test_input")
eval_train_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_test_input") # Input sequence for the final model to evaluate on test set
test = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "clim_test") #Test set for evaluation

#SERIES USED FOR CLIMATE FORECASTING - FOR DENGUE PIPELINE
#For tuning
#Input 1a, b, c to generate forecasts for the validation sets in hyperparameter tuning
clim_input_1a = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_1a")
clim_input_1b = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_1b")
clim_input_1c = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_1c")

#To compare with forecasts using input 1a,b,c
clim_gap_1a = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_gap_1a")
clim_gap_1b = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_gap_1b")
clim_gap_1c = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_gap_1c")


#To compare with forecasts using input 1a,b,c
clim_val_1a = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_val_1a")
clim_val_1b = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_val_1b")
clim_val_1c = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_val_1c")

#For training
#Input 1, 2, 3 to generate forecasts for the validation sets used in early stopping of the main model training process
clim_input_1 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_1")
clim_input_2 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_2")
clim_input_3 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_3")

#Gap preceding each main val set
clim_gap_1 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_gap_1")
clim_gap_2 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_gap_2")
clim_gap_3 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_gap_3")

#Climate during the dengue validation sets (can evaluate against output from clim_input 1, 2, 3)
clim_val_1 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_val_1")
clim_val_2 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_val_2")
clim_val_3 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_val_3")

#For evaluation
#Hold 1, 2, 3 will also be used as input to generate 
clim_hold_1 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_hold_1")
clim_hold_2 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_hold_2")
clim_hold_3 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_hold_3")

clim_target_gap_1 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_target_gap_1")
clim_target_gap_2 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_target_gap_2")
clim_target_gap_3 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_target_gap_3")

clim_target_1 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_target_1")
clim_target_2 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_target_2")
clim_target_3 = hp.create_ts_list(climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_target_3")

#Log transformed versions of the TimeSeries that will be used as model input
clim_input_1a_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_1a")
clim_input_1b_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_1b")
clim_input_1c_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_1c")

clim_input_1_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_1")
clim_input_2_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_2")
clim_input_3_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_input_3")

clim_hold_1_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_hold_1")
clim_hold_2_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_hold_2")
clim_hold_3_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = input_set_info, set_column = "clim_hold_3")

## 3.2 Scaling ----
#Original scaler
clim_scaler = Scaler(global_fit = global_scale)
clim_scaler.fit(train1_log)

#Only scale training sets 
train1_log_s = clim_scaler.transform(train1_log)
train2_log_s = clim_scaler.transform(train2_log)
train3_log_s = clim_scaler.transform(train3_log)

#New Scaler
clim_eval_scaler = Scaler(global_fit = global_scale)
if use_orig_scaler:
    eval_train_log_s = clim_scaler.transform(eval_train_log)
else:
    eval_train_log_s = clim_eval_scaler.fit_transform(eval_train_log)

if use_orig_normer:
    normer = train1
else:
    normer = eval_train


#%%
## 3.3 Building list sets ----
#Building ts list sets 
#To simplify evaluation, we group lists of series together into sets 
# We can also change the naming scheme here to those in the data split slides
# 

#Tuning series
den_tune_train = [clim_input_1a, clim_input_1b, clim_input_1c]
den_tune_gap = [clim_gap_1a, clim_gap_1b, clim_gap_1c]
den_tune_val = [clim_val_1a, clim_val_1b, clim_val_1c]

den_tune_gap_val = nhp.ts_list_set_concatenate([den_tune_gap, den_tune_val])
den_tune_gap_val_preds = [eval_dict["clim_preds_1a"], eval_dict["clim_preds_1b"], eval_dict["clim_preds_1c"]]


#Main 
den_train = [clim_input_1, clim_input_2, clim_input_3]
den_gap = [clim_gap_1, clim_gap_2, clim_gap_3]
den_val = [clim_val_1, clim_val_2, clim_val_3]

den_hold = [clim_hold_1, clim_hold_2, clim_hold_3]
den_targ_gap = [clim_target_gap_1, clim_target_gap_2, clim_target_gap_3]
den_targ = [clim_target_1, clim_target_2, clim_target_3]


den_gap_val = nhp.ts_list_set_concatenate([den_gap, den_val])
den_gap_val_preds = [eval_dict["clim_preds_1"], eval_dict["clim_preds_2"], eval_dict["clim_preds_3"]]

den_gap_targ = nhp.ts_list_set_concatenate([den_targ_gap, den_targ])
den_gap_targ_preds = [eval_dict["clim_preds_targ1"], eval_dict["clim_preds_targ2"], eval_dict["clim_preds_targ3"]]


#%%
# 4. Baseline Models ----
## 4.1 Naive Seasonal Average Class ----
# We define a class with similar fit and predict interfaces as the Darts models, but with some requirements in the constructor
class NaiveSeasonalAverage:
    def __init__(self, calendar, comp_names, date_name = "WeekStart"): 
        self.calendar = calendar.copy() 
        self.date_name = date_name
        #Assumes dayfirst date format in the weekly calendar
        self.calendar[self.date_name] = pd.to_datetime(self.calendar[self.date_name], format = "%d/%m/%Y")
        self.comp_names = comp_names
        self.weekly_avg_list = []


    def fit(self, series):
        #Build a list of DataFrames containing the average per week for each series in the list
        #This then assumes that the input admin units are in the same order as during training 
        series_list = series
        self.weekly_avg_list = [] 
        for curr_train_ts in series_list:

            curr_train_ts_df = curr_train_ts.to_dataframe().reset_index()

            week_label = self.calendar[[self.date_name, "Week"]]

            #Add the week numbers to each time point
            curr_train_ts_df = curr_train_ts_df.merge(week_label, how = "left", on = self.date_name)

            #Compute the average per week number 
            curr_train_ts_df = curr_train_ts_df.drop(columns = [self.date_name])
            curr_weekly_avg = curr_train_ts_df.groupby("Week").mean(numeric_only=True).reset_index()

            #Remove the value for Week 53 and replace it with the value for Week 52
            curr_weekly_avg = curr_weekly_avg[curr_weekly_avg["Week"] != 53]
            week53 = curr_weekly_avg[curr_weekly_avg["Week"] == 52].copy()
            week53["Week"] = 53
            curr_weekly_avg = pd.concat([curr_weekly_avg, week53], ignore_index=True).sort_values("Week")
            
            self.weekly_avg_list.append(curr_weekly_avg)

    def predict(self, series, n):
        forecast_result = []
        for curr_ind in range(len(series)):
            curr_input_ts = series[curr_ind]
            curr_weekly_avg = self.weekly_avg_list[curr_ind]

            input_end = curr_input_ts.end_time()
            start_index = self.calendar[self.calendar[self.date_name] == input_end].index.tolist()
            idx = start_index[0]
            predict_period = self.calendar.iloc[idx + 1 : idx + 1 + n]


            forecast_df = predict_period.merge(curr_weekly_avg, how = "left", on = "Week").set_index(self.date_name)
            forecast_ts = TimeSeries.from_times_and_values(times = forecast_df.index, values = forecast_df[self.comp_names], columns = self.comp_names)

            forecast_result.append(forecast_ts)
        return forecast_result



# %%
#We make an averager for each ts series list in each set from DenTuneTrain, DenTrain, and DenHold
#then we use the same training sets as input sequences to generate forecasts of length 68 - same as the main forecasts
#Note we use unscaled versions of all the data for this (averaging occurs with the same magnitude as the original variables)

den_tune_train_models = [NaiveSeasonalAverage(calendar = ext_weekly_cal, comp_names = clim_var_names) for curr in den_tune_train] 
den_train_models = [NaiveSeasonalAverage(calendar = ext_weekly_cal, comp_names = clim_var_names) for curr in den_train] 
den_hold_models = [NaiveSeasonalAverage(calendar = ext_weekly_cal, comp_names = clim_var_names) for curr in den_hold] 


#Fit all the models
for model, ts_list in zip(den_tune_train_models, den_tune_train):
    model.fit(ts_list)

for model, ts_list in zip(den_train_models, den_train):
    model.fit(ts_list)

for model, ts_list in zip(den_hold_models, den_hold):
    model.fit(ts_list)

forecast_length = 68

den_tune_gap_val_preds = [model.predict(ts_list, forecast_length) for model, ts_list in zip(den_tune_train_models, den_tune_train)]
den_gap_val_preds = [model.predict(ts_list, forecast_length) for model, ts_list in zip(den_train_models, den_train)]
den_gap_targ_preds = [model.predict(ts_list, forecast_length) for model, ts_list in zip(den_hold_models, den_hold)]

#Save the forecast results for later use through a dictionary
forecast_dict = {
    "den_tune_gap_val_preds": den_tune_gap_val_preds,
    "den_gap_val_preds": den_gap_val_preds,
    "den_gap_targ_preds": den_gap_targ_preds
}
#Pickle
nhp.save_pickle(forecast_dict, os.path.join(output_dir, "seasonal_average.pkl"))