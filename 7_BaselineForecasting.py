#%%

import pandas as pd
import numpy as np
import torch 
import os
import json

from darts import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.metrics import mape, smape, mse, rmse, mae, mql, ql
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import MinMaxScaler
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from darts.models import (
    LinearRegressionModel,
    Prophet,
    TFTModel,
)
from darts.utils.likelihood_models import GaussianLikelihood, PoissonLikelihood, QuantileRegression
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback, EarlyStopping
import pickle
import helpers as hp
import new_helpers as nhp

torch.set_float32_matmul_precision("medium")
include_stringency_index = False
accel = "gpu"

quantiles = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

gpu_id = 0
tuning_name = "Tuning_30_07_2025" #Name of the tuning scenario to use
random_seed = 20
#%%
#1. Reading Data and Setup ----
base_dir = os.getcwd()

#We define a boolean here, relevant to model evaluation
#If this is true, then we train a model with the tuned hyperparameters from scratch - i.e. using train1, 2, and 3. This also means that
#we can train a separate scaler on train1, which is then applied to train 1,2,3
#If this is false, then we mirror the tuning process first, training on 1a,b,c then the model will be fine tuned using train 1,2,3.
#This means that we apply the same scaler, trained only on train1a to train1a,b,c and train1,2,3. 
train_from_scratch = False
global_fit = False
use_multi_climate_final_targ = False

base_dir = os.getcwd()
model_input_dir = os.path.join(base_dir, "ModelInput_Round2")
tuning_dir = os.path.join(base_dir, "Dengue Tuning", tuning_name)
climate_tuning_dir = os.path.join(base_dir, "Climate Tuning_Round2", "Notebook_24_07_2025")
climate_model_name = "RandomForest"
climate_forecast_dir = os.path.join(climate_tuning_dir, climate_model_name, "Forecasts_Round2")

output_dir = os.path.join(base_dir, "Baselines")

os.makedirs(tuning_dir, exist_ok = True)

weekly_cal = pd.read_csv(os.path.join(model_input_dir, "Calendar.csv"))
weekly_cal = weekly_cal.iloc[1:]
climate_df = pd.read_csv(os.path.join(model_input_dir, "TimeVaryingCovs.csv"))
dengue_df = pd.read_csv(os.path.join(model_input_dir, "DengueCases.csv"))

#Remove uf = "ES" since there is no dengue data for Espirito Santo
climate_df = climate_df[climate_df["uf"] != "ES"]
dengue_df = dengue_df[dengue_df["uf"] != "ES"]

dengue_df = dengue_df[["epiweek", "Year", "Week", "uf", "uf_name", "Cases"]]

#%%
#Read the climate forecasts

#We use the climate forecasts from the first round of tuning

clim_forecast_dict = hp.load_pickle(os.path.join(climate_forecast_dir, "forecast_dict_round1.pkl"))
clim_forecast_log_dict = hp.load_pickle(os.path.join(climate_forecast_dir, "forecast_dict_log_round1.pkl"))

clim_forecast_dict_multi = hp.load_pickle(os.path.join(climate_forecast_dir, "Multi_forecast_dict.pkl"))
clim_forecast_log_dict_multi = hp.load_pickle(os.path.join(climate_forecast_dir, "Multi_forecast_dict_log.pkl"))

clim_forecast_dict_single = hp.load_pickle(os.path.join(climate_forecast_dir, "forecast_dict.pkl"))
clim_forecast_log_dict_single = hp.load_pickle(os.path.join(climate_forecast_dir, "forecast_dict_log.pkl"))

clim_tuning_config = hp.read_json(os.path.join(climate_tuning_dir,"tuning_details.json"))
clim_var_names = clim_tuning_config["clim_var_names"]

#If boolean says we should include stringency index, then we add it into the climate variable list
if include_stringency_index:
    clim_var_names.append("StringencyIndex")


#%%
#Reading the dengue model tuning config
den_tuning_config = hp.read_json(os.path.join(tuning_dir, "tuning_config.json"))
#global_fit = den_tuning_config["global_fit"] #Can read global fit from the config file to mirror tuning or ignore this and set it manually 
den_tuning_config

#%%

set_info = weekly_cal[["epiweek", "Year", "Week", "WeekStart", "WeekMid", "WeekEnd"]]

#Set up all the set information for splitting the dengue and climate data
#For tuning training 1a,b,c - val1a,b,c
set_info["den_train1a"] = set_info["epiweek"] <= 201825
set_info["den_train1b"] = set_info["epiweek"] <= 201925
set_info["den_train1c"] = set_info["epiweek"] <= 202025

set_info["den_gap1a"] = (set_info["epiweek"] >= 201826) & (set_info["epiweek"] <= 201840)
set_info["den_gap1b"] = (set_info["epiweek"] >= 201926) & (set_info["epiweek"] <= 201940)
set_info["den_gap1c"] = (set_info["epiweek"] >= 202026) & (set_info["epiweek"] <= 202040)

set_info["den_val1a"] = (set_info["epiweek"] >= 201841) & (set_info["epiweek"] <= 201940)
set_info["den_val1b"] = (set_info["epiweek"] >= 201941) & (set_info["epiweek"] <= 202040)
set_info["den_val1c"] = (set_info["epiweek"] >= 202041) & (set_info["epiweek"] <= 202140)

set_info["den_ex"] = (set_info["epiweek"] >= 202141) & (set_info["epiweek"] <= 202225)

#Train - Val - Test for model evaluation
set_info["den_train1"] = set_info["epiweek"] <= 202110
set_info["den_train2"] = set_info["epiweek"] <= 202210
set_info["den_train3"] = set_info["epiweek"] <= 202310
set_info["den_train4"] = set_info["epiweek"] <= 202410 #For fine-tuning the model for the 2nd round

set_info["den_gap1"] = (set_info["epiweek"] >= 202111) & (set_info["epiweek"] <= 202125)
set_info["den_gap2"] = (set_info["epiweek"] >= 202211) & (set_info["epiweek"] <= 202225)
set_info["den_gap3"] = (set_info["epiweek"] >= 202311) & (set_info["epiweek"] <= 202325)
set_info["den_gap4"] = (set_info["epiweek"] >= 202411) & (set_info["epiweek"] <= 202425)

set_info["den_val1"] = (set_info["epiweek"] >= 202126) & (set_info["epiweek"] <= 202225)
set_info["den_val2"] = (set_info["epiweek"] >= 202226) & (set_info["epiweek"] <= 202325)
set_info["den_val3"] = (set_info["epiweek"] >= 202326) & (set_info["epiweek"] <= 202425)
set_info["den_val4"] = (set_info["epiweek"] >= 202426) & (set_info["epiweek"] <= 202525)

#Hold sets - can be used to retrieve climate covariates for evaluation time
set_info["den_hold1"] = set_info["epiweek"] <= 202225
set_info["den_hold2"] = set_info["epiweek"] <= 202325
set_info["den_hold3"] = set_info["epiweek"] <= 202425
set_info["den_hold4"] = set_info["epiweek"] <= 202525

#Targets for model evaluation
set_info["den_targ1"] = (set_info["epiweek"] >= 202241) & (set_info["epiweek"] <= 202340)
set_info["den_targ2"] = (set_info["epiweek"] >= 202341) & (set_info["epiweek"] <= 202440)
set_info["den_targ3"] = (set_info["epiweek"] >= 202441) & (set_info["epiweek"] <= 202540)
set_info["den_targ4"] = (set_info["epiweek"] >= 202541) & (set_info["epiweek"] <= 202640) #Note that we do not have data for den_targ4

#%%
#Set up the names of the Brazilian states 
uf_list = climate_df["uf"].unique()
uf_mapper = climate_df[["uf", "uf_name"]].copy().drop_duplicates()
uf_dict = dict(zip(uf_mapper["uf"], uf_mapper["uf_name"]))
uf_name_list = [uf_dict[curr_uf] for curr_uf in uf_list]


#%%
# 2. Preparing TimeSeries

#Log transformed climate (just the precipitation columns)
log_trans_cols = ["precip_min", "precip_med", "precip_max"]
proc_climate_df = hp.transform_df_cols(climate_df, log_trans_cols, np.log1p)

#Log transformed dengue cases
proc_dengue_df = hp.transform_df_cols(dengue_df, ["Cases"], np.log1p)

#Create all the TimeSeries objects we need for tuning first
group_col = "uf"

#Dengue case TimeSeries - without any log transforms - mostly here for visualisations and loss computations
train1a = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1a")
train1b = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1b")
train1c = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1c")

val1a = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1a")
val1b = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1b")
val1c = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1c")

train1 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1")
train2 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train2")
train3 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train3")
train4 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train4")

val1 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1")
val2 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val2")
val3 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val3")
val4 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val4")

hold1 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold1")
hold2 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold2")
hold3 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold3")
hold4 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold4")

targ1 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ1")
targ2 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ2")
targ3 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ3")

#Log transformed version of the cases TimeSeries for actual training
train1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1a")
train1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1b")
train1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1c")

gap1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1a")
gap1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1b")
gap1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1c")

val1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1a")
val1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1b")
val1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1c")


#Evaluation sets
train1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1")
train2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train2")
train3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train3")
train4_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train4")

gap1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1")
gap2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap2")
gap3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap3")
gap4_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap4")

val1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1")
val2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val2")
val3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val3")
val4_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val4")

hold1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold1")
hold2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold2")
hold3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold3")
hold4_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold4")

targ1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ1")
targ2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ2")
targ3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ3")



#Climate values to use as covariates 
clim_train1a_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1a")
clim_train1b_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1b")
clim_train1c_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1c")

clim_train1_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1")
clim_train2_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train2")
clim_train3_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train3")
clim_train4_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train4")

#Used for evaluation over test set
clim_hold1_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_hold1")
clim_hold2_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_hold2")
clim_hold3_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_hold3")
clim_hold4_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_hold4")

#Get the forecasts over the validation set to be used as future covariates
clim_preds_1a_log = clim_forecast_log_dict["clim_preds_1a"]
clim_preds_1b_log = clim_forecast_log_dict["clim_preds_1b"]
clim_preds_1c_log = clim_forecast_log_dict["clim_preds_1c"]

clim_preds_1_log = clim_forecast_log_dict["clim_preds_1"]
clim_preds_2_log = clim_forecast_log_dict["clim_preds_2"]
clim_preds_3_log = clim_forecast_log_dict["clim_preds_3"]


clim_preds_targ1_log = clim_forecast_log_dict["clim_preds_targ1"]
clim_preds_targ2_log = clim_forecast_log_dict["clim_preds_targ2"]
clim_preds_targ3_log = clim_forecast_log_dict["clim_preds_targ3"]

clim_preds_targ3_log = clim_forecast_log_dict["clim_preds_targ3"]

#Climate forecasts for 2nd round (can be from mutli or single model set-up depending on Boolean setting).
#For all other targets, we use the single model set-up, since that was used in the first round. 
if not use_multi_climate_final_targ:
    print("Using climate forecasts from single main model for final target")
    clim_preds_4_log = clim_forecast_log_dict_single["clim_preds_4"]
    clim_preds_targ4_log = clim_forecast_log_dict_single["clim_preds_targ4"]
else:
    print("Using final target-specific climate forecasts (multiple models) for Targ. 4")
    clim_preds_4_log = clim_forecast_log_dict_multi["clim_preds_4"]
    clim_preds_targ4_log = clim_forecast_log_dict_multi["clim_preds_targ4"]

#Processing with COVID-19 stringency index (optional)
def intersect_stack(curr_shorter, curr_longer): 
    to_stack = curr_longer.slice_intersect(curr_shorter)
    return curr_shorter.stack(to_stack)

if include_stringency_index:
    #If including the StringencyIndex, then should at it to the climate forecasts
    #Since there is no StringencyIndex forecasting model, we assume it is 0 in forecasts
    temp_cal = weekly_cal.copy()
    temp_cal["WeekStart"] = pd.to_datetime(temp_cal["WeekStart"], format = "%Y-%m-%d")
    temp_cal = temp_cal.set_index("WeekStart")
    zero_series = TimeSeries.from_times_and_values(
        times = temp_cal.index,
        values = np.zeros(len(temp_cal)),
        columns=["StringencyIndex"]
    )
    
    #Add the zero series into the TimeSeries with predicted climate
    clim_preds_1a_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_1a_log]
    clim_preds_1b_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_1b_log]
    clim_preds_1c_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_1c_log]

    clim_preds_1_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_1_log]
    clim_preds_2_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_2_log]
    clim_preds_3_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_3_log]
    clim_preds_4_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_4_log]

    clim_preds_targ1_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_targ1_log]
    clim_preds_targ2_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_targ2_log]
    clim_preds_targ3_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_targ3_log]
    clim_preds_targ4_log = [intersect_stack(curr_series, zero_series) for curr_series in clim_preds_targ4_log]


#%%
## 2.1 Add static covariates ----

stat_covs = pd.read_csv(os.path.join(model_input_dir, "StaticCovs.csv"))
stat_covs_names = stat_covs.columns[2:]

#generate min-max scaled version of the static covariates
stat_cov_scaler = MinMaxScaler()
stat_cov_scaler.fit(stat_covs[stat_covs_names])
stat_covs_vals_s = pd.DataFrame(stat_cov_scaler.transform(stat_covs[stat_covs_names]), columns = stat_covs_names)
stat_covs_s = pd.concat([stat_covs[["uf", "uf_name"]], stat_covs_vals_s], axis = 1)


#Add static coariates to all input data used in either training or as input sequence during prediction time
train1a_log_sc = hp.add_stat_covs(train1a_log, stat_covs_s, uf_list, stat_covs_names)
train1b_log_sc = hp.add_stat_covs(train1b_log, stat_covs_s, uf_list, stat_covs_names)
train1c_log_sc = hp.add_stat_covs(train1c_log, stat_covs_s, uf_list, stat_covs_names)

train1_log_sc = hp.add_stat_covs(train1_log, stat_covs_s, uf_list, stat_covs_names)
train2_log_sc = hp.add_stat_covs(train2_log, stat_covs_s, uf_list, stat_covs_names)
train3_log_sc = hp.add_stat_covs(train3_log, stat_covs_s, uf_list, stat_covs_names)
train4_log_sc = hp.add_stat_covs(train4_log, stat_covs_s, uf_list, stat_covs_names)

hold1_log_sc = hp.add_stat_covs(hold1_log, stat_covs_s, uf_list, stat_covs_names)
hold2_log_sc = hp.add_stat_covs(hold2_log, stat_covs_s, uf_list, stat_covs_names)
hold3_log_sc = hp.add_stat_covs(hold3_log, stat_covs_s, uf_list, stat_covs_names)
hold4_log_sc = hp.add_stat_covs(hold4_log, stat_covs_s, uf_list, stat_covs_names)

#%%
## 2.3 Rescaling TimeSeries ----

#Rescale the cases and the covariates
den_tune_scaler = Scaler(global_fit = global_fit)
den_scratch_scaler = Scaler(global_fit = global_fit)
clim_tune_scaler = Scaler(global_fit = global_fit)
clim_scratch_scaler = Scaler(global_fit = global_fit)

#Fit the scalers to the training sets from the first folds
den_tune_scaler.fit(train1a_log_sc)
den_scratch_scaler.fit(train1_log_sc)
clim_tune_scaler.fit(clim_train1a_log)
clim_scratch_scaler.fit(clim_train1_log)

#If we are training from scratch, we use the scaler fit only on train1
if train_from_scratch:
    den_eval_scaler = den_scratch_scaler
    clim_eval_scaler = clim_scratch_scaler
else:
    #If we are not starting from scratch i.e. we are just fine tuning the model after mirroring the tuning process,
    #then we have to use the same scaler that we setup during the tuning process
    den_eval_scaler = den_tune_scaler
    clim_eval_scaler = clim_tune_scaler

#Generate the scaled TimeSeries - 1st set used to mirror the tuning process
train1a_log_sc_s = den_tune_scaler.transform(train1a_log_sc)
train1b_log_sc_s = den_tune_scaler.transform(train1b_log_sc)
train1c_log_sc_s = den_tune_scaler.transform(train1c_log_sc)

gap1a_log_s = den_tune_scaler.transform(gap1a_log)
gap1b_log_s = den_tune_scaler.transform(gap1b_log)
gap1c_log_s = den_tune_scaler.transform(gap1c_log)

val1a_log_s = den_tune_scaler.transform(val1a_log)
val1b_log_s = den_tune_scaler.transform(val1b_log)
val1c_log_s = den_tune_scaler.transform(val1c_log)

#Scale the main TimeSeries we use for primary training and evaluation (post-tuning fits)
train1_log_sc_s = den_eval_scaler.transform(train1_log_sc)
train2_log_sc_s = den_eval_scaler.transform(train2_log_sc)
train3_log_sc_s = den_eval_scaler.transform(train3_log_sc)
train4_log_sc_s = den_eval_scaler.transform(train4_log_sc)

gap1_log_s = den_eval_scaler.transform(gap1_log)
gap2_log_s = den_eval_scaler.transform(gap2_log)
gap3_log_s = den_eval_scaler.transform(gap3_log)
gap4_log_s = den_eval_scaler.transform(gap4_log)

val1_log_s = den_eval_scaler.transform(val1_log)
val2_log_s = den_eval_scaler.transform(val2_log)
val3_log_s = den_eval_scaler.transform(val3_log)
val4_log_s = den_eval_scaler.transform(val4_log)

hold1_log_sc_s = den_eval_scaler.transform(hold1_log_sc)
hold2_log_sc_s = den_eval_scaler.transform(hold2_log_sc)
hold3_log_sc_s = den_eval_scaler.transform(hold3_log_sc)
hold4_log_sc_s = den_eval_scaler.transform(hold4_log_sc)

targ1_log_s = den_eval_scaler.transform(targ1_log)
targ2_log_s = den_eval_scaler.transform(targ2_log)
targ3_log_s = den_eval_scaler.transform(targ3_log)


#Scaled climate covariates - for tuning (mirroring tuning process)
clim_train1a_log_s = clim_tune_scaler.transform(clim_train1a_log)
clim_train1b_log_s = clim_tune_scaler.transform(clim_train1b_log)
clim_train1c_log_s = clim_tune_scaler.transform(clim_train1c_log)

clim_preds_1a_log_s = clim_tune_scaler.transform(clim_preds_1a_log)
clim_preds_1b_log_s = clim_tune_scaler.transform(clim_preds_1b_log)
clim_preds_1c_log_s = clim_tune_scaler.transform(clim_preds_1c_log)

#Scaled climate covariates - for evaluation (values post-tuning, for forecasting over the test set)
clim_train1_log_s = clim_eval_scaler.transform(clim_train1_log)
clim_train2_log_s = clim_eval_scaler.transform(clim_train2_log)
clim_train3_log_s = clim_eval_scaler.transform(clim_train3_log)
clim_train4_log_s = clim_eval_scaler.transform(clim_train4_log)

clim_preds_1_log_s = clim_eval_scaler.transform(clim_preds_1_log)
clim_preds_2_log_s = clim_eval_scaler.transform(clim_preds_2_log)
clim_preds_3_log_s = clim_eval_scaler.transform(clim_preds_3_log)
clim_preds_4_log_s = clim_eval_scaler.transform(clim_preds_4_log)

clim_hold1_log_s = clim_eval_scaler.transform(clim_hold1_log)
clim_hold2_log_s = clim_eval_scaler.transform(clim_hold2_log)
clim_hold3_log_s = clim_eval_scaler.transform(clim_hold3_log)
clim_hold4_log_s = clim_eval_scaler.transform(clim_hold4_log)

clim_preds_targ1_log_s = clim_eval_scaler.transform(clim_preds_targ1_log)
clim_preds_targ2_log_s = clim_eval_scaler.transform(clim_preds_targ2_log)
clim_preds_targ3_log_s = clim_eval_scaler.transform(clim_preds_targ3_log)
clim_preds_targ4_log_s = clim_eval_scaler.transform(clim_preds_targ4_log)

#Merge the training and validation covariates to be passed into the predict function for validation loss evaluation
clim_train_val1a_log_s = hp.ts_list_concatenate([clim_train1a_log_s, clim_preds_1a_log_s])
clim_train_val1b_log_s = hp.ts_list_concatenate([clim_train1b_log_s, clim_preds_1b_log_s])
clim_train_val1c_log_s = hp.ts_list_concatenate([clim_train1c_log_s, clim_preds_1c_log_s])

clim_train_val1_log_s = hp.ts_list_concatenate([clim_train1_log_s, clim_preds_1_log_s])
clim_train_val2_log_s = hp.ts_list_concatenate([clim_train2_log_s, clim_preds_2_log_s])
clim_train_val3_log_s = hp.ts_list_concatenate([clim_train3_log_s, clim_preds_3_log_s])
clim_train_val4_log_s = hp.ts_list_concatenate([clim_train4_log_s, clim_preds_4_log_s])

clim_hold_targ1_log_s = hp.ts_list_concatenate([clim_hold1_log_s, clim_preds_targ1_log_s])
clim_hold_targ2_log_s = hp.ts_list_concatenate([clim_hold2_log_s, clim_preds_targ2_log_s])
clim_hold_targ3_log_s = hp.ts_list_concatenate([clim_hold3_log_s, clim_preds_targ3_log_s])
clim_hold_targ4_log_s = hp.ts_list_concatenate([clim_hold4_log_s, clim_preds_targ4_log_s])

#%%
# Grouping series into sets for easier handling

train = [train1, train2, train3]
train_log_sc = [train1_log_sc, train2_log_sc, train3_log_sc]
train_log_sc_s = [train1_log_sc_s, train2_log_sc_s, train3_log_sc_s]

gap_log = [gap1_log, gap2_log, gap3_log]
gap_log_s = [gap1_log_s, gap2_log_s, gap3_log_s]

val = [val1, val2, val3]
val_log = [val1_log, val2_log, val3_log]
val_log_s = [val1_log_s, val2_log_s, val3_log_s]

hold = [hold1, hold2, hold3]
hold_log_sc = [hold1_log_sc, hold2_log_sc, hold3_log_sc]
hold_log_sc_s = [hold1_log_sc_s, hold2_log_sc_s, hold3_log_sc_s]

targ = [targ1, targ2, targ3]
targ_log = [targ1_log, targ2_log, targ3_log]
targ_log_s = [targ1_log_s, targ2_log_s, targ3_log_s]

clim_train_val_log_s = [clim_train_val1_log_s, clim_train_val2_log_s, clim_train_val3_log_s]
clim_hold_targ_log_s = [clim_hold_targ1_log_s, clim_hold_targ2_log_s, clim_hold_targ3_log_s]
#%%
# 3. Modelling ----

#Below are some helper functions 
## 3.1 Helpers ----
def series_difference(curr_longer, curr_shorter, input_chunk_length = 0, replay = 0):
    """
    Gets the difference between two series, with one longer than the other (points in longer that are not in shorter).
    Additionally gets the last input_chunk_length + replay data points from the shorter series and adds them to the series difference.
    
    """
    shorter_end = curr_shorter.end_time()
    long_before, long_after = curr_longer.split_after(shorter_end)
    to_ret = long_after
    to_add = 0 
    to_add += input_chunk_length
    to_add += replay
    
    if to_add != 0:
        to_ret = concatenate([long_before[-to_add:], to_ret])
    return to_ret

def series_difference_list(curr_longer_list, curr_shorter_list, input_chunk_length = 0, replay = 0):
    """
    Applies the series difference function to two lists of TimeSeries, where one list always has longer series
    than the corresponding series in the other list. 
    """
    to_ret = [series_difference(curr_longer, curr_shorter, input_chunk_length = input_chunk_length, replay = replay) for curr_longer, curr_shorter in zip(curr_longer_list, curr_shorter_list)]
    return to_ret

def num_windows(curr_ts, input_chunk_length, output_chunk_length):
    """
    Returns number of input/output chunk pairs that could be derived from a TimeSeries assuming stride = 1
    """
    return len(curr_ts) - (input_chunk_length + output_chunk_length) + 1

def get_replay_length(curr_ts, input_chunk_length, output_chunk_length, p):
    """
    Assuming stride=1, answers the question "how many points should we add to the input TimeSeries
    such that the number of i/o chunk pairs is increased by p" where p is some percentage. 
    """
    s = len(curr_ts)
    win_size = (input_chunk_length + output_chunk_length) 
    return int(np.ceil((1+p) * (s - win_size + 1) + (win_size) - 1)) - s
    
class NaiveSeasonalWeeklyForecaster:
    """
    Class defining a Naive Seasonal Weekly Forecaster with fit and predict functions.
    """
    def __init__(self, calendar, exclude_years = None, week_col = "WeekStart"):
        self.calendar = calendar.copy()
        
        #Convert the week columns to dates
        self.calendar["WeekStart"] = pd.to_datetime(self.calendar["WeekStart"], format = "%Y-%m-%d")
        self.calendar["WeekMid"] = pd.to_datetime(self.calendar["WeekMid"], format = "%Y-%m-%d")
        self.calendar["WeekEnd"] = pd.to_datetime(self.calendar["WeekEnd"], format = "%Y-%m-%d")

        self.weekly_cases_list = []
        self.exclude_years = exclude_years
        self.week_col = week_col

    def fit(self, series):
        self.weekly_cases_list = [] 
        for curr_train_ts in series: 
            #Convert each training TimeSeries into a DataFrame
            curr_train_ts_df = curr_train_ts.to_dataframe().reset_index()
            #Change the name of the columns to "Date" and "Cases"
            curr_train_ts_df.columns = ["Date", "Cases"]


            
            #Merge the calendar with the TimeSeries DataFrame which will give us the week numbers
            curr_train_ts_df = curr_train_ts_df.merge(self.calendar, how = "left", left_on = "Date", right_on = self.week_col)

            #IF there are any years to exclude, we remove them from the DataFrame so they aren't averaged
            if self.exclude_years:
                curr_train_ts_df = curr_train_ts_df[~curr_train_ts_df["Year"].isin(self.exclude_years)]

            #Get the weekly average of the cases and store the DataFrame of averages in weekly_cases_list
            weekly_cases = curr_train_ts_df[["Cases", "Week"]]
            weekly_cases = weekly_cases.groupby("Week").agg(
                                Cases = ("Cases", "mean")
                            ).reset_index()
            
            #Some years have 53 weeks - use the value for week 52 as the value for week 53
            #We have to add this as a row or replace the value for week 53 if it already exists
            week_52_cases = weekly_cases[weekly_cases["Week"] == 52]["Cases"].values[0]
            if 53 not in weekly_cases["Week"].values:
                weekly_cases = pd.concat([weekly_cases, pd.DataFrame({"Week": [53], "Cases": [week_52_cases]})], ignore_index=True)
            else:
                weekly_cases.loc[weekly_cases["Week"] == 53, "Cases"] = week_52_cases

            self.weekly_cases_list.append(weekly_cases)

    def predict(self, n, series):
        #Create a list for the forecast result
        forecast_result = []
        
        #For each pair of input TimeSeries and weekly cases DataFrame, we generate the forecast
        for curr_input_ts, curr_weekly_cases in zip(series, self.weekly_cases_list):
            input_end = curr_input_ts.end_time()
            curr_col_name = curr_input_ts.components.tolist()[0]

            #The start of the forecast is the week after the last input week
            start_index = self.calendar[self.calendar[self.week_col] == input_end].index.tolist()

            #Get the prediction period
            idx = start_index[0]
            predict_period = self.calendar.iloc[idx + 1 : idx + 1 + n]

            #The forecast is just the weekly average cases for the corresponding week numbers in the prediction period
            forecast_df = predict_period.merge(curr_weekly_cases, how = "left", on = "Week").set_index(self.week_col)
            forecast_ts = TimeSeries.from_times_and_values(times = forecast_df.index, values = forecast_df["Cases"], columns = [curr_col_name])
            forecast_result.append(forecast_ts)
        return forecast_result
#%%
#Read the original tuning results
storage = JournalStorage(
    JournalFileBackend(os.path.join(tuning_dir, den_tuning_config["OptunaLog"]))
)
opt_study = optuna.load_study(study_name = den_tuning_config["StudyName"], storage = storage)
best_params = opt_study.best_params

#%%
# 3.2 Building Models ----
# In this section we build the baseline models.
# Importantly, as baseline models, we do not use replay and train from scratch (since these are not neural networks)
# In addition, we do not need a validation set and instead train the models using all available data
# prior to the gap preceding each test set 
# 

#We implement 3 baseline models, prophet, linear regression, and seasonal average - we then use each to generate forecasts
#That is gap + test_set long 

#For linear regression and prophet - we include month and year encoders - which should ideally capture some trend and seasonality

num_models = 3

#Function to build TFT models with the given hyperparameters
def build_tft_model(model_name, batch_size, input_chunk_length,
                     output_chunk_length, hidden_size, 
                     hidden_continuous_size, lstm_layers, 
                     num_attention_heads, dropout, learning_rate, 
                     output_chunk_shift = 0, force_reset = True, 
                     progress_bars = True, n_epochs = 100, early_stopping = True):
    torch.manual_seed(random_seed)
    curr_encoders = {"datetime_attribute": {"future": ["month", "year"]}, 
                    "transformer": Scaler()}

    pl_trainer_kwargs = {"accelerator": accel,
                             "devices": [gpu_id]}

    if not progress_bars:
        pl_trainer_kwargs["enable_progress_bar"] = False

    if early_stopping:
        early_stopper = EarlyStopping("val_loss", min_delta = 0.0001, patience = 8)
        pl_trainer_kwargs["callbacks"] = [early_stopper]

    curr_model = TFTModel(
        model_name = model_name,
        batch_size = batch_size,
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        output_chunk_shift = output_chunk_shift,
        hidden_size = hidden_size,
        hidden_continuous_size = hidden_continuous_size,
        lstm_layers = lstm_layers,
        num_attention_heads = num_attention_heads,
        dropout = dropout,
        n_epochs = n_epochs,
        add_encoders = curr_encoders,
        likelihood = QuantileRegression(quantiles = quantiles), 
        optimizer_kwargs = {"lr": learning_rate},
        pl_trainer_kwargs = pl_trainer_kwargs,
        show_warnings = True,
        save_checkpoints = True,
        use_static_covariates = True,
        force_reset = force_reset,
        random_state = random_seed
    )
    return curr_model


#%%
#Here we build the models -- we try with maximum data (training on hold)
#LR with fixed lag = 52
lr_models = [LinearRegressionModel(lags = 52, lags_future_covariates = (52,1),
 add_encoders = {"datetime_attribute": {"future": ["month", "year"]}, 
                    "transformer": Scaler()}) for i in range(num_models)]
proph_models = [[Prophet(random_state = random_seed, add_encoders = {"datetime_attribute": {"future": ["month", "year"]}, 
                    "transformer": Scaler()}) for j in range(len(uf_list))] for i in range(num_models)] #Prophet is local only, so one has to be created per uf
seas_avg_models = [NaiveSeasonalWeeklyForecaster(calendar = weekly_cal) for i in range(num_models)] #This behaves like a global model - but internally acts local

#But as sensitivity analysis, we create a version training on just train -- the same data used by the TFT 
lr_models_sens = [LinearRegressionModel(lags = 52, lags_future_covariates = (52,1),
 add_encoders = {"datetime_attribute": {"future": ["month", "year"]}, 
                    "transformer": Scaler()}) for i in range(num_models)]
proph_models_sens = [[Prophet(random_state = random_seed, add_encoders = {"datetime_attribute": {"future": ["month", "year"]}, 
                    "transformer": Scaler()}) for j in range(len(uf_list))] for i in range(num_models)] #Prophet is local only, so one has to be created per uf
seas_avg_models_sens = [NaiveSeasonalWeeklyForecaster(calendar = weekly_cal) for i in range(num_models)] #This behaves like a global model - but internally acts local

#%%
#We also build TFTs training from scratch as global models 
#For this, we load the best hyperparameters -- same as those used in the main forecast

tft_models_sens = []
for i in range(num_models):
    curr_model = build_tft_model(model_name = "TFT_Sens" + str(i), 
                                 batch_size = best_params["batch_size"], 
                                 input_chunk_length = best_params["input_chunk_length"], 
                                 output_chunk_length = best_params["output_chunk_length"], 
                                 hidden_size = best_params["hidden_size"], 
                                 hidden_continuous_size = best_params["hidden_size"], 
                                 lstm_layers = best_params["lstm_layers"], 
                                 num_attention_heads = best_params["num_attention_heads"], 
                                 dropout = best_params["dropout"], 
                                 learning_rate = best_params["learning_rate"])
    tft_models_sens.append(curr_model)


#%%
## 3.3 Fitting Models ----

### 3.3.1 Fitting Main ----
#First, we fit the models maximising the training data (using hold as both training and input series)
for i in range(num_models):
    print(f"Fitting Linear Regression model {i+1} of {num_models}")
    lr_models[i].fit(series = hold_log_sc_s[i], 
                     future_covariates = clim_hold_targ_log_s[i])
#Fit the prophet models -- these are local models 
for i in range(num_models):
    for j in range(len(uf_list)):
        print(f"Fitting Prophet model {i+1} of {num_models} for uf {uf_list[j]}")
        proph_models[i][j].fit(series = hold_log_sc_s[i][j], 
                               future_covariates = clim_hold_targ_log_s[i][j])

#Fit the seasonal average models -- these are also local models 
for i in range(num_models):
    print(f"Fitting Seasonal Average model {i+1} of {num_models}")
    seas_avg_models[i].fit(series = hold[i])


### 3.3.2 Fitting Sensitivity ----
for i in range(num_models):
    print(f"Fitting Linear Regression model {i+1} of {num_models}")
    lr_models_sens[i].fit(series = train_log_sc_s[i], 
                     future_covariates = clim_train_val_log_s[i])
#Fit the prophet models -- these are local models 
for i in range(num_models):
    for j in range(len(uf_list)):
        print(f"Fitting Prophet model {i+1} of {num_models} for uf {uf_list[j]}")
        proph_models_sens[i][j].fit(series = train_log_sc_s[i][j], 
                               future_covariates = clim_train_val_log_s[i][j])

#Fit the seasonal average models -- these are also local models 
for i in range(num_models):
    print(f"Fitting Seasonal Average model {i+1} of {num_models}")
    seas_avg_models_sens[i].fit(series = train[i])

#%%
#Also fit the sensitivity analysis TFT

#We have to convert the relevant series into float32 
train_log_sc_s_f32 = nhp.ts_list_set_conv_dtype(train_log_sc_s)
gap_log_s_f32 = nhp.ts_list_set_conv_dtype(gap_log_s)
val_log_s_f32 = nhp.ts_list_set_conv_dtype(val_log_s)
clim_train_val_log_s_f32 = nhp.ts_list_set_conv_dtype(clim_train_val_log_s)
hold_log_sc_s_f32 = nhp.ts_list_set_conv_dtype(hold_log_sc_s)
clim_hold_targ_log_s_f32 = nhp.ts_list_set_conv_dtype(clim_hold_targ_log_s)

input_chunk_length = best_params["input_chunk_length"]
dyn_val_sets_log_s_f32 = []
for i in range(num_models):
    curr_train_list = train_log_sc_s_f32[i]
    curr_gap_list = gap_log_s_f32[i]
    curr_val_list = val_log_s_f32[i]
    curr_dyn_val_list = []
    for j in range(len(curr_train_list)):
        curr_train = curr_train_list[j]
        curr_gap = curr_gap_list[j]
        curr_val = curr_val_list[j]

        curr_dyn_val_ts = concatenate([curr_train[-input_chunk_length:], curr_gap, curr_val])

        curr_dyn_val_list.append(curr_dyn_val_ts)
    dyn_val_sets_log_s_f32.append(curr_dyn_val_list)


for i in range(num_models):
    print(f"Fitting TFT model {i+1} of {num_models}")
    tft_models_sens[i].fit(series = train_log_sc_s_f32[i], val_series = dyn_val_sets_log_s_f32[i],
                           future_covariates = clim_train_val_log_s_f32[i], val_future_covariates = clim_train_val_log_s_f32[i])

#%%

#TFT_sens_max is a version of the TFT using the number of epochs used in early stopping by the original TFT_sens models
#This means it can train on all the hold data as it does not need a validation set 
epochs_trained = [curr_model.epochs_trained for curr_model in tft_models_sens]
tft_models_sens_max = []
for i in range(num_models):
    curr_epochs_trained = epochs_trained[i]
    curr_model = build_tft_model(model_name = "TFT_SensMax" + str(i), 
                                 batch_size = best_params["batch_size"], 
                                 input_chunk_length = best_params["input_chunk_length"], 
                                 output_chunk_length = best_params["output_chunk_length"], 
                                 hidden_size = best_params["hidden_size"], 
                                 hidden_continuous_size = best_params["hidden_size"], 
                                 lstm_layers = best_params["lstm_layers"], 
                                 num_attention_heads = best_params["num_attention_heads"], 
                                 dropout = best_params["dropout"], 
                                 learning_rate = best_params["learning_rate"],
                                 early_stopping = False,
                                 n_epochs = curr_epochs_trained,
                                 )
    tft_models_sens_max.append(curr_model)

#%%
for i in range(num_models):
    print(f"Fitting TFT model {i+1} of {num_models}")
    tft_models_sens_max[i].fit(series = hold_log_sc_s_f32[i],
                           future_covariates = clim_hold_targ_log_s_f32[i])

#%%
# 3.4 Generating Forecasts ----
gap_length = 15 

forecast_lengths = [len(targ[i][0]) + gap_length for i in range(num_models)] #The forecast length is the length of the target set plus the gap length

#Main forecasts
lr_preds_log_s = []
proph_preds_log_s = []
seas_avg_preds = [] 
for i in range(num_models):
    print(f"Generating forecasts for model {i+1} of {num_models}")
    
    #Linear Regression
    curr_lr_preds = lr_models[i].predict(n = forecast_lengths[i], 
                                         series = hold_log_sc_s[i], 
                                         future_covariates = clim_hold_targ_log_s[i])
    
    #Prophet
    curr_proph_preds = [proph_models[i][j].predict(n = forecast_lengths[i], num_samples = 200, 
                                                   series = hold_log_sc_s[i][j], 
                                                   future_covariates = clim_hold_targ_log_s[i][j]) for j in range(len(uf_list))]
    
    #Seasonal Average
    curr_seas_avg_preds = seas_avg_models[i].predict(n = forecast_lengths[i], 
                                                     series = hold[i])

   #Add the predictions to the lists 
    lr_preds_log_s.append(curr_lr_preds)
    proph_preds_log_s.append(curr_proph_preds)
    seas_avg_preds.append(curr_seas_avg_preds)

#%%
# Sensitivity forecasts -- input sequences should be the same as main 
lr_sens_preds_log_s = []
proph_sens_preds_log_s = []
seas_avg_sens_preds = [] 
for i in range(num_models):
    print(f"Generating forecasts for model {i+1} of {num_models}")
    
    #Linear Regression
    curr_lr_preds = lr_models_sens[i].predict(n = forecast_lengths[i], 
                                         series = hold_log_sc_s[i], 
                                         future_covariates = clim_hold_targ_log_s[i])
    
    #Prophet
    curr_proph_preds = [proph_models_sens[i][j].predict(n = forecast_lengths[i], num_samples = 200, 
                                                   series = hold_log_sc_s[i][j], 
                                                   future_covariates = clim_hold_targ_log_s[i][j]) for j in range(len(uf_list))]
    
    #Seasonal Average
    curr_seas_avg_preds = seas_avg_models_sens[i].predict(n = forecast_lengths[i], 
                                                     series = hold[i])

   #Add the predictions to the lists 
    lr_sens_preds_log_s.append(curr_lr_preds)
    proph_sens_preds_log_s.append(curr_proph_preds)
    seas_avg_sens_preds.append(curr_seas_avg_preds)

#%%
# TFT forecasts - load the best models first
best_tft_models_sens = [curr_model.load_from_checkpoint(curr_model.model_name, best = True) for curr_model in tft_models_sens]

#Generate forecasts using the best models 
tft_sens_preds_log_s = []
for i in range(num_models):
    print(f"Generating TFT forecasts for model {i+1} of {num_models}")
    curr_tft_preds = best_tft_models_sens[i].predict(n = forecast_lengths[i], 
                                                     series = hold_log_sc_s_f32[i], 
                                                     future_covariates = clim_hold_targ_log_s_f32[i],
                                                     num_samples = 200
                                                     )
    tft_sens_preds_log_s.append(curr_tft_preds)

#%%
#For the version with no early stopping and trained on all the hold data, no need to load the best model
tft_sens_preds_max_log_s = []
for i in range(num_models):
    print(f"Generating TFT forecasts for model {i+1} of {num_models}")
    curr_tft_preds = tft_models_sens_max[i].predict(n = forecast_lengths[i], 
                                                     series = hold_log_sc_s_f32[i], 
                                                     future_covariates = clim_hold_targ_log_s_f32[i],
                                                     num_samples = 200
                                                     )
    tft_sens_preds_max_log_s.append(curr_tft_preds)

#%%
# 3.5 Forecast post-processing ----

## 3.5.1 Post-process main
#For LR and prophet predictions - invert transforms to get back to natural scale
lr_preds_log = [den_eval_scaler.inverse_transform(curr_preds) for curr_preds in lr_preds_log_s]
proph_preds_log = [den_eval_scaler.inverse_transform(curr_preds) for curr_preds in proph_preds_log_s]

#Invert the log transform - map np.expm1 to each individual ts 
lr_preds = [[curr_ts.map(np.expm1) for curr_ts in curr_preds] for curr_preds in lr_preds_log]
proph_preds = [[curr_ts.map(np.expm1) for curr_ts in curr_preds] for curr_preds in proph_preds_log]

#For the naive seasonal forecasts - add the log transform then scale
seas_avg_preds_log = [[curr_ts.map(np.log1p) for curr_ts in curr_preds] for curr_preds in seas_avg_preds]
seas_avg_preds_log_s = [den_eval_scaler.transform(curr_preds) for curr_preds in seas_avg_preds_log]

## 3.5.2 Post-process sensitivity 
#Same post-processing as above but we run it on the results of the sensitivity analysis, including the TFT
lr_sens_preds_log = [den_eval_scaler.inverse_transform(curr_preds) for curr_preds in lr_sens_preds_log_s]
proph_sens_preds_log = [den_eval_scaler.inverse_transform(curr_preds) for curr_preds in proph_sens_preds_log_s]
tft_sens_preds_log = [den_eval_scaler.inverse_transform(curr_preds) for curr_preds in tft_sens_preds_log_s]
tft_sens_preds_max_log = [den_eval_scaler.inverse_transform(curr_preds) for curr_preds in tft_sens_preds_max_log_s]


lr_sens_preds = [[curr_ts.map(np.expm1) for curr_ts in curr_preds] for curr_preds in lr_sens_preds_log]
proph_sens_preds = [[curr_ts.map(np.expm1) for curr_ts in curr_preds] for curr_preds in proph_sens_preds_log]
tft_sens_preds = [[curr_ts.map(np.expm1) for curr_ts in curr_preds] for curr_preds in tft_sens_preds_log]
tft_sens_preds_max = [[curr_ts.map(np.expm1) for curr_ts in curr_preds] for curr_preds in tft_sens_preds_max_log]

seas_avg_sens_preds_log = [[curr_ts.map(np.log1p) for curr_ts in curr_preds] for curr_preds in seas_avg_sens_preds]
seas_avg_sens_preds_log_s = [den_eval_scaler.transform(curr_preds) for curr_preds in seas_avg_sens_preds_log]

#%%
# 4. Outputting Baseline Forecasts

#Create some dictionaries to output the forecasts 
lr_preds_dict = {
    "preds": lr_preds,
    "preds_log": lr_preds_log,
    "preds_log_s": lr_preds_log_s
}
proph_preds_dict = {
    "preds": proph_preds,
    "preds_log": proph_preds_log,
    "preds_log_s": proph_preds_log_s
}
seas_avg_preds_dict = {
    "preds": seas_avg_preds,
    "preds_log": seas_avg_preds_log,
    "preds_log_s": seas_avg_preds_log_s
}

lr_sens_preds_dict = {
    "preds": lr_sens_preds,
    "preds_log": lr_sens_preds_log,
    "preds_log_s": lr_sens_preds_log_s
}

proph_sens_preds_dict = {
    "preds": proph_sens_preds,
    "preds_log": proph_sens_preds_log,
    "preds_log_s": proph_sens_preds_log_s
}

seas_avg_sens_preds_dict = {
    "preds": seas_avg_sens_preds,
    "preds_log": seas_avg_sens_preds_log,
    "preds_log_s": seas_avg_sens_preds_log_s
}

tft_sens_preds_dict = {
    "preds": tft_sens_preds,
    "preds_log": tft_sens_preds_log,
    "preds_log_s": tft_sens_preds_log_s
}

#Create a dict for the max TFT sensitivity forecasts
tft_sens_preds_max_dict = {
    "preds": tft_sens_preds_max,
    "preds_log": tft_sens_preds_max_log,
    "preds_log_s": tft_sens_preds_max_log_s
}

#Save the dicts as pickles to the output directory
hp.save_pickle(lr_preds_dict, os.path.join(output_dir, "lr_preds_dict.pkl"))
hp.save_pickle(proph_preds_dict, os.path.join(output_dir, "proph_preds_dict.pkl"))
hp.save_pickle(seas_avg_preds_dict, os.path.join(output_dir, "seas_avg_preds_dict.pkl"))

hp.save_pickle(lr_sens_preds_dict, os.path.join(output_dir, "lr_sens_preds_dict.pkl"))
hp.save_pickle(proph_sens_preds_dict, os.path.join(output_dir, "proph_sens_preds_dict.pkl"))
hp.save_pickle(seas_avg_sens_preds_dict, os.path.join(output_dir, "seas_avg_sens_preds_dict.pkl"))

hp.save_pickle(tft_sens_preds_dict, os.path.join(output_dir, "tft_sens_preds_dict.pkl"))
hp.save_pickle(tft_sens_preds_max_dict, os.path.join(output_dir, "tft_sens_preds_max_dict.pkl"))

# %%
