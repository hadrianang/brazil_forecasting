#%%

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

rd1_dir = os.path.join(tuning_dir, "Results")
rd2_dir = os.path.join(tuning_dir, "Results_Round2")

climate_tuning_dir = os.path.join(base_dir, "Climate Tuning_Round2", "Notebook_24_07_2025")
climate_model_name = "RandomForest"
climate_forecast_dir = os.path.join(climate_tuning_dir, climate_model_name, "Forecasts_Round2")

baseline_dir = os.path.join(base_dir, "Baselines")
output_dir = os.path.join(base_dir, "DenEvalRes")

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

admin_level_list = ["State" for curr_uf in uf_list]


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

hold = [hold1, hold2, hold3]
hold_log_sc = [hold1_log_sc, hold2_log_sc, hold3_log_sc]
hold_log_sc_s = [hold1_log_sc_s, hold2_log_sc_s, hold3_log_sc_s]

targ = [targ1, targ2, targ3]
targ_log = [targ1_log, targ2_log, targ3_log]
targ_log_s = [targ1_log_s, targ2_log_s, targ3_log_s]

clim_hold_targ_log_s = [clim_hold_targ1_log_s, clim_hold_targ2_log_s, clim_hold_targ3_log_s]

#%%
# 3. Reading forecasts to evaluate ----
#Read pickles with the main forecast values
tag = "GlobalFit=False_TrainFromScratch=False"
tag_multi = "GlobalFit=False_TrainFromScratch=False_MultiClimate=True"
tag_single = "GlobalFit=False_TrainFromScratch=False_MultiClimate=False"
main_fc_multi = hp.load_pickle(os.path.join(rd2_dir, f"Targ_{tag_multi}.pkl"))
main_fc_single = hp.load_pickle(os.path.join(rd2_dir, f"Targ_{tag_single}.pkl"))

#%%
#Read the baseline forecasts as well
lr_preds_dict = hp.load_pickle(os.path.join(baseline_dir, "lr_preds_dict.pkl"))
proph_preds_dict = hp.load_pickle(os.path.join(baseline_dir, "proph_preds_dict.pkl"))
seas_avg_preds_dict = hp.load_pickle(os.path.join(baseline_dir, "seas_avg_preds_dict.pkl"))

lr_sens_preds_dict = hp.load_pickle(os.path.join(baseline_dir, "lr_sens_preds_dict.pkl"))
proph_sens_preds_dict = hp.load_pickle(os.path.join(baseline_dir, "proph_sens_preds_dict.pkl"))
seas_avg_sens_preds_dict = hp.load_pickle(os.path.join(baseline_dir, "seas_avg_sens_preds_dict.pkl"))

tft_sens_preds_dict = hp.load_pickle(os.path.join(baseline_dir, "tft_sens_preds_dict.pkl"))
tft_sens_preds_max_dict = hp.load_pickle(os.path.join(baseline_dir, "tft_sens_preds_max_dict.pkl"))


#%%

#Main model values
main_preds_dict = main_fc_multi

main_eval_scaler = den_eval_scaler
main_preds = [main_preds_dict["targ1_preds"], main_preds_dict["targ2_preds"], main_preds_dict["targ3_preds"]]
main_preds_log_s = [main_preds_dict["targ1_preds_log_s"], main_preds_dict["targ2_preds_log_s"], main_preds_dict["targ3_preds_log_s"]]
main_preds_log = nhp.transform_comps_ts_list_set(main_preds, comps_to_trans = ["Cases"], trans_func = np.log1p)

#%% 
#Define a function that returns the set dict with nat, log, scaled keys
def get_set_dict(preds_dict):
    #Assumes preds, preds_log, preds_log_s keys
    set_dict = {
        "nat": preds_dict["preds"],
        "log": preds_dict["preds_log"],
        "scaled": preds_dict["preds_log_s"]
    }
    return set_dict


obs_list_set_dict = {"nat": targ, "log": targ_log, "scaled": targ_log_s}
main_preds_set_dict = {"nat": main_preds, "log": main_preds, "scaled": main_preds_log_s}

lr_preds_set_dict = get_set_dict(lr_preds_dict)
proph_preds_set_dict = get_set_dict(proph_preds_dict)
seas_avg_preds_set_dict = get_set_dict(seas_avg_preds_dict)

lr_sens_preds_set_dict = get_set_dict(lr_sens_preds_dict)
proph_sens_preds_set_dict = get_set_dict(proph_sens_preds_dict)
seas_avg_sens_preds_set_dict = get_set_dict(seas_avg_sens_preds_dict)

tft_sens_preds_set_dict = get_set_dict(tft_sens_preds_dict)
tft_sens_preds_max_set_dict = get_set_dict(tft_sens_preds_max_dict)


#%%
# 4. Computing Error metrics ----
# To compute error metrics, we compare values to Targ1, 2, and 3
# Note that targ3 is shorter than the others due to the time when the competition ran
# 
#collect_set_metrics(obs_list_set_dict, preds_list_set_dict, normer_list = None, comp_summ_func = None, series_summ_func = None, meta_metrics = ["mae"], meta_norms = ["min_max"], manual_log = True, extras = {}, strict = True)


normer_list = train1
meta_metrics = ["mcrps", "mae", "mic"]
meta_metrics_det = ["mcrps", "mae"]
meta_norms = ["min_max", "log", "nat", "scaled"]
extras = {"quantiles": quantiles, "mic_q_interval_list": [[0.025, 0.975], [0.25, 0.75]]}

main_mets = nhp.collect_set_metrics(obs_list_set_dict, main_preds_set_dict, normer_list = normer_list, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)
lr_mets = nhp.collect_set_metrics(obs_list_set_dict, lr_preds_set_dict, normer_list = normer_list, meta_metrics = meta_metrics_det, meta_norms = meta_norms, extras = extras, strict = False)
proph_mets = nhp.collect_set_metrics(obs_list_set_dict, proph_preds_set_dict, normer_list = normer_list, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)
seas_avg_mets = nhp.collect_set_metrics(obs_list_set_dict, seas_avg_preds_set_dict, normer_list = normer_list, meta_metrics = meta_metrics_det, meta_norms = meta_norms, extras = extras, strict = False)

#No sens_mets for Prophet, which requires that predictions immediately follow the training series
lr_sens_mets = nhp.collect_set_metrics(obs_list_set_dict, lr_sens_preds_set_dict, normer_list = normer_list, meta_metrics = meta_metrics_det, meta_norms = meta_norms, extras = extras, strict = False)
# proph_sens_mets = nhp.collect_set_metrics(obs_list_set_dict, proph_sens_preds_set_dict, normer_list = normer_list, meta_metrics = meta_metrics_det, meta_norms = meta_norms, extras = extras, strict = False)
seas_avg_sens_mets = nhp.collect_set_metrics(obs_list_set_dict, seas_avg_sens_preds_set_dict, normer_list = normer_list, meta_metrics = meta_metrics_det, meta_norms = meta_norms, extras = extras, strict = False)

tft_sens_mets = nhp.collect_set_metrics(obs_list_set_dict, tft_sens_preds_set_dict, normer_list = normer_list, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)
tft_sens_max_mets = nhp.collect_set_metrics(obs_list_set_dict, tft_sens_preds_max_set_dict, normer_list = normer_list, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)

#%% Compute the MCRPSS versus each baseline model
model_mets_dicts = {
    "MainTFT": main_mets,
    "LR": lr_mets,
    "Prophet": proph_mets,
    "SeasAvg": seas_avg_mets,
    "LR_Sens": lr_sens_mets,
    "SeasAvg_Sens": seas_avg_sens_mets,
    "TFT_Sens": tft_sens_mets,
    "TFT_Sens_Max": tft_sens_max_mets
}

main_eval_dict = {
    "obs": obs_list_set_dict,
}

loc_mets_dict = {curr_key: nhp.get_den_loc_mets_tables(model_mets_dicts[curr_key], main_eval_dict, admin_level_list, uf_name_list, fold_weighted = True) for curr_key in model_mets_dicts.keys()}
#%%
#Compute skill scores to add to each of the TFTs - main, tft_sens, tft_sens_max versus every other model


def trunc_intersect(obs_set, preds_set):
    #Truncate predictions to get only those that overlap with the observed values
    #Assumes both are ts_list_sets
    
    ts_list_set = []
    for curr_obs_list, curr_preds_list in zip(obs_set, preds_set):
        ts_list = [] 
        for curr_obs, curr_preds in zip(curr_obs_list, curr_preds_list):
            curr_preds_trunc = curr_preds.slice_intersect(curr_obs)
            ts_list.append(curr_preds_trunc)
        ts_list_set.append(ts_list)

    return ts_list_set

def trunc_intersect_set_dict(obs_set_dict, preds_set_dict):
    key_list = obs_set_dict.keys()
    to_ret = {}
    for curr_key in key_list:
        to_ret[curr_key] = trunc_intersect(obs_set_dict[curr_key], preds_set_dict[curr_key])
    return to_ret

def ts_list_set_dict_collapse(preds_set_dict):
    key_list = preds_set_dict.keys()
    to_ret = {}
    for curr_key in key_list:
        to_ret[curr_key] = nhp.ts_list_set_collapse(preds_set_dict[curr_key])
    return to_ret


#Sanity check 
# main_preds_set_dict_trunc = trunc_intersect_set_dict(obs_list_set_dict, main_preds_set_dict)



# #Concatenate all the main predictions 
# main_preds_set_dict_concat = ts_list_set_dict_collapse(main_preds_set_dict_trunc)
# obs_preds_set_dict_concat = ts_list_set_dict_collapse(obs_list_set_dict)

# curr_key = "nat"
# curr_ind = 0 
# curr_preds = main_preds_set_dict_concat[curr_key][curr_ind]
# curr_obs = obs_preds_set_dict_concat[curr_key][curr_ind]


# curr_mcrps = mcrps(curr_obs, curr_preds)

#%%
#Write a function that accepts one main prediction mets dict and multiple baselines (with names) then outputs
#a DataFrame containing the skill scores for each location and baseline model together with mean, min, and max summary rows

def compute_skill_scores(curr_main_mets, baselines, base_score_names, weights, curr_main_name = None):
    baselines = baselines.copy() 
    if curr_main_name is not None and curr_main_name in baselines:
        baselines.pop(curr_main_name)
    
    #Get number of sets 
    num_sets = curr_main_mets[base_score_names[0]].shape[0]
    baseline_names = list(baselines.keys())

    #The return is a list of DataFrames of length num_sets + 1 containing
    #skill scores per set and overall
    to_ret = [{} for i in range(num_sets + 1)]

    #For each base score and other baseline model
    for curr_base_score in base_score_names:

        for curr_baseline_key in baseline_names:
            curr_score_name = curr_base_score + "_" + curr_baseline_key + "_skill"
            curr_baseline_mets = baselines[curr_baseline_key]
            curr_baseline_score = curr_baseline_mets[curr_base_score]
            curr_main_score = curr_main_mets[curr_base_score]

            # Compute the skill scores for each set
            curr_score_ratios = curr_main_score / curr_baseline_score 

            #Add the skill scores to the return dictionaries
            for i in range(num_sets):
                curr_set_ratios = curr_score_ratios[i,]

                #Compute geometric mean 
                curr_set_gmean = gmean(curr_set_ratios)
                curr_set_min = np.min(curr_set_ratios)
                curr_set_max = np.max(curr_set_ratios)
                summ_arr = np.array([curr_set_gmean, curr_set_min, curr_set_max])

                curr_set_ratios = np.append(curr_set_ratios, summ_arr)
                
                to_ret[i][curr_score_name] = 1 - curr_set_ratios

            #Compute the overall values across all sets 

            curr_main_means = np.average(curr_main_score, axis=0, weights = weights)
            curr_baseline_means = np.average(curr_baseline_score, axis=0, weights = weights)
            curr_overall_ratios = curr_main_means / curr_baseline_means

            curr_overall_gmean = gmean(curr_overall_ratios)
            curr_overall_min = np.min(curr_overall_ratios)
            curr_overall_max = np.max(curr_overall_ratios)
            curr_overall_summ = np.array([curr_overall_gmean, curr_overall_min, curr_overall_max])

            curr_overall_ratios = np.append(curr_overall_ratios, curr_overall_summ)

            to_ret[-1][curr_score_name] = 1 - curr_overall_ratios

    to_ret_dfs = [pd.DataFrame(curr_dict) for curr_dict in to_ret]
    return to_ret_dfs




weights = [len(curr_ts_list[0]) for curr_ts_list in targ]
base_score_names = ["nat_mcrps", "log_mcrps"]

curr_model_names = list(loc_mets_dict.keys())
final_mets_dict = {}

for curr_model_name in curr_model_names:
    curr_loc_mets = loc_mets_dict[curr_model_name]
    curr_main_mets = model_mets_dicts[curr_model_name]
    curr_skill_scores = compute_skill_scores(curr_main_mets = curr_main_mets,
                                            baselines = model_mets_dicts, 
                                            base_score_names = base_score_names, 
                                            weights = weights, 
                                            curr_main_name = curr_model_name)

    curr_final_mets = []
    #Merge the skill scores into the current loc mets tables
    for i in range(len(curr_loc_mets["Overall"])):
        temp = pd.concat([curr_loc_mets["Overall"][i], curr_skill_scores[i]], axis = 1)
        curr_final_mets.append(temp)

    final_mets_dict[curr_model_name] = curr_final_mets


curr_sheet_names = ["Targ1", "Targ2", "Targ3", "TargOverall"]
for curr_model_name in curr_model_names:
    curr_output_dir = os.path.join(output_dir, curr_model_name)
    os.makedirs(curr_output_dir, exist_ok=True)

    curr_output_path = os.path.join(curr_output_dir, f"{curr_model_name}_Mets.xlsx")
    nhp.df_list_to_excel(df_list = final_mets_dict[curr_model_name], 
                        sheet_names = curr_sheet_names,
                        path = curr_output_path)




#%%
# 5. Generating Visuals ---- 
# obs_list_set_dict = {"nat": targ, "log": targ_log, "scaled": targ_log_s}
# main_preds_set_dict = {"nat": main_preds, "log": main_preds, "scaled": main_preds_log_s}


# lr_preds_set_dict = get_set_dict(lr_preds_dict)
# proph_preds_set_dict = get_set_dict(proph_preds_dict)
# seas_avg_preds_set_dict = get_set_dict(seas_avg_preds_dict)

# lr_sens_preds_set_dict = get_set_dict(lr_sens_preds_dict)
# proph_sens_preds_set_dict = get_set_dict(proph_sens_preds_dict)
# seas_avg_sens_preds_set_dict = get_set_dict(seas_avg_sens_preds_dict)

# tft_sens_preds_set_dict = get_set_dict(tft_sens_preds_dict)
# tft_sens_preds_max_set_dict = get_set_dict(tft_sens_preds_max_dict)


vis_key = "nat"
model_preds_dict = {
    "MainTFT": main_preds_set_dict,
    "LR": lr_preds_set_dict,
    "Prophet": proph_preds_set_dict,
    "SeasAvg": seas_avg_preds_set_dict,
    "LR_Sens": lr_sens_preds_set_dict,
    "SeasAvg_Sens": seas_avg_sens_preds_set_dict,
    "TFT_Sens": tft_sens_preds_set_dict,
    "TFT_Sens_Max": tft_sens_preds_max_set_dict
}
input_set_dict = {"nat": hold, "log": hold_log_sc, "scaled": hold_log_sc_s}

# For each model, we generate 2 kinds of visuals - one per individual target showing the input, test set and forecast 
# One showing concatenated targets and forecasts
# 

## 5.1 Targ-specific Visuals ----
curr_model_names = list(model_preds_dict.keys())
num_sets = len(input_set_dict[vis_key])
for curr_model_name in curr_model_names:
    for curr_ind in range(num_sets):
        curr_input = input_set_dict[vis_key][curr_ind]
        curr_targ = obs_list_set_dict[vis_key][curr_ind]
        curr_preds = model_preds_dict[curr_model_name][vis_key][curr_ind]

        curr_ts_dict_list = nhp.create_ts_dict_list([curr_input, curr_targ, curr_preds], ["Input", "Test", "Forecast (95% UI)"], uf_name_list)

        curr_fig, curr_ax = nhp.plot_dict_list(curr_ts_dict_list, low_quantile = 0.025, central_quantile = 0.5, high_quantile = 0.975, 
                        width = 12, height = 16.97, curr_fig_dims = None, limit_dict = {"Input": 104})

        curr_ax = curr_ax.flatten()
        for ax in curr_ax:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        handles, labels = curr_ax[0].get_legend_handles_labels()
        curr_fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol = 3, fontsize = 14)
        for ax in curr_ax:
            curr_leg = ax.get_legend()
            if curr_leg is not None:
                curr_leg.remove()

        
        curr_fig.delaxes(curr_ax[-2])
        curr_fig.delaxes(curr_ax[-1])

        curr_fig.tight_layout(pad=0.5)
        curr_fig_name = f"{curr_model_name}_Targ{curr_ind}.png"
        curr_output_path = os.path.join(output_dir, curr_model_name, curr_fig_name)
        curr_fig.savefig(curr_output_path, bbox_inches='tight', dpi=300)


#%%
## 5.2 Overall Visuals ----
curr_model_names = list(model_preds_dict.keys())
vis_key = "nat"

for curr_model_name in curr_model_names:
    curr_input = input_set_dict[vis_key]
    curr_targ = obs_list_set_dict[vis_key]
    curr_preds = model_preds_dict[curr_model_name][vis_key]

    line_dates = [curr_targ[1][0].start_time(), curr_targ[2][0].start_time()]
    curr_targ_collapse = nhp.ts_list_set_collapse(curr_targ)
    curr_preds_trunc = trunc_intersect(curr_targ, curr_preds)
    curr_preds_trunc_collapse = nhp.ts_list_set_collapse(curr_preds_trunc)

    curr_ts_dict_list = nhp.create_ts_dict_list([curr_targ_collapse, curr_preds_trunc_collapse], ["Test", "Forecast (95% UI)"], uf_name_list)
    curr_fig, curr_ax  = nhp.plot_dict_list(curr_ts_dict_list, low_quantile = 0.025, central_quantile = 0.5, high_quantile = 0.975, 
                    width = 12, height = 16.97, curr_fig_dims = None, limit_dict = {}, colour_dict = {"Test": "C1", "Forecast (95% UI)": "C2"})

    curr_ax = curr_ax.flatten()

    handles, labels = curr_ax[0].get_legend_handles_labels()
    curr_fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol = 3, fontsize = 14)
    for ax in curr_ax:
        curr_leg = ax.get_legend()
        if curr_leg is not None:
            curr_leg.remove()

    for ax in curr_ax:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    for ax in curr_ax:
        ax.axvline(x=line_dates[0], color='gray', linestyle='--', linewidth=1)
        ax.axvline(x=line_dates[1], color='gray', linestyle='--', linewidth=1)

    curr_fig.delaxes(curr_ax[-2])
    curr_fig.delaxes(curr_ax[-1])

    curr_fig.tight_layout(pad=0.5)
    curr_fig_name = f"{curr_model_name}_Overall.png"
    curr_output_path = os.path.join(output_dir, curr_model_name, curr_fig_name)
    curr_fig.savefig(curr_output_path, bbox_inches='tight', dpi=300)


