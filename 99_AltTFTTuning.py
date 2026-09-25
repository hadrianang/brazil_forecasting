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
import random

torch.set_float32_matmul_precision("medium")
include_stringency_index = False
accel = "gpu"

quantiles = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

from IPython import get_ipython
ip = get_ipython()
if ip:
    print("Running in iPython")
    ip.run_line_magic("load_ext", "autoreload")
    ip.run_line_magic("autoreload", "1")
    ip.run_line_magic("aimport", "helpers")
    ip.run_line_magic("aimport", "new_helpers")

    #INPUTS - can be hardcoded or command line args
    den_set_name = "AltTFTTuning"
    den_model_name = "AltTFT_25_09_2026"
    random_seed = 3
    n_trials = 2
    gpu_id = 0
    progress_bars = True
    # set_name = "Base"
    # gpu_id = 7
    # accel = "gpu"
else:
    print("Running from command line")
    den_set_name = sys.argv[1]
    den_model_name = sys.argv[2]
    random_seed = int(sys.argv[3])
    n_trials = int(sys.argv[4])
    gpu_id = int(sys.argv[5])
    progress_bars = False

print("den_set_name: " + den_set_name)
print("den_model_name: " + den_model_name)
print("random_seed: " + str(random_seed))
print("n_trials: " + str(n_trials))
print("gpu_id: " + str(gpu_id))

if gpu_id < 0:
    accelerator = "cpu"
else:
    accelerator = "gpu"

import helpers as hp
import new_helpers as nhp

#Set torch prevision to speed things up
torch.set_float32_matmul_precision('medium')

#Set the seed
def set_seed (seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    #Set cuda seed, silently ignored if unavailable
    torch.cuda.manual_seed_all(seed)

set_seed(random_seed)

gpu_id = 0
tuning_name = den_model_name #Name of the tuning scenario to use

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
tuning_dir = os.path.join(base_dir, den_set_name, tuning_name)
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
train1a = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1a", to_float32 = True)
train1b = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1b", to_float32 = True)
train1c = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1c", to_float32 = True)

val1a = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1a", to_float32 = True)
val1b = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1b", to_float32 = True)
val1c = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1c", to_float32 = True)

train1 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1", to_float32 = True)
train2 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train2", to_float32 = True)
train3 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train3", to_float32 = True)
train4 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train4", to_float32 = True)

val1 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1", to_float32 = True)
val2 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val2", to_float32 = True)
val3 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val3", to_float32 = True)
val4 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val4", to_float32 = True)

hold1 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold1", to_float32 = True)
hold2 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold2", to_float32 = True)
hold3 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold3", to_float32 = True)
hold4 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold4", to_float32 = True)

targ1 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ1", to_float32 = True)
targ2 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ2", to_float32 = True)
targ3 = hp.create_ts_list(dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ3", to_float32 = True)

#Log transformed version of the cases TimeSeries for actual training
train1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1a", to_float32 = True)
train1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1b", to_float32 = True)
train1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1c", to_float32 = True)

gap1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1a", to_float32 = True)
gap1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1b", to_float32 = True)
gap1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1c", to_float32 = True)

val1a_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1a", to_float32 = True)
val1b_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1b", to_float32 = True)
val1c_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1c", to_float32 = True)


#Evaluation sets
train1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train1", to_float32 = True)
train2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train2", to_float32 = True)
train3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train3", to_float32 = True)
train4_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_train4", to_float32 = True)

gap1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap1", to_float32 = True)
gap2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap2", to_float32 = True)
gap3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap3", to_float32 = True)
gap4_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_gap4", to_float32 = True)

val1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val1", to_float32 = True)
val2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val2", to_float32 = True)
val3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val3", to_float32 = True)
val4_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_val4", to_float32 = True)

hold1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold1", to_float32 = True)
hold2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold2", to_float32 = True)
hold3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold3", to_float32 = True)
hold4_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_hold4", to_float32 = True)

targ1_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ1", to_float32 = True)
targ2_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ2", to_float32 = True)
targ3_log = hp.create_ts_list(proc_dengue_df, ["Cases"], group_col, uf_list, set_info = set_info, set_column = "den_targ3", to_float32 = True)



#Climate values to use as covariates 
clim_train1a_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1a", to_float32 = True)
clim_train1b_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1b", to_float32 = True)
clim_train1c_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1c", to_float32 = True)

clim_train1_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train1", to_float32 = True)
clim_train2_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train2", to_float32 = True)
clim_train3_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train3", to_float32 = True)
clim_train4_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_train4", to_float32 = True)

#Used for evaluation over test set
clim_hold1_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_hold1", to_float32 = True)
clim_hold2_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_hold2", to_float32 = True)
clim_hold3_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_hold3", to_float32 = True)
clim_hold4_log = hp.create_ts_list(proc_climate_df, clim_var_names, group_col, uf_list, set_info = set_info, set_column = "den_hold4", to_float32 = True)

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
## 2.2 Building sets ----

#Tune train
tune_train = [train1a, train1b, train1c] 
tune_train_log_sc = [train1a_log_sc, train1b_log_sc, train1c_log_sc]

#Main training
train = [train1, train2, train3]
train_log_sc = [train1_log_sc, train2_log_sc, train3_log_sc]

#Tune gap
tune_gap_log = [gap1a_log, gap1b_log, gap1c_log]

#Tune val and main val
tune_val = [val1a, val1b, val1c]
tune_val_log = [val1a_log, val1b_log, val1c_log]

val = [val1, val2, val3]
val_log = [val1_log, val2_log, val3_log]

#Hold
hold = [hold1, hold2, hold3]
hold_log_sc = [hold1_log_sc, hold2_log_sc, hold3_log_sc]

#Targ
targ = [targ1, targ2, targ3]
targ_log = [targ1_log, targ2_log, targ3_log]


#Covariates
clim_tune_train = [clim_train1a_log, clim_train1b_log, clim_train1c_log]
clim_train = [clim_train1_log, clim_train2_log, clim_train3_log]
clim_hold = [clim_hold1_log, clim_hold2_log, clim_hold3_log]

clim_tune_val_preds = [clim_preds_1a_log, clim_preds_1b_log, clim_preds_1c_log]
clim_val_preds = [clim_preds_1_log, clim_preds_2_log, clim_preds_3_log]
clim_targ_preds = [clim_preds_targ1_log, clim_preds_targ2_log, clim_preds_targ3_log]

clim_tune_val_preds = nhp.ts_list_set_conv_dtype(clim_tune_val_preds)
clim_val_preds = nhp.ts_list_set_conv_dtype(clim_val_preds)
clim_targ_preds = nhp.ts_list_set_conv_dtype(clim_targ_preds)

# %%
## 2.3 Scaling data ----

# First we scale all dengue data 
tune_train_log_s, den_tune_scalers = nhp.create_fit_scaler_set(tune_train_log_sc)
train_log_s, den_train_scalers = nhp.create_fit_scaler_set(train_log_sc)

tune_gap_log_s = nhp.scaler_transform_ts_list_set(tune_gap_log, den_tune_scalers)
tune_val_log_s = nhp.scaler_transform_ts_list_set(tune_val_log, den_tune_scalers)
val_log_s = nhp.scaler_transform_ts_list_set(val_log, den_train_scalers)

hold_log_s = nhp.scaler_transform_ts_list_set(hold_log_sc, den_train_scalers)
targ_log_s = nhp.scaler_transform_ts_list_set(targ_log, den_train_scalers)


#Scale the covariates
clim_tune_train_log_s, clim_tune_train_scalers = nhp.create_fit_scaler_set(clim_tune_train)
clim_train_log_s, clim_train_scalers = nhp.create_fit_scaler_set(clim_train)

clim_tune_val_preds_log_s = nhp.scaler_transform_ts_list_set(clim_tune_val_preds, clim_tune_train_scalers)
clim_val_preds_log_s = nhp.scaler_transform_ts_list_set(clim_val_preds, clim_train_scalers)
clim_hold_log_s = nhp.scaler_transform_ts_list_set(clim_hold, clim_train_scalers)
clim_targ_preds_log_s = nhp.scaler_transform_ts_list_set(clim_targ_preds, clim_train_scalers)

#Concatenate to create covariate sets 

covs_tune_train_log_s = nhp.ts_list_set_pair_concat(clim_tune_train_log_s, clim_tune_val_preds_log_s)
covs_train_log_s = nhp.ts_list_set_pair_concat(clim_train_log_s, clim_val_preds_log_s)
covs_eval_log_s = nhp.ts_list_set_pair_concat(clim_hold_log_s, clim_targ_preds_log_s)


# %%
# 3. Modelling ----

gap_length = 15
if den_tuning_config["val_metric"] == "mql":
    val_metric = mql
else:
    val_metric = mae

num_tune_models = len(tune_train_log_s)

val_lengths = [len(curr_val[0]) for curr_val in tune_val_log_s]
forecast_lengths = [gap_length + curr_val_length for curr_val_length in val_lengths]

def build_tft_model(model_name, batch_size, input_chunk_length, output_chunk_length, hidden_size, hidden_continuous_size, lstm_layers, num_attention_heads, dropout, learning_rate, output_chunk_shift = 0, force_reset = True):
    torch.manual_seed(random_seed)
    curr_encoders = {"datetime_attribute": {"future": ["month", "year"]}, 
                    "transformer": Scaler()}
    early_stopper = EarlyStopping("val_loss", min_delta = 0.0001, patience = 8)
    pl_trainer_kwargs = {"accelerator": accel,
                             "devices": [gpu_id],
                             "callbacks": [early_stopper]}

    if not progress_bars:
        pl_trainer_kwargs["enable_progress_bar"] = False

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
        n_epochs = 100,
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

# Now that the data has been scaled and covariate sets have been created, we can proceed with modelling.
#%%
## 3.1 Objective function
test_study = optuna.create_study()
trial = test_study.ask()

model_name = "BrazilModel_" + den_tuning_config["ModelSuffix"] + "_" + str(den_model_name) + "_" + str(trial._trial_id)
batch_size = trial.suggest_categorical("batch_size", den_tuning_config["batch_size"])
input_chunk_length = trial.suggest_int("input_chunk_length", den_tuning_config["input_chunk_length_low"], den_tuning_config["input_chunk_length_high"])

output_chunk_length_config = den_tuning_config["output_chunk_length_high"]
if output_chunk_length_config[0] == "min_vs_input":
    #We can set the output_chunk_length upper end to ensure it is smaller than the proposed input_chunk_length 
    output_chunk_length_high = min(input_chunk_length, output_chunk_length_config[1])
else: #If not labelled as min_vs_input, we take the value of output_chunk_length_high directly and the entire range is sampled (not dependent on input_chunk_length for the trial) 
    output_chunk_length_high = output_chunk_length_config[1]

#If output_chunk_length_high is not fixed, then we have to tune it 
if output_chunk_length_config[0] != "fixed":
    output_chunk_length = trial.suggest_int("output_chunk_length", den_tuning_config["output_chunk_length_low"], output_chunk_length_high)
else:
    #Rather than tuning output_chunk_length, if the config file says it's fixed, the value is directly taken from output_chunk_length_high and the "low" value is ignored
    output_chunk_length = output_chunk_length_high

hidden_size = trial.suggest_categorical("hidden_size", den_tuning_config["hidden_size"])
hidden_continuous_size = hidden_size #We assume that hidden_continuous_size is the same as hidden_size
lstm_layers = trial.suggest_int("lstm_layers",  den_tuning_config["lstm_layers_low"], den_tuning_config["lstm_layers_high"])
num_attention_heads = trial.suggest_categorical("num_attention_heads",  den_tuning_config["num_attention_heads"])
dropout = trial.suggest_float("dropout",  den_tuning_config["dropout_low"],  den_tuning_config["dropout_high"])
learning_rate = trial.suggest_float("learning_rate",  den_tuning_config["learning_rate_low"],  den_tuning_config["learning_rate_high"], log = True)
output_chunk_shift = 0


tune_models = []
for i in range(num_tune_models):
    start_model = build_tft_model(model_name = model_name + "_" + str(i),
                                batch_size = batch_size,
                                input_chunk_length = input_chunk_length,
                                output_chunk_length = output_chunk_length, 
                                output_chunk_shift = output_chunk_shift,
                                hidden_size = hidden_size,
                                hidden_continuous_size = hidden_size,
                                lstm_layers = lstm_layers,
                                num_attention_heads = num_attention_heads,
                                dropout = dropout,
                                learning_rate = learning_rate
                            )
    tune_models.append(start_model)

dyn_val_log_sc_s = []
for curr_tune_train, curr_tune_gap, curr_tune_val in zip(tune_train_log_s, tune_gap_log_s, tune_val_log_s):
    curr_dyn_val = []
    for k in range(len(curr_tune_train)):
        curr_tune_train_ts = curr_tune_train[k]
        curr_tune_gap_ts = curr_tune_gap[k]
        curr_tune_val_ts = curr_tune_val[k]
        
        curr_dyn_val_ts = concatenate([curr_tune_train_ts[-(input_chunk_length):], curr_tune_gap_ts, curr_tune_val_ts])
        curr_dyn_val.append(curr_dyn_val_ts)

    dyn_val_log_sc_s.append(curr_dyn_val)


#Fit the models
for i in range(num_tune_models):
    curr_tune_train = tune_train_log_s[i]
    curr_dyn_val = dyn_val_log_sc_s[i]
    curr_covs = covs_tune_train_log_s[i]
    tune_models[i].fit(series = curr_tune_train, val_series = curr_dyn_val, future_covariates = curr_covs, val_future_covariates = curr_covs)

#Load the best models
best_tune_models = [curr_model.load_from_checkpoint(curr_model.model_name, best = True) for curr_model in tune_models]

#Generate predictions for the validation set
tune_preds_log_s = []
for i in range(num_tune_models):
    curr_tune_train = tune_train_log_s[i]
    curr_forecast_length = forecast_lengths[i]
    curr_preds = best_tune_models[i].predict(n = curr_forecast_length, series = curr_tune_train, future_covariates= curr_covs, num_samples = 200)
    tune_preds_log_s.append(curr_preds)
    
#Invert the scaling of predictions - back to log scale
tune_preds_log = nhp.scaler_inverse_transform_ts_list_set(tune_preds_log_s, den_tune_scalers)
tune_preds = nhp.map_func_ts_set(tune_preds_log, func = np.expm1)

#Add some info to the trial on epochs trained and the random seed

epochs_trained = [curr_model.epochs_trained for curr_model in tune_models]
trial.set_user_attr("epochs_trained", epochs_trained)
trial.set_user_attr("random_seed", random_seed)

val_losses = []
for curr_val, curr_preds in zip(tune_val, tune_preds):
    curr_loss = hp.norm_metric_list(curr_val, curr_preds, train1a, series_summ_func = np.mean, metric = val_metric, quantiles = quantiles)
    val_losses.append(curr_loss)

trial.set_user_attr("val_losses", val_losses)

val_losses = np.array(val_losses)
val_loss = np.mean(val_losses)



#%%
## 3.2 Tuning Setup
# storage = JournalStorage(
#     JournalFileBackend(os.path.join(tuning_dir, den_tuning_config["OptunaLog"]))
# )

# multivariate_tpe = False
# constant_liar_tpe = False

# if("Multivariate_TPE" in den_tuning_config):
#     multivariate_tpe = den_tuning_config["Multivariate_TPE"]
# if("ConstantLiar_TPE" in den_tuning_config):
#     constant_liar_tpe = den_tuning_config["ConstantLiar_TPE"]


# #Create a sampler 
# tuning_sampler = TPESampler(seed = random_seed, multivariate = multivariate_tpe, constant_liar = constant_liar_tpe)

# #We then create the Optuna study, or load it if it exists
# tuning_study = optuna.create_study(
#     study_name = den_tuning_config["StudyName"],
#     direction = "minimize",
#     sampler = tuning_sampler,
#     storage = storage,
#     load_if_exists = True
# )