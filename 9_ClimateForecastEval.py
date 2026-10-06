#%%
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
output_dir = os.path.join(base_dir, "ClimEvalRes", tuning_name, model_name)
plots_output_dir = os.path.join(output_dir, "Plots")

bl_dir = os.path.join(base_dir, "ClimBaselines")

os.makedirs(plots_output_dir, exist_ok = True)
os.makedirs(output_dir, exist_ok = True)

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
## 3.3 Building list sets and loading baselines ----
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

seasonal_avg_dict = nhp.load_pickle(os.path.join(bl_dir, "seasonal_average.pkl"))



#%%
## 3.4 Computing Metrics ----
#Computing metrics - MASE, Nat MAE, MIN-MAX MAE
#We need two views from this
# (1) Location x metrics for overall + each individual den_tune_val - separate file per component
# (2) Components (inc. Overall) x metrics for overall + each individual den_tune_val

extras = {"m_steps": 1}
meta_metrics = ["mae"]
meta_norms = ["nat", "min_max", "naive_forecast_mae"]

den_tune_gap_val_mets = nhp.collect_set_metrics({"nat":den_tune_gap_val}, {"nat":den_tune_gap_val_preds}, normer_list = normer, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)
den_gap_val_mets = nhp.collect_set_metrics({"nat":den_gap_val}, {"nat":den_gap_val_preds}, normer_list = normer, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)
den_gap_targ_mets = nhp.collect_set_metrics({"nat":den_gap_targ}, {"nat":den_gap_targ_preds}, normer_list = normer, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)


bl_den_tune_gap_val_preds = seasonal_avg_dict["den_tune_gap_val_preds"]
bl_den_gap_val_preds = seasonal_avg_dict["den_gap_val_preds"]
bl_den_gap_targ_preds = seasonal_avg_dict["den_gap_targ_preds"]

bl_den_tune_gap_val_mets = nhp.collect_set_metrics({"nat":den_tune_gap_val}, {"nat":bl_den_tune_gap_val_preds}, normer_list = normer, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)
bl_den_gap_val_mets = nhp.collect_set_metrics({"nat":den_gap_val}, {"nat":bl_den_gap_val_preds}, normer_list = normer, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)
bl_den_gap_targ_mets = nhp.collect_set_metrics({"nat":den_gap_targ}, {"nat":bl_den_gap_targ_preds}, normer_list = normer, meta_metrics = meta_metrics, meta_norms = meta_norms, extras = extras, strict = False)



#%%
### 3.4.1 Loc x Mets Table ----
den_tune_gap_val_loc_mets = nhp.get_loc_mets_tables(den_tune_gap_val_mets, curr_eval_dict = {"obs": {"nat": den_tune_gap_val}}, admin_list = uf_name_list, admin_level_list = admin_level_list, clim_var_names = clim_var_names)
den_gap_val_loc_mets = nhp.get_loc_mets_tables(den_gap_val_mets, curr_eval_dict = {"obs": {"nat": den_gap_val}}, admin_list = uf_name_list, admin_level_list = admin_level_list, clim_var_names = clim_var_names)
den_gap_targ_loc_mets = nhp.get_loc_mets_tables(den_gap_targ_mets, curr_eval_dict = {"obs": {"nat": den_gap_targ}}, admin_list = uf_name_list, admin_level_list = admin_level_list, clim_var_names = clim_var_names)

bl_den_tune_gap_val_loc_mets = nhp.get_loc_mets_tables(bl_den_tune_gap_val_mets, curr_eval_dict = {"obs": {"nat": den_tune_gap_val}}, admin_list = uf_name_list, admin_level_list = admin_level_list, clim_var_names = clim_var_names)
bl_den_gap_val_loc_mets = nhp.get_loc_mets_tables(bl_den_gap_val_mets, curr_eval_dict = {"obs": {"nat": den_gap_val}}, admin_list = uf_name_list, admin_level_list = admin_level_list, clim_var_names = clim_var_names)
bl_den_gap_targ_loc_mets = nhp.get_loc_mets_tables(bl_den_gap_targ_mets, curr_eval_dict = {"obs": {"nat": den_gap_targ}}, admin_list = uf_name_list, admin_level_list = admin_level_list, clim_var_names = clim_var_names)

# #Add skill scores to each loc_mets table to help contextualise results - note this is per component
# curr_main = den_tune_gap_val_mets
# curr_bl = bl_den_tune_gap_val_mets
# comp_names = clim_var_names + ["Overall"]
# admin_list = uf_name_list

# ret_dict = {}
# # Goal here is return a dict where the values are lists of dataframes of length curr_main + 1, one for each ts_list (e.g. DenVal1) + an overall summary
# # The keys should be the component names + "Overall" for easy merging into the loc_mets tables
# # In each dataframe with have an admin column + a column for each skill score which is computed as 1 - ratio

# #We compute skill scores for each metric
# met_keys = list(curr_main.keys())
# curr_met_key = met_keys[0]

# curr_main_mets = curr_main[curr_met_key]
# curr_bl_mets = curr_bl[curr_met_key]

# #Ratios give per set, per admin, per component skill scores
# #
# curr_comp = comp_names[0]

# curr_ratios = curr_main_mets / curr_bl_mets


#%%
#Compile all the Loc x mets tables for outputting into Excel files
#Each one has keys based on the component name + Overall
comp_key_list = clim_var_names + ["Overall"]
set_names = ["DenTuneGapVal", "DenGapVal", "DenGapTarg"]
set_suffix = [1, 2, 3, "Overall"] #Assumed same across sets
sheet_names = []
for set_name in set_names:
    curr_sheet_names = [set_name + "_" + str(suffix) for suffix in set_suffix]
    sheet_names.extend(curr_sheet_names)

for comp_key in comp_key_list:
    den_tune_gap_val_sheets = den_tune_gap_val_loc_mets[comp_key]
    den_gap_val_sheets = den_gap_val_loc_mets[comp_key]
    den_gap_targ_sheets = den_gap_targ_loc_mets[comp_key]


    bl_den_tune_gap_val_sheets = bl_den_tune_gap_val_loc_mets[comp_key]
    bl_den_gap_val_sheets = bl_den_gap_val_loc_mets[comp_key]
    bl_den_gap_targ_sheets = bl_den_gap_targ_loc_mets[comp_key]
    bl_df_output_list = bl_den_tune_gap_val_sheets + bl_den_gap_val_sheets + bl_den_gap_targ_sheets
    df_output_list = den_tune_gap_val_sheets + den_gap_val_sheets + den_gap_targ_sheets

    #Output to an Excel file 
    output_path = os.path.join(output_dir, comp_key + "_LocMets.xlsx")
    nhp.df_list_to_excel(df_output_list, output_path, sheet_names = sheet_names)

    bl_output_path = os.path.join(output_dir, "BL_" + comp_key + "_LocMets.xlsx")
    nhp.df_list_to_excel(bl_df_output_list, bl_output_path, sheet_names = sheet_names)

#%%
### 3.4.2 Comps x Mets Table ----
#Here we create a table across components 
#With one component per row and metrics per column already averaged across locations
#Like the loc x mets tables, we add a sheet with the weights

def get_comps_mets_tables(curr_metric_dict, curr_eval_dict, comp_names, fold_weighted = True):
    met_keys = list(curr_metric_dict.keys())
    comp_list = comp_names + ["Mean", "Min", "Max"]
    to_ret = [{"Component": comp_list} for i in range(len(curr_eval_dict["obs"]["nat"]) + 1)]

    val_series_set = curr_eval_dict["obs"]["nat"]
    if fold_weighted:
        weights = [len(val_series_set[i][0]) for i in range(len(val_series_set))]
    else:
        weights = None

    for curr_met_key in met_keys:
        curr_mets = curr_metric_dict[curr_met_key] #This is a num_sets x num_admin x num_components array of metrics

        #First, we average the metrics across all locations (resulting in num_sets x num_components array)
        curr_mets_sets_comps = np.mean(curr_mets, axis=1) #Averaging across the admin/location axis
        num_sets = curr_mets_sets_comps.shape[0]

        for i in range(num_sets):
            curr_vals = curr_mets_sets_comps[i, :]
            curr_mean = np.mean(curr_vals)
            curr_min = np.min(curr_vals)
            curr_max = np.max(curr_vals)
            curr_summ = [curr_mean, curr_min, curr_max]
            
            to_ret[i][curr_met_key] =  np.concatenate([curr_vals, curr_summ])

        curr_mets_sets_comps_overall = np.average(curr_mets_sets_comps, axis=0, weights=weights)

        curr_mean = np.mean(curr_mets_sets_comps_overall)
        curr_min = np.min(curr_mets_sets_comps_overall)
        curr_max = np.max(curr_mets_sets_comps_overall)
        curr_summ = [curr_mean, curr_min, curr_max]

        to_ret[-1][curr_met_key] =  np.concatenate([curr_mets_sets_comps_overall, curr_summ])

    to_ret_dfs = [pd.DataFrame(curr) for curr in to_ret]
    return to_ret_dfs


den_tune_gap_val_comps_mets = get_comps_mets_tables(curr_metric_dict = den_tune_gap_val_mets, curr_eval_dict = {"obs": {"nat": den_tune_gap_val}}, comp_names = clim_var_names, fold_weighted = True)
den_gap_val_comps_mets = get_comps_mets_tables(curr_metric_dict = den_gap_val_mets, curr_eval_dict = {"obs": {"nat": den_gap_val}}, comp_names = clim_var_names, fold_weighted = True)
den_gap_targ_comps_mets = get_comps_mets_tables(curr_metric_dict = den_gap_targ_mets, curr_eval_dict = {"obs": {"nat": den_gap_targ}}, comp_names = clim_var_names, fold_weighted = True)

bl_den_tune_gap_val_comps_mets = get_comps_mets_tables(curr_metric_dict = bl_den_tune_gap_val_mets, curr_eval_dict = {"obs": {"nat": den_tune_gap_val}}, comp_names = clim_var_names, fold_weighted = True)
bl_den_gap_val_comps_mets = get_comps_mets_tables(curr_metric_dict = bl_den_gap_val_mets, curr_eval_dict = {"obs": {"nat": den_gap_val}}, comp_names = clim_var_names, fold_weighted = True)
bl_den_gap_targ_comps_mets = get_comps_mets_tables(curr_metric_dict = bl_den_gap_targ_mets, curr_eval_dict = {"obs": {"nat": den_gap_targ}}, comp_names = clim_var_names, fold_weighted = True)

#Output to Excel using the sheet names already built
comp_mets_df_list = den_tune_gap_val_comps_mets + den_gap_val_comps_mets + den_gap_targ_comps_mets
nhp.df_list_to_excel(comp_mets_df_list, os.path.join(output_dir, "CompMets.xlsx"), sheet_names = sheet_names)

bl_comp_mets_df_list = bl_den_tune_gap_val_comps_mets + bl_den_gap_val_comps_mets + bl_den_gap_targ_comps_mets
nhp.df_list_to_excel(bl_comp_mets_df_list, os.path.join(output_dir, "BL_CompMets.xlsx"), sheet_names = sheet_names)
#%%
#Creating plots 
# 4. Creating Plots ----

## 4.1 Per Set Plots
#Generate a plot for every validation and test set we generate climate forecasts over
#We cannot generate summary plots since some forecasts intersect when gaps are included (e.g.: gap before DenTuneVal2 intersects with DenTuneVal1)
select_admin = uf_name_list[:5]
input_sets = den_tune_train + den_train + den_hold
test_sets = den_tune_gap_val + den_gap_val + den_gap_targ
pred_sets = den_tune_gap_val_preds + den_gap_val_preds + den_gap_targ_preds
plot_names = [f"{curr_set}_{i}" for curr_set in set_names for i in range(3)]

for curr_ind in range(len(input_sets)):
    curr_input = input_sets[curr_ind]
    curr_test = test_sets[curr_ind]
    curr_pred = pred_sets[curr_ind]

    get_inds = [uf_name_list.index(admin) for admin in select_admin]
    curr_input_filt = [curr_input[i] for i in get_inds]
    curr_test_filt = [curr_test[i] for i in get_inds]
    curr_pred_filt = [curr_pred[i] for i in get_inds]

    #Flatten across components 
    def component_flatten(ts_list):
        to_ret = []
        comp_names = ts_list[0].components
        for curr_comp in comp_names:
            curr_comp_ts_list = [curr_ts[curr_comp] for curr_ts in ts_list]
            to_ret += curr_comp_ts_list
        return to_ret

    curr_input_filt_flat = component_flatten(curr_input_filt)
    curr_test_filt_flat = component_flatten(curr_test_filt)
    curr_pred_filt_flat = component_flatten(curr_pred_filt)

    num_comps = len(curr_input_filt[0].components)

    clim_var_disp_names = ["Min. Temp.", "Med. Temp.", "Max. Temp.", 
                        "Min. Precip.", "Med. Precip.", "Max. Precip.",
                        "Min. Pressure", "Med. Pressure", "Max. Pressure",
                        "Min. Rel. Hum.", "Med. Rel. Hum.", "Max. Rel. Hum."]
    curr_vis_dict = nhp.create_ts_dict_list(ts_lists = [curr_input_filt_flat, curr_test_filt_flat, curr_pred_filt_flat], 
                                            ts_names = ["Input", "Test", "Forecast"], unit_names = select_admin * num_comps)


    curr_fig, curr_ax = nhp.plot_dict_list(curr_vis_dict, low_quantile = 0.025, central_quantile = 0.5, high_quantile = 0.975, 
                    width = 16, height = 23, curr_fig_dims = (12,5), limit_dict = {"Input": 104})

    for ax, disp_name in zip(curr_ax[:,0], clim_var_disp_names):
        ax.set_ylabel(disp_name, fontsize = 14)

    curr_ax = curr_ax.flatten()
    handles, labels = curr_ax[0].get_legend_handles_labels()
    curr_fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol = 3, fontsize = 14)
    for ax in curr_ax:
        curr_leg = ax.get_legend()
        if curr_leg is not None:
            curr_leg.remove()

    curr_fig.tight_layout(pad=0.5)
    output_path = os.path.join(plots_output_dir, "ClimFC_" + plot_names[curr_ind] + ".png")  # Specify the desired output path
    curr_fig.savefig(output_path, dpi=300)

# %%
## 4.2 Summary Plots
den_tune_val_concat = nhp.ts_list_concatenate([den_tune_gap_val[0], den_tune_val[1], den_tune_gap_val[2]])
den_val_concat = nhp.ts_list_concatenate([den_gap_val[0], den_val[1], den_gap_val[2]])
den_targ_concat = nhp.ts_list_concatenate([den_gap_targ[0] + den_targ[1] + den_gap_targ[2]])

# %%
