# helpers.py - Helper functions for all scripts
import pandas as pd
import numpy as np
from scipy.stats import gmean
import torch 
import os
import json
import pickle

from darts import TimeSeries, concatenate
from darts.utils.callbacks import TFMProgressBar
from darts.metrics import mape, smape, mse, rmse, ae, mae, mql, ql, miw, miws, crps, mcrps, ic, mic
from scoringrules import interval_score, weighted_interval_score

from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

def read_json(path):
    """
    Function reads a json file in a given path and returns it as a dictionary
    
    Parameters:
    path (string): Path to the json file
    
    Returns:
    data (dict): Dictionary containing the json data
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_dict_as_json(curr_json, path):
    """
    Function saves a dictionary as a json file in a given path
    
    Parameters:
    curr_json (dict): Dictionary to be saved as json
    path (string): Path to save the json file
    """
    with open(path, "w") as f:
        json.dump(curr_json, f, indent = 4)

def save_pickle(to_save, save_path):
    with open(save_path, "wb") as file:
        pickle.dump(to_save, file)

def load_pickle(load_path):
    to_ret = None 
    with open(load_path, "rb") as file:
        to_ret = pickle.load(file)
    return(to_ret)

def create_ts(data_df, val_cols, set_info = None, set_column = None, time_col = "WeekMid", time_format = "%Y-%m-%d", col_names = None):
    """
    Function creates a TimeSeries object from a DataFrame using only rows where set_column is true

    Parameters:
    data_df (DataFrame): DataFrame containing TimeSeries data (assumed in chronological order)
    val_cols (list or string): Name(s) of the column(s) in data_df containing the values to make a TimeSeries (will be multivariate if a list with more than 1 element)
    set_info (DataFrame): DataFrame to merge into data_df (assumed with Year, Week, and epiweek columns) containg set information
    set_column (string): Name of the column containing booleans to be used for inclusion/exclusion in the output TimeSeries
    time_col (string): Name of the column to be used as the time-axis of the output series
    time_format (string): Format of the time column

    Returns:
    curr_ts: TimeSeries filtered based on set_column with values based on val_cols
    """
    data_df = data_df.copy()
    data_df[time_col] = pd.to_datetime(data_df[time_col], format = time_format)
    data_df = data_df.set_index(time_col)

    if set_info is not None:
        #We assume the set_info DataFrame has Year, Week, and epiweek columns we can join on 
        data_df = pd.merge(data_df, set_info, how = "left", on = ["Year", "Week", "epiweek"]) 
    if set_column is not None:
        data_df = data_df[data_df[set_column]]
    if col_names is None:
        col_names = val_cols
    curr_ts = TimeSeries.from_times_and_values(times = data_df.index, values = data_df[val_cols], columns = col_names).astype(np.float32)
    return curr_ts


def create_ts_list(data_df, val_cols, group_col, group_vals, set_info = None, set_column = None, time_col = "WeekMid", time_format = "%Y-%m-%d", unit_as_col_name = False):
    """
    Function creates a list TimeSeries objects from a DataFrame. Each list element is a TimeSeries for a particular unit from group_vals, based on the group_col

    Parameters:
    data_df (DataFrame): DataFrame containing TimeSeries data (assumed in chronological order)
    val_cols (list or string): Name(s) of the column(s) in data_df containing the values to make a TimeSeries (will be multivariate if a list with more than 1 element)
    group_col (string): Name of the column to match with the units from group_vals
    group_vals (list): List of values in the group_col to make a TimeSeries for. The list returned by this function will have a TimeSeries for each element in group_vals.
    set_info (DataFrame): DataFrame to merge into data_df (assumed with Year, Week, and epiweek columns) containg set information
    set_column (string): Name of the column containing booleans to be used for inclusion/exclusion in the output TimeSeries
    time_col (string): Name of the column to be used as the time-axis of the output series
    time_format (string): Format of the time column

    Returns:
    to_ret: list of TimeSeries, one for each element in group_vals
    """
    data_df = data_df.copy()
    to_ret = []
    if set_info is not None:
        data_df = pd.merge(data_df, set_info, how = "left", on = ["Year", "Week", "epiweek"])
        
    for curr_unit in group_vals:
        curr_unit_df = data_df[data_df[group_col] == curr_unit]
        if unit_as_col_name:
            curr_unit_ts = create_ts(curr_unit_df, val_cols, set_column = set_column, time_col = time_col, 
                                     time_format = time_format, col_names = curr_unit)
        else:
            curr_unit_ts = create_ts(curr_unit_df, val_cols, set_column = set_column, time_col = time_col, time_format = time_format) #No need to pass set_info here since we have already merged in this function
        to_ret.append(curr_unit_ts)
    return to_ret

def create_ts_list_set(data_df, val_cols, group_col, group_vals,
                      set_info = None, set_cols = None,
                      time_col = "WeekMid", time_format = "%Y-%m-%d", 
                      unit_as_col_name = False):
    """
    Function creates a set of TimeSeries lists using the create_ts_list function.
    These sets are based on the set_cols, a list of set_columns passed to create_ts_list. This function then
    returns a list where each element is a list of TimeSeries, each corresponding to one set_column from 
    the values of set_cols.
    """
    ts_set = []
    for curr_set_col in set_cols:
        curr_ts_list = create_ts_list(data_df, val_cols, group_col, group_vals, 
                                      set_info = set_info, set_column = curr_set_col, 
                                      time_col = time_col, time_format = time_format, 
                                      unit_as_col_name = unit_as_col_name)
        ts_set.append(curr_ts_list)
    return ts_set
    
# def create_fit_scaler_set(ts_list_set, global_scale = False, norm_mode = "min_max", scale = True):
#     """
#     Creates a list of scalers, one for each list of TimeSeries in the ts_list_set.
#     The scalers are then fitted to each of the corresponding lists. If scale is True
#     then this returns both a scaled version of the set and the scalers themselves. If
#     scale if False, then only the fitted scalers are returned.
#     """
#     if norm_mode == "min_max":
#         scalers = [Scaler(global_fit = global_scale) for curr_ts_list in ts_list_set]
#     elif norm_mode == "std":
#         scalers = [Scaler(scaler = StandardScaler(), global_fit = global_scale) for curr_ts_list in ts_list_set]
#     else:
#         raise ValueError("Invalid norm_mode found while creating scaler set")
#     ts_list_set_s = []
#     for curr_ts_list, curr_scaler in zip(ts_list_set, scalers):
#         curr_scaler.fit(curr_ts_list)
#         if scale:
#             curr_ts_list_s = curr_scaler.transform(curr_ts_list)
#             ts_list_set_s.append(curr_ts_list_s)
#     if scale:
#         return ts_list_set_s, scalers
#     else:
#         return scalers

def split_ts_list(ts_list, num_moh, num_dist):
    """
    Splits a TimeSeries list into Sri Lankan admin levels based on the number of MOHs and 
    Districts entered (function assumes that everything is in MOH-District-Country order)
    """
    moh_ts = ts_list[:num_moh]
    dist_ts = ts_list[num_moh:num_moh + num_dist]
    nat_ts = [ts_list[-1]]
    return moh_ts, dist_ts, nat_ts

def merge_ts_lists(series_lists):
    """
    Given a list of lists, returns a single list that is a combination of all the given lists.
    Example usage: we can merge TimeSeries from MOH, District, and National-levels into a single list of TS

    Parameters:
    series_lists (list of list of TimeSeries): List of lists of TimeSeries to merge
    Returns:
    merged (list of TimeSeries): Single list of TimeSeries containing all the elements from original list of lists
    """
    merged = [item for sublist in series_lists for item in sublist]
    return merged

def merge_ts_list_set(series_list_set):
    """
    Given a set of lists, returns a single list that is a combination of all the given lists.
    Example usage: we can merge all training sets from MOH, District, and National-levels into a single training set 
    (composed of multiple TS lists).

    Parameters:
    series_list_set (list of list of list of TimeSeries): List of lists of TimeSeries to merge
    Returns:
    merged (lsit of list of TimeSeries): List containing list of TimeSeries containing all the elements from original set of lists
    """
    #Assume that all the input sets have the same length
    set_len = len(series_list_set[0])
    to_ret = []
    for i in range(set_len):
        curr_set_to_merge = [curr_set[i] for curr_set in series_list_set]
        curr_set_merged = merge_ts_lists(curr_set_to_merge)
        to_ret.append(curr_set_merged)
    return to_ret
   

def ts_list_concatenate(series_lists):
    """
    Returns a list of TimeSeries concatenating each set of series from the input list of lists (aligning on index)
    This assumes all input lists are of the same length
    """
    to_ret = []
    list_len = len(series_lists[0]) #Assume that all the input lists have the same length
    for i in range(list_len):
        to_concat = [curr_list[i] for curr_list in series_lists]
        to_ret.append(concatenate(to_concat))
    return to_ret

def ts_list_set_collapse(ts_list_set):
    """
    Function takes a set of TimeSeries lists and returns a single list with all the TimeSeries concatenated together (aligning on index). This assumes all input lists are of the same length.
    """
    
    to_ret = []
    set_len = len(ts_list_set[0]) #Assume that all the input sets have the same length
    for i in range(set_len):
        curr_set_to_concat = [curr_set[i] for curr_set in ts_list_set]
        curr_set_concatenated = concatenate(curr_set_to_concat)
        to_ret.append(curr_set_concatenated)
    return to_ret


def ts_list_set_concatenate(series_set_lists):
    """
    Given a set of TimeSeries lists, returns a single set of TimeSeries lists concatenating each set of series from the input set of lists (aligning on index)
    This assumes all input sets are of the same length and that each set contains lists of the same length.
    
    We can, for example, merge the MOH, District, and National-level TimeSeries sets into a single set.
    """
    to_ret = []
    set_len = len(series_set_lists[0]) #Assume that all the input sets have the same length
    for i in range(set_len):
        curr_set_to_concat = [curr_set[i] for curr_set in series_set_lists]
        curr_set_concatenated = ts_list_concatenate(curr_set_to_concat)
        to_ret.append(curr_set_concatenated)
    return to_ret 

def ts_list_pair_concat(list_a, list_b, lim_a = None):
    """
    Accepts two lists of TimeSeries and returns a single list concatenating corresponding elements of each. 
    Optionally accepts a value lim_a, meaning that only the last lim_a time points in each series of list_a 
    should be attached to b. We use this primarily for dynamic validation set generation and training sets with replay.
    
    If lim_a is None, then the entire series in list_a is concatenated to the corresponding series in list_b.

    Parameters:
    list_a (list of TimeSeries): First list of TimeSeries to concatenate
    list_b (list of TimeSeries): Second list of TimeSeries to concatenate
    lim_a (int): Optional limit on the number of time points to take from list_a

    Returns:
    to_ret (list of TimeSeries): List of concatenated TimeSeries
    """
    if lim_a is None:
        to_ret = ts_list_concatenate([list_a, list_b])
    else:
        to_ret = [concatenate([curr_a[-(lim_a):], curr_b]) for curr_a, curr_b in zip(list_a, list_b)]
    return to_ret


def ts_list_set_pair_concat(set_a, set_b, lim_a = None):
    """
    Accepts two sets of TimeSeries lists and returns a single set of TimeSeries lists concatenating corresponding elements of each.
    Optionally accepts a value lim_a, meaning that only the last lim_a time points in each series of set_a should be attached to set_b.

    Parameters:
    set_a (list of list of TimeSeries): First set of TimeSeries lists to concatenate
    set_b (list of list of TimeSeries): Second set of TimeSeries lists to concatenate
    lim_a (int): Optional limit on the number of time points to take from set_a

    Returns:
    to_ret (list of list of TimeSeries): List of concatenated TimeSeries lists
    """
    to_ret = [] 
    for curr_list_a, curr_list_b in zip(set_a, set_b):
        curr_list_ab = ts_list_pair_concat(curr_list_a, curr_list_b, lim_a = lim_a)
        to_ret.append(curr_list_ab)
    return to_ret


def create_fit_scaler_set(ts_list_set, global_scale = False, norm_mode = "min_max", scale = True):
    """
    Creates a list of scalers, one for each list of TimeSeries in the ts_list_set.
    The scalers are then fitted to each of the corresponding lists. If scale is True
    then this returns both a scaled version of the set and the scalers themselves. If
    scale if False, then only the fitted scalers are returned.
    """
    if norm_mode == "min_max":
        scalers = [Scaler(global_fit = global_scale) for curr_ts_list in ts_list_set]
    elif norm_mode == "std":
        scalers = [Scaler(scaler = StandardScaler(), global_fit = global_scale) for curr_ts_list in ts_list_set]
    elif norm_mode == "nat":
        #If norm_mode is nat, which stands for natural scale, then we just return the input and a list of None for the scalers
        scalers = [None for curr_ts_list in ts_list_set]
        if scale:
            return ts_list_set, scalers
        else:
            return scalers
    else:
        raise ValueError("Invalid norm_mode found while creating scaler set")
    ts_list_set_s = []
    for curr_ts_list, curr_scaler in zip(ts_list_set, scalers):
        curr_scaler.fit(curr_ts_list)
        if scale:
            curr_ts_list_s = curr_scaler.transform(curr_ts_list)
            ts_list_set_s.append(curr_ts_list_s)
    if scale:
        return ts_list_set_s, scalers
    else:
        return scalers

def scaler_transform_ts_list_set(ts_list_set, scalers):
    """
    Given a set of TimeSeries lists and a corresponding set of fitted scalers, returns a set of scaled TimeSeries lists.
    """
    ts_list_set_s = []
    for curr_ts_list, curr_scaler in zip(ts_list_set, scalers):
        if curr_scaler is not None:
            curr_ts_list_s = curr_scaler.transform(curr_ts_list)
        else:
            curr_ts_list_s = curr_ts_list
        ts_list_set_s.append(curr_ts_list_s)
    return ts_list_set_s



def scaler_inverse_transform_ts_list_set(ts_list_set_s, scalers):
    """
    Given a set of scaled TimeSeries lists and a corresponding set of fitted scalers, returns a set of unscaled TimeSeries lists.
    """
    ts_list_set = []
    for curr_ts_list_s, curr_scaler in zip(ts_list_set_s, scalers):
        if curr_scaler is not None:
            curr_ts_list = curr_scaler.inverse_transform(curr_ts_list_s)
        else:
            curr_ts_list = curr_ts_list_s
        ts_list_set.append(curr_ts_list)
    return ts_list_set


def transform_comps_ts(curr_ts, comps_to_trans, trans_func):
    """
    Transforms selected components from a TimeSeries using trans_func. Ex: can invert the log transform for only a given components in a multivariate series.
    Parameters:
    curr_ts (TimeSeries): TimeSeries to transform
    comps_to_trans (list): List of component names to transform
    trans_func (function): Function to apply to the selected components
    
    Returns:
    result_ts (TimeSeries): TimeSeries with selected components transformed
    """
    curr_builder = []
    comp_names = curr_ts.components
    for curr_var in comp_names:
        temp = curr_ts[curr_var] 
        if curr_var in comps_to_trans:
            temp = temp.map(trans_func)
            
        curr_builder.append(temp)
    
    result_ts = concatenate(curr_builder, axis = 1)
    return result_ts

def transform_comps_ts_list(curr_ts_list, comps_to_trans, trans_func):
    """
    Transforms selected components for each TimeSeries in a given list using trans_func
    """
    to_ret = []
    for curr_ts in curr_ts_list:
        result = transform_comps_ts(curr_ts, comps_to_trans = comps_to_trans, trans_func = trans_func)
        to_ret.append(result)
    return to_ret

def transform_comps_ts_list_set(curr_ts_list_set, comps_to_trans, trans_func):
    """
    Transforms selected components for each TimeSeries in a given set of lists using trans_func
    """
    to_ret = []
    for curr_ts_list in curr_ts_list_set:
        result = transform_comps_ts_list(curr_ts_list, comps_to_trans = comps_to_trans, trans_func = trans_func)
        to_ret.append(result)
    return to_ret

def compute_scaling_factor(insample, m_steps):
    """
    Function to compute scaling factors based on the mean absolute error of an m_steps ahead naive forecast over
    some insample series.
    Parameters:
    insample (list): List of TimeSeries containing insample values. The scaling factor will then be based on this. 
    m_steps (int): Naive forecast is based on guessing that values are the same as m_steps ago. 

    Return:
    Mean absolute error of a naive forecast to be used as scaling factors for errors
    """
    insample_preds = [curr_ts.shift(m_steps) for curr_ts in insample] #We can generate a naive prediction by shifting the timeseries
    scaling_factor = mae(insample, insample_preds, intersect = True, component_reduction = None) #The scaling factor is the mae 
    return scaling_factor


def get_error_func(func_name):
    """
    Returns the error function corresponding to the given name. This is used to dynamically select error functions for model evaluation.
    """
    if func_name == "mape":
        return mape
    elif func_name == "smape":
        return smape
    elif func_name == "mse":
        return mse
    elif func_name == "rmse":
        return rmse
    elif func_name == "ae":
        return ae
    elif func_name == "mae":
        return mae
    elif func_name == "mql":
        return mql
    elif func_name == "ql":
        return ql
    elif func_name == "crps":
        return crps
    elif func_name == "mcrps":
        return mcrps
    elif func_name == "miw":
        return miw
    elif func_name == "miws":
        return miws
    elif func_name == "mic":
        return mic
    elif func_name == "ic":
        return ic
    else:
        raise ValueError(f"Error function {func_name} not recognized.")
        

quantile_funcs = [mql, ql]
q_interval_funcs = [miws, miw, ic, mic]
def norm_metric(curr_obs, curr_preds, curr_normer = None, metric = None, norm_mode = "min_max", summ_func = None, extras = {}, strict = True):
    """
    Computes a normalised version of some Darts metric given an observed, predicted, and normalising TimeSeries.
    Note that this function assumes that all inputs are on their natural scales (should invert any log transforms or min-max scaling).

    Parameters:
    curr_obs (TimeSeries): Darts TimeSeries of observed values to compute metrics against
    curr_preds (TimeSeries): Darts TimeSeries forecasted values to commpute metrics with
    curr_normer (TimeSeries): Darts TimeSeries that serves as the basis of min-max values to normalise metrics (usually the training set)
    metric (string or function): If function, we use it directly, but if string, we use the get_error_func to get the actual function
    summ_func (function): Function to summarise the normalised per component metrics
    norm_mode (string): Value in ["min_max", "std", "naive_forecast_mae", "log"]. If min_max, use the minimum and maximum value in the norm series to normalise metrics, if std use standard dev, 
                            if naive_forecast_mae, generate an m_steps naive forecast over the normer series and use the MAE as the scaling factor.
    extras (dict): Dictionary of extra arguments to pass to the metric function such as "q_interval" or "quantiles".
    strict (bool): If True, then observed and predicted series must have same start and end times. If false, just take the intersection.
    Returns:
    Value of the computed metric summarised (if summ_func is not None) or an array containing the metric per component (if summ_func is None)
    """
    if metric is None:
        raise ValueError("Metric must be specified")
    #Input checking and setup
    if strict:
        assert curr_obs.start_time() == curr_preds.start_time(), "Strict set to true, observed and predicted series must have the same start time"
        assert curr_obs.end_time() == curr_preds.end_time(), "Strict set to true, observed and predicted series must have the same end time"
    else:
        curr_preds = curr_preds.slice_intersect(curr_obs) #If strict is false, then take the intersection of observed and predicted for eval
    
    #metric is a string we have to get the function
    if isinstance(metric, str):
        metric = get_error_func(metric)

    assert len(curr_preds) == len(curr_obs) #Check that the forecasts and observations match in length
    assert norm_mode in ["min_max", "std", "naive_forecast_mae", "log", None], f"Invalid norm_mode: {norm_mode}"
    
    if metric in q_interval_funcs:
        assert "q_interval" in extras, "q_interval must be specified for MIC, IC, MIW, or MIWS metrics"
        q_interval = extras["q_interval"]
        
        if metric in [mic, ic]:
            assert norm_mode is None, "Should not normalise MIC or IC metrics"

    if norm_mode == "naive_forecast_mae": 
        assert "m_steps" in extras, "m_steps must be specified for naive_forecast_mae normalisation"
        m_steps = extras["m_steps"]

    if metric in quantile_funcs:
        assert "quantiles" in extras, "Quantiles must be specified for quantile metrics"
        quantiles = extras["quantiles"]
    else:
        quantiles = None

    if norm_mode == "log":
        #If norm_mode is log, then we assume the inputs are initially on the natural scale and we log transform them prior to computing the metrics
        curr_obs = curr_obs.map(np.log1p)
        curr_preds = curr_preds.map(np.log1p)
        #If we are computing metrics on log scale then no need for normalisation
        curr_normer = None #Ignore the insample normalisation series
        norm_mode = None #Once we have log transformed, proceed as though computing natural scale metrics

    if norm_mode is None: #If norm_mode is None, no normalisation and we just return the metrics computed directly, ignoring curr_normer
        curr_normer = None
    
    #Main metric computation
    if ((metric is mql) or (metric is ql)) and (quantiles is not None):
        #If metric is MQL, we return the mean across quantiles

        #Compute a n_components x n_quantiles metrics array
        n_components = curr_obs.n_components
        met_per_comp = metric(curr_obs, curr_preds, component_reduction = None, q = quantiles)
        #Reshape to 2D
        met_per_comp = np.array(met_per_comp).reshape(n_components, len(quantiles))
        
        #Compute the mean across quantiles - this will then be an array with length equal to n_components
        met_per_comp = np.mean(met_per_comp, axis = -1)

    elif metric in q_interval_funcs: #Coverage or interval width computations using Darts functions
        met_per_comp = metric(curr_obs, curr_preds, q_interval = q_interval, component_reduction = None)
    elif isinstance(metric, tuple): #If metric is a tuple then this is coverage in (func, low, high) format
        #Like CRPS, coverage is currently computed only for univariate series
        if len(curr_obs.components) > 1:
            raise ValueError("Computing coverage for multivariate series not implemented yet")
        
        low_quantile = metric[1]
        high_quantile = metric[2] 

        met_per_comp = [metric[0](curr_obs, curr_preds, low_quantile, high_quantile)]
        #If we are computing coverage, then there is no normalisation required 
        curr_normer = None
    else:
        met_per_comp = metric(curr_obs, curr_preds, component_reduction = None) #Get the metric value per component
        
    #If there is only one component in the series being evaluated, put the metric in a list by itself
    if len(curr_obs.components) == 1:
        met_per_comp = [met_per_comp]
    
    #Normalisation/Transforms
    norm_met_vals = []
    if norm_mode in ["min_max", "std", "naive_forecast_mae"]:
        assert curr_normer is not None, "norm_mode is set to a normalisation method but no insample normer series was provided"
        if norm_mode == "min_max": #min_max per component
            bounds_list = [] 
            for comp in curr_normer.components: #Use the normer series to find the min and max per component
                values = curr_normer[comp].values()
                comp_min = values.min()
                comp_max = values.max()
                bounds_list.append((comp_min, comp_max))
                
            for met, min_max in zip(met_per_comp, bounds_list):
                curr_min = min_max[0]
                curr_max = min_max[1]
                norm_met_vals.append(met / (curr_max - curr_min)) #Normalise the component error based on the min and max for each component
        elif norm_mode == "std":
            std_list = [] #List of standard deviations per component
            for comp in curr_normer.components:
                values = curr_normer[comp].values()
                curr_std = values.std()
                if curr_std == 0:
                    raise ValueError("Could not normalise metric, found a standard deviation of 0")
                std_list.append(curr_std)
    
            for met, curr_std in zip(met_per_comp, std_list):
                #Divide the metric values by the std of the insample normer series
                norm_met_vals.append(met / curr_std)
        elif norm_mode == "naive_forecast_mae":
            #If we're normalising by the m_steps ahead forecast over some insample
            scale_factors = compute_scaling_factor([curr_normer], m_steps)[0]
            norm_met_vals = (met_per_comp/scale_factors).tolist()
    else: #Norm modes that do not require an insample series / if no normalisation needed
        for met in met_per_comp:
            norm_met_vals.append(met)
        
    #Set metrics as numpy array
    norm_met_vals = np.array(norm_met_vals)

    #Whether to summarise the metric values or to just return the whole array
    if summ_func is None:
        to_ret = norm_met_vals
        if curr_obs.n_components == 1: #If there is only one component, flatten to simplify the output
            to_ret = to_ret.flatten()
        return to_ret
    else:
        return summ_func(norm_met_vals)

def norm_metric_list(obs_list, preds_list, normer_list = None, comp_summ_func = None, series_summ_func = None, metric = None, norm_mode = "min_max", extras = {}, strict = True):
    """
    Computes a normalised metric using a list of observed, predicted, and normalising series. Function assumes that all series are on
    their natural scales. 

    Parameters:
    obs_list (list): List of Darts TimeSeries containing observed values to compute metrics against
    preds_list (list): List of Darts TimeSeries containing forecasted values to compute metrics with
    normer_list (list): List of Darts TimeSeries that serve as the basis of min-max values to normalise metrics (usually the training set)
    comp_summ_func (function): Function to summarise the normalised per component metrics
    series_summ_func (function): Function to summarise metrics across different series
    metric (string or function): If function, we use it directly, but if string, we use the get_error_func to get the actual function
    norm_mode (string): Value in ["min_max", "std", "naive_forecast_mae"]. If min_max, use the minimum and maximum value in the norm series to normalise metrics, if std use standard dev, if naive_forecast_mae, generate an m_steps naive forecast over the normer series and use the MAE as the scaling factor.
    extras (dict): Dictionary of extra arguments to pass to the metric function such as "q_interval" or "quantiles".
    """
    #metric is a string we have to get the function
    if isinstance(metric, str):
        metric = get_error_func(metric)

    result = []
    normer_list = normer_list if normer_list is not None else [None] * len(obs_list)
    for curr_obs, curr_preds, curr_normer in zip(obs_list, preds_list, normer_list):
        met_val = norm_metric(curr_obs = curr_obs, curr_preds = curr_preds, curr_normer = curr_normer, summ_func = comp_summ_func, metric = metric, norm_mode = norm_mode, extras = extras, strict = strict)
        result.append(met_val)
    result = np.array(result)

    if series_summ_func is None:
        to_ret = result
        if to_ret.ndim == 2 and to_ret.shape[1] == 1: #If the results are two dimensional, but the 2nd dimension is 1, flatten to simplify the output
            to_ret = to_ret.flatten()
        return to_ret
    else:
        return series_summ_func(result)

def norm_metric_list_set(obs_list_set, preds_list_set, normer_list = None, comp_summ_func = None, series_summ_func = None, metric = None, norm_mode = "min_max", extras = {}, strict = True):
    """
    Computes a normalised metric using a set of lists of observed, predicted, and normalising series. Function assumes that all series are on
    their natural scales. 

    Parameters:
    obs_list_set (list): List of lists of Darts TimeSeries containing observed values to compute metrics against
    preds_list_set (list): List of lists of Darts TimeSeries containing forecasted values to compute metrics with
    normer_list (list): List of lists of Darts TimeSeries that serve as the basis of min-max values to normalise metrics (usually the training set)
    comp_summ_func (function): Function to summarise the normalised per component metrics
    series_summ_func (function): Function to summarise metrics across different series
    metric (string or function): If function, we use it directly, but if string, we use the get_error_func to get the actual function
    norm_mode (string): Value in ["min_max", "std", "naive_forecast_mae"]. If min_max, use the minimum and maximum value in the norm series to normalise metrics, if std use standard dev, if naive_forecast_mae, generate an m_steps naive forecast over the normer series and use the MAE as the scaling factor.
    extras (dict): Dictionary of extra arguments to pass to the metric function such as "q_interval" or "quantiles".
    """
    #metric is a string we have to get the function
    if isinstance(metric, str):
        metric = get_error_func(metric)

    result = []
    normer_list = normer_list if normer_list is not None else [None] * len(obs_list_set)
    for curr_obs_list, curr_preds_list in zip(obs_list_set, preds_list_set):
        met_val = norm_metric_list(curr_obs_list, curr_preds_list, normer_list = normer_list, comp_summ_func = comp_summ_func, series_summ_func = series_summ_func, metric = metric, norm_mode = norm_mode, extras = extras, strict = strict)
        result.append(met_val)
    result = np.array(result)

    return result

def collect_set_metrics(obs_list_set_dict, preds_list_set_dict, normer_list = None, comp_summ_func = None, series_summ_func = None, meta_metrics = ["mae"], meta_norms = ["min_max"], manual_log = True, extras = {}, strict = True):
    """
    Computes a collection of metrics (every pair of metric and norm_mode in meta_metrics and meta_norms) for a set of lists of observed and predicted series.
    This uses the norm_metric_list_set function. 
    
    Parameters:
    obs_list_set_dict (dict): Dictionary of list of lists of Darts TimeSeries containing observed values to compute metrics against. Keys are usually "nat", "log", and "scaled".
    preds_list_set_dict (dict): Dictionary of list of lists of Darts TimeSeries containing forecasted values to compute metrics with. Keys are usually, "nat", "log", and "scaled". 
    normer_list (list): List of lists of Darts TimeSeries that serve as insample series to compute normalisation values (usually the training set)
    comp_summ_func (function): Function to summarise the normalised per component metrics
    series_summ_func (function): Function to summarise metrics across different series
    meta_metrics (list): List of metrics to compute
    meta_norms (list): List of normalisation modes to use
    extras (dict): Dictionary of extra arguments to pass to the metric function such as "q_interval" or "quantiles".
    strict (bool): If True, then observed and predicted series must have same start and end times. If false, just take the intersection.

    Returns:
    result_dict (dict): Dictionary containing the computed metrics for each pair of metric and norm_mode. The keys will be (metric, norm_mode) tuples and the values will be the computed metrics.
    """
    #Boolean to check if the forecast is probabilistic
    is_prob_fc = preds_list_set_dict["nat"][0][0].n_samples > 1

    result_dict = {}

    met_comp_params = {
        "normer_list": normer_list,
        "comp_summ_func": comp_summ_func,
        "series_summ_func": series_summ_func,
        "extras": extras,
        "strict": strict
    }
    for curr_metric in meta_metrics:
        # print(curr_metric)
        #Special case for mic and ic since these do not need normalisation and may have multiple q_interval params
        if curr_metric in ["mic", "ic"]:
            assert "mic_q_interval_list" in extras, "Need to provide a list of q_intervals for mic and ic metrics"
            for curr_qs in extras["mic_q_interval_list"]:
                curr_q_interval = (curr_qs[0], curr_qs[1])
                mic_cov_name = str(int((curr_qs[1]-curr_qs[0]) * 100))
                curr_key = f"{curr_metric}_{mic_cov_name}"
                
                curr_met_comp_params = met_comp_params.copy()
                curr_met_comp_params["obs_list_set"] = obs_list_set_dict["nat"]
                curr_met_comp_params["preds_list_set"] = preds_list_set_dict["nat"]
                curr_met_comp_params["extras"] = {"q_interval": curr_q_interval}
                curr_met_comp_params["norm_mode"] = None #No normalisation for mic and ic
                curr_met_comp_params["metric"] = curr_metric
                temp = norm_metric_list_set(**curr_met_comp_params)
                result_dict[curr_key] = temp
        else:
            for curr_norm_mode in meta_norms:
                # curr_key = (curr_metric, curr_norm_mode)
                curr_key = f"{curr_norm_mode}_{curr_metric}"
                curr_met_comp_params = met_comp_params.copy()
                #Some routing here to select the appropriate observed and predicted series based on the norm_mode
                if curr_norm_mode in ["min_max", "std", "naive_forecast_mae", "nat", None]:
                    #If min_max, std, naive_forecast_mae, or None, then we use the natural scale series
                    curr_met_comp_params["obs_list_set"] = obs_list_set_dict["nat"]
                    curr_met_comp_params["preds_list_set"] = preds_list_set_dict["nat"]
                    if curr_norm_mode == "nat": #Nat means natural scale so we retrieve nat values and set norm_mode to None since we don't want to normalise the metrics
                        curr_norm_mode = None
                    curr_met_comp_params["norm_mode"] = curr_norm_mode
                elif curr_norm_mode == "log":
                    if not manual_log: #If not manual log, assume that the inputted dict already has a properly computed log transform
                        curr_met_comp_params["obs_list_set"] = obs_list_set_dict["log"]
                        curr_met_comp_params["preds_list_set"] = preds_list_set_dict["log"]
                        curr_met_comp_params["norm_mode"] = None
                    else: #If manual log, get the natural scale values and pass norm_mode = "log" so series are log transformed by the metrics function
                        curr_met_comp_params["obs_list_set"] = obs_list_set_dict["nat"]
                        curr_met_comp_params["preds_list_set"] = preds_list_set_dict["nat"]
                        curr_met_comp_params["norm_mode"] = "log"
                elif curr_norm_mode == "scaled":
                    #scaled refers to computing the metrics on the min-max + log transformed series
                    #If scaled, then we use the scaled series and set norm_mode to None since we have already scaled the series
                    curr_met_comp_params["obs_list_set"] = obs_list_set_dict["scaled"]
                    curr_met_comp_params["preds_list_set"] = preds_list_set_dict["scaled"]
                    curr_met_comp_params["norm_mode"] = None
                else:
                    raise ValueError(f"Invalid norm_mode {curr_norm_mode} found while computing metrics") 
                    
                #If the inputted series are deterministic and curr_metric is mcrps then we return MAE instead 
                if curr_metric == "mcrps" and not is_prob_fc:
                    # print("Requested mcrps but forecast is deterministic, returning mae instead")
                    curr_met_comp_params["metric"] = "mae"
                else: #Otherwise just compute the specified metric
                    curr_met_comp_params["metric"] = curr_metric
                temp = norm_metric_list_set(**curr_met_comp_params)
                result_dict[curr_key] = temp

    return result_dict

def ts_list_set_conv_dtype(ts_set, dtype_set = np.float32):
    """
    Given a set of TimeSeries lists, returns a set of TimeSeries lists with the values converted to the given dtype.
    This is useful for ensuring that all TimeSeries in a set have the same dtype, especially when working with PyTorch models.

    Parameters:
    ts_set (list of list of TimeSeries): Set of TimeSeries lists to convert
    dtype (numpy dtype): Desired numpy dtype for the TimeSeries values

    Returns:
    ts_set_converted (list of list of TimeSeries): Set of TimeSeries lists with values converted to the given dtype
    """
    ts_set_converted = []
    for curr_ts_list in ts_set:
        curr_ts_list_converted = []
        for curr_ts in curr_ts_list:
            curr_ts_converted = curr_ts.astype(dtype_set)
            curr_ts_list_converted.append(curr_ts_converted)
        ts_set_converted.append(curr_ts_list_converted)
    return ts_set_converted


def roll_dict_conv_dtype(curr_roll_dict, dtype_set = np.float32):
    for curr_key in curr_roll_dict.keys():
        curr_roll_set = curr_roll_dict[curr_key]
        new_vals = []
        for curr_ts_set in curr_roll_set:
            new_ts_set = ts_list_set_conv_dtype(curr_ts_set, dtype_set = np.float32)
            new_vals.append(new_ts_set)
        curr_roll_dict[curr_key] = new_vals

    return curr_roll_dict

def save_model_set(model_set, model_set_name, output_dir):
    for i, model in enumerate(model_set):
        model_filename = f"{model_set_name}_{i}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        model_set[i].save(model_path)

def df_list_to_excel(df_list, path, sheet_names = None):
    """
    Writes a list of DataFrames as separate sheets in one Excel file given some path.
    """
    if sheet_names is None:
        sheet_names = ["Sheet " + str(i) for i in range(len(df_list))]
    with pd.ExcelWriter(path, engine = "openpyxl") as writer:
        for curr_df, curr_sheet in zip(df_list, sheet_names):
            curr_df.to_excel(writer, sheet_name = curr_sheet, index = False)

def get_loc_mets_tables(curr_metric_dict, curr_eval_dict, admin_level_list, admin_list, clim_var_names, add_col_summ = True, add_set_summ = True, fold_weighted = True):
    """
    Function to create a dictionary of DataFrames containing metrics for each set and each component, as well as summary DataFrames.

    Parameters:
    curr_metric_dict (dict): Dictionary containing metric arrays for each metric key. Each array is in the shape (num_sets, num_admin_units, num_components). Usually from the collect_set_metrics function
    curr_eval_dict (dict): Dictionary containing evaluation data for each set
    admin_level_list (list): List of admin levels
    admin_list (list): List of admin units
    clim_var_names (list): List of climate variable names
    add_col_summ (bool): Whether to add column summaries
    add_set_summ (bool): Whether to add set summaries
    fold_weighted (bool): Whether to weight folds by the number of time points in each evaluation set
    """
    metric_keys = list(curr_metric_dict.keys())
    sample_arr = curr_metric_dict[metric_keys[0]] #Assume all metrics same shape
    num_sets = sample_arr.shape[0]
    num_admin = sample_arr.shape[1]
    num_comps = sample_arr.shape[2]
    curr_comp_ind = -1

    ret_dict = {}
    comp_ind_list = list(range(num_comps)) + [-1] #Go through each component, and lastly, summarise
    for curr_comp_ind in comp_ind_list:
        #Create a DataFrame with AdminLevel and Admin columns for each set
        to_ret = [pd.DataFrame({"AdminLevel": admin_level_list, "Admin": admin_list}) for i in range(num_sets)]

        #Metrics arrays are in (num_sets, num_admin_units, num_components) format
        #For each set
        for curr_set_ind in range(num_sets):
            #Get the metrics DataFrame for the set
            curr_set_df = to_ret[curr_set_ind]
            for curr_key in metric_keys:
                #For each metric key, get the corresponding array and extract the values for the current set and component
                curr_metric_arr = curr_metric_dict[curr_key]
                if curr_comp_ind == -1:
                    #If curr_comp_ind is -1, we want to summarise across components, so we take the mean across the last axis
                    curr_metric_arr = curr_metric_arr.mean(axis=-1)
                    curr_metric_vals = curr_metric_arr[curr_set_ind, :]
                else:
                    curr_metric_vals = curr_metric_arr[curr_set_ind, :, curr_comp_ind]
                curr_set_df[curr_key] = curr_metric_vals

        if not fold_weighted: 
            weights = None
        else:
            #If fold_weighted, then we weight the losses by the length of each validation set
            val_set_lens = [len(curr_set[0]) for curr_set in curr_eval_dict["obs"]["nat"]] 
            weights = val_set_lens

        #If add_set_summ 
        if add_set_summ:
            summ_df = pd.DataFrame({"AdminLevel": admin_level_list, "Admin": admin_list})
            for curr_key in metric_keys: 
                curr_metric_arr = curr_metric_dict[curr_key]
                curr_metric_arr = np.average(curr_metric_arr, axis=0, weights=weights) 
                if curr_comp_ind == -1:
                    #If curr_comp_ind is -1, we want to summarise across components, so we take the mean across the last axis
                    curr_metric_arr = curr_metric_arr.mean(axis=-1)
                    curr_metric_vals = curr_metric_arr.flatten()
                else:
                    curr_metric_vals = curr_metric_arr[:, curr_comp_ind]
                summ_df[curr_key] = curr_metric_vals
            to_ret.append(summ_df)

        #If add_col_summ, we add a row for the mean of each metric across all admin units
        if add_col_summ:
            for curr_set_ind in range(len(to_ret)):
                curr_set_df = to_ret[curr_set_ind]
                summ_row = {"AdminLevel": "Summary", "Admin": "Summary"}
                for curr_key in metric_keys:
                    summ_row[curr_key] = curr_set_df[curr_key].mean()
                to_ret[curr_set_ind] = pd.concat([curr_set_df, pd.DataFrame([summ_row])], ignore_index=True)
        
        ret_key = "Overall" if curr_comp_ind == -1 else clim_var_names[curr_comp_ind]
        ret_dict[ret_key] = to_ret
    return ret_dict

def create_ts_dict_list(ts_lists, ts_names, unit_names, comp_name = None):
    """
    Creates a list of dicts per unit in unit_names. Each dict contains a TimeSeries from each list in ts_lists where the keys are
    given by corresponding elements in ts_names
    """
    assert len(ts_lists) == len(ts_names)
    assert len(unit_names) == len(ts_lists[0])
    
    result = []
    for i in range(len(unit_names)):
        curr_title = unit_names[i]
        curr_dict = {"title": curr_title}
        for curr_ts_list, curr_ts_name in zip(ts_lists, ts_names):
            curr_val = curr_ts_list[i]
            if comp_name is not None: #If only a specific component to visualise
                curr_val = curr_val[comp_name]
            curr_dict[curr_ts_name] = curr_val
        result.append(curr_dict)
    return result

def plot_dict_list(dict_list, low_quantile = 0.025, central_quantile = 0.5, high_quantile = 0.975, 
                   width = 20, height = 36, curr_fig_dims = None, limit_dict = {}, interval_dict = {}, colour_dict = {}):
    """
    Plots a list of dicts, where each dict contains multiple keys and values. All values are TimeSeries to be plotted, with the exception of 
    the key "title", which is then used as the plot title.

    Parameters:
    dict_list (list): List of dictionaries to plot. Each dictionary contains TimeSeries to plot, with an optional "title" key for the plot title.
    low_quantile (float): Lower quantile for prediction intervals.
    central_quantile (float): Central quantile for prediction intervals.
    high_quantile (float): Upper quantile for prediction intervals.
    width (int): Width of the entire figure.
    height (int): Height of the entire figure.
    curr_fig_dims (tuple): Dimensions of the figure in terms of number of rows and columns of subplots. If None, it is automatically determined.
    limit_dict (dict): Dictionary specifying limits on the number of time points to plot for specific keys (e.g. {"Input": 104} means only plot the last 104 points of the "Input" series).
    interval_dict (dict): Dictionary specifying custom intervals for specific keys. Each key maps to a list of tuples representing intervals. If the key is not found, the default intervals are used
                            (i.e. use low_quantile and high_quantile).

    """
    if curr_fig_dims is None:
        if len(dict_list) > 1:
            curr_fig_dims = (int(np.ceil(len(dict_list)/4)) ,4)
        else:
            curr_fig_dims = (1,1)
    fig, axs = plt.subplots(curr_fig_dims[0], curr_fig_dims[1], figsize = (width,  height))
    if len(dict_list) > 1:
        ax_list = axs.flatten()
    else:
        ax_list = [axs]
    #If there is only one dictionary in the list (such as with national-level forecasts), just create one plot
    for curr_dict, curr_ax in zip(dict_list, ax_list):
        #Generate a plot per dictionary
        for curr_key in curr_dict.keys():
            if curr_key == "title":
                continue
            curr_ts = curr_dict[curr_key]
            #If there's a limit 
            if curr_key in limit_dict: 
                curr_ts = curr_ts[-limit_dict[curr_key]:]
            if curr_key not in interval_dict:
                curr_ts.plot(ax = curr_ax, label = curr_key, low_quantile = low_quantile, 
                             high_quantile = high_quantile, central_quantile = central_quantile,
                             color = colour_dict.get(curr_key, None))
            else:
                for curr_interval in interval_dict[curr_key]:
                    interval_label = curr_key +  " (" + str(int(100 * (curr_interval[1]-curr_interval[0]))) + "% UI)"
                    curr_ts.plot(ax = curr_ax, label = interval_label, low_quantile = curr_interval[0], 
                                high_quantile = curr_interval[1], central_quantile = central_quantile)
        if "title" in curr_dict: 
            curr_ax.set_title(curr_dict["title"])
        curr_ax.set_xlabel("")
    return fig, axs


def get_vals_ind_list(val_list, ind_list):
    """
    Get the values from val_list corresponding to indices in ind_list
    """
    return [val_list[i] for i in ind_list]

def flatten_list(nested_list):
    """
    Flattens a nested list into a single list.
    """
    return [item for sublist in nested_list for item in sublist]

def summarise_df_list(df_list, id_cols, weights):
    """
    Summarises a list of DataFrames by averaging their numeric columns with given weights.
    
    Parameters:
    df_list: List of pandas DataFrames to summarise.
    id_cols: List of column names that are identifiers (non-numeric) and should be preserved in the output.
    weights: List of weights for averaging the numeric columns. Must
        
    Returns:
    A new pandas DataFrame containing the averaged numeric columns and the identifier columns from the first DataFrame in df_list.
    """
    if weights is not None:
        weights = np.array(weights) #Turn weights into numpy array in case it's a list
        assert len(weights) == len(df_list), "Weights must have the same length as df_list"
    
    numeric_cols = df_list[0].select_dtypes(include=[np.number]).columns.tolist()

    #Assert that rows are in the same order based on id_cols of the first df in the list 
    ref = df_list[0][id_cols].reset_index(drop=True)
    for df in df_list[1:]:
        assert ref.equals(df[id_cols].reset_index(drop=True)), "DataFrames must have the same rows in the same order for id_cols"

    #Stack the numeric values so we can average them
    stack_vals = np.stack([df[numeric_cols].values for df in df_list], axis=0)  # Shape: (num_dfs, num_rows, num_numeric_cols)
    weighted_vals = np.average(stack_vals, axis=0, weights=weights)  #Shape: (num_rows, num_numeric_cols)
    
    #Create a new DataFrame with the weighted values and the id_cols from the first DataFrame
    to_ret = pd.DataFrame(weighted_vals, columns=numeric_cols)
    for col in reversed(id_cols): #Insert the id_cols back (in reverse order to maintain original order)
        to_ret.insert(0, col, df_list[0][col].values)
    return to_ret

def summarise_prob_ts_list_set(curr_prob_ts_set, summ_stat):
    """
    Summarises a set of probabilistic TimeSeries by applying a specified summary statistic.
    
    Parameters:
    curr_prob_ts_set (list of list of TimeSeries): Set of probabilistic TimeSeries to summarise.
    summ_stat (str): Summary statistic to apply. Options are "median", "mean", or "q_<value>" for quantiles (e.g., "q_0.025" for the 2.5th percentile).

    Returns:
    to_ret (list of list of TimeSeries): Set of summarised TimeSeries after applying the specified summary statistic.
    """
    to_ret = []
    for curr_ts_list in curr_prob_ts_set:
        curr_ret_list = []
        for curr_ts in curr_ts_list:
            if summ_stat == "median":
                curr_ret_ts = curr_ts.median()
            elif summ_stat == "mean":
                curr_ret_ts = curr_ts.mean()
            elif summ_stat.startswith("q"):
                q_value = summ_stat.split("_")[1]
                curr_ret_ts = curr_ts.quantile(float(q_value))
            else:
                raise ValueError(f"Unknown summary statistic: {summ_stat}")
            
            curr_ret_list.append(curr_ret_ts)
        to_ret.append(curr_ret_list)
    return to_ret

def add_comp_intersect_ts_list(curr_ts_list, curr_add_comp): 
    """
    Function adds a component to every TimeSeries in a list of TimeSeries. Automatically takes the intersection
    of the base series and the additional component.
    
    Parameters:
    curr_ts_set (list of TimeSeries): List of TimeSeries to which the additional component will be added.
    curr_add_comp (TimeSeries): TimeSeries containing the additional component to be added to each TimeSeries in the list.
    
    Returns:
    to_ret (list of TimeSeries): List of TimeSeries with the additional component
    """
    to_ret = []
    for curr_ts in curr_ts_list:
        curr_add_ts = curr_add_comp.slice_intersect(curr_ts)
        ret_ts = concatenate([curr_ts, curr_add_ts], axis=1)
        to_ret.append(ret_ts)
    return to_ret

def add_comp_intersect_ts_list_set(curr_ts_set, curr_add_comp):
    """
    Function adds a component to every TimeSeries in a set of TimeSeries lists. Automatically takes the intersection
    of the base series and the additional component.
    
    Parameters:
    curr_ts_set (list of list of TimeSeries): Set of TimeSeries lists to which the additional component will be added.
    curr_add_comp (TimeSeries): TimeSeries containing the additional component to be added to each TimeSeries in the set.
    
    Returns:
    to_ret (list of list of TimeSeries): Set of TimeSeries lists with the additional component
    """

    to_ret = []
    for curr_ts_list in curr_ts_set:
        ret_ts_list = []
        for curr_ts in curr_ts_list:
            curr_add_ts = curr_add_comp.slice_intersect(curr_ts)
            ret_ts = concatenate([curr_ts, curr_add_ts], axis=1)
            ret_ts_list.append(ret_ts)
        to_ret.append(ret_ts_list)
    return to_ret

def ts_set_add_stat_cov(curr_ts_set, stat_cov):
    """
    Function adds a stat_cov component to every TimeSeries in a set of TimeSeries lists. Automatically takes the intersection
    of the base series and the additional component.
    
    Parameters:
    curr_ts_set (list of list of TimeSeries): Set of TimeSeries lists to which the additional component will be added.
    stat_cov (dict): Dictionary containing the stat_cov component to be added to each TimeSeries in the set. Auto-converted into pandas series
    
    Returns:
    to_ret (list of list of TimeSeries): Set of TimeSeries lists with the stat_cov component
    """
    stat_cov = pd.Series(stat_cov)
    to_ret = []
    for curr_ts_list in curr_ts_set:
        ret_ts_list = []
        for curr_ts in curr_ts_list:
            curr_ts_sc = curr_ts.with_static_covariates(stat_cov)
            ret_ts_list.append(curr_ts_sc)
        to_ret.append(ret_ts_list)
    return to_ret

def fit_reconciler_set(curr_reconcilers, curr_ts_set, curr_hierarchy):
    """
    Fits reconcilers using the corresponding TimeSeries sets and hierarchy.
    
    Parameters:
    curr_reconcilers: list of MinTReconciliator objects
    curr_ts_set: list of TimeSeries objects
    curr_hierarchy: dict, hierarchy for the TimeSeries
    """
    assert len(curr_reconcilers) == len(curr_ts_set), "Number of reconcilers must match number of TimeSeries in the set"
    
    for curr_ind in range(len(curr_ts_set)):
        curr_reconciler = curr_reconcilers[curr_ind]
        curr_ts_list = curr_ts_set[curr_ind]

        #Convert the TimeSeries list into a multivariate series 
        curr_mvts = concatenate(curr_ts_list, axis = "component").with_hierarchy(curr_hierarchy)
        
        #Fit the reconciler
        curr_reconciler.fit(curr_mvts)


#We can then use this to transform static covariates for all train/input series
def set_stat_cov_trans(curr_ts_set, stat_cov_trans):
    """
    Transforms static covariates for a set of TimeSeries lists.
    
    Parameters:
    curr_ts_set (list of list of TimeSeries): Set of TimeSeries lists to transform.
    stat_cov_trans (StaticCovariatesTransformer): Transformer for static covariates.
    
    Returns:
    to_ret (list of list of TimeSeries): Set of TimeSeries lists with transformed static covariates.
    """
    to_ret = []
    for curr_ts_list in curr_ts_set:
        to_ret.append(stat_cov_trans.transform(curr_ts_list))
    return to_ret

def map_func_ts_set(curr_ts_list_set, func):
    """
    Map a function to every individual TimeSeries in a set of TimeSeries lists. One example is inverting log scaling. 
    
    Parameters:
    curr_ts_list_set (list of list of TimeSeries): Set of TimeSeries lists to which the function will be applied.
    func (function): Function to apply to each TimeSeries.

    Returns:
    to_ret (list of list of TimeSeries): Set of TimeSeries lists with the function mapped to each individual series.
    """
    to_ret = []
    for curr_ts_list in curr_ts_list_set:
        curr_ret = [curr_ts.map(func) for curr_ts in curr_ts_list]
        to_ret.append(curr_ret)
    return to_ret

def reconcile_corr_ts_list(curr_ts_list, curr_reconciler, hierarchy, moh_list, dist_list):
    """
    Reonciles a ts_list_set using a given reconciler then truncates values at 0 and propagates the correction 
    through the hierarchy to maintain reconciled results. 

    Parameters:
    curr_ts_list (list of TimeSeries): List of TimeSeries to reconcile and correct.
    curr_reconciler (Reconciler): Reconciler object to use for reconciliation.
    hierarchy: Hierarchy for the TimeSeries.
    moh_list (list): List of MOH units.
    dist_list (list): List of district units.

    Returns:
    to_ret (list of TimeSeries): List of reconciled and corrected TimeSeries.
    """
    #First, direct reconcile the natural scale predictions
    #Create multivariate series
    curr_mvts = concatenate(curr_ts_list, axis = "component").with_hierarchy(hierarchy)

    #Reconcile
    curr_mvts_rec = curr_reconciler.transform(curr_mvts)

    #Create a dictionary with admin units as keys and reconciled TimeSeries as values
    preds_dict = {comp: curr_mvts_rec.univariate_component(comp) for comp in curr_mvts_rec.components}

    #Correct potential negative values coming from the reconciliation
    preds_dict_corr = preds_dict.copy()

    #Iterate through base of hierarchy (MOHs)
    for curr_moh in moh_list:
        #Get the TimeSeries for the current MOH
        curr_ts = preds_dict_corr[curr_moh]

        #Convert TimeSeries to numpy array
        curr_arr = curr_ts.all_values()

        #Create a TimeSeries containing only negative values in the series
        neg_mask = np.where(curr_arr < 0, curr_arr, 0)
        neg_mask_ts = curr_ts.with_values(neg_mask)

        #Get the parent district and its corresponding TimeSeries
        par_dist = hierarchy[curr_moh]
        par_ts = preds_dict_corr[par_dist]

        #Correct the current MOH TimeSeries - clipping at 0
        corr_ts = curr_ts - neg_mask_ts 
        #Correct the parent district by the same amount - propagating changes up the hierarchy
        corr_par_ts = par_ts - neg_mask_ts 
        #Correct the national-level series
        corr_nat_ts = preds_dict_corr["Sri Lanka"] - neg_mask_ts

        #Set the corrected TimeSeries back into the dictionary
        preds_dict_corr[curr_moh] = corr_ts
        preds_dict_corr[par_dist] = corr_par_ts
        preds_dict_corr["Sri Lanka"] = corr_nat_ts
        
    #Propagate corrections up from district level
    for curr_dist in dist_list:
        #Get TimeSeries for current district
        curr_dist_ts = preds_dict_corr[curr_dist]
        #Convert to numpy array, get negative values, then convert to TimeSeries
        curr_arr = curr_dist_ts.all_values()
        neg_mask = np.where(curr_arr < 0, curr_arr, 0)
        neg_mask_ts = curr_dist_ts.with_values(neg_mask)

        #Correct the district- and national-level TimeSeries
        corr_ts = curr_dist_ts - neg_mask_ts
        corr_nat_ts = preds_dict_corr["Sri Lanka"] - neg_mask_ts


        #Set values in the dict to the corrected values
        preds_dict_corr[curr_dist] = corr_ts
        preds_dict_corr["Sri Lanka"] = corr_nat_ts

    key_list = moh_list + dist_list + ["Sri Lanka"]
    #Convert back to a list of reconciled + corrected TimeSeries
    to_ret = [preds_dict_corr[key] for key in key_list]
    return to_ret

def reconcile_corr_ts_list_set(curr_ts_list_set, curr_reconciler_list, hierarchy, moh_list, dist_list):
    """
    Reconciles and corrects a set of TimeSeries lists using a given reconciler and hierarchy. 
    This function applies the reconcile_corr_ts_list function to each list in the set.

    Parameters:
    curr_ts_list_set (list of list of TimeSeries): Set of TimeSeries lists to reconcile and correct.
    curr_reconciler_list (list of Reconciler): List of Reconciler objects to use for reconciliation.
    curr_hierarchy: Hierarchy for the TimeSeries.
    moh_list (list): List of MOH units.
    dist_list (list): List of district units.

    Returns:
    to_ret (list of list of TimeSeries): Set of reconciled and corrected TimeSeries lists.
    """
    to_ret = []
    for curr_ts_list, curr_reconciler in zip(curr_ts_list_set, curr_reconciler_list):
        curr_ret = reconcile_corr_ts_list(curr_ts_list, curr_reconciler, hierarchy, moh_list, dist_list)
        to_ret.append(curr_ret)
    return to_ret




def model_roll_forecast(curr_model, curr_series_list, max_horizon, num_samples, log_trans_cols, curr_input_scaler, curr_output_scaler = None, overlap_end = True):
    """
    Function generates a rolling forecast assuming no covariates (outside of encoders built into the model).

    Parameters:
    curr_model: Darts forecasting model object
    curr_series_list: List of Darts TimeSeries to generate rolling forecasts with and across (combined input + output series)
    max_horizon: Maximum forecast horizon to generate
    num_samples: Number of samples to generate for probabilistic forecasts
    log_trans_cols: List of column names that were log transformed, or None if no log transformation was applied
    curr_input_scaler: Scaler object used to scale the input series
    curr_output_scaler: Scaler object used to scale the output series, or None if the same as the input scaler

    Returns:
    ret_dict: Dictionary containing rolling forecasts in natural scale, log scale, and scaled values.
    """

    #Build a dictionary of parameters to pass to rolling forecast
    #No retraining and we set last_points_only to False so we get the full rolling forecasts 
    hist_fc_params_dict = {
        "series": curr_series_list,
        "forecast_horizon": max_horizon,
        "retrain": False,
        "last_points_only": False,
        "overlap_end": overlap_end
    }
    if num_samples is not None:
        hist_fc_params_dict["num_samples"] = num_samples

    #The rolling forecasts are generated as log + scaled values
    #This will give us a list of lists of TimeSeries - in num_admin x num_forecasts order
    curr_roll_res = curr_model.historical_forecasts(**hist_fc_params_dict)
    #We flip the output into num_forecast x num_admin order so that a single index gives the same forecast across all admin units
    #Raw num_forecast x num_admin list of lists
    curr_roll_fc_log_s = [list(curr_fc) for curr_fc in zip(*curr_roll_res)]

    #With this, we can use list-based helper functions to scale and transform values
    curr_roll_fc_log = [curr_input_scaler.inverse_transform(curr_fc) for curr_fc in curr_roll_fc_log_s]

    if log_trans_cols is not None:
        curr_roll_fc = transform_comps_ts_list_set(curr_roll_fc_log, log_trans_cols, trans_func = np.expm1)
    else: #If log_trans_cols is None, we assume that this is univariate and we just invert the log transform
        curr_roll_fc = [curr_fc.map(np.expm1) for curr_fc in curr_roll_fc_log]


    #If output scaler is None, then we assume it's same as input scaler and we just return the formatted raw forecasts
    if curr_output_scaler is None:
        curr_roll_fc_log_out_s = curr_roll_fc_log_s
    else:
        curr_roll_fc_log_out_s = [curr_output_scaler.transform(curr_fc) for curr_fc in curr_roll_fc_log]

    ret_dict = {
        "nat": curr_roll_fc,
        "log": curr_roll_fc_log,
        "scaled": curr_roll_fc_log_out_s
    }
    return ret_dict #Return dict with "nat", "log", "scaled" keys for natural, log, and scaled forecasts

def model_set_roll_forecast(curr_model_list, curr_input_series_set, curr_output_series_set, curr_horizon_list, num_samples, curr_input_chunk_length, log_trans_cols, input_scaler_set, output_scaler_set = None):
    """
    Function uses model_roll_forecast to generate rolling forecasts for a set of models and corresponding input/output series. Note that this assumes no covariates outside of encoders built into the models. 
    
    Parameters:
    curr_model_list: List of Darts forecasting model objects
    curr_input_series_set: List of lists of Darts TimeSeries containing input series for each model
    curr_output_series_set: List of lists of Darts TimeSeries containing output series for each model
    curr_horizon_list: List of forecast horizons for each model
    num_samples: Number of samples to generate for probabilistic forecasts
    curr_input_chunk_length: Length of the input chunk used for each model
    log_trans_cols: List of column names that were log transformed, or None if no log transformation was applied
    input_scaler_set: List of scaler objects used to scale the input series for each model
    output_scaler_set: List of scaler objects used to scale the output series for each model
    """
    max_horizon = max(curr_horizon_list)
    #We combine the last input chunk of the input series with the output series - ensuring rolling forecasts start where the output series starts.
    curr_series_set = ts_list_set_pair_concat(curr_input_series_set, curr_output_series_set, lim_a = curr_input_chunk_length + max_horizon - 1)

    #Builder lists that we return as a dict
    nat_builder = []
    log_builder = []
    scaled_builder = []

    #For each set index, we generate rolling forecasts
    for curr_ind in range(len(curr_model_list)):
        curr_series_list = curr_series_set[curr_ind]
        curr_model = curr_model_list[curr_ind]
        curr_input_scaler = input_scaler_set[curr_ind]
        curr_output_scaler = output_scaler_set[curr_ind] if output_scaler_set is not None else None

        #We generate the rolling forecasts for the current model
        curr_roll_fc_dict = model_roll_forecast(
            curr_model = curr_model, 
            curr_series_list = curr_series_list, 
            max_horizon = max_horizon, 
            num_samples = num_samples, 
            log_trans_cols = log_trans_cols, 
            curr_input_scaler = curr_input_scaler, 
            curr_output_scaler = curr_output_scaler
        )
        nat_builder.append(curr_roll_fc_dict["nat"])
        log_builder.append(curr_roll_fc_dict["log"])
        scaled_builder.append(curr_roll_fc_dict["scaled"])

    to_ret = {
        "nat": nat_builder,
        "log": log_builder,
        "scaled": scaled_builder

    }

    return to_ret

def map_func_ts_dict_set(curr_ts_dict_set, func):
    """
    Takes a dictionary as input where keys are horizons and values are ts list sets 
    and applies a function to each ts in the ts list sets for each horizon

    Parameters: 
    curr_ts_dict_set: A dictionary where keys are horizons and values are ts list sets
    func: A function to apply to each ts in the ts list sets
    
    Returns:
    A dictionary where keys are horizons and values are ts list sets with the function applied
    """
    to_ret = {}
    for curr_key in curr_ts_dict_set:
        to_ret[curr_key] = map_func_ts_set(curr_ts_dict_set[curr_key], func = func)
    return to_ret

def scaler_transform_ts_dict_set(curr_ts_dict_set, curr_scaler_set):
    """
    Takes a dictionary as input where keys are horizons and values are ts list sets 
    and applies a scaler transform to each ts in the ts list sets for each horizon

    Parameters: 
    curr_ts_dict_set: A dictionary where keys are horizons and values are ts list sets
    curr_scaler_set: A list of scalers to apply to each ts in the ts list sets
    
    Returns:
    A dictionary where keys are horizons and values are ts list sets with the scaler transform applied
    """
    to_ret = {}
    for curr_key in curr_ts_dict_set:
        to_ret[curr_key] = scaler_transform_ts_list_set(curr_ts_dict_set[curr_key], curr_scaler_set)
    return to_ret

def scaler_inverse_transform_ts_dict_set(curr_ts_dict_set, curr_scaler_set):
    """
    Takes a dictionary as input where keys are horizons and values are ts list sets 
    and applies a scaler inverse transform to each ts in the ts list sets for each horizon

    Parameters: 
    curr_ts_dict_set: A dictionary where keys are horizons and values are ts list sets
    curr_scaler_set: A list of scalers to apply to each ts in the ts list sets
    
    Returns:
    A dictionary where keys are horizons and values are ts list sets with the scaler inverse transform applied
    """
    to_ret = {}
    for curr_key in curr_ts_dict_set:
        to_ret[curr_key] = scaler_inverse_transform_ts_list_set(curr_ts_dict_set[curr_key], curr_scaler_set)
    return to_ret

def reconcile_ts_dict_set(curr_ts_dict_set, curr_reconciler_list, hierarchy, moh_list, dist_list):
    """
    Takes a dictionary as input where keys are horizons and values are ts list sets 
    and applies a reconciler to each ts in the ts list sets for each horizon

    Parameters: 
    curr_ts_dict_set: A dictionary where keys are horizons and values are ts list sets
    curr_reconciler_list: A list of reconcilers to apply to each ts in the ts list sets
    hierarchy: Hierarchy for the TimeSeries
    moh_list: List of MOH units
    dist_list: List of district units
    
    Returns:
    A dictionary where keys are horizons and values are ts list sets with the reconciler applied
    """
    to_ret = {}
    for curr_key in curr_ts_dict_set:
        to_ret[curr_key] = reconcile_corr_ts_list_set(curr_ts_dict_set[curr_key], curr_reconciler_list, hierarchy, moh_list, dist_list)
    return to_ret


def get_den_loc_mets_tables(curr_metric_dict, curr_eval_dict, admin_level_list, admin_list, fold_weighted = True):
    """
    Generate a num_admin x num_metrics table for each set of observed/forecasted values. An additional summary
    table is generated which averages across folds. The function returns a dictionary with keys "Overall" and each admin level, where the 
    values are lists of DataFrames (one per set) with the last DataFrame being the summary table.

    Parameters:
    curr_metric_dict (dict): A dictionary containing the metrics for each set of observed/forecasted values.
    curr_eval_dict (dict): A dictionary containing the observed and forecasted values for each set of observed/forecasted values.
    admin_level_list (list): A list of admin levels corresponding to the observed and forecasted values.
    admin_list (list): A list of admin units corresponding to the observed and forecasted values.
    fold_weighted (bool): Whether to weight the metrics by the length of the TimeSeries in each set when averaging across folds.
    
    Returns:
    dict: A dictionary with keys "Overall" and each admin level, where the values are
    """
    metric_keys = list(curr_metric_dict.keys())
    sample_arr = curr_metric_dict[metric_keys[0]] #Assume all metrics same shape
    num_sets = sample_arr.shape[0]

    #DataFrame with AdminLedvel and Admin columns - one per set
    to_ret = [pd.DataFrame({"AdminLevel": admin_level_list, "Admin": admin_list}) for i in range(num_sets)]

    #For each set, get metrics arrays in (num_sets, num_admin_units) format - assumed here that only 1 component (hence no 3rd dim)
    for curr_set_ind in range(num_sets):
        curr_set_df = to_ret[curr_set_ind]
        for curr_key in metric_keys: 
            curr_metric_arr = curr_metric_dict[curr_key]
            curr_set_df[curr_key] = curr_metric_arr[curr_set_ind, :]

    if not fold_weighted:
        weights = None
    else:
        #If we weight measures by length of TimeSeries in each set
        val_set_lens = [len(curr_set[0]) for curr_set in curr_eval_dict["obs"]["nat"]] 
        weights = val_set_lens

    summ_df = pd.DataFrame({"AdminLevel": admin_level_list, "Admin": admin_list})
    for curr_key in metric_keys:
        curr_metric_arr = curr_metric_dict[curr_key]
        curr_metric_arr = np.average(curr_metric_arr, axis = 0, weights = weights) #Average across folds
        
        #Add the average to the summary DataFrame
        summ_df[curr_key] = curr_metric_arr

    #Add summ_df to our main list
    to_ret.append(summ_df)

    #We then build admin-level specific versions of this main list
    ret_dict = {"Overall": to_ret}

    #For each admin level
    for curr_admin_level in set(admin_level_list):
        ret_builder = [] 
        for curr_df in to_ret:
            #Filter the current DataFrame to only the specified admin-level
            curr_df_admin = curr_df[curr_df["AdminLevel"] == curr_admin_level]
            ret_builder.append(curr_df_admin)
        ret_dict[curr_admin_level] = ret_builder
        
    #Add a summary row to every DataFrame
    for curr_admin_level in ret_dict:
        #Get the DataFrame list for the current admin unit
        curr_df_list = ret_dict[curr_admin_level]
        
        #For each DataFrame in the DataFrame list
        for curr_ind in range(len(curr_df_list)):
            curr_set_df = curr_df_list[curr_ind]
            #Create a summary row as a dictionary and iterate over each metric key, taking the mean from the DF
            mean_row = {"AdminLevel": "Summary", "Admin": "Mean"}
            min_row = {"AdminLevel": "Summary", "Admin": "Min"}
            max_row = {"AdminLevel": "Summary", "Admin": "Max"}

            for curr_met_key in metric_keys:
                mean_row[curr_met_key] = curr_set_df[curr_met_key].mean()
                min_row[curr_met_key] = curr_set_df[curr_met_key].min()
                max_row[curr_met_key] = curr_set_df[curr_met_key].max()
            summ_row = pd.DataFrame([mean_row, min_row, max_row])
            #Replace the DataFrame in the list with the concatenation of the current DataFrame and the summary row
            curr_df_list[curr_ind] = pd.concat([curr_set_df, summ_row], ignore_index=True)
    
    return ret_dict
            

def ts_list_has_non_finite(ts_list):
    """
    Check if any TimeSeries in the list has non-finite values (NaN or Inf).
    
    Parameters:
    ts_list (list): List of TimeSeries objects.
    
    Returns:
    bool: True if any TimeSeries has non-finite values, False otherwise.
    """
    return any(not np.isfinite(ts.all_values()).all() for ts in ts_list)

def ts_list_set_has_non_finite(ts_list_set):
    """
    Check if any TimeSeries in the list of lists has non-finite values (NaN or Inf).
    
    Parameters:
    ts_list_set (list of lists): List containing lists of TimeSeries objects.
    
    Returns:
    bool: True if any TimeSeries in any list has non-finite values, False otherwise.
    """
    return any(ts_list_has_non_finite(ts_list) for ts_list in ts_list_set)

def format_summ_row(summ_df, add_ids, precision = 4):
    """
    Function takes a DataFrame with the last three rows of a loc x mets table
    where AdminLevel = Summary and AdminUnit = Mean, Min, Max and formats it into
    a single row where values are in the format Mean (Min to Max) 
    """
    val_cols = [c for c in summ_df.columns if c not in ["AdminLevel", "Admin"]]

    mean_row = summ_df.query("Admin == 'Mean'").copy().iloc[0]
    min_row = summ_df.query("Admin == 'Min'").copy().iloc[0]
    max_row = summ_df.query("Admin == 'Max'").copy().iloc[0]
    #Build summary row based on mean, min, and max 
    #If the metric name contains "mic_" - this is "Mean Interval Coverage" and we display it as a percentage
    ret_builder = {
        col: f"{mean_row[col]:.{precision}f} ({min_row[col]:.{precision}f} to {max_row[col]:.{precision}f})" if "mic_" not in col else
                f"{mean_row[col] * 100:.{precision}f}% ({min_row[col] * 100:.{precision}f}% to {max_row[col] * 100:.{precision}f}%)"
        for col in val_cols
    }
    ret_builder = {**add_ids, **ret_builder}
    ret_df = pd.DataFrame([ret_builder])
    return ret_df

def stack_ts_list_set(curr_ts_list_set):
    """
    Takes a list of TimeSeries lists and stacks them along the sample dimension to create a single list of TimeSeries objects.
    Each TimeSeries in the input lists is assumed to have the same time index and shape.

    Parameters:
    curr_ts_list_set (list of list of TimeSeries): A list where each element is a list of TimeSeries objects to be stacked.

    Returns:
    list of TimeSeries: A list of TimeSeries objects where each TimeSeries is the result of stacking the corresponding TimeSeries from the input lists along the sample dimension.
    """
    #Ensure that all the component lists are the same length
    list_lengths = [len(ts_list) for ts_list in curr_ts_list_set]
    assert len(set(list_lengths)) == 1, "All input lists must have the same length."

    to_ret = []
    for i in range(list_lengths[0]):
        #Get the i-th TimeSeries from each list
        ts_to_stack = [ts_list[i] for ts_list in curr_ts_list_set]
        #Stack them along the sample dimension
        stacked_ts = concatenate(ts_to_stack, axis=2)  # Concatenate along the sample dimension
        to_ret.append(stacked_ts)

    return to_ret


#We define a class to hold information on model forecasts which helps in ensembling
class ModelForecast:
    def __init__(self, forecast_name, tuning_recipe, root_dir, cand_name = None):
        self.forecast_name = forecast_name
        self.tuning_recipe = tuning_recipe
        self.model_type = tuning_recipe["model_type"]
        self.root_dir = root_dir
        self.cand_name = cand_name
        self.forecast_dict = None
        self.roll_forecast_dict = None
        self.seas_metrics_dict = None
        self.roll_metrics_dict = None
        
        self.seas_rel_metrics_dict = None
        self.roll_rel_metrics_dict = None


        self.seas_comps = None
        self.roll_comps = None
        self.perf_metric_name = None
        self.perf_mapping = None
        self.roll_perf_mapping = None
    
    def __str__(self):
        return str(self.forecast_name)
    
    def __repr__(self):
        return str(self.forecast_name)
    
    def output_forecasts(self, seas_fc_key, roll_fc_key, den_forecast_pickle_name, den_tuning_recipe_name, forecast_dir_name = "Forecasts"):
        assert self.root_dir is not None, "Root directory is None, cannot output forecasts"
        assert self.model_type == "Ensemble", "We only output ensemble forecasts to prevent overwriting individual model forecasts"
        
        forecast_output_dir = os.path.join(self.root_dir, forecast_dir_name)
        
        #If root_dir does not exist, create it
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        #If forecast_dir does not exist, create it
        if not os.path.exists(forecast_output_dir):
            os.makedirs(forecast_output_dir)

        #Output the forecasts to a pickle file  
        output_dict = {seas_fc_key: self.forecast_dict, roll_fc_key: self.roll_forecast_dict}
        save_pickle(output_dict, os.path.join(forecast_output_dir, den_forecast_pickle_name))
        
        #Also have to output a tuning recipe - this one goes to the root_dir
        output_recipe = {
            "model_type": self.model_type,
            "model_fc_type": "quantile",
            "model_nn": True,
            "model_covs_type": "none"
        }
        save_dict_as_json(output_recipe, os.path.join(self.root_dir, den_tuning_recipe_name))

        #Output components for performance-based ensembles
        #This won't be available for other ensembles
        if self.seas_comps is not None and self.roll_comps is not None:

            seas_comps = self.seas_comps
            roll_comps = self.roll_comps

            #Output components of the ensemble as an Excel file with forecast_key + index as sheet names
            #Rolling forecast ensemble components are labelled with the horizon
            comp_name_builder = []
            comp_sheet_names = []
            for ind, (curr_comps, curr_mapping) in enumerate(zip(seas_comps, self.perf_mapping)):
                #Create DataFrame
                comp_name_builder.append(pd.DataFrame(curr_comps, columns = ["Model", self.perf_metric_name]))
                comp_sheet_names.append(f"{curr_mapping[1]}{curr_mapping[2]}")
                
            curr_horizon_list = list(roll_comps.keys())
            for curr_horizon in curr_horizon_list:
                curr_roll_comps = roll_comps[curr_horizon]
                for ind, (curr_comps, curr_mapping) in enumerate(zip(curr_roll_comps, self.roll_perf_mapping)):
                    #Create DataFrame
                    comp_name_builder.append(pd.DataFrame(curr_comps, columns = ["Model", self.perf_metric_name]))
                    comp_sheet_names.append(f"H{curr_horizon}_{curr_mapping[1]}{curr_mapping[2]}")

            #Output the DataFrame containing component model names and summary metrics to an Excel file
            #This should be in the same directory as all the forecasts
            df_list_to_excel(
                df_list = comp_name_builder,
                path = os.path.join(self.root_dir, "ensemble_components.xlsx"),
                sheet_names = comp_sheet_names
            )

def generate_ensemble(curr_model_list, curr_ensemble_name, curr_ensemble_base_dir, no_log = True, no_tune = True):
    """
    Generates an ensemble of models by combining their forecasts. This is done in a 
    naive manner - not looking at any metrics 
    
    Parameters:
    curr_model_list: List of ModelForecast objects to combine into an ensemble.
    curr_ensemble_name: Name of the ensemble model. Also used to name the output directory for forecasts.
    curr_ensemble_base_dir: Base directory where the ensemble forecasts will be saved.
    no_log: If True, exclude log-scaled forecasts from the ensemble.
    no_tune: If True, exclude forecasts over the tuning set from the ensemble (taking just test forecasts)

    Returns:
    ModelForecast: A new ModelForecast object representing the ensemble, with combined forecasts and metrics.
    """
    #Create an ensemble object 
    curr_ensemble_obj = ModelForecast(curr_ensemble_name, {"model_type": "Ensemble"}, None)

    #Get a sample model so we know the keys to iterate over when building forecast dicts
    sample_model = curr_model_list[0]

    #Combine seasonal forecasts, these are forecast_dict
    ensemble_forecast_dict = {}
    #Get the names of the keys for forecast_dict - these are assumed to be the same across all the models
    forecast_keys = list(sample_model.forecast_dict.keys())

    #Loop through forecast keys and combine all the models
    for curr_forecast_key in forecast_keys:

        if no_tune and "tune" in curr_forecast_key:
            continue
            
        if no_log and curr_forecast_key.endswith("_log"):
            continue

        #Loop through indices for the current forecast key
        curr_set_len = len(sample_model.forecast_dict[curr_forecast_key])
        curr_set_combined = []
        for curr_set_ind in range(curr_set_len):
            #Combine the forecast for models across the current forecast key and index
            to_combine = [curr_model.forecast_dict[curr_forecast_key][curr_set_ind] for curr_model in curr_model_list]
            stacked_fcs = stack_ts_list_set(to_combine)
            curr_set_combined.append(stacked_fcs)
        #Add the combined set to the ensemble_forecast_dict with the current forecast key
        ensemble_forecast_dict[curr_forecast_key] = curr_set_combined

    #Combine rolling forecasts
    roll_forecast_keys = list(sample_model.roll_forecast_dict.keys())
    roll_ensemble_forecast_dict = {}

    #Loop through forecast_keys
    for curr_roll_forecast_key in roll_forecast_keys:
        if no_tune and "tune" in curr_roll_forecast_key:
            continue
            
        if no_log and curr_roll_forecast_key.endswith("_log"):
            continue

        #Loop through horizons for the current forecast key
        curr_horizon_list = list(sample_model.roll_forecast_dict[curr_roll_forecast_key].keys())
        curr_horizon_combined_dict = {}
        for curr_horizon in curr_horizon_list:
            #Get the sample model's horizon forecast to determine the number of sets
            sample_horizon_fc = sample_model.roll_forecast_dict[curr_roll_forecast_key][curr_horizon]
            curr_set_len = len(sample_horizon_fc)
            curr_set_combined = []
            for curr_set_ind in range(curr_set_len):
                #Combine the forecast for models across the current forecast key, horizon, and index
                to_combine = [curr_model.roll_forecast_dict[curr_roll_forecast_key][curr_horizon][curr_set_ind] for curr_model in curr_model_list]
                stacked_fcs = stack_ts_list_set(to_combine)
                curr_set_combined.append(stacked_fcs)
            #Add the combined set to the current horizon's combined dict
            curr_horizon_combined_dict[curr_horizon] = curr_set_combined
        #Add the current horizon's combined dict to the roll_ensemble_forecast_dict with the current forecast key
        roll_ensemble_forecast_dict[curr_roll_forecast_key] = curr_horizon_combined_dict

    #Set the forecast_dict and roll_forecast_dict for the ensemble object
    curr_ensemble_obj.forecast_dict = ensemble_forecast_dict
    curr_ensemble_obj.roll_forecast_dict = roll_ensemble_forecast_dict

    #Set the root dir for the ensemble object
    curr_ensemble_obj.root_dir = os.path.join(curr_ensemble_base_dir, curr_ensemble_name)
    return curr_ensemble_obj

def get_fc_summary_metric(curr_forecast, curr_metrics_sheet_name, metric_name, summ_name = "Mean", summ_col = "Admin", roll = False, horizon = None, rel = False):
    """
    Function takes a ModelForecast object and returns a summary metric from the given sheet with the given metric name.
    summ_name and summ_col are keys to help find the summary row. If roll is False, this is retrieved from the seas_metrics_dict of the 
    ModelForecast object. If roll is True, this is retrieved from the roll_metrics_dict of the ModelForecast object for the given horizon.
    
    Parameters:
    curr_forecast: ModelForecast object assumed to have forecasts and metrics populated
    """
    #Get the DataFrame with metrics
    if not rel:
        if not roll:
            curr_metrics_sheet = curr_forecast.seas_metrics_dict[curr_metrics_sheet_name]
        else:
            assert horizon is not None, "If roll is True, horizon must be specified"
            curr_metrics_sheet = curr_forecast.roll_metrics_dict[horizon][curr_metrics_sheet_name]
    else:
        if not roll:
            curr_metrics_sheet = curr_forecast.seas_rel_metrics_dict[curr_metrics_sheet_name]
        else:
            assert horizon is not None, "If roll is True, horizon must be specified"
            curr_metrics_sheet = curr_forecast.roll_rel_metrics_dict[horizon][curr_metrics_sheet_name]

    #Filter to just the summary row - just one row
    curr_summ_row = curr_metrics_sheet.query(f"{summ_col} == @summ_name")
    
    assert len(curr_summ_row) == 1, f"Expected one summary row for {summ_name} in {curr_metrics_sheet_name}, but found {len(curr_summ_row)} rows."
    
    #Get the summary for the metric we want
    # print(str(curr_forecast) + " " + str(curr_metrics_sheet_name) + " " + str(metric_name) + " " + str(summ_name) + " ROLL: " + str(roll))
    curr_metric = curr_summ_row[metric_name].values[0]
    return curr_metric

def fc_met_pair_merge(top_fcs, base_forecast_key, curr_forecast_ind, roll = False, curr_horizon = None):
    """
    Function takes a list of ModelForecast objects and merges the forecasts and metrics for a given forecast key and index.
    If roll is True, it merges the rolling forecasts and metrics for the given horizon.

    Assumes that base_forecast_key = "_log" and "_log_s" are also present in the forecast_dict and
    roll_forecast_dict of the ModelForecast objects.

    Parameters:
    top_fcs: List of tuples, each a (ModelForecast, metric) pair
    base_forecast_key: Forecast key to merge (e.g., "test", "tune")
    curr_forecast_ind: Index of the forecast to merge
    roll: Boolean indicating whether to merge rolling forecasts (default: False)
    curr_horizon: Horizon for rolling forecasts (required if roll is True)

    Returns:
    Dictionary with keys "nat", "log", and "scaled" containing the merged forecasts

    """
    log_key = base_forecast_key + "_log"
    scaled_key = log_key + "_s"

    top_fc_list = []
    top_fc_list_log = []
    top_fc_list_log_s = []
    
    for fc_met_pair in top_fcs:
        #Get the actual for case object, index[1] is just the metric value
        fc_obj = fc_met_pair[0]

        #If roll is True, get the rolling forecasts for the specified horizon
        if roll:
            assert curr_horizon is not None, "If roll is True, curr_horizon must be specified"
            fc_list = fc_obj.roll_forecast_dict[base_forecast_key][curr_horizon]
            fc_list_log = fc_obj.roll_forecast_dict[log_key][curr_horizon]
            fc_list_log_s = fc_obj.roll_forecast_dict[scaled_key][curr_horizon]
        else:
            #If roll is false, get the seasonal forecasts for the given key
            fc_list = fc_obj.forecast_dict[base_forecast_key]
            fc_list_log = fc_obj.forecast_dict[log_key]
            fc_list_log_s = fc_obj.forecast_dict[scaled_key]
            
        top_fc_list.append(fc_list[curr_forecast_ind])
        top_fc_list_log.append(fc_list_log[curr_forecast_ind])
        top_fc_list_log_s.append(fc_list_log_s[curr_forecast_ind])

    #Stack the forecasts across the models to create ensemble forecasts
    top_fc_ensemble = stack_ts_list_set(top_fc_list)
    top_fc_ensemble_log = stack_ts_list_set(top_fc_list_log)
    top_fc_ensemble_log_s = stack_ts_list_set(top_fc_list_log_s)
    
    return {"nat": top_fc_ensemble, "log": top_fc_ensemble_log, "scaled": top_fc_ensemble_log_s}
    
def gen_perf_ensemble(
    curr_cand_forecasts,
    curr_metric_name,
    top_n,
    curr_horizon_list,
    curr_ensemble_base_dir,
    perf_mapping,
    roll_perf_mapping,
    prefix="EN_Perf_",
    no_log=True,
    rel=False,
    summ_name = "Mean",
    summ_col = "Admin"
):
    """
    Function generates a performance-based ensemble forecast by pooling the top_n candidate forecasts
    based on the specified metric. This function generates a single ModelForecast object, but can select
    different component models for seasonal and rolling forecasts depending on the performance metric selected.
    
    Parameters:
    curr_cand_forecasts: List of ModelForecast objects to consider for the ensemble.
    curr_metric_name: The performance metric to use for selecting top_n forecasts (e.g. log_mcrps)
    top_n: The number of top forecasts to pool for the ensemble.
    curr_horizon_list: List of horizons for which to generate forecasts.
    curr_ensemble_base_dir: Base directory for saving the ensemble forecasts.
    prefix: Prefix for the ensemble forecast names.
    no_log: Flag indicating whether to exclude log forecasts.
    rel: Boolean flag on whether to use relative metrics. If false, use raw metrics.
    summ_name: Name of the summary row to use for metric selection (default is "Mean").
    summ_col: Name of the column to use for filtering the summary row (default is "Admin").

    Returns:
    ModelForecast object representing the performance-based ensemble forecast
    """
    assert len(curr_cand_forecasts) >= top_n, "Number of candidate forecasts must be greater than or equal to top_n"
    
    #Tackle seasonal forecast ensemble first
    seas_forecast_ensemble = {}
    seas_comps = []
    for curr_perf_map in perf_mapping:
        curr_metrics_sheet_name = curr_perf_map[0]
        curr_forecast_key = curr_perf_map[1]
        curr_forecast_ind = curr_perf_map[2]

        log_key = curr_forecast_key + "_log"
        scaled_key = log_key + "_s"

        if curr_forecast_key not in seas_forecast_ensemble:
            seas_forecast_ensemble[curr_forecast_key] = []
        if scaled_key not in seas_forecast_ensemble:
            seas_forecast_ensemble[scaled_key] = []

        if (not no_log) and (log_key not in seas_forecast_ensemble):
            #This means we also need the log forecasts 
            seas_forecast_ensemble[log_key] = []

        #We generate a sorted list of tuples with the first element being the candidate model and the second element being the metric value for that model
        #With this we can select the top_n models
        fc_met_pairs = sorted(
            [
                (
                    cand,
                    get_fc_summary_metric(
                        curr_forecast=cand,
                        curr_metrics_sheet_name=curr_metrics_sheet_name,
                        metric_name=curr_metric_name,
                        rel = rel,
                        summ_name = summ_name,
                        summ_col = summ_col
                    ),
                )
                for cand in curr_cand_forecasts
            ],
            key=lambda x: x[1],
        )
        top_fcs = fc_met_pairs[:top_n]

        top_fc_merge = fc_met_pair_merge(top_fcs, curr_forecast_key, curr_forecast_ind)
        top_seas_ensemble = top_fc_merge["nat"]
        top_seas_ensemble_log = top_fc_merge["log"]
        top_seas_ensemble_log_s = top_fc_merge["scaled"]

        seas_forecast_ensemble[curr_forecast_key].append(top_seas_ensemble)
        seas_forecast_ensemble[scaled_key].append(top_seas_ensemble_log_s)
        
        if not no_log:
            #If we need the log values, retrieve them, stack into an ensemble, and add to the seas_forecast_dict
            seas_forecast_ensemble[log_key].append(top_seas_ensemble_log)
        #Append the names of the model forecast components so we can output them later 
        seas_comps.append(top_fcs)

    #%%
    #Dict in [key][horizon][set_index] order to be used a roll_forecast_dict in output ensemble
    roll_forecast_ensemble = {} 

    #We also create a dict to hold the components of the rolling forecast ensembles for each horizon
    roll_forecast_comps = {curr_horizon: [] for curr_horizon in curr_horizon_list}

    #Tackle rolling forecast ensemble
    for curr_roll_perf_map in roll_perf_mapping:
        curr_metrics_sheet_name = curr_roll_perf_map[0]
        curr_roll_forecast_key = curr_roll_perf_map[1]
        curr_roll_forecast_ind = curr_roll_perf_map[2]

        log_key = curr_roll_forecast_key + "_log"
        scaled_key = log_key + "_s"

        #Create horizon specific lists and attach to the rolling forecast keys
        #E.g. den_test_roll_preds: {1: [], 2:[], 4:[], ....}
        if curr_roll_forecast_key not in roll_forecast_ensemble:
            roll_forecast_ensemble[curr_roll_forecast_key] = {curr_horizon:[] for curr_horizon in curr_horizon_list}
        if scaled_key not in roll_forecast_ensemble:
            roll_forecast_ensemble[scaled_key] = {curr_horizon:[] for curr_horizon in curr_horizon_list}
        if (not no_log) and (log_key not in roll_forecast_ensemble):
            roll_forecast_ensemble[log_key] = {curr_horizon:[] for curr_horizon in curr_horizon_list}
            
        #Loop through horizons here
        for curr_horizon in curr_horizon_list:
            #For each horizon, we generate our sorted list of tuples - 
            #first element being the candidate model and the second element being the metric value for that model
            #With this we can select the top_n models for the given horizon
            fc_met_pairs = sorted(
                [
                    (
                        cand,
                        get_fc_summary_metric(
                            curr_forecast = cand,
                            curr_metrics_sheet_name = curr_metrics_sheet_name,
                            metric_name = curr_metric_name,
                            roll=True,
                            horizon=curr_horizon,
                            rel = rel,
                            summ_name = summ_name,
                            summ_col = summ_col,
                        ),
                    )
                    for cand in curr_cand_forecasts
                ],
                key=lambda x: x[1],
            )
            
            #Get the top_n - already sorted
            top_fcs = fc_met_pairs[:top_n]

            #Combine the rolling forecasts here for the specified horizon 
            top_fc_merge = fc_met_pair_merge(
                top_fcs,
                curr_roll_forecast_key,
                curr_roll_forecast_ind,
                roll=True,
                curr_horizon=curr_horizon,
            )

            top_roll_ensemble = top_fc_merge["nat"]
            top_roll_ensemble_log = top_fc_merge["log"]
            top_roll_ensemble_log_s = top_fc_merge["scaled"]

            #Add the ensemble forecasts to the roll_forecast_ensemble dict for the current horizon
            roll_forecast_ensemble[curr_roll_forecast_key][curr_horizon].append(top_roll_ensemble)
            roll_forecast_ensemble[scaled_key][curr_horizon].append(top_roll_ensemble_log_s)

            if not no_log:
                #If we need the log values, retrieve them, stack into an ensemble, and add to the seas_forecast_dict
                roll_forecast_ensemble[log_key][curr_horizon].append(top_roll_ensemble_log)
                
            #Add the components of the rolling forecasts per horizon 
            roll_forecast_comps[curr_horizon].append(top_fcs)

    #Build the ModelForecast object for the resulting performance-based ensemble
    ret_forecast_name = f"{prefix}{curr_metric_name}_Top{top_n}"
    curr_root_dir = os.path.join(curr_ensemble_base_dir, ret_forecast_name)
    res_obj = ModelForecast(
        forecast_name=ret_forecast_name,
        tuning_recipe={"model_type": "Ensemble"},
        root_dir=curr_root_dir,
    )
    res_obj.forecast_dict = seas_forecast_ensemble
    res_obj.roll_forecast_dict = roll_forecast_ensemble
    res_obj.seas_comps = seas_comps
    res_obj.roll_comps = roll_forecast_comps
    res_obj.perf_metric_name = curr_metric_name
    res_obj.perf_mapping = perf_mapping
    res_obj.roll_perf_mapping = roll_perf_mapping

    return res_obj


def gaverage(ratios, axis=0, weights = None):
    """
    Function to compute the weighted geometric average of a list of ratios.
    If weights is None, the geometric average is unweighted. 
    Otherwise, weights should be a list of the same length as ratios.
    """
    if weights is None:
        #If unweighted, just use the scipy function
        return gmean(ratios, axis=axis)
    else:
        assert len(weights) == ratios.shape[axis], "Weights must be the same length as the number of ratios along the specified axis."
        
        #If weighted, then use np.average on the log transformed values then exponentiate the result to get the geometric mean
        result = np.exp(np.average(np.log(ratios), axis=axis, weights=weights))
        return result

def get_model_set_rel_mets(
    curr_set_key,
    curr_rel_met_key,
    model_metrics_dicts,
    admin_level_list,
    admin_list,
    set_length_dict,
    roll=False,
    curr_horizon=None,
):
    """
    Function to compute relative metrics for different models within a single set. 
    Silently ignores models that do not contain the specified key (e.g. ensembles do not have DenTuneVal)
    Parameters:
    curr_set_key: str, the key for the current set (e.g. "DenTest")
    curr_rel_met_key: str, the metric_key to use to compute the relative metric (e.g. "log_mcrps" to compute "rel_log_mcrps")
    model_metrics_dicts: dict, the dictionary containing the metrics for each model, set, and metric
    admin_level_list: list, the list of admin levels (e.g. ["MOH", "District", "Country"])
    admin_list: list, the list of admin units (e.g. ["MOH1", "MOH2", "District1", "District2", "Sri Lanka"])
    set_length_dict: dict, the dictionary containing the lengths of each set (e.g. {"DenTest": [10, 20, 30]})
    roll: bool, whether to compute relative metrics for rolling forecasts (default: False)
    curr_horizon: int, the current horizon for rolling forecasts (required if roll is True)

    Returns:
    rel_mets_sheets: List of DataFrames with relative metrics (to be outputted)
    rel_mets_sheet_names: List with a name for each of the outputted sheets (based on the set name, e.g. DenTest0, DenTest1, DenTest2, DenTest_Overall, DenTest_GOverall)

    """
    if roll:
        assert curr_horizon is not None, "If roll is True, curr_horizon must be specified"
    #Create a list of dictionaries where each key is a column
    #We then add top this as we iterate through models
    num_sheets = len(set_length_dict[curr_set_key]) + 2 #Number of folds + summary + gsummary
    sheet_names = [curr_set_key + str(i+1) for i in range(num_sheets-2)] + [curr_set_key + "_Overall", curr_set_key + "_GOverall"]
    model_rel_mets_builder = []
    for i in range(num_sheets):
        curr_dict = {"AdminLevel": admin_level_list.copy(), "Admin": admin_list.copy()}
        if i < (num_sheets-2):
            curr_dict["AdminLevel"] += ["Summary", "GSummary"]
            curr_dict["Admin"] += ["Summary", "GSummary"]
        elif i == (num_sheets-2):
            curr_dict["AdminLevel"] += ["Summary"]
            curr_dict["Admin"] += ["Summary"]
        else:
            curr_dict["AdminLevel"] += ["GSummary"]
            curr_dict["Admin"] += ["GSummary"]
        
        model_rel_mets_builder.append(curr_dict)


    # Loop through the current models
    # curr_model_key = model_keys[0]
    model_keys = list(model_metrics_dicts.keys())
    for curr_model_key in model_keys:
        
        if not roll:
            if curr_set_key not in model_metrics_dicts[curr_model_key]:
                print(f"Skipping {curr_model_key} as it does not have metrics for {curr_set_key}")
                continue
            curr_model_mets = model_metrics_dicts[curr_model_key][curr_set_key][curr_rel_met_key]
        else:
            if curr_set_key not in model_metrics_dicts[curr_model_key][curr_horizon]:
                print(f"Skipping {curr_model_key} as it does not have metrics for {curr_set_key}")
                continue
            curr_model_mets = model_metrics_dicts[curr_model_key][curr_horizon][curr_set_key][curr_rel_met_key]

        # Mean performance per admin across all folds in set
        curr_model_mets_per_admin = np.mean(curr_model_mets, axis=0)

        # Mean performance across all admins per fold in set
        curr_model_mets_per_fold = np.mean(curr_model_mets, axis=1)

        # Mean across all admin and folds in set
        curr_model_mets_overall = np.mean(curr_model_mets_per_fold)

        curr_mets_ratios = []
        curr_mets_per_admin_ratios = []
        curr_mets_per_fold_ratios = []
        curr_mets_overall_ratios = []


        # Collect the same values from other models, then compute the ratios and store
        other_model_key = model_keys[1]
        for other_model_key in model_keys:
            # Skip if other_model is the same as curr_model
            if other_model_key == curr_model_key:
                continue
            if not roll:
                if curr_set_key not in model_metrics_dicts[other_model_key]:
                    continue
                other_model_mets = model_metrics_dicts[other_model_key][curr_set_key][curr_rel_met_key]
            else:
                if curr_set_key not in model_metrics_dicts[other_model_key][curr_horizon]:
                    continue
                other_model_mets = model_metrics_dicts[other_model_key][curr_horizon][curr_set_key][curr_rel_met_key]
                
            other_model_mets_per_admin = np.mean(other_model_mets, axis=0)
            other_model_mets_per_fold = np.mean(other_model_mets, axis=1)
            other_model_mets_overall = np.mean(other_model_mets_per_fold)

            # We compute ratios with curr in numerator and other in denominator
            # When ratio is less than 1 then curr outperforms other
            mets_ratio = curr_model_mets / other_model_mets
            mets_per_admin_ratio = curr_model_mets_per_admin / other_model_mets_per_admin
            mets_per_fold_ratio = curr_model_mets_per_fold / other_model_mets_per_fold
            mets_overall_ratio = curr_model_mets_overall / other_model_mets_overall

            # Gather the ratios so we can compute the geometric mean to populate the main sheet
            curr_mets_ratios.append(mets_ratio)
            curr_mets_per_admin_ratios.append(mets_per_admin_ratio)
            curr_mets_per_fold_ratios.append(mets_per_fold_ratio)
            curr_mets_overall_ratios.append(mets_overall_ratio)
            
        curr_mets_ratios_arr = np.array(curr_mets_ratios)
        curr_mets_per_admin_ratios_arr = np.array(curr_mets_per_admin_ratios)
        curr_mets_per_fold_ratios_arr = np.array(curr_mets_per_fold_ratios)
        curr_mets_overall_ratios_arr = np.array(curr_mets_overall_ratios)

        #These are gmeans taken from ratios of arithmetic means (i.e. averaging was already done across folds/admin and we gmean over models)
        curr_mets_gmeans = gmean(curr_mets_ratios_arr, axis=0) #Geometric mean across models - also used for gsumm metrics
        curr_mets_per_admin_gmeans = gmean(curr_mets_per_admin_ratios_arr, axis=0)
        curr_mets_per_fold_gmeans = gmean(curr_mets_per_fold_ratios_arr, axis=0)
        curr_mets_overall_gmeans = gmean(curr_mets_overall_ratios_arr, axis=0)

        #These are direct gmeans (i.e. gmean across folds/admins, then gmean across models)
        #Overall performance across all folds per admin using gmeans
        curr_mets_gaverage_per_admin = gaverage(curr_mets_gmeans, axis = 0, weights = set_length_dict[curr_set_key])
        #Overall performance across all admins per fold using gmeans
        curr_mets_gaverage_per_fold = gaverage(curr_mets_gmeans, axis = 1, weights = None)
        #Single number overall performance across all admins and folds using gmeans
        curr_mets_gaverage_overall = gaverage(curr_mets_gaverage_per_admin, axis = 0, weights = None)

        #To form the non-summary sheets, combine curr_mets_gmeans (3 x 61) with curr_mets_per_fold_gmeans (3, )
        #and curr_mets_gaverage_per_fold (3, ) to form a 3 x 63 array

        non_summ_sheets = np.concatenate(
            [
                curr_mets_gmeans,
                curr_mets_per_fold_gmeans[:, np.newaxis],
                curr_mets_gaverage_per_fold[:, np.newaxis],
            ],
            axis=1,
        )
        summ_sheet = np.append(curr_mets_per_admin_gmeans, curr_mets_overall_gmeans)
        gsumm_sheet = np.append(curr_mets_gaverage_per_admin, curr_mets_gaverage_overall)

        #Add as columns in our DataFrame builders 
        for i in range(non_summ_sheets.shape[0]):
            model_rel_mets_builder[i][curr_model_key] = non_summ_sheets[i, :]
        model_rel_mets_builder[-2][curr_model_key] = summ_sheet
        model_rel_mets_builder[-1][curr_model_key] = gsumm_sheet
        
    rel_mets_sheets = [pd.DataFrame(curr_builder) for curr_builder in model_rel_mets_builder]
    rel_mets_sheet_names = sheet_names
    return rel_mets_sheets, rel_mets_sheet_names

def get_model_rel_mets(curr_forecast_name, rel_mets, seas_rel_mets_dict_raw, rolling_rel_mets_dict_raw, horizon_list): 
    """
    Extracts the specified relative metrics for a given candidate model from the raw relative metrics dictionaries for both seasonal and rolling forecasts.
    
    Parameters:
    - curr_forecast_name (str): The name of the forecast for which to extract metrics.
    - rel_mets (list): A list of relative metric names to extract.
    - seas_rel_mets_dict_raw (dict): A dictionary containing raw seasonal relative metrics, where keys are metric names and values are dictionaries of DataFrames for each sheet.
    - rolling_rel_mets_dict_raw (dict): A dictionary containing raw rolling relative metrics, where keys are metric names and values are dictionaries by horizon, each containing dictionaries of DataFrames for each sheet.
    - horizon_list (list): A list of horizons for which to extract rolling metrics.

    Returns:
    - curr_cand_seas_builder (dict): A dictionary where keys are sheet names and values are DataFrames containing the extracted seasonal metrics for the specified candidate model.
    - curr_cand_roll_builder (dict): A dictionary where keys are horizons, and values are dictionaries where keys are sheet names and values are DataFrames containing the extracted rolling metrics for the specified candidate model.
    """
    curr_cand_seas_builder = {}
    curr_cand_roll_builder = {curr_horizon: {} for curr_horizon in horizon_list}

    #For each metric, retrieve the value for the specified candidate model 
    #This should be just taking one column from each sheet in the Excel file - with each metric
    #having a separate Excel file
    for curr_rel_met in rel_mets:
        curr_rel_met_df_list = seas_rel_mets_dict_raw[curr_rel_met]
        for curr_sheet_name, curr_df in curr_rel_met_df_list.items():
            if curr_sheet_name == "WinCounter":
                continue #Skip the WinCounter sheet as it does not contain metrics
            if curr_sheet_name not in curr_cand_seas_builder:
                curr_cand_seas_builder[curr_sheet_name] = {}
                curr_cand_seas_builder[curr_sheet_name]["AdminLevel"] = curr_df["AdminLevel"]
                curr_cand_seas_builder[curr_sheet_name]["Admin"] = curr_df["Admin"]
                
            curr_cand_seas_builder[curr_sheet_name][curr_rel_met] = curr_df[curr_forecast_name]
                
        #Do the same but for rolling forecasts
        #The main difference is that we have horizon-specific Excel files and thus add an additional 
        #horizon index with the builder
        for curr_horizon in horizon_list:
            curr_rel_met_df_list = rolling_rel_mets_dict_raw[curr_rel_met][curr_horizon]
            for curr_sheet_name, curr_df in curr_rel_met_df_list.items():
                if curr_sheet_name == "WinCounter":
                    continue #Skip the WinCounter sheet as it does not contain metrics
                if curr_sheet_name not in curr_cand_roll_builder[curr_horizon]:
                    curr_cand_roll_builder[curr_horizon][curr_sheet_name] = {}
                    curr_cand_roll_builder[curr_horizon][curr_sheet_name]["AdminLevel"] = curr_df["AdminLevel"]
                    curr_cand_roll_builder[curr_horizon][curr_sheet_name]["Admin"] = curr_df["Admin"]
                
                curr_cand_roll_builder[curr_horizon][curr_sheet_name][curr_rel_met] = curr_df[curr_forecast_name]
        
    #Convert builders into DataFrames
    for curr_sheet_name, curr_dict in curr_cand_seas_builder.items():
        curr_cand_seas_builder[curr_sheet_name] = pd.DataFrame(curr_dict)
    for curr_horizon in horizon_list:
        for curr_sheet_name, curr_dict in curr_cand_roll_builder[curr_horizon].items():
            curr_cand_roll_builder[curr_horizon][curr_sheet_name] = pd.DataFrame(curr_dict)
    
    return curr_cand_seas_builder, curr_cand_roll_builder


def slice_before_ts_list(curr_ts_list, slice_loc, inclusive = True):
    """
    Function to slice a list of time series objects before a given location.
    This is useful for creating training and testing sets from a list of time series.

    Parameters:
    curr_ts_list: List of time series objects
    slice_loc: Location of slice (based on Darts slice_before)
    inclusive: Boolean indicating whether the slice location is inclusive

    Returns:
    sliced_ts_list: List of sliced time series objects
    """
    if inclusive:
        sliced_ts_list = [curr_ts.split_after(slice_loc)[0] for curr_ts in curr_ts_list] 
    else:
        sliced_ts_list = [curr_ts.split_before(slice_loc)[0] for curr_ts in curr_ts_list]

    
    return sliced_ts_list

def get_win_counter(df_list, sheet_names, exc_levels = ["Summary", "GSummary"], non_mod_cols = ["AdminLevel", "Admin"]):
    
    all_model_cols = sorted(set().union(*[set(curr_df.columns) for curr_df in df_list]) - set(non_mod_cols))
    win_counter_builder = {
        sheet: (df.query("AdminLevel not in @exc_levels")
            .drop(columns=non_mod_cols)
            .idxmin(axis=1)
            .value_counts()
            .reindex(all_model_cols, fill_value=0)
        )
        for df, sheet in zip(df_list, sheet_names)
    }
    win_counter_df = pd.DataFrame(win_counter_builder).reset_index().rename(columns={"index": "Model"})
    return win_counter_df