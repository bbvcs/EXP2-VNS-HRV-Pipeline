import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import biosppy
import pyhrv

import shelve
import math

import json

def produce_timeseries(df, segments, column, function, v=True):
    timeseries = []

    i = 0
    for segment_interval in segments:
        
        if v:
            print(f"\r{i}/{len(segments)-1}", end="")
            i+=1
        
        segment = df[(df["timestamp"] >= segment_interval[0]) & (df["timestamp"] <= segment_interval[-1])]
        
        timeseries.append(function(segment[column]))

    return timeseries



if __name__ == "__main__":

    subject = "taVNS003" 
        


    save_merged_df = True

    with open("subject_mapping.json") as sm:
        subject_to_files = json.load(sm)


    subject_dir = f"subject_data/{subject}"

    ecg_filename = f"{subject_dir}/{subject_to_files[subject][0]}"
    vitals_filename = f"{subject_dir}/{subject_to_files[subject][1]}" 
    ax3_filename = f"{subject_dir}/{subject_to_files[subject][2]}"


    print("Gathering Vitalpatch ECG Data...")
    # get ecg data
    ecg_df = pd.read_csv(ecg_filename)
    ecg_nd = ecg_df.to_numpy()
    ecg = ecg_nd.flatten() # ecg data in unusual format

    timestamps = ecg[0:len(ecg):2]
    values = ecg[1:len(ecg):2]

    # reconstruct ecg_df in normal format
    ecg_df = pd.DataFrame({"timestamp": timestamps, "ecg": values})


    print("Gathering Vitalpatch Vitals Data...")
    # get vitals data
    vitals_df = pd.read_csv(vitals_filename, index_col=None, usecols=range(0,8), names=["timestamp", "heart_rate", "respiration_rate", "vitalpatch_temp", "steps", "posture", "RRIP", "patch_battery"])

    # get ax3 data
    print("Gathering AX3 Data...")
    ax3_df = pd.read_csv(ax3_filename)

    # remove any ax3 data recorded after vitalpatch finished recording (necessary for 27F/taVNS003)
    vitalpatch_end = max(ecg_df.iloc[-1]["timestamp"], vitals_df.iloc[-1]["timestamp"])
    ax3_extra = ax3_df.iloc[-1]['timestamp'] - vitalpatch_end
    if ax3_extra > 0:
        print(f"AX3 data contains {ax3_extra} samples following the final vitalpatch sample. Removing these samples...")
        ax3_df = ax3_df.drop(ax3_df.index[ax3_df["timestamp"] > vitalpatch_end])

    print("Merging Vitalpatch and AX3 dataframes ...")
    # create dataframe of times in timeline -> ecg and vital properties (NaN if not present)
    # join tables so we have table of timestamp -> any ecg/vitals/ax3 (either or all) that occured at that time
    merged_df = ecg_df.merge(vitals_df, how="outer", left_on="timestamp", right_on="timestamp", sort=True)
    merged_df = merged_df.merge(ax3_df, how="outer", left_on="timestamp", right_on="timestamp", sort=True)

    if save_merged_df:
        print("Saving Merged Data ...")
        merged_df.to_csv(f"{subject_dir}/{subject}_AX3Vital_MERGED.csv")

    

