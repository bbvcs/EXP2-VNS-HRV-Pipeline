import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import json

import time
import datetime

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


def remove_NaN(arr):
    return arr[np.invert(np.isnan(arr))]

def convert_pyHRV_freqdom_tuplestr_to_tuple(tuple_as_str):
    # for example, fft_log entries look like : '(8.901840598362377, 5.9300714514495, 5.228194636918156)' (strings, not tuple)
    # so convert to a tuple of (8.901840598362377, 5.9300714514495, 5.228194636918156)

    tuple_as_str = tuple_as_str[1:-2] # remove brackets
    tuple_as_str = tuple_as_str.split(", ") 

    return tuple([float(s) for s in tuple_as_str])

def convert_unixtime_ms_to_datetime(unix_epoch_ms):

    # remove ms from unix epoch
    unix_epoch = np.floor(unix_epoch_ms / 1000)

    # convert to datetime format
    timestamp = datetime.datetime.fromtimestamp(unix_epoch)

    # get ms 
    ms = unix_epoch_ms - (unix_epoch * 1000)

    # add ms
    return timestamp + datetime.timedelta(milliseconds=ms) 


def find_nearest(array, value):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array (modified)
    idx = np.nanargmin((np.abs(array - value)))

    return idx


def legend_without_duplicate_labels(ax, **kwargs):
    # https://stackoverflow.com/a/56253636/12909146
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), **kwargs)


if __name__ == "__main__":


    with open("setup.json") as setup:
        setup_dict = json.load(setup)

        subject = setup_dict["current_subject"]
        all_subjects_dir = setup_dict["all_subjects_dir"]
        subject_mapping = setup_dict["subject_mapping"]


    subject_dir = f"{all_subjects_dir}/{subject}"
    summary_spreadsheet = "EXP2_data_summary.xlsx"

    print("Reading TIMESTAMPS from Merged AX3/Vitalpatch Data ...")
    merged_df = pd.read_csv(f"{subject_dir}/{subject}_AX3Vital_MERGED.csv", usecols=["timestamp"]) # if you want all of the merged data, remove usecols parameter
    
    print("Reading Frequency Domain HRV Metrics ...")
    freq_dom_df = pd.read_csv(f"{subject_dir}/{subject}_FREQDOM.csv") 
     
    print("Reading Time Domain HRV Metrics ... ")
    time_dom_df = pd.read_csv(f"{subject_dir}/{subject}_TIMEDOM.csv")


    # timestamp is used as the index column, so has label "Unnamed: 0"
    freq_dom_df = freq_dom_df.rename(columns={"Unnamed: 0" : "timestamp"})
    time_dom_df = time_dom_df.rename(columns={"Unnamed: 0" : "timestamp"})


    all_timestamps = merged_df["timestamp"].to_numpy()
    hrv_timestamps = time_dom_df["timestamp"].to_numpy() # timestamps are the same in freq & time dom dfs


    print("Converting Timestamps ...")
    # WARNING; formatting all timestamps takes a VERY long time!
    # TODO save results to a JSON file/CSV/Python shelve once done?
#    all_timestamps_formatted = np.zeros(shape = (len(all_timestamps)), dtype = datetime.datetime)
#    for i in range(0, len(all_timestamps)):                         
#        all_timestamps_formatted[i] = (convert_unixtime_ms_to_datetime(all_timestamps[i]))
#
#        # print for every integer percentage done (1%, 2%, ... 100%)
#        if i % np.floor(len(all_timestamps) / 100) == 0:            
#            print(f"{i / np.floor(len(all_timestamps) / 100)}%")

    # Formatting only HRV Timestamps doesn't take long though
    hrv_timestamps_formatted = np.vectorize(convert_unixtime_ms_to_datetime)(hrv_timestamps) # timestamps in HRV Dataframes represent the starting time of each 5 min segment 



    basic_id = pd.read_excel(summary_spreadsheet, sheet_name="Ex2 Basic ID") # Contains study start/end dates
    if not subject in list(basic_id["Study ID"]):
        raise Exception(f"{subject} not found in {summary_spreadsheet} - do you have the most up-to-date copy?")

    stimulation_times = pd.read_excel(summary_spreadsheet, sheet_name="Ex2 Stimulation") # Contains times of VNS
    joined_df = basic_id.merge(stimulation_times, how="outer", left_on="Study ID", right_on="Study ID", sort=True)
    
    # get subject info on start/end date and VNS times
    subject_df = joined_df[joined_df["Study ID"] == subject]

    """
    recording_start_day = subject_df["Data start"].iloc[0]
    recording_end_day   = subject_df["Data end"].iloc[0]
    if isinstance(recording_start_day, str):
        recording_start_day = datetime.datetime.strptime(subject_df["Data start"].iloc[0],  "%d/%m/%Y")
    if isinstance(recording_end_day, str):
        recording_end_day   = datetime.datetime.strptime(subject_df["Data end"].iloc[0],    "%d/%m/%Y")
    
    """
    recording_start_day = hrv_timestamps_formatted[0].date()
    recording_end_day = hrv_timestamps_formatted[-1].date()
    
    # get AM/PM VNS times. ASSUMPTIONS; VNS starts the day after recording.
    try:
        #day_2_AM = datetime.datetime.combine(recording_start_day.date() + datetime.timedelta(days=1), subject_df["Day 2 AM"].iloc[0])
        day_2_AM = datetime.datetime.combine(recording_start_day + datetime.timedelta(days=1), subject_df["Day 2 AM"].iloc[0])
    except Exception:
        day_2_AM = None    
    try:
        day_2_PM = datetime.datetime.combine(recording_start_day + datetime.timedelta(days=1), subject_df["Day 2 PM"].iloc[0])
    except Exception:
        day_2_PM = None  

    try:
        day_3_AM = datetime.datetime.combine(recording_start_day + datetime.timedelta(days=2), subject_df["Day 3 AM"].iloc[0])
    except Exception:
        day_3_AM = None    
    try:
        day_3_PM = datetime.datetime.combine(recording_start_day + datetime.timedelta(days=2), subject_df["Day 3 PM"].iloc[0])
    except Exception:
        day_3_PM = None 


    try:
        day_5_AM = datetime.datetime.combine(recording_start_day + datetime.timedelta(days=4), subject_df["Day 5 AM"].iloc[0])
    except Exception:
        day_5_AM = None    
    try:
        day_5_PM = datetime.datetime.combine(recording_start_day + datetime.timedelta(days=4), subject_df["Day 5 PM"].iloc[0])
    except Exception:
        day_5_PM = None 


    try:
        day_6_AM = datetime.datetime.combine(recording_start_day + datetime.timedelta(days=5), subject_df["Day 6 AM"].iloc[0])
    except Exception:
        day_6_AM = None    
    try:
        day_6_PM = datetime.datetime.combine(recording_start_day + datetime.timedelta(days=5), subject_df["Day 6 PM"].iloc[0])
    except Exception:
        day_6_PM = None 



    """ Produce Plot """

    PROPERTY = "fft_ratio"
    FUNC = np.mean # ensure to do without brackets e.g np.mean, not np.mean()
    FUNC_PROPERTY_DESC = "Mean LF/HF Ratio" # what is FUNC doing to PROPERTY?
    DF = freq_dom_df

    # for each property, call FUNC over the property (e.g np.mean over the LF/HF Ratio in this period)
    try:
        day_2_AM_value = FUNC(DF[PROPERTY]
            .iloc[find_nearest(hrv_timestamps_formatted, day_2_AM):find_nearest(hrv_timestamps_formatted, day_2_AM + datetime.timedelta(hours=1))])
    except Exception:
        day_2_AM_value = None
    try:
        day_2_PM_value = FUNC(DF[PROPERTY]
            .iloc[find_nearest(hrv_timestamps_formatted, day_2_PM):find_nearest(hrv_timestamps_formatted, day_2_PM + datetime.timedelta(hours=1))])
    except Exception:
        day_2_PM_value = None        


    try:
        day_3_AM_value = FUNC(DF[PROPERTY]
            .iloc[find_nearest(hrv_timestamps_formatted, day_3_AM):find_nearest(hrv_timestamps_formatted, day_3_AM + datetime.timedelta(hours=1))])
    except Exception:
        day_3_AM_value = None
    try:
        day_3_PM_value = FUNC(DF[PROPERTY]
            .iloc[find_nearest(hrv_timestamps_formatted, day_3_PM):find_nearest(hrv_timestamps_formatted, day_3_PM + datetime.timedelta(hours=1))])
    except Exception:
        day_3_PM_value = None 

    try:
        day_5_AM_value = FUNC(DF[PROPERTY]
            .iloc[find_nearest(hrv_timestamps_formatted, day_5_AM):find_nearest(hrv_timestamps_formatted, day_5_AM + datetime.timedelta(hours=1))])
    except Exception:
        day_5_AM_value = None
    try:
        day_5_PM_value = FUNC(DF[PROPERTY]
            .iloc[find_nearest(hrv_timestamps_formatted, day_5_PM):find_nearest(hrv_timestamps_formatted, day_5_PM + datetime.timedelta(hours=1))])
    except Exception:
        day_5_PM_value = None 

    try:
        day_6_AM_value = FUNC(DF[PROPERTY]
            .iloc[find_nearest(hrv_timestamps_formatted, day_6_AM):find_nearest(hrv_timestamps_formatted, day_6_AM + datetime.timedelta(hours=1))])
    except Exception:
        day_6_AM_value = None
    try:
        day_6_PM_value = FUNC(DF[PROPERTY]
            .iloc[find_nearest(hrv_timestamps_formatted, day_6_PM):find_nearest(hrv_timestamps_formatted, day_6_PM + datetime.timedelta(hours=1))])
    except Exception:
        day_6_PM_value = None 

    
    if (subject_df["Round 1 (Day 2 + 3)"].iloc[0] == "Active") and (subject_df["Round 2 (Day 5 + 6)"].iloc[0] == "Sham"):
        round_1_label = "Round 1 (Day 2 + 3) (Active)"
        round_1_color = "palegreen"    
        round_2_label = "Round 2 (Day 5 + 6) (Sham)"
        round_2_color = "salmon"    

    elif (subject_df["Round 1 (Day 2 + 3)"].iloc[0] == "Sham") and (subject_df["Round 2 (Day 5 + 6)"].iloc[0] == "Active"):
        round_1_label = "Round 1 (Day 2 + 3) (Sham)"
        round_1_color = "salmon"         
        round_2_label = "Round 2 (Day 5 + 6) (Active)"
        round_2_color = "palegreen"  
    else:
        print("ERROR: Please ensure labelling of sham/active VNS is correct.")


    fig, axs = plt.subplots(2, 1)
    axs[0].plot(hrv_timestamps_formatted, DF[PROPERTY])

    if day_2_AM: axs[0].axvspan(day_2_AM, day_2_AM+datetime.timedelta(hours=1), label=round_1_label, color=round_1_color)
    if day_2_PM: axs[0].axvspan(day_2_PM, day_2_PM+datetime.timedelta(hours=1), label=round_1_label, color=round_1_color)
    if day_3_AM: axs[0].axvspan(day_3_AM, day_3_AM+datetime.timedelta(hours=1), label=round_1_label, color=round_1_color)
    if day_3_PM: axs[0].axvspan(day_3_PM, day_3_PM+datetime.timedelta(hours=1), label=round_1_label, color=round_1_color)

    if day_5_AM: axs[0].axvspan(day_5_AM, day_5_AM+datetime.timedelta(hours=1), label=round_2_label, color=round_2_color)
    if day_5_PM: axs[0].axvspan(day_5_PM, day_5_PM+datetime.timedelta(hours=1), label=round_2_label, color=round_2_color)
    if day_6_AM: axs[0].axvspan(day_6_AM, day_6_AM+datetime.timedelta(hours=1), label=round_2_label, color=round_2_color)
    if day_6_PM: axs[0].axvspan(day_6_PM, day_6_PM+datetime.timedelta(hours=1), label=round_2_label, color=round_2_color)

    TEXT_Y = np.nanmax(DF[PROPERTY]) - 10
    TEXT_SIZE = "small"
    TEXT_ROTATION = 90
    TEXT_N_DP = 2
    if day_2_AM: axs[0].text(day_2_AM, TEXT_Y, round(day_2_AM_value, TEXT_N_DP), fontsize=TEXT_SIZE, rotation=TEXT_ROTATION)
    if day_2_PM: axs[0].text(day_2_PM, TEXT_Y, round(day_2_PM_value, TEXT_N_DP), fontsize=TEXT_SIZE, rotation=TEXT_ROTATION)
    if day_3_AM: axs[0].text(day_3_AM, TEXT_Y, round(day_3_AM_value, TEXT_N_DP), fontsize=TEXT_SIZE, rotation=TEXT_ROTATION)
    if day_3_PM: axs[0].text(day_3_PM, TEXT_Y, round(day_3_PM_value, TEXT_N_DP), fontsize=TEXT_SIZE, rotation=TEXT_ROTATION)
    if day_5_AM: axs[0].text(day_5_AM, TEXT_Y, round(day_5_AM_value, TEXT_N_DP), fontsize=TEXT_SIZE, rotation=TEXT_ROTATION)
    if day_5_PM: axs[0].text(day_5_PM, TEXT_Y, round(day_5_PM_value, TEXT_N_DP), fontsize=TEXT_SIZE, rotation=TEXT_ROTATION)
    if day_6_AM: axs[0].text(day_6_AM, TEXT_Y, round(day_6_AM_value, TEXT_N_DP), fontsize=TEXT_SIZE, rotation=TEXT_ROTATION)
    if day_6_PM: axs[0].text(day_6_PM, TEXT_Y, round(day_6_PM_value, TEXT_N_DP), fontsize=TEXT_SIZE, rotation=TEXT_ROTATION)
    

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("LF/HF Ratio")
    axs[0].set_xlim([min(hrv_timestamps_formatted), max(hrv_timestamps_formatted)])
    axs[0].set_title(f"VNS EXP2 {FUNC_PROPERTY_DESC} for Subject {subject}")

    legend_without_duplicate_labels(axs[0], fontsize="x-small")
    
    # scatter the values
    axs[1].scatter(day_2_AM, day_2_AM_value, label=round_1_label, color=round_1_color)
    axs[1].scatter(day_2_PM, day_2_PM_value, label=round_1_label, color=round_1_color)
    axs[1].scatter(day_3_AM, day_3_AM_value, label=round_1_label, color=round_1_color)
    axs[1].scatter(day_3_PM, day_3_PM_value, label=round_1_label, color=round_1_color)

    axs[1].scatter(day_5_AM, day_5_AM_value, label=round_2_label, color=round_2_color)
    axs[1].scatter(day_5_PM, day_5_PM_value, label=round_2_label, color=round_2_color)
    axs[1].scatter(day_6_AM, day_6_AM_value, label=round_2_label, color=round_2_color)
    axs[1].scatter(day_6_PM, day_6_PM_value, label=round_2_label, color=round_2_color)

    axs[1].set_xlabel("Time")
    axs[1].set_ylabel(FUNC_PROPERTY_DESC)
    axs[1].set_xlim([min(hrv_timestamps_formatted), max(hrv_timestamps_formatted)])


    # should be 1280x720 (720p)
    fig.set_size_inches(12.80, 7.2)
    fig.savefig(f"{subject}_plot_data_out", dpi=100)


    # some examples of plotting the data interactively:

    

