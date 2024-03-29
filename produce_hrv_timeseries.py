import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, interpolate, signal
from scipy.signal import butter, sosfiltfilt, sosfreqz


import biosppy
import pyhrv
from dtw import *
import emd

import shelve
import math
import time
import json
import os

from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import seaborn as sns



sys.path.append("/data/billy/VNS_2/")
from hrv_preprocessor.hrv_preprocessor import hrv_per_segment, produce_hrv_dataframes, save_hrv_dataframes, load_hrv_dataframes, time_dom_keys, freq_dom_keys

def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        sos = butter(order, normal_cutoff, btype="low", analog=False,output="sos")
        return sos

def butter_lowpass_filter(data, cutoff, fs, order=5):
        sos = butter_lowpass(cutoff, fs, order=order)
        y = sosfiltfilt(sos, data)
        return y

# TODO remove this function
def hrv_timeseries(df, segments, segment_onsets, ecg_srate, segment_len_min, v=True):

    save_plots = True
    
    
    QRS_MAX_DIST_THRESH = 0.30 # the minimum cross-correlation value between each QRS and avg QRS(0-1) for a QRS to be considered valid
    RRI_OUTLIER_PERCENTAGE_DIFF_THRESH = 0.30 # when checking RRI against mean of 2 surrounding it
    MAX_RRI_MS = 2200 * 2 # the guinness world record for lowest heart rate is 27bpm. 60/27 = 2.2 / 2200ms, assuming beats are evenly spread. For safety, add multiplier.
  
    def find_nearest(array, value):
            # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array (modified)
            idx = np.nanargmin((np.abs(array - value)))

            return idx

    # produce dataframe w/ cols (timestamp representing start of 5min segment, hrv analysis metrics for that 5 min segment ...)) 
    freq_dom_hrv = []
    time_dom_hrv = []

    segment_labels = []

    # 1996 european heart journal, heart rate variability standards of measurements paper (task force of european society of cardiology ...)
    # reccomends (pg. 360, "Frequency Domain Methods"-> "Technical requirements and reccomendations")
    # to keep track of the "relative" number of RRI that have been interpolated, and their durations
    modification_report_list = [] 

    #print("REMOVE TESTING #1")
    #TESTING_MEAN = []
    #TESTING_STD = []
    #TESTING_2M = []
    #TESTING_NOPEAKS = []


    for i in range(0,len(segments)):
        segment_interval = segments[i]

        #segment_labels.append(segment_interval[0]) # use first timestamp of 5min interval *found in data* as label for segment
        segment_labels.append(segment_onsets[i]) # use exact start timestamp of the 5min segment (value which may not exist in data) in case no timestamps found in interval

        # keep a report of what happens to this to this segment
        modification_report = {}
        modification_report["seg_idx"] = i
        modification_report["excluded"] = False
        modification_report["n_rpeaks_noisy"] = np.NaN
        modification_report["n_RRI_detected"] = np.NaN 
        modification_report["n_RRI_suprathresh"] = np.NaN
        modification_report["suprathresh_values"] = np.NaN
        modification_report["notes"] = ""
        
        if v:
            print(f"\r{i}/{len(segments)-1}", end="")


        # <EXIT_CONDITION>
        # may get the case that no timestamps were found in the data for a segment interval
        # the number found can vary, I assume due to recording issues
        # so if there isn't even enough samples to make 2min (minimum for LF), exit early
        # note; this doesn't account for whether timestamps in the segment are associated with ecg, or with NaN (were a timestamp for another reading)
        if len(segment_interval) < ecg_srate * (60 * 2):

            freq_dom_hrv.append(np.NaN)
            time_dom_hrv.append(np.NaN)

            modification_report["excluded"] = True
            modification_report["notes"] = "Not enough data recorded in this segment interval BEFORE NaN removed"
            modification_report_list.append(modification_report)
		
           
            continue
        # </EXIT_CONDITION>

            

        # to get segment, get the section of the DF between the first timestamp found within this segment interval, and the last
        segment = df[(df["timestamp"] >= segment_interval[0]) & (df["timestamp"] <= segment_interval[-1])]
        
        # replace values that cannot be converted to float with np.NaN
        error_vals = ["-"]  # this was found in taVNS002
        segment["ecg"] = segment["ecg"].replace(error_vals, np.NaN)        

        ecg = segment["ecg"].to_numpy(dtype=np.float64)

        non_NaN_idx = np.invert(pd.isnull(ecg)) #np.isnan replaced with pd.isnull
        ecg = ecg[non_NaN_idx] # remove NaNs (samples of other properties in between ecg samples)
        ecg_timestamps = segment["timestamp"].to_numpy()[non_NaN_idx] # also remove timestamps corresponding to ECG NaN's 


        # <EXIT_CONDITION>
        # if there isn't enough data in the segment AFTER we remove the NaNs (so looking only at ECG)
        if len(ecg) < ecg_srate * (60 * 2):

            freq_dom_hrv.append(np.NaN)
            time_dom_hrv.append(np.NaN)

            modification_report["excluded"] = True
            modification_report["notes"] = "Not enough data recorded in this segment interval AFTER NaN removed"
            modification_report_list.append(modification_report)

            continue
        # </EXIT_CONDITION>



        timevec = np.cumsum(np.concatenate(([0], np.diff(ecg_timestamps)))) # 0 -> ~300,000 (time in ms)


        """ Apply Empirical Mode Decomposition (EMD) to detrend the ECG Signal (remove low freq drift) """

        # perform EMD on the ecg, and take ecg as sum of IMFS 1-3; this is to remove low frequency drift from the signal, hopefully help R peak detection
        imfs = emd.sift.sift(ecg).T

        # <EXIT_CONDITION>
        # if not enough imfs can be detected (this can happen if the data is mostly zeros)
        if len(imfs) < 3:

            freq_dom_hrv.append(np.NaN)
            time_dom_hrv.append(np.NaN)

            modification_report["excluded"] = True
            modification_report["notes"] = "Less than 3 IMFs were produced by EMD"
            modification_report_list.append(modification_report)

            continue
        # </EXIT_CONDITION>

        ecg_emd = sum(imfs[[0, 1, 2]])

        if save_plots:
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(timevec, ecg, c="lightgrey", label="Raw ECG Signal")
            axs[0].plot(timevec, ecg_emd, c="lightcoral", label="ECG Signal w/ EMD Applied")

        # replace the ecg with the detrended signal
        ecg = ecg_emd


        """ Get ECG RPeaks """

        # with reflection to remove edge effects
        reflection_order = math.floor(len(ecg) / 2)
        ecg_reflected = np.concatenate(
            (ecg[reflection_order:0:-1], ecg, ecg[-2:len(ecg) - reflection_order - 2:-1]))
        

        # get rpeak locations, using a "segmenter algorithm" (algorithm to detect R peaks in ECG)
        rpeaks = biosppy.signals.ecg.engzee_segmenter(signal=ecg_reflected, sampling_rate=ecg_srate)["rpeaks"]
        # NOTE: biosppy provides other segmenters. method "biosppy.signals.ecg.ecg()" uses the hamilton segmenter.
        # christov and hamilton are likely valid alternatives to engzee segmenter, but I haven't thoroughly tested.
        # others (e.g ssf and gamboa) didn't seem great
        
        # how many rpeaks should we expect the alg to detect for a reflected piece of ecg?
            # if lowest bpm ever achieved was 27, expect 27 peaks per min, 27*5 for 5 min
            # then use reflection order to work out how many we might expect in the length of reflected ECG we have
        min_rpeaks = (27*5)
        min_rpeaks_in_reflected = min_rpeaks * (len(ecg) / reflection_order) 
        if len(rpeaks) < min_rpeaks_in_reflected: 
            # TODO log these events?

            # likely something has gone wrong: try again with other algorithms as backup
            rpeaks = biosppy.signals.ecg.hamilton_segmenter(signal=ecg_reflected, sampling_rate=ecg_srate)["rpeaks"]

            if len(rpeaks) < min_rpeaks_in_reflected:

                rpeaks = biosppy.signals.ecg.christov_segmenter(signal=ecg_reflected, sampling_rate=ecg_srate)["rpeaks"]
                
                # <EXIT_CONDITION>
                if len(rpeaks) < min_rpeaks_in_reflected:
                    
                    print("LEN RPEAKS (after segmenting) TOO LOW EARLY EXIT")
                    freq_dom_hrv.append(np.NaN)
                    time_dom_hrv.append(np.NaN)

                    #print("REMOVE TESTING #1.5")
                    #TESTING_NOPEAKS.append(i)

                    # produce a report of what has been done to this segment
                    modification_report["excluded"] = True
                    modification_report["notes"] = "Segmenter detected no Rpeaks"
                    modification_report_list.append(modification_report) 


                    continue
                # </EXIT_CONDITION>


        # need to chop off the reflected parts before and after original signal
        original_begins = reflection_order
        original_ends = original_begins + len(ecg)-1

        rpeaks_begins = find_nearest(rpeaks, original_begins)
        rpeaks_ends = find_nearest(rpeaks, original_ends)
        rpeaks = rpeaks[rpeaks_begins:rpeaks_ends]

        # get their position in the original
        rpeaks = rpeaks - original_begins

        # find_nearest may return the first as an element before original_begins
        # as we flipped, the last r peak of the flipped data put before original
        # will be the negative of the first r peak in the original data
        # as we are using argmin(), this will be returned first
        # so, remove any negative indices (r peaks before original begins)
        rpeaks = rpeaks[rpeaks > 0]


        # correct candidate rpeaks to the maximum ECG value within a time tolerance (0.05s by default)
        rpeaks = biosppy.signals.ecg.correct_rpeaks(ecg, rpeaks, sampling_rate = ecg_srate, tol = 0.05)["rpeaks"]

        """ Attempt to remove noise that has been incorrectly identified as QRS """

        # look for noise in the ECG signal by checking if each detected QRS complex is similar enough to the average QRS in this segment
        beats = biosppy.signals.ecg.extract_heartbeats(ecg, rpeaks, ecg_srate)["templates"] # get ECG signal a small amount of time around detected Rpeaks
        avg_beat = np.mean(beats, axis=0) # produce the average/'typical' beat within the segment
        
        # produce a vector of 1 value per QRS of how similar it is to the avg
        # use dynamic time warping (DTW)
        # use z-score to eliminate difference in amplitude
        beats_distance = np.array([dtw(stats.zscore(beats[x]), stats.zscore(avg_beat), keep_internals=True).normalizedDistance for x in range(0, len(beats))])

        # how many beats are too distant from the average
        noisy_beats_idx = np.where(beats_distance > QRS_MAX_DIST_THRESH)[0]

        # get indices of longest consecutive run of valid rpeaks
        run_start = run_end = 0
        runs = [] # (start, end)
        on_noise = True
        for j in range(0, len(rpeaks)):

          
            if j in noisy_beats_idx:
                if on_noise:
                    # we're still on noise
                    pass
                else:
                    # run has ended
                    run_end = j
                    runs.append((run_start, run_end))
                    on_noise = True
            
            else: # a run has begun/continues

                if on_noise: # a run has begun
                    run_start = j
                    on_noise = False

                if j == len(rpeaks) - 1:
                    # we've reached end of segment
                    run_end = j
                    runs.append((run_start, run_end))
                        
        #print(runs)
        

        # <EXIT_CONDITION>
        # discard as NaN if no valid rpeaks were found
        if len(runs) == 0:

            freq_dom_hrv.append(np.NaN)
            time_dom_hrv.append(np.NaN)

            modification_report["excluded"] = True
            modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
            modification_report["notes"] = f"No runs detected - so likely signal was all noise."
            modification_report_list.append(modification_report) 

            continue
        # </EXIT_CONDITION>


        run_lengths = [np.abs(run[1] - run[0]) for run in runs] 
        longest_consecutive = runs[run_lengths.index(max(run_lengths))]
        print(f"Longest run = {longest_consecutive[0]} -> {longest_consecutive[1]}")
        ####

        noisy_rpeaks = rpeaks[noisy_beats_idx] # keep a copy for plotting
        rpeaks = np.delete(rpeaks, noisy_beats_idx)
        

        # <EXIT_CONDITION>
        # if too great a percentage were due to noise
        snr = len(noisy_beats_idx) / len(beats)
        if snr > 0.40:

            freq_dom_hrv.append(np.NaN)
            time_dom_hrv.append(np.NaN)

            modification_report["excluded"] = True
            modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
            modification_report["notes"] = f"Noisy beats {snr}"
            modification_report_list.append(modification_report) 

            continue
        # </EXIT_CONDITION>

       
        # <EXIT_CONDITION>
        # if there isn't enough 
        if len(rpeaks) < min_rpeaks:
                    
            print("LEN RPEAKS (post-noise removal) TOO LOW EARLY EXIT")
            freq_dom_hrv.append(np.NaN)
            time_dom_hrv.append(np.NaN)

            #print("REMOVE TESTING #1.5 (2)")
            #TESTING_NOPEAKS.append(i)

            modification_report["excluded"] = True
            modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
            modification_report["notes"] = "No rpeaks left after noisy rpeaks removed"
            modification_report_list.append(modification_report) 

            continue
        # </EXIT_CONDITION>


        """ Calculate and correct R-R Intervals """
        
        rri = (np.diff(rpeaks) / ecg_srate) * 1000 # ms

	# TEMP; raw RRI plot
        fig3, ax3 = plt.subplots()
        ax3.plot(timevec[rpeaks][:-1], rri, c="black")
        ax3.set_xlabel("Time (ms)")	
        ax3.set_ylabel("ms")
        fig3.savefig(f"saved_plots/{subject}_{i}_RAW_RRI")

        # TEMP: Poincare and clustering
        fig2, ax2 = plt.subplots(1, 2)
        mu, sigma = stats.norm.fit(rri)
        points = np.linspace(stats.norm.ppf(0.01,loc=mu,scale=sigma), stats.norm.ppf(0.9999,loc=mu,scale=sigma),100)
        pdf = stats.norm.pdf(points,loc=mu,scale=sigma)
        ax2[0].hist(rri, len(rri)//10, density=True)
        ax2[0].plot(points, pdf, color='g')
        ax2[0].axvline(mu + (4 * sigma), color="r")


        poincare = np.array([rri[:-1],rri[1:]], dtype=np.float32)
        #poincare = StandardScaler().fit_transform(poincare.T) # set no scale, so can use ms epsilon that is constant across different data  

        # DBSCAN clustering of poincare plot; outliers should be far from main cluster
        #db = DBSCAN(eps=1.5, min_samples=100).fit(poincare)
        #db = OPTICS(min_samples=100, max_eps=1.5, cluster_method="dbscan").fit(poincare) 
        thresh = np.mean(rri) * 0.25
        print(f"THRESH: {thresh}")
        db = DBSCAN(eps=thresh, min_samples=100).fit(poincare.T)
        labels = db.labels_

        poincare=poincare.T
        
        poincare_outliers = np.zeros(len(rri))
        for j in range(0, len(rri)):
            if j == 0:
                if labels[j] == -1:
                    poincare_outliers[j] = 1                         

            if j == len(rri)-1:
               if labels[-1] == -1:
                    poincare_outliers[j] = 1  
                
            # every RRI except first/last will have 2 poincare pairs associated
                # one where it is the second point in the pair, and one where it is the first
                # if there are BOTH marked as outliers, then this is likely a spike
            else:
                if (labels[j-1] == -1) and (labels[j] == -1):
                    poincare_outliers[j] = 1

        # plot non-scaled poincare
        labels_text = ["Valid" if label >= 0 else "Outlier" for label in labels]
        sns.scatterplot(x=rri[:-1], y=rri[1:], hue=labels_text, palette={"Valid": "#000000", "Outlier": "#FF0000"}, ax=ax2[1])
        fig2.savefig(f"saved_plots/{subject}_{i}_HIST_NEW")

        """        
        fig3, ax3 = plt.subplots()
        neighbors = NearestNeighbors(n_neighbors=15).fit(poincare.T)
        distances, indices = neighbors.kneighbors(poincare.T)

        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        ax3.plot(distances)
        ax3.set_ylim(0, 1)
        fig3.savefig(f"saved_plots/{subject}_{i}_ELBOW")
        """

        # <EXIT_CONDITION> # TODO not sure if this situation is possible
        if sum(rri) < (1000 * 60) * 2:
            #TESTING_2M.append(i)

            freq_dom_hrv.append(np.NaN)
            time_dom_hrv.append(np.NaN)

            modification_report["excluded"] = True
            modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
            modification_report["n_RRI_detected"] = len(rri)
            modification_report["notes"] = "Sum of RRI was less than 2Mins"
            modification_report_list.append(modification_report) 

            continue
        # </EXIT_CONDITION>
        

        # <EXIT_CONDITION>
        # if there is only a bit of noise, is there enough consecutive for LFHF?
        # i guess this covers spread? slightly
        # TODO surely we should only keep the longest consecutive
            # - if it was e.g 3m 1m and 1m runs for example, pick only 3m? (use the 3m as the RRI for next)
        if snr > 0.20:
            k = np.sum(rri[longest_consecutive[0]:longest_consecutive[1]])

            if k < (1000 * 60) * 2:
                freq_dom_hrv.append(np.NaN)
                time_dom_hrv.append(np.NaN)

                modification_report["excluded"] = True
                modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
                modification_report["n_RRI_detected"] = len(rri)
                modification_report["notes"] = "Sum of RRI in LONGEST CONSECUTIVE was less than 2Mins"
                modification_report_list.append(modification_report) 

                continue
        # </EXIT_CONDITION>



        # Often RRI contain spikes, when algorithm error has resulted in a beat being missed/noise has mean extra beat detected
        # These spikes must be removed as they will affect HRV metrics
        rri_corrected = np.copy(rri)

        """
        # instead of TKEO to remove spikes, use approach inspired by:
        # https://www.hrv4training.com/blog/issues-in-heart-rate-variability-hrv-analysis-motion-artifacts-ectopic-beats
        #   - which said any RR intervals that differ 20-25% more than the previous
        # and Karlsson et. al "Automatic filtering of outliers in RR intervals before analysis of HRV in Holter recordings"
        #   - which said "RR intervals that differ to mean of *surrounding* intervals", testing from 30-50%.
        # to remove spikes & ectopic beats by:
        #       - checking how each RR interval compares to mean of the previous/next interval
        #       - if difference between RR interval and surrounding mean exceeds 30% of surrounding mean, exclude as spike.
        # (note; ectopic beats in ECG show up as spikes in RR intervals, as do missed beats)
        suprathresh_idx = []
        
        max_traverse = 5
        for j in range(0, len(rri)):

            previous_idx = j-1
 
            while (previous_idx in suprathresh_idx) and (previous_idx > 0):
                previous_idx -= 1

                if (previous_idx < j-max_traverse):
                    # stuck in a rut of going back; just use the one previous
                    previous_idx = j-1
                    break

            if j == 0:
                surrounding_mean = rri[j+1]
            elif j == len(rri)-1:
                surrounding_mean = rri[previous_idx]
            else:
                surrounding_mean = np.mean([rri[previous_idx], rri[j+1]])            

            diff = abs(rri[j] - surrounding_mean)
            #diff = max(abs(rri[j] - rri[j-1]), abs(rri[j] - rri[j+1]))
            
            if rri[j] > MAX_RRI_MS or (diff > surrounding_mean * RRI_OUTLIER_PERCENTAGE_DIFF_THRESH):
                suprathresh_idx.append(j)


        # do in reverse
        for j in np.flip(range(0, len(rri))):

            previous_idx = j+1
 
            while (previous_idx in suprathresh_idx) and (previous_idx < len(rri)-1):
                previous_idx += 1

                if (previous_idx > j+max_traverse):
                    previous_idx = j+1
                    break
            

            if j == 0:
                surrounding_mean = rri[previous_idx]
            elif j == len(rri)-1:
                surrounding_mean = rri[j-1]
            else:
                surrounding_mean = np.mean([rri[previous_idx], rri[j-1]])            

            diff = abs(rri[j] - surrounding_mean)
            
            if rri[j] > MAX_RRI_MS or (diff > surrounding_mean * RRI_OUTLIER_PERCENTAGE_DIFF_THRESH):
                suprathresh_idx.append(j)

        # remove duplicates and sort
        suprathresh_idx = sorted(set(suprathresh_idx))
        """
        suprathresh_idx = np.where(poincare_outliers == 1)[0]
        
        # produce a copy without the RRIs exceeding the threshold, for use in interpolation
        rri_corrected_supra_removed = np.delete(rri_corrected, suprathresh_idx)
        rri_corrected_supra_idx_removed = np.delete(np.array(range(0, len(rri_corrected))), suprathresh_idx)

        # <EXIT_CONDITION>
        # if too many have been detected as outliers
        if sum(rri_corrected_supra_removed) < (1000 * 60) * 3:
            #TESTING_2M.append(i)

            freq_dom_hrv.append(np.NaN)
            time_dom_hrv.append(np.NaN)

            modification_report["excluded"] = True
            modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
            modification_report["n_RRI_detected"] = len(rri)
            modification_report["notes"] = "Sum of corrected RRI with outliers removed was less than 3Mins"
            modification_report_list.append(modification_report) 

            continue
        # </EXIT_CONDITION>
        
        # interpolate points above threshold
        rri_corrected[suprathresh_idx] = np.interp(suprathresh_idx, rri_corrected_supra_idx_removed, rri_corrected_supra_removed)

        modification_report["excluded"] = False
        modification_report["n_rpeaks_noisy"] = len(noisy_beats_idx)
        modification_report["n_RRI_detected"] = len(rri) # how many RRI were detected for the segment originally
        modification_report["n_RRI_suprathresh"] = len(suprathresh_idx)
        modification_report["suprathresh_values"] = rri[suprathresh_idx]
        modification_report["notes"] = ""
        modification_report_list.append(modification_report) 

        if save_plots:
            axs[0].scatter(timevec[rpeaks], ecg[rpeaks], c='springgreen', label="Valid R Peaks")
            axs[0].scatter(timevec[noisy_rpeaks], ecg[noisy_rpeaks], c="r", label="Noisy R Peaks (Removed)")
            axs[0].set_title(f"ECG Data with Detected R Peaks")

            if len(suprathresh_idx) > 0:
                axs[1].plot(timevec[rpeaks][:-1], rri, c="dimgray", label="Pre-processed HRV")
                axs[1].plot(timevec[rpeaks][:-1], rri_corrected, c="crimson", label="Processed HRV")

                

            else:
                axs[1].plot(timevec[rpeaks][:-1], rri, c="crimson", label="HRV")

            # temp poincare
            #axs[1].scatter(timevec[rpeaks][:-1][np.where(poincare_outliers==1)], rri[np.where(poincare_outliers==1)], c="black", label="Poincare Outliers")

            axs[1].set_title(f"HRV Signal")

            fig.suptitle(f"Segment {i} starting {segment_interval[0]}")

            axs[0].set_ylabel("uV")
            axs[1].set_ylabel("ms")

            axs[1].set_xlabel("Time (ms)") # TODO is this correct (previous "Datapoint No.")

            axs[0].legend()
            axs[1].legend()
            
            # save figure to shelf so it can be opened later (similar to MATLAB save plot)
            """
            with shelve.open(f"saved_plots/{subject}_saved_plots") as shelf:
                shelf[f"{i}"] = fig
            """
            
            # save to image for inspection
            # should be 1280x720 (720p)
            fig.set_size_inches(12.80, 7.2)
            fig.savefig(f"saved_plots/{subject}_{i}_NEW", dpi=100)
    

        """ Calculate HRV Metrics """
      
        freq_dom_hrv.append(pyhrv.frequency_domain.welch_psd(nni=rri_corrected, show=False))
        try:
            time_dom_hrv.append(pyhrv.time_domain.time_domain(nni=rri_corrected, sampling_rate = ecg_srate, show=False, plot=False))
        except ZeroDivisionError: # temporary until bug fixed in sdnn_index()
            modification_report_list[-1]["notes"] = "Zero Division Error (probably bug in sdnn_index()), so time domain excluded."
            time_dom_hrv.append(np.NaN)
    

    #print("REMOVE TESTING #3")
    #print(TESTING_MEAN)
    #print(TESTING_STD)
    #print(TESTING_2M)
    #print(TESTING_NOPEAKS)


    if not all(pd.isnull(time_dom_hrv)):
	# get the index of a non-NaN entry (so we can take columns)
        v = np.where(~pd.isnull(time_dom_hrv))[0][0]
        
        # replace any NaNs with a list of NaNs the same shape as valid data
        nan_idx = np.where(pd.isnull(time_dom_hrv))[0] # get the idx of NaNs as list
        nan_replacement = np.full(shape=np.shape(time_dom_hrv[v]), fill_value=np.NaN)
        for nan in nan_idx:
            time_dom_hrv[nan] = nan_replacement

        time_dom_df = pd.DataFrame(time_dom_hrv, index=segment_labels, columns=list(time_dom_hrv[v].keys()))
    else:
        raise Exception("ALL time_dom_hrv are NaN!")

    if not all(pd.isnull(freq_dom_hrv)):
        v = np.where(~pd.isnull(freq_dom_hrv))[0][0]
        
        nan_idx = np.where(pd.isnull(freq_dom_hrv))[0]
        nan_replacement = np.full(shape=np.shape(freq_dom_hrv[v]), fill_value=np.NaN)
        for nan in nan_idx:
            freq_dom_hrv[nan] = nan_replacement
        
        freq_dom_df = pd.DataFrame(freq_dom_hrv, index=segment_labels, columns=list(freq_dom_hrv[v].keys()))
    else:
        raise Exception("ALL freq_dom_hrv are NaN!")

    modification_report_df = pd.DataFrame({"segment_idx":   [i["seg_idx"] for i in modification_report_list], 
                            "excluded":                 [i["excluded"] for i in modification_report_list],
                            "n_rpeaks_noisy":           [i["n_rpeaks_noisy"] for i in modification_report_list],
                            "n_RRI_detected":           [i["n_RRI_detected"] for i in modification_report_list], 
                            "n_RRI_suprathresh":        [i["n_RRI_suprathresh"] for i in modification_report_list], 
                            "suprathresh_RRI_values":   [i["suprathresh_values"] for i in modification_report_list],
                            "notes":                    [i["notes"] for i in modification_report_list]})
   


    return time_dom_df, freq_dom_df, modification_report_df


if __name__ == "__main__":

    """
        Example SUBJECT_ID = taVNS003
    
        Necessary Folder Structure:
            extract_and_merge_data.py
            produce_hrv_timeseries.py
            subject_data/
                SUBJECT_ID/
                    SUBJECT_ID_AX3Vital_MERGED.csv

    """


    ecg_srate = 125

    # extract_and_merge.py must be ran prior to this
    with open("setup.json") as setup:
        setup_dict = json.load(setup)

        #subject = setup_dict["current_subject"]
        all_subjects_dir = setup_dict["all_subjects_dir"]
        subject_mapping = setup_dict["subject_mapping"]



    logfile_loc = "LOGFILE"
    with open(logfile_loc, "w") as logfile:
        logfile.write(f"{time.time()}")
        logfile.write("\n")

    for subject in [subj for subj in sorted(os.listdir("subject_data")) if subj in subject_mapping.keys()]:

        try:

            rng = np.random.default_rng(1905)

            subject_dir = f"{all_subjects_dir}/{subject}"
            print(subject_dir)

            saved_plots_dir = os.path.join(subject_dir, "saved_plots")
            if not os.path.exists(saved_plots_dir):
                    os.makedirs(saved_plots_dir, exist_ok=True)


            START = time.time()


            print("Reading Merged Data ... ")
            merged_df = pd.read_csv(f"{subject_dir}/{subject}_AX3Vital_MERGED.csv", usecols = ["timestamp", "ecg"])


            start = merged_df["timestamp"].iloc[0]
            end = merged_df["timestamp"].iloc[-1]

            duration_ms = end-start
            duration_s = duration_ms / 1000
            duration_m = duration_s / 60
            duration_h = duration_m / 60
            duration_d = duration_h / 24
            print(f"Data spans {duration_d} days.")

            timeline = np.arange(start, end+1)
            

            window_length_min = 5
            window_length_ms = (window_length_min * 60) * 1000

            # create non-overlapping segments, and look for data occuring within these segment intervals (data is unevenly sampled)
            
            #timestamp is an int64, and is ms
            onsets = np.arange(start, end, window_length_ms)

            #print("Gathering Timestamps Segments...")
            print("Gathering Segment Timestamp Start-stops...")
            
            # break timestamps into n-minute segments (each seg is list of timestamps)
            # when we want the data for a timestamp in a segment, we can just go get it from df
            timestamp_startstops = [] 
            timestamps = merged_df["timestamp"].to_numpy()
            seg_start = 0
            seg_end   = 0
            #subspace_start = 0 # what is the idx in timestamps of the start of the current search subspace?
            for i in range(0, len(onsets)-1):

                if i % 100 == 0:
                    print(f"\r{i}/{len(onsets)-1}", end="")

                # define the start/end intervals of this segment (these intervals may/may not exist in the data)
                interval_start = onsets[i]
                interval_end = onsets[i+1]
                
                #timestamps_in_interval = []

                # find any timestamps that DO exist in the data that fall between these intervals
                
                #while timestamps[pointer] >= interval_start and timestamps[pointer] <= interval_end:
                    #timestamps_in_interval.append(timestamps[pointer])
           

                # find the closest start/end timestamps that actually exist in the data
                seg_start = seg_end  # segment will likely start soon after the last one
                while timestamps[seg_start] < interval_start:
                    seg_start +=1
            
                seg_end = seg_start # segment end will definitely be after start, so bring forward
                while timestamps[seg_end] < interval_end:
                    seg_end +=1


            

                """
                expected_delta = (window_length_min * 60) * ecg_srate
                search_subspace =  timestamps[seg_end:seg_end+np.int64(np.ceil(expected_delta*1.5))]
                seg_start = subspace_start + np.nanargmin(np.abs(search_subspace-interval_start))
                seg_end = subspace_start + np.nanargmin(np.abs(search_subspace-interval_end))
                subspace_start += len(search_subspace)
                """
               
                #timestamps_in_interval = timestamps[(timestamps >= interval_start) & (timestamps < interval_end)]
                #timestamps_in_interval = timestamps[seg_start:seg_end]	            

                #timestamp_segments.append(timestamps_in_interval)
                
                timestamp_startstops.append([seg_start, seg_end])
             
            print("\n")

            """
            check_segment_validity = False
            if check_segment_validity:
                print("Checking values collected in segments correspond to original timestamp list...") 
                # iterate over every timestamp value and then each timestamp in each segment in turn,
                # and check that each value corresponds and lines up correctly.
                # i.e test that segments contain the right data, and no values missed (does NOT check that they are the right length)
                ti = si = 0;
                while ti < len(timestamps):
                    curr_timestamp_val = timestamps[ti]
                
                    if si < len(segments):
                        current_segment = segments[si]
                    else:
                        # there will likely be data left over in timestamps (list of all timestamps in data)
                        # that don't fit into strict 5min segmenting, which will be left over when end of segments reached
                        print(f"End of segments reached at timestamp idx {ti}")
                        print(f"Final timestamp in segments was {curr_seg_val}, {timestamps[-1] - curr_seg_val} ms prior to final timestamp in data.")
                        ti = len(timestamps) # exit condition
                    
                    for curr_seg_val in current_segment:
                        print(f"\r{curr_seg_val} -> {curr_timestamp_val}\t\t{ti}/{len(timestamps)-1}\t\t\t\t\t", end="")
                        
                        if curr_seg_val == curr_timestamp_val:
                            ti += 1
                            curr_timestamp_val = timestamps[ti]
                            pass
                        else:
                            print(f"Values were not equal at ti = {ti}")
                            ti = len(timestamps) # exit condition
                    si += 1
            """

            
            # calculate HRV !

            """
            time_dom_df, freq_dom_df, modification_report_df = hrv_timeseries(merged_df, segments, segment_onsets = onsets, ecg_srate=ecg_srate, segment_len_min = window_length_min)

            time_dom_df.to_csv(f"{subject_dir}/{subject}_TIMEDOM_NEW.csv")
            freq_dom_df.to_csv(f"{subject_dir}/{subject}_FREQDOM_NEW.csv")
            modification_report_df.to_csv(f"{subject_dir}/{subject}_MODIFICATION_REPORT_NEW.csv")
            """


            freq_dom_hrvs = []
            time_dom_hrvs = []
            modification_reports = [] 

            segment_labels = []

            print("Gathering Segments...")
            for i, timestamp_startstop in enumerate(timestamp_startstops):

                #segment_labels.append(segment_interval[0]) # use first timestamp of 5min interval *found in data* as label for segment
                segment_labels.append(onsets[i]) # use exact start timestamp of the 5min segment (value which may not exist in data) in case no timestamps found in interval

                print(f"\r{i}/{len(timestamp_startstops)-1}", end="")

                # to get segment, get the section of the DF between the first timestamp found within this segment interval, and the last
                segment = merged_df[(merged_df["timestamp"] >= timestamps[timestamp_startstop[0]]) & (merged_df["timestamp"] <= timestamps[timestamp_startstop[-1]])]
                
                # replace values that cannot be converted to float with np.NaN
                error_vals = ["-"]  # this was found in taVNS002
                segment["ecg"] = segment["ecg"].replace(error_vals, np.NaN)        

                ecg = segment["ecg"].to_numpy(dtype=np.float64)

                ecg = ecg[~np.isnan(ecg)]

                # lowpass filter with cutoff of 22Hz to remove high frequency noise in ECG, make QRS detector's job easier
                if len(ecg) != 0:
                        if len(ecg.shape) > 1:
                                ecg = ecg.T
                                for ch in range(0, ecg.shape[0]):
                                        ecg[ch] = butter_lowpass_filter(ecg[ch], 22, ecg_srate, order=4)
                        else:
                                ecg = butter_lowpass_filter(ecg, 22, ecg_srate, order=4)

                eps = 0.14
                min_samp = 35

                rpeaks, rri, rri_corrected, freq_dom_hrv, time_dom_hrv, modification_report = hrv_per_segment(ecg, ecg_srate, window_length_min, segment_idx=i, save_plots_dir=saved_plots_dir, save_plots=True, save_plot_filename=i, use_segmenter="engzee", DBSCAN_RRI_EPSILON_MEAN_MULTIPLIER=eps, DBSCAN_MIN_SAMPLES=min_samp, rng=rng)
                
                if not isinstance(freq_dom_hrv, float):
                        freq_dom_hrvs.append(np.array(freq_dom_hrv, dtype="object"))
                else:
                        freq_dom_hrvs.append(np.full(shape=freq_dom_keys.shape, fill_value=np.NaN))

                if not isinstance(time_dom_hrv, float):
                        time_dom_hrvs.append(np.array(time_dom_hrv))
                else:
                        time_dom_hrvs.append(np.full(shape=time_dom_keys.shape, fill_value=np.NaN))

                modification_reports.append(modification_report)


            time_dom_df, freq_dom_df, modification_report_df = produce_hrv_dataframes(time_dom_hrvs, freq_dom_hrvs, modification_reports, segment_labels)
            save_hrv_dataframes(time_dom_df, freq_dom_df, modification_report_df, subject_dir)




            END = time.time()
            print(f"START: {START}, END: {END}, END-START: {END-START}")

            with open(logfile_loc, "a") as logfile:
                logfile.write(f"\n{subject}: Success!")

        except Exception as e:

            with open(logfile_loc, "a") as logfile:
                logfile.write(f"\n{subject}: {e}")
