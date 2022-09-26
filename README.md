# HRV Processing Pipeline for VNS EXP 2

### Setup:

I would recommend running on an Ubuntu/Debian based OS, with at least Python 3.8 installed. If you run into any problems running these scripts please let me know.

1. Follow the instructions at https://github.com/bbvcs/axivity-ax3-collator to setup the script to convert .cwa files to CSV.
2. Enter the following commands: "pipenv shell", followed by "pipenv update". This will setup a pipenv virtual environment, and install all necessary packages for the python scripts. Pipenv combines the "pip" python package manager and the "venv" virtual environment tool. Pipenv uses a "Pipfile" to keep track of dependencies for the project, this file is provided in this repository. For more info on pipenv (and how to install if you don't have it): https://pipenv.pypa.io/en/latest/
3. Ensure all files/folders are in the correct structure as shown in the diagram.
    - Please create 2 folders "subject_data" and "saved_plots" in this directory. 
    - For each subject, create a folder in "subject_data" with the study ID, for example "subject_data/taVNS003"
4. Pipeline Usage Overview:
    - Convert the AX3 .cwa to a .csv file.
    - Before running any python scripts, ensure you have ran "pipenv shell"; this puts you in a virtual environment, with all the necessary modules/libraries available.
    - Check setup.json; ensure the data directory is correct, and set the subject to use (e.g taVNS006).
    - Run `python3 extract_and_merge_data.py` to merge all the AX3 & Vitalpatch data into a single CSV.
    - Run `python3 produce_hrv_timeseries.py` to produce HRV Frequency & Time Domain Metrics CSVs. 
    - Run `python3 -i plot_data.py`. This loads the merged CSV and the HRV CSV's so that all data is available to be plotted/processed. Running in interactive mode (`-i`) means you can make any  matplotlib calls after all data has been loaded. 






### Files

##### SETUP .JSON
- A JSON file, containing the current subject to use, the location on disk where all subjects are kept, and a mapping of `study ID -> [ecg, vitals, ax3]` .csv filenames
- Example entry for subject mapping: `"taVNS001": ["VC2B008BF_02CA07_ecgs.csv",  "VC2B008BF_02CA07_vitals.csv", "AX3_taVNS001.csv"]`
    - The ecg & vitals filenames are as they are in the data OneDrive, under Patch_data/output
    - All these files should be placed in the "subject_data" folder, under a folder for the subject ID (e.g "taVNS001")
    - The ax3 .csv files are produced by using a script (see step 1 and next step "**AX3 COLLATOR**")
        

##### AX3 COLLATOR
- "*A simple C++ command line tool designed to read Light, Temperature and Accelerometry (Gyro on AX6 and Gyro/Mag on AX9 should work also) data from Axivity AX-Series Data Logger .CWA file and merge into a single .CSV file*"
- Example usage: `./ax-collator -i "../subject_data/taVNS003/taVNS003_46902_0000000003.cwa" -o "../subject_data/taVNS003/AX3_taVNS003.csv` (run this inside the folder "axivity-ax3-collator")
- compared to the AX3 CSV's available on the OneDrive, this script will produce a CSV with temp/light values (not just accel), and timestamps in the same format as the vitalpatch.
    - **Make sure to use the correct output ax3 .csv filename** (as specified by the subject mapping JSON file) when running the ax3 C++ script, and that it is placed in the correct location.
- I'm not sure that I have the light values correct in this script - I used the formula provided, but they seem a bit high. Please let me know if you think this is the case also.

##### EXTRACT AND MERGE DATA
- This script will load the ECG, Vitals .csv's and the AX3 .csv produced in the last stage, and merge them using the timestamp column.
- The merged pandas dataframe will be saved as a .csv to the subject_data folder, for that subject.
- Warning; this script will use a lot of memory (assuming 7-day long datasets, as in VNS EXP2), you will probably need at least 32GB. 

##### PRODUCE HRV TIMESERIES
- This will load the merged .csv (but only the ECG column)
    - Actually, this could be modified to just load the vitalpatch ecg .csv (see how this is loaded in extract_and_merge_data.py to do so)
1. Break the ECG into n-minute long segments (by default, 5min)
    - Going from the first timestamp in the data, a list of segment intervals/onsets are produced; take a timestamp, determine what the timestamp 5 minutes after this timestamp would be.
    - So, timestamps in this list may/may not actually be in the data.
    - To work out the timestamps in each 5 min segment that are actually in the data, take 2 successive onset timestamps, and find timestamps in the data between these 2 values.
    - So due to different sample rates between different metrics, there may be a slightly different number of readings in each segment.
        - But, for example, if you only take readings of a particular metric (e.g ECG) in each segment, the number readings per segment should be fairly similar.
2. For each segment:
    - Use Empirical Mode Decomposition (EMD) to remove low-frequency drift from/straighten-out the signal, intended to help the R peak detection algorithm.
    - Use an R peak detection algorithm provided by *biosppy* (I'm currently using EngZee, but many others are available - Christov & Hamilton's also look good)
    - Use *biosppy*'s rpeak correction - this basically just moves the detected R peak to a local maximum in the ECG signal, useful as sometimes the detection algorithm misses it slightly (e.g places it on the T wave)
    - Attempt to remove R peaks that may have mistakenly been placed on a noisy QRS complex.
        - *biosppy* provides a function to get the "templates" - the ECG snippets associated with each R peak it detected.
        - store all of these in an array, then determine the "average QRS" for the segment, by taking the mean of each template.
        - Use Dynamic Time Warping (DTW) to get a distance value for each beat from this average - the noisy beats should correlate poorly, and can be removed using a threshold (currently, any that have a distance value above 0.30 are removed.)
    - Use the R peak locations to calculate the R-R interval (RRI) series for this segment.
    - Noise/Misidentified R peaks/Ectopic beats etc all cause spikes in the RRI series, which are detrimental to HRV calculation - so we must try to correct these.
        - Iterate forwards over the RRI series, for each point:
            - Calculate the mean of the RRI either side of it.
                - if the point behind was determined to be an outlier/spikes, take the point before that instead.
                - Only try this 5 times - If all 5 behind the point are outliers, just use the point immediately behind it.
            - Calculate the difference between this RRI and the surrounding mean.
            - If this difference is greater than a percentage (30%) of the surrounding mean, then this RRI is an outlier.
        - Do the same, iterating backwards over the series - so now if the point in front of an RRI was previously determined as an outlier, we can take the next point instead, but again only try for 5 points.
        - Remove duplicates from the list of outlier points built up over both runs
        - Interpolate points we have determined as outliers.  

3. NaN Exit Conditions
    - There are some cases where we would want to discard a segment from HRV calculation.
    - The following are the current conditions for this, and are fairly arbitrary:
        - If, for whatever reason, not enough ECG readings occurred in the segment interval (lost data due to recording issues)
        - If EMD produced less than 3 IMFs.
        - The segmenter algorithms (multiple are tried in case of a fault in any particular one) didn't give an R Peak count above the *minimum*.
        - All R Peaks were determined to be associated with noise.
        - If more than 40% of all R Peaks were determined to be associated with noise.
        - If the number of R Peaks did not meet the *minimum* after removal of those associated with noise.
        - The sum of the RRI is less than 2 minutes, the minimum for calculation of LF HRV Frequency Domain Metrics.
        - If more than 20% of all R Peaks were determined to be associated with noise, check that the sum of the RRI related to the longest consecutive run of Valid R Peaks (not associated with noise) is greater than 2 minutes - if not, remove.
            - I did this to try to remove segments where the noise is spread evenly/frequently throughout a segment, meaning interpolation occurs too often, so won't resemble real data.
            - If the noise is only contained to a small part of the segment, and most of the data is valid, it may still be OK for HRV.
    - Note the "*minimum*" number of R Peaks is currently 27 per minute (27bpm).
    - Usually, a dictionary of time/frequency domain HRV metrics is saved per segment. Discarding a segment is currently implemented by saving a NaN instead.
    - These definitely require improvement and refinement:
        - A histogram could be used to determine the reasonable number of R Peaks per beat for each subject.
        - If, for example, the longest consecutive run of valid R Peaks is 2 minutes, and all other consecutive runs are less than 2 minutes, we should discard the rest of the data in this segment, only keeping the consecutive run.
	 	- and much more!
    - A note of the problem that caused the segment to be removed is stored in the **Modifications CSV**.
    
4. Return Values
    - CSVs for Segment Start Time ID -> Time/Frequency Domain Metrics (seperately)
        - Note - ignore Time Domain SDNN and TINN; these cannot be calculated currently (SDNN requires 24 hour long data, *pyHRV* throws a warning that a bug is causing TINN values to be incorrectly calculated) (I'll remove these at some point, and make the HRV CSV's more nicely formatted)
    - **Modifications CSV**; for each Segment, was it excluded or not (T/F), what was the number of noisy QRS, what was the number of RRI determined as spikes/outliers in the HRV Signal, and some brief notes if it was excluded.

##### PLOT DATA
- You'll need *EXP2_data_summary.xlsx* (EXP2 OneDrive) in the same directory as the script for this, to get info on the experiment start/end times and VNS times. 
- All data relevant for further processing/producing plots for one subject is loaded in this script; the experiment summary sheet, the HRV Time & Frequency metrics, and the Merged AX3/Vitalpatch .csv. 
- NOTE; if you're plotting frequency domain values where results are stored in a tuple per frequency band (e.g `fft_log`), this script has a function `convert_pyHRV_freqdom_tuplestr_to_tuple()` which will allow you to access these values. For instance, an `fft_log` entry looks like: `"(8.90, 5.93, 5.23)"`; while it looks like a tuple, it is actually a string. This method will convert it into a usable tuple. 

##### MISC
- Plots of the ECG segments, the corresponding HRV signal, and corrections made to both of these, are saved to the "saved_plots" directory. Currently this is only segments that are not excluded (NaN Exit Conditions)

##### FUTURE WORK (also see presentation)
- A way to automate this for all subjects.
- Improve & perfect RRI outlier/spike detection and correction.
- Investigate usage of Outlier Detection Algorithms (*pyOD* library provides many) for:
    - Replacing a Threshold in the DTW Distance vector for ECG Noise detection.
    - Determining RRI outliers/spikes in the HRV signal.
- Modifications CSV may need more information to be useful.
- I'm not sure, but the only barrier to running this on Windows may be file path strings - these should be replaced using the *os* module (*os.path*). 


Billy C. Smith, 14/09/2022
