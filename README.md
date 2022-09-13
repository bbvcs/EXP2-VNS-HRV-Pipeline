### Setup:

I would recommend running on an Ubuntu/Debian based OS, with at least Python 3.8 installed. If you run into any problems running these scripts please let me know.

1. Follow the instructions at https://github.com/bbvcs/axivity-ax3-collator to setup the script to convert .cwa files to CSV.
2. Enter the following commands: "pipenv shell", followed by "pipenv update". This will setup a pipenv virtual environment, and install all necessary packages for the python scripts. Pipenv combines the "pip" python package manager and the "venv" virtual environment tool. Pipenv uses a "Pipfile" to keep track of dependencies for the project, this file is provided. For more info on pipenv (and how to install if you don't have it): https://pipenv.pypa.io/en/latest/
3. Ensure all files/folders are in the correct structure as shown in the diagram.
    - Please create 2 folders "subject_data" and "saved_plots" in this directory. 
    - For each subject, create a folder in "subject_data" with the study ID, for example "subject_data/taVNS003"
4. Pipeline Usage:
    - Convert the AX3 .cwa to a .csv file.
    - Before running any python scripts, ensure you have ran "pipenv shell"; this puts you in a virtual environment, with all the necessary modules/libraries available.
    - There is a "subject" variable at the top of each python script - change this depending on the subject you're running it over. 
    - Run "python3 extract_and_merge_data.py" to merge all the AX3 & Vitalpatch data into a single CSV.
    - Run "python3 produce_hrv_timeseries.py" to produce HRV Frequency & Time Domain Metrics CSVs. 
    - Run "python3 -i plot_data.py". This loads the merged CSV and the HRV CSV's so that all data is available to be plotted/processed. Running in interactive mode (-i) means you can make any  matplotlib calls after all data has been loaded. 






### Files

##### SUBJECT MAPPING
- a json file, containing study ID -> ecg, vitals, and ax3 .csv filenames
- example entry: "taVNS001": ["VC2B008BF_02CA07_ecgs.csv",  "VC2B008BF_02CA07_vitals.csv", "AX3_taVNS001.csv"]
- the ecg & vitals filenames are as they are in the data OneDrive, under Patch_data/output
- all these files should be placed in the "subject_data" folder, under a folder for the subject ID (e.g "taVNS001")
- the ax3 .csv files are produced by using a script (see step 1 and next step "**AX3 COLLATOR**")
    

##### AX3 COLLATOR
- A simple C++ command line tool designed to read Light, Temperature and Accelerometry (Gyro on AX6 and Gyro/Mag on AX9 should work also) data from Axivity AX-Series Data Logger .CWA file and merge into a single .CSV file.
- Example usage: "./ax-collator -i "subject_data/taVNS003/taVNS003_46902_0000000003.cwa" -o "subject_data/taVNS003/AX3_taVNS003.csv"
- compared to the AX3 CSV's available on the OneDrive, this script will produce a CSV with temp/light values (not just accel), and timestamps in the same format as the vitalpatch.
    - ENSURE to use the correct output ax3 .csv filename (as specified by the subject mapping json file) when running the ax3 C++ script, and that it is placed in the correct location.
- I'm not sure that I have the light scaling correct in this script - I used the formula provided, but it seems a bit high. Please let me know if this is the case.

##### EXTRACT AND MERGE DATA
- This script will load the ECG, Vitals .csv's and the AX3 .csv produced in the last stage, and merge them using the timestamp column.
- The merged pandas dataframe will be saved as a .csv to the subject_data folder, for that subject.
- Warning; this script will use a lot of memory (assuming 7-day long datasets, as in VNS EXP2), you will probably need at least 32GB. 

##### PRODUCE HRV TIMESERIES
- This will load the merged .csv (only the ECG column)
    - This could be modified to just load the vitalpatch ecg .csv (see how this is loaded in extract_and_merge_data.py to do so)
1. Break the ECG into n-minute long segments (by default, 5min)
2. For each segment:
    - Use Empirical Mode Decomposition (EMD) to remove low-frequency drif from/straighten-out the signal, intended to help the R peak detection algorithm.
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
        - Interpolate points we determined as outliers.  

3. NaN Exit Conditions
    - There are some cases where we would want to discard a segment from HRV calculation. I've done this by setting the HRV Time/Freq value as NaN for certain segments, rather than a dictionary of HRV metrics.
    - The following are the current conditions for this, and are fairly arbitrary:
        - The segmenter algorithms (multiple are tried in case of a fault in any particular one) didn't give an R Peak count above the *minimum*.
        - All R Peaks were determined to be associated with noise.
        - If more than 40% of all R Peaks were determined to be associated with noise.
        - If the number of R Peaks did not meet the *minimum* after removal of those associated with noise.
        - The sum of the RRI is less than 2 minutes, the minimum for calculation of LF HRV Frequency Domain Metrics.
        - If more than 20% of all R Peaks were determined to be associated with noise, check that the sum of the RRI related to the longest consecutive run of Valid R Peaks (not associated with noise) is greater than 2 minutes - if not, remove.
            - I did this to try to remove segments where the noise is spread evenly/frequently throughout a segment, meaning interpolation occurs too often.
            - If the noise is only contained to a small part of the segment, and most of the data is valid, it may still be OK for HRV.
    - Note the "*minimum*" number of R Peaks is currently 27 per minute (27bpm).
    - These definitely require improvement and refinement:
        - A histogram could be used to determine the reasonable number of R Peaks per beat for each subject.
        - If, for example, the longest consecutive run of valid R Peaks is 2 minutes, and all other consecutive runs are less than 2 minutes, we should discard the rest of the data in this segment, only keeping the consecutive run.
    - A note of the problem that caused the segment to be removed is stored in the **Modifications CSV**.
    
4. Return Values
    - CSVs for Segment Start Time ID -> Time/Frequency Domain Metrics (seperately)
        - Note - ignore Time Domain SDNN and TINN; these cannot be calculated currently (SDNN requires 24 hour long data, *pyHRV* throws a warning that a bug is causing TINN values to be incorrectly calculated) (I'll remove these)
    - **Modifications CSV**; for each Segment, was it excluded or not (T/F), what was the number of noisy QRS, what was the number of RRI determined as spikes/outliers in the HRV Signal, and some brief notes if it was excluded.

##### PLOT DATA
- You'll need *EXP2_data_summary.xlsx* (EXP2 OneDrive) in the same directory as the script for this, to get info on the experiment start/end times and VNS times. 
- All data I believe is relevant for further processing/producing plots for one subject is loaded in this script; the experiment summary sheet, the HRV Time & Frequency metrics, and the Merged AX3/Vitalpatch .csv.  

##### MISC
- You can set the subject used in each of the Python scripts, by changing the "subject" variable at the top of the "if __name__ == '__main__':" section.
- Plots of the ECG segments, the corresponding HRV signal, and corrections made to both of these, are saved to the "saved_plots" directory. Currently this is only segments that are not excluded (NaN Exit Conditions)

##### FUTURE WORK
- A way to automate this for all subjects.
- Improve & perfect RRI outlier/spike detection and correction.
- Investigate usage of Outlier Detection Algorithms (*pyOD* library provides many) for:
    - Replacing a Threshold in the DTW Distance vector for ECG Noise detection.
    - Determining RRI outliers/spikes in the HRV signal.
- Modifications CSV may need more information to be useful.


Billy C. Smith, 13/09/2022
