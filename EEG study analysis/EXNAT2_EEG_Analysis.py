#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

EXNAT-2 EEG DATA ANALYSIS PIPELINE

Author: Merle Schuckart
Version: November, 2023


"""


# ----------------------------------------------------- # 

""" General information / random stuff that's maybe good to know: """

# This is how the EEG data were recorded in the lab:
# sampling rate: 1000 Hz
# nr of channels: 64, one of which is the reference
# reference electrode: TP9 --> channel not included in the raw data!
# filter: DC-280 Hz

# We used the BrainVision Recorder for recording the EEG.

# For each participant, we get 3 files from the EEG, one .eeg file (containing the raw eeg data), 
# one .vhdr file (containing metadata and configuration information for the EEG recording, 
# such as the sampling rate, number of channels, electrode locations, ...), and
# one .vmrk file (containining triggers). 

# So for participant 001, it would look like this:
# "part_001/EXNAT_part001_1.eeg",
# "part_001/EXNAT_part001_1.vhdr",
# "part_001/EXNAT_part001_1.vmrk"


# Please make sure to pip install MNE python 
# (or updating your version if it's older) 
# before running this script.


# ----------------------------------------------------- # 

""" Import Packages """

# for setting paths:
import os
#os.chdir('/Volumes/MERLE 1/EXNAT-2/EEG_study_EXNAT2/EXNAT-2 Win7 Experiment EYELINK/Analysis/')

import sys

import numpy as np

# for dataframes
import pandas as pd

# import MNE
# --> make sure it's pip installed 
!pip install mne
import mne

from mne.io import read_raw_eyelink

import re # for regular expressions

# for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap #  for plotting bridging between electrodes


# ----------------------------------------------------- # 

# set path to data file here:
#curr_data_path = "/Users/merleschuckart/Github/PhD/EXNAT/EEG_study_EXNAT2/EEG study analysis/Data/"
curr_data_path = "/Users/merle/Github/PhD/EXNAT/EEG_study_EXNAT2/EEG study analysis/Data/"

# get list of files in data directory: 
file_list = os.listdir(curr_data_path)

# Only get files that don't end with some file suffix like .py or .DS_Store, 
# those are the folders for each participant.
file_list = [item for item in file_list if ".py" not in item and ".DS_Store" not in item and ".fif" not in item]


# ----------------------------------------------------- # 


# loop folders & read in data for each participant.

for curr_file in file_list: 
    
    # get ID of current participant
    curr_id = curr_file.replace("part_", "")
    print("reading in data for participant with ID", curr_id)
    
    # skip pilot datasets
    if curr_id == "eg": 
        print("skipping test dataset")
        continue
    
    
    """ create MNE raw object with raw EEG data + triggers + metadata """
    
    # read in the 3 EEG-related datasets
    curr_vhdr_file = curr_data_path + "part_" + curr_id + "/part" + curr_id + ".vhdr"
        
    # read in .eeg file with EEG data EEG data using MNE
    raw = mne.io.read_raw_brainvision(curr_vhdr_file, preload = True)
    
    # Print information about the Raw object: Now you can check if all looks 
    # as intended or if channels are missing for example.
    #print(raw.info)
    #print(raw.ch_names)
    
    # We can't see the channel TP9, but that's fine.
    # We used it as a reference channel during recording, so it should be completely flat.
    # We can now add a completely flat channel to the channel list in case we want to re-reference later.
    # I was a bit confused about the missing channel, but it's all described here: https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html
    
    # add TP9 reference channel (all zeros)
    raw = mne.add_reference_channels(raw, ref_channels = ["TP9"])

    # re-reference using a common average reference. This might take a while:
    raw.set_eeg_reference(ref_channels = 'average')
        
    
    # save backup of raw object in the data folder: 
    raw.save((curr_data_path + "part_" + curr_id + "/backup_raw.fif"), overwrite = True)
    
    
    """ Set Montage (aka Sensor Locations) """
    
    # use basic 10_20 system as montage for all participants:
    #montage = mne.channels.make_standard_montage("standard_1020") 
    #raw.set_montage(montage, verbose = False)
    #montage.plot()
    
    # use custom montage:
        
    # path to .elc file with participant's head shape points & sensor locations:
    curr_elc_file = curr_data_path + "part_" + curr_id + "/part" + curr_id + ".elc"
    
    # extract information from .elc file
    with open(curr_elc_file, 'r') as file:
        lines = file.readlines()
        
    # get header info
    header_info = lines[0:3]
    
    # Extract channel positions:
    # get the raw lines containing information on each channel:
    channel_lines = lines[3:3 + int(header_info[0].split('=')[1])]
    # split strings if there's a '\t' in there:
    channel_data = [line.strip().split('\t') for line in channel_lines]
    # exclude strings containing just ":":
    channel_data = [[item for item in line if ":" not in item] for line in channel_data]
    
    # MNE wants montage channel labels that are not completely in caps, so change them:
    name_mapping = {'FP1': 'Fp1', 
                    'FZ': 'Fz', 
                    'PZ': 'Pz', 
                    'OZ': 'Oz', 
                    'CZ': 'Cz',
                    'FP2': 'Fp2', 
                    'AFZ': 'AFz',
                    'FCZ': 'FCz', 
                    'POZ': 'POz', 
                    'CPZ': 'CPz'}

    # update the channel labels in channel_data
    for i, channel_info in enumerate(channel_data):
        old_name = channel_info[0]
        new_name = name_mapping.get(old_name, old_name)  # Use the new name if it exists, otherwise keep the old name
        channel_data[i][0] = new_name
    
    # convert data into dictionary with Keys = channel names and values = 3D coordinates - array of shape (3,):
    channel_dict = {line[0]: list(map(float, line[1:])) for line in channel_data}


    # extract information on nasion position, 
    # position of the left periauricular fiducial point 
    # in shape (3,)
    nasion_pos = channel_dict.get('Nasion')
    lpa_pos = channel_dict.get('LeftEar')
    rpa_pos = channel_dict.get('RightEar')

    # remove the entries containing info on nasion, lpa and rpa from dict 
    # as they aren't EEG channels.
    channel_dict = {key: value for key, value in channel_dict.items() if key not in ['Nasion', 'LeftEar', 'RightEar']}
    
    
    # extract head shape points, but create an array of shape (n_points, 3) 
    # instead of a dict as they have no labels.

    # get index of row containing header "HeadShapePoints":
    head_shape_start = lines.index('HeadShapePoints\n')

    # extract head shape points
    head_shape_data = lines[head_shape_start +1 : ]

    # Split the data into columns
    head_shape_matrix = [list(map(float, line.split())) for line in head_shape_data]
    
    # convert to numpy array of shape (n_points, 3):
    head_shape_array = np.array([list(map(float, point.split('\t'))) for point in head_shape_data], dtype=float)


    custom_montage = mne.channels.make_dig_montage(ch_pos = channel_dict, 
                                          nasion = nasion_pos,
                                          lpa = lpa_pos,
                                          rpa = rpa_pos,
                                          hsp = head_shape_array,
                                          hpi = None,
                                          coord_frame = "unknown")
                                          
    
    # Plot the 3D montage (hint: you can move it with your cursor)
    #mne.viz.plot_montage(montage = custom_montage, 
    #                     scale_factor = 20, 
    #                     show_names = True, 
    #                     kind = '3d', 
    #                     show = True, 
    #                     sphere = "auto", 
    #                     axes = None, 
    #                     verbose = None)
    

    # Apply your custom montage to the raw object
    raw.set_montage(custom_montage)

    # Plot the sensors with the applied montage
    #raw.plot_sensors(kind='3d', ch_type='eeg', show_names=True)



    
    # for source localisation (later): make a head sphere model
    #sphere_model = make_sphere_model(info = raw.info, 
    #                                 r0 = 'auto', 
    #                                 head_radius = 'auto')
        
        
    # save backup of raw object in the data folder: 
    raw.save((curr_data_path + "part_" + curr_id + "/backup_raw.fif"), overwrite = True)



    """ Delete 'New Segment/' Trigger """
    
    # So first, check how often the weird additional trigger occurs and delete it from the annotations.
    new_segment_count = sum(1 for trigger in raw.annotations.description if 'New Segment/' in trigger)
    # just a sanity check to see if my code works properly: For trial onsets, we should have way more triggers:
    #new_segment_count = sum(1 for trigger in raw.annotations.description if 'Stimulus/S  8' in trigger)
    #print(new_segment_count)
    # As you can see, the trigger occurs only once, so I'm probably right and we can exclude it. 
    annotations = [trigger for trigger in raw.annotations.description if 'New Segment/' not in trigger]
    #Check if it's gone:
    #print('New Segment/' in annotations)
    
    # build a new annotations object without the weird 'New Segment/' trigger
    new_annotations = mne.Annotations(onset = raw.annotations.onset[raw.annotations.description != 'New Segment/'],
                                      duration = raw.annotations.duration[raw.annotations.description != 'New Segment/'],
                                      description = annotations)

    # assign new annotations object to the raw object
    raw.set_annotations(new_annotations)
    
    # check if the weird trigger is still there:
    #print(set(raw.annotations.description))
    # Doesn't seem to be the case so let's hope I did everything correctly.
    
    
    """ Change Trigger Labels """
    
    # Now change the labels of the remaining annotations. 
    
    # Unfortunately, we can only use trigger labels up to a certain length, 
    # so we have to shorten the labels from the original trigger map from the experiment
    # a bit:
    trigger_map = {'block_on': 2,
                   'resp_target': 4,
                   'resp_continue': 6,
                   'trial_on': 8,
                   'click_t_on': 10,
                   'BL_t_on': 12,
                   'BL_m_on': 14,
                   '1back_t1_on': 16,
                   '1back_t2_on': 18,
                   '1back_s_m_on': 20,
                   '1back_d_m_on': 22,
                   '2back_t1_on': 24,
                   '2back_t2_on': 26,
                   '2back_s_m_on': 28,
                   '2back_d_m_on': 30,
                   'pt_task_on': 32,
                   'vtask_main_on': 34,
                   'vtask_t_on': 36,
                   'block_off': 38,
                   '440_on': 40,
                   '440_off': 42,
                   '587_on': 44,
                   '587_off': 46,
                   '782_on': 48,
                   '782_off': 50,
                   '1043_on': 52,
                   '1043_off': 54,
                   'ordered': 56,
                   'random': 58,
                   'start_exp': 60,
                   'end_exp': 62,
                   'trial_off': 64, 
                   'placeholder_1': 66, # look up what these mean
                   'placeholder_2': 68} # look up what these mean
    
    
    # loop annotations in raw object:
    for old_annotation in set(raw.annotations.description):
        #print(old_annotation)
        
        # use regex to extract the number of the trigger & convert it to int:
        trigger_value = int(re.findall(r'\d+', old_annotation)[0])
        #print(trigger_value)
        
        # find correct trigger label for the trigger value we extracted:
        trigger_label = list(trigger_map.keys())[list(trigger_map.values()).index(trigger_value)] 
        #print(trigger_label)
        
        # change label in the annotations:
        raw.annotations.description[raw.annotations.description == old_annotation] = trigger_label

    # print annotations again to check if it worked:
    #print(set(raw.annotations.description))


    # save backup of raw object in the data folder: 
    raw.save((curr_data_path + "part_" + curr_id + "/backup_raw.fif"), overwrite = True)


    """ Get Eyetracking Data: Read in ascii Dataset as MNE Raw Object """
    
    # we might not have eyetracking data for each participant:
    try:
        eyelink_raw = read_raw_eyelink(curr_data_path + "part_" + curr_id + "/" + curr_id + ".asc", 
                                       create_annotations = ["blinks", "messages"], # mark blinks in the stream & add trigger messages
                                       preload = True,
                                       apply_offsets = True) # adjust onset time of the mne.Annotations created from exp. messages (= triggers)
    
        # Check out info to see if everything looks fine:
        eyelink_raw.info
    
        # plot raw eyetracking data from pupil size channel of right eye 
        # to see if everything's there and the triggers were recorded properly:
        pupil_channel_idx = mne.pick_channels(eyelink_raw.ch_names, ["pupil_right"])
        #eyelink_raw.plot(scalings = dict(eyegaze = 1e3), order = pupil_channel_idx)
        

    

    

    """ Look at Raw Data """
    # Filter & downsample the data for plotting (this is not applied to the actual data)
    #raw_resampled = raw.copy().resample(sfreq = 250)
    #raw_resampled.plot(n_channels = 63, duration = 1, scalings = 'auto', highpass = 2, lowpass = 12, filtorder = 4)
    
    
    """ Check If We Have EEG Electrodes That Formed Bridges Due to Too Much Gel """
    
    # If we find some, there's not a lot we can do, but maybe we can exlude them as bad channels 
    # or interpolating them.
    
    # first check electrical distance metrics to estimate electrode bridging
    # "The electrical distance is just the variance of signals subtracted pairwise. 
    # Channels with activity that mirror another channel nearly exactly will have 
    # very low electrical distance." - https://mne.tools/stable/auto_examples/preprocessing/eeg_bridging.html
    
    # It's sufficient to use about 3 min of data to detect bridging, but we need a segment from the end of the 
    # recording where the gel is already set.
    
    
    # get the duration of the recording in seconds
    total_duration = raw.times[-1]

    # compute the start time stamp for the last 3 min
    start_time_last_3_min = total_duration - 180  # 180 s = 3 min

    # get the last 3 minutes of the recording
    raw_bridges = raw.copy().crop(tmin = start_time_last_3_min, tmax = total_duration)
    
    # check which electrodes correlate highly:
    ed_data = mne.preprocessing.compute_bridged_electrodes(raw_bridges)

    # plot potential bridges in red on topoplot:
    bridged_idx, ed_matrix = ed_data
    
    mne.viz.plot_bridged_electrodes(raw_bridges.info,
                                    bridged_idx,
                                    ed_matrix,
                                    title = "Bridged Electrodes of Participant " + curr_id,
                                    topomap_args = dict(vlim = (None, 5)),
                                    )
    
    
    """ Interpolate Bridged Electrodes or Exclude Data of Current Participant """
    
    # if there are bridges, but not too many (let's say 3, 
    # that would be 6 affected electrodes at most), interpolate them: 
    if len(bridged_idx) > 0 and len(bridged_idx) < 3:
        print("interpolating " + str(len(bridged_idx)) + " bridged electrodes!" )
        raw_interpolated_bridges = mne.preprocessing.interpolate_bridged_electrodes(raw_bridges.copy(), 
                                                                                    bridged_idx = bridged_idx)
        # overwrite raw object with data with fixed channels
        raw = raw_interpolated_bridges.copy()
    
    # if there are more than 3 bridges, exclude participant from further analysis and go to next one:
    elif len(bridged_idx) > 3:
        print("Detected " + str(len(bridged_idx)) +  " electrode bridges for current participant! Excluding dataset from the analysis!")
        next
        
    # if there are no bridges, just keep the df as is
    elif len(bridged_idx) == 0:
        print("no electrode bridging detected here :-)" )        
    
    
    # save backup of raw object in the data folder: 
    raw.save((curr_data_path + "part_" + curr_id + "/backup_raw.fif"), overwrite = True)

    



    """ High-Pass Filter to get rid of slow drifts"""
    
    raw = raw.filter(l_freq = 0.2, 
                     h_freq = None)
    
    
    # --> this needs to happen before the ICA!
    #raw = raw.filter(l_freq = 1, 
    #                 h_freq = None)
    
    # save backup of raw object in the data folder: 
    raw.save((curr_data_path + "part_" + curr_id + "/backup_raw.fif"), overwrite = True)












    """ ICA """
    
    # in case the script crashed here again during testing: read in .fif with the raw object:
    #raw = mne.io.read_raw_fif(curr_data_path + "part_" + curr_id + "/backup_raw.fif", preload=True)
    
    # --> get rid of ICs that represent motor artifacts like 
    # head & eye movements or blinks.

    # Initialize ICA with a desired number of components
    # How many components should I use here?!
    ica = mne.preprocessing.ICA(n_components = 20, # 
                                random_state = 42, # set seed
                                #method="picard") # less memory-intensive method
                                method = "fastica") # use Fast ICA
                                
        
    # Fit ICA to the raw data
    ica.fit(raw)
    
    # this takes a while, so make computer remind me to check the ICA output:
    os.system('say "hey queen go check your screen"')
    
    # save backups again:
    raw.save((curr_data_path + "part_" + curr_id + "/backup_ica.fif"), overwrite = True)
    raw.save((curr_data_path + "part_" + curr_id + "/backup_raw.fif"), overwrite = True)
  
    


    # Exclude ICA components manually:
        
    # 1. inspect topoplots: Where do we have weird frontal activity?
    # 2. inspect signal of all components and check where blinks or eye movements are visible.
    # 3. inspect power spectrum: Where do we have a weird skewed spectrum, 50 Hz peaks or no (!) alpha? --> probably artifacts

    excl_components = [] # collect indices of all components that should be excluded here
    
    # 1. Plot the ICA components as topoplots
    ica.plot_components()
    # ask user to set the components that should be excluded:
    user_input_excl_components = input("Which components should be excluded? Please provide the corresponding indices separated by commas: ")
    print("You told me to exclude the following components:" + user_input_excl_components)
    # append chosen components to list of components we want to exclude:
    excl_components = excl_components + list(map(int, user_input_excl_components.split(', ')))
    
    
    # 2. Plot the time series of the components
    ica.plot_sources(raw, show_scrollbars = True)
    # ask user to set the components that should be excluded:
    user_input_excl_components = input("Which components should be excluded? Please provide the corresponding indices separated by commas: ")
    print("You told me to exclude the following components:" + user_input_excl_components)
    # append chosen components to list of components we want to exclude:
    excl_components = excl_components + list(map(int, user_input_excl_components.split(', ')))
    
    
    # 3. Plot power spectrum of components
    user_input_excl_components = []
    # Get the power spectrum of the first five ICA components
    ica_sources = ica.get_sources(raw)  # extract ICA sources from raw data
    ica_sources_data = ica_sources.get_data()  # get NumPy array of ICA sources
    freqs, psds = mne.time_frequency.psd_array_welch(ica_sources_data, raw.info['sfreq'], fmin=1, fmax=80, n_fft=2048, window='hamming')
    # Plot the power spectrum of all ICA components
    for i in range(np.shape(ica_sources_data)[0]):
        fig, ax = plt.subplots()
        ax.plot(psds, freqs[i])  # transpose psds[i] to match the shape of freqs
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title('ICA component {}'.format(i))
        # Show plot
        plt.show() 
         # ask user to set the components that should be excluded:
        user_input_excl_curr_component = input("Exclude this component? (y/n)")
        if user_input_excl_curr_component == "y":
            user_input_excl_components = user_input_excl_components + [i]
    
    print("You told me to exclude the following components: " + ', '.join(map(str, user_input_excl_components)))
    # append chosen components to list of components we want to exclude:
    excl_components = excl_components + user_input_excl_components




    # Exclude ICA components automatically if they have a high correlation with EOG events:

    # Find & exclude ICA components that match EOG events from frontopolar electrodes.
    
    # set channels here that we can use to detect 
    # electrooculogram (EOG) events aka blinks & eye movements
    eog_channels = ["Fp1", "Fp2"] # we don't really have face electrodes, but I guess Fp1 & Fp2 works too
        
    #
    eog_indices, eog_scores = ica.find_bads_eog(raw, 
                                                ch_name = eog_channels, 
                                                reject_by_annotation = False,
                                                measure = "correlation",
                                                threshold = .8)
        
    # The eog indices tell how much each ICA component 
    # correlates with activity in the EOG channels.
    # If the correlation is > .8 for some component, we exclude it.
        
        
    # if we found some matches, visualise eog indices and exclude matching ICA components
    if len(eog_indices) > 0:
            
        # visualise EOG indices (aka correlations of each ICA component with EOG activity)     
        plt.figure(figsize = (12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(eog_indices, marker = 'o', linestyle = '-', color = 'red')
        plt.title('EOG Indices')
        plt.xlabel('Index of ICA Component')
        plt.ylabel('Correlation with EOG activity')
        plt.grid(True)
        
        print("automatically identified " + len(eog_indices) + " components for exclusion.")
    
        # exclude all ICs that match the EOG artifacts from back transformation:
        excl_components = excl_components + eog_indices



    
    # Exclude Components:
    # remove all duplicated component indices from list of components that should be excluded:
    print("excluding the following components from data: " + ', '.join(map(str, excl_components)))
    excl_components = list(set(excl_components))
    ica.exclude = excl_components  # Remove components 1, 3, and 5
    # remove components from the EEG data
    ica.apply(raw)










    
    """ Filtering """
    
    # plot PSD before low-pass filtering:
    #raw.plot_psd(fmax = 50, average = True, spatial_colors = False)
    
    # Visualise filter parameters
    #filter_params = mne.filter.create_filter(data = raw.get_data(),
    #                                         sfreq = raw.info["sfreq"],
    #                                         h_freq = 15,
    #                                         l_freq = None, 
    #                                         phase = 'zero', 
    #                                         fir_window ='hamming',
    #                                         verbose = None)
    #mne.viz.plot_filter(filter_params, raw.info["sfreq"])

    # Apply low-pass filter (we already applied a high-pass filter before):
    raw = raw.filter(h_freq = 15,
                     l_freq = None,
                     verbose = None)


    # plot PSD after filtering
    #raw.plot_psd(fmax = 50, average = True, spatial_colors = False)

    
    """ 4.4 Select relevant channels """

    raw_selected_channels = raw.copy().pick_channels(['O1', 'O2', 'Oz', #'P07', 
                                                       'C1','CP1', 'C3', 'CP3', 'C5','CP5','C6','CP6','Cz','C2', 'CP2', 'C4', 'CP4',
                                                       'CPz', 'Pz', 'POz', 'PO3','P2','P6','P5','P1'])

    # check if it worked:
    #raw_selected_channels.ch_names
    #raw.ch_names
        
     
        
    """ Epoching """
    
    # get trial onsets:    
    curr_events, _ = mne.events_from_annotations(raw_selected_channels, event_id = {"trial_on": trigger_map["trial_on"],
                                                                  "resp_continue": trigger_map["resp_continue"]})

    # convert to np array
    events_array = np.array(curr_events)

    
    # cut epochs around the response events:        
    epochs = mne.Epochs(raw_selected_channels, 
                        events = curr_events,
                        event_id = trigger_map['resp_continue'], 
                        tmin = -0.2, 
                        tmax = 0.7,
                        reject = dict(eeg = 40e-6#,      # unit: V (EEG channels)
                                      #eog = 250e-6      # unit: V (EOG channels)
                                      ),
                        detrend = 1, # 1 = linear detrend --> performed before BL correction
                        baseline = (-0.1, 0), 
                        preload = True)
    # plot them:
    epochs.plot_image(
                      combine = None, 
                      picks = ['O1', 'O2', 'Oz', #'P07', 
                               'C1','CP1', 'C3', 'CP3', 'C5','CP5','C6','CP6','Cz','C2', 'CP2', 'C4', 'CP4',
                               'CPz', 'Pz', 'POz', 'PO3','P2','P6','P5','P1'],
                      group_by=dict(central_ROI = [3,4,5,6,7,8,9,10,11,12,13,14,15],
                                    parietal_ROI = [16,17,18,19,20,21,22,23],
                                    occipital_ROI = [0, 1, 2]),
                      title = "Reponse ERPs across conditions, occipital & central electrodes, data only filtered to 0.2-15 Hz, noisy epochs automatically rejected")
    
    
    
    
    # cut epochs around the word-onset events:        
    epochs = mne.Epochs(raw_selected_channels, 
                        events = curr_events,
                        event_id = trigger_map['trial_on'], 
                        tmin = -0.2, 
                        tmax = 0.7,
                        reject = dict(eeg = 40e-6#,      # unit: V (EEG channels)
                                      #eog = 250e-6      # unit: V (EOG channels)
                                      ),
                        detrend = 1, # 1 = linear detrend --> performed before BL correction
                        baseline = (-0.1, 0), 
                        preload = True)
    # plot them:
    epochs.plot_image(
                      combine = "mean", 
                      picks = ['O1', 'O2', 'Oz', #'P07', 
                               'C1','CP1','C3', 'CP3', 'C5','CP5','C6','CP6','Cz','C2', 'CP2', 'C4', 'CP4',
                               'CPz', 'Pz','POz', 'PO3','P2','P6','P5','P1'],
                      group_by=dict(central_ROI = [3,4,5,6,7,8,9,10,11,12,13,14,15],
                                    parietal_ROI = [16,17,18,19,20,21,22,23],
                                    occipital_ROI = [0, 1, 2]),
                      title = "Word-Onset ERPs across conditions, occipital & central electrodes, data only filtered to 0.2-15 Hz, noisy epochs automatically rejected")
    
    
    
    
    raw_selected_channels.compute_psd(fmax=50).plot()
    
    
    
    """ --> cut data into blocks """
    
    # Create Events from Annotations
    events, event_id = mne.events_from_annotations(raw, event_id = trigger_map)
    #print(event_id)
    #print(events) 
    
    # define onset triggers for all blocks you want to get
    onset_triggers = ['BL_m_on', '1back_d_m_on', '2back_d_m_on']
    
    # get the corresponding values from the trigger_map
    onset_trigger_values = [trigger_map[trigger] for trigger in onset_triggers]
    
    # Find indices where the last value corresponds to the value of one of the onset triggers
    onset_indices = np.where(np.isin(events[:, -1], onset_trigger_values))[0]


    # Now this is where it gets tricky: 
    # For each onset trigger, I want to get the next offset trigger in the data.
    
    # So first, we'll get ALL offset triggers, also the offset triggers for the blocks we're not interested in:
    offset_indices = np.where(np.isin(events[:, -1], trigger_map["block_off"]))[0]

    # Next, we'll get their indices, then we'll merge the onset and offset indices, sort them in ascending order 
    # and we can extract the index following each onset index as the corresponding offset-index:
    
    all_indices = np.sort(np.concatenate((onset_indices, offset_indices)))

    # create a dict for saving onset + offset indices
    onset_to_offset_dict = {}
    
    # loop onset indices:
    for onset_index in onset_indices:
        
        # get the index after the current onset index in "all_indices"
        next_index = all_indices[all_indices > onset_index][0]
        
        # check if the next index is from an offset trigger
        if events[next_index, -1] == trigger_map["block_off"]:
            
            # put both the on- and the offset trigger in the dict
            onset_to_offset_dict[onset_index] = next_index
    
    #print(onset_to_offset_dict)

    # Worked! Now we can cut the data into smaller segments, ranging from 
    # each onset to the corresponding offset trigger.


    # empty dict for storing the segmented data:
    main_blocks_segments = []
    main_block_names = []
    # loop dict with onset and offset pairs:
    for onset_index, offset_index in onset_to_offset_dict.items():
        print("onset index:" + str(onset_index) + ", offset index: " + str(offset_index))

        # get the time points corresponding to the onset and offset indices
        onset_time = events[onset_index, 0] / raw.info["sfreq"]
        offset_time = events[offset_index, 0] / raw.info["sfreq"]
    
        # cut the raw data into a segment from onset to offset
        segment = raw.copy().crop(tmin = onset_time, tmax = offset_time)
    
        # find out which block this belongs to:
        onset_annotation_value = events[onset_index, -1]
        block_name = next((key for key, value in trigger_map.items() if value == onset_annotation_value), None)
    
        # Append the segment to the list of segments
        main_blocks_segments.append(segment)
        # append block name to the list of segment names
        main_block_names.append(block_name)


    # Sanity check: 
    # Check number of trials in one of the segments to see if we did everything correctly.
    
    # loop blocks
    exclude_block_indices = []
    for i in range(0, len(main_block_names)):

        # get segment
        segment_to_plot = main_blocks_segments[i]

        # count the number of 'trial_on' triggers
        count_trials = sum(1 for trigger in segment_to_plot.annotations.description if 'trial_on' in trigger)

        print("number of trials in block " + main_block_names[i] + ": " + str(count_trials))
    
        # check if current block has less than 300 trials aka was ended prematurely:
        if count_trials < 300:
            # exclude block at index i from lists "block_names" and "main_blocks_segments"
            exclude_block_indices.append(i)
            
        # Plot the data
        #segment_to_plot.plot()

    
    # Exclude blocks with less than 300 trials:
    main_block_names = [main_block_names[i] for i in range(len(main_block_names)) if i not in exclude_block_indices]
    main_blocks_segments = [main_blocks_segments[i] for i in range(len(main_blocks_segments)) if i not in exclude_block_indices]


    
    """ --> cut blocks into trials """
    # window: - 200 to 700 ms around stimulus onset
    
    # loop blocks again & cut epochs:
    epochs_list = []     
    for curr_segment, curr_block_name, curr_block_idx in zip(main_blocks_segments, main_block_names, range(0, len(main_block_names))):
        
        print(curr_block_name)

        """ Cut Epochs & Apply Baseline Correction """
        # Concerning the BL Correction:
        # --> does it make sense to use a baseline period before stimulus onset, 
        # bc this is basically during the presentation of another stimulus or possibly 
        # even more than one?
        
        # From Ditman, T., Holcomb, P. J., & Kuperberg, G. R. (2007). 
        # An investigation of concurrent ERP and self-paced reading methodologies. 
        # Psychophysiology, 44(6), 927–935. doi:10.1111/j.1469-8986.2007.00593.x     
        # "[...] using the 100ms that preceded word onset as a baseline."
        # If they did it, I can do this, too. 
            
    
        """ create events from annotations """
        
        # get timestamps for all "trial onset" events:
        curr_events, _ = mne.events_from_annotations(curr_segment,event_id = {"trial_on": trigger_map["trial_on"]})
        
        curr_events = np.array(curr_events)

        curr_events, _ = mne.events_from_annotations(curr_segment, event_id={"trial_on": trigger_map["trial_on"]})

        # convert to np array
        events_array = np.array(curr_events)

        
        # cut epochs around those events:        
        epochs = mne.Epochs(curr_segment, 
                            events = curr_events,
                            event_id = trigger_map['trial_on'], 
                            tmin = -0.2, 
                            tmax = 0.7, 
                            detrend = 1, # 1 = linear detrend --> performed before BL correction
                            baseline = (-0.1, 0), 
                            preload = True)
        
        
        """ Downsample Data """
        # Important: We need to do this after epoching so triggers are not jittered
        # --> https://gist.github.com/larsoner/01642cb3789992fbca59
        
        # downsample the data a bit so it's easier to work with:
        epochs.resample(sfreq = 250)
        print(epochs.info)
    
    
        """ Sanity check: Create word-onset ERP for current block """
        if curr_block_name == "BL_m_on":    
            epochs.plot_image(combine="mean", title = "BL (single task)")
        if curr_block_name == "1back_d_m_on":    
            epochs.plot_image(combine="mean", title = "1-back (dual task)")
        if curr_block_name == "2back_d_m_on":    
            epochs.plot_image(combine="mean", title = "2-back (dual task)")
        
        
        
    
        """ Add Metadata for Each Epoch """
        
        # loop epochs
        for i, epoch in enumerate(epochs):
            
            epoch.info['trial_nr'] = i # trial nr
            epoch.info['block_nr'] = curr_block_idx # block nr
            
            # assign block name
            epoch.info['block_name'] = curr_block_name
            
            # extract a few information from the block name:
                
            # if we have a BL block, it can only be a single-task:
            if curr_block_name[0] == "B":
                epoch.info['cognitive_load'] = "BL"
                epoch.info['task_kind'] = "single_task"
            
            # if we have a 1-back block, check if it's single or dual task:
            elif curr_block_name[0] == "1":
                epoch.info['cognitive_load'] = "1back"
                if curr_block_name[6] == "d":
                    epoch.info['task_kind'] = "dual_task"
                else: epoch.info['task_kind'] = "single_task"
                
            # if we have a 2-back block, check if it's single or dual task:
            elif curr_block_name[0] == "2":
                epoch.info['cognitive_load'] = "2back"
                if curr_block_name[6] == "d":
                    epoch.info['task_kind'] = "dual_task"
                else: epoch.info['task_kind'] = "single_task"
                
          
            # add information on the stimulus material:    
            epoch.info['word'] = None
            
            epoch.info['surprisal_1'] = None
            epoch.info['surprisal_4'] = None
            epoch.info['surprisal_12'] = None
            epoch.info['surprisal_60'] = None
            
            epoch.info['word_length'] = None
            epoch.info['reaction'] = None
            epoch.info['word_freq'] = None

            

        # If epochs object is ready, append to list for all epochs:
        epochs_list.append(epochs)
    
    
    
    
    

    
    
    """ Exclude noisy trials """
    
    # Plot individual trials for manual inspection and rejection
    #epochs.plot(n_epochs=5, n_channels=30, title='Manual Trial Inspection', scalings=dict(eeg=50e-6), reject_by_annotation=False)
    
    # After plotting, you can manually reject trials by clicking on them in the plot
    # and then pressing the 'D' key (to mark as bad) or 'G' key (to mark as good).
    
    # To get the list of rejected epochs
    #rejected_epochs = epochs.drop_log
    
    # To get the cleaned epochs (epochs without rejected trials)
    #clean_epochs = epochs.copy().drop(reject='manually')

    
 
    
    """ Compute Word Onset ERPs across conditions """
    
    
























# FOR LATER:

""" Align EEG & Eyetracking Data """
# Convert event onsets from samples to seconds
et_flash_times = et_events[:, 0] / raw_et.info["sfreq"]
eeg_flash_times = eeg_events[:, 0] / raw_eeg.info["sfreq"]
# Align the data
mne.preprocessing.realign_raw(
    raw_et, raw_eeg, et_flash_times, eeg_flash_times, verbose="error"
)
# Add EEG channels to the eye-tracking raw object
raw_et.add_channels([raw_eeg], force_update_info=True)

# Define a few channel groups of interest and plot the data
frontal = ["E19", "E11", "E4", "E12", "E5"]
occipital = ["E61", "E62", "E78", "E67", "E72", "E77"]
pupil = ["pupil_right"]
picks_idx = mne.pick_channels(
    raw_et.ch_names, frontal + occipital + pupil, ordered=True
)
raw_et.plot(events=et_events, event_id=event_dict, event_color="g", order=picks_idx)








