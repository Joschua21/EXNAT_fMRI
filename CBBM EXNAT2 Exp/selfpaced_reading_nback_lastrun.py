﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on Fri Apr 14 14:44:34 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

import psychopy
psychopy.useVersion('2021.2.3')


from psychopy import locale_setup
from psychopy import prefs
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'EXNAT-2'  # from the Builder filename that created this script
expInfo = {'participant': '', 'age': '', 'handedness': 'r', 'gender': 'w', 'testing_mode': 'yes'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/merle/Github/PhD/EXNAT/EEG_study_EXNAT2/CBBM EXNAT2 Exp/selfpaced_reading_nback_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1500, 1000], fullscr=False, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color='', colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='deg')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "settings_2"
settings_2Clock = core.Clock()
### import packages:

# for setting the output encoding to UTF-8
import sys
# --> if you don't do this, German "Umlaute" can't be displayed correctly:
sys.stdout = open(sys.stdout.fileno(), mode = 'w', encoding = 'utf8', buffering = 1)

# for getting current date & time:
import datetime
# numpy for being able to calculate
import numpy as np
# for random number generator:
import random
# for saving data in csv:
import pandas as pd
# additional timing package (I know we have core.wait, but I also want this one)
import time
# pylsl for pushing triggers to lsl stream:
# from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
# for connecting to serial ports:
# import serial

# from my custom scripts...
# import all texts
from EXNAT2_texts_MC_Qs import *
# import some additional functions I wrote for the experiment:
from EXNAT2_study_components import change_bg_colour
from nback_colour_generator import create_nback_stimlist, draw_without_replacement, get_targets

# build little function to flatten nested lists:
def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_list(item))
        else:
            flattened_list.append(item)
    return flattened_list
    

### Setup LSL Stream
print("create trigger stream") 
# Create trigger stream:
#global out_marker
#info_marker_stream = StreamInfo('PsychoPyMarkers', 'Marker', 1, 0, 'string')
#out_marker = StreamOutlet(info_marker_stream)
#out_marker.push_sample(["TEST MARKER"])

### Stimulus settings

# set flicker frequency (in Hz)
flicker_freq = 10
# set frame rate (in Hz)
frame_rate = 60

# set colours you want to use for background:
#light_bg_col_hex = "#FDFBF0" # ivory instructions background
#dark_bg_col_hex  = "#505050" # dark grey background for stimuli
light_bg_col = [(x / 127.5) - 1 for x in (253, 251, 240)] # ivory instructions background (use RGB -1:1)
dark_bg_col  = [(x / 127.5) - 1 for x in (80, 80, 80)] # dark grey background for stimuli (use RGB -1:1)

# make background light for a start - use rgb -1:1 colour codes
win.setColor(light_bg_col, colorSpace='rgb')

# set colours you want to use for the stimuli:
colours = ["#D292F3", "#F989A2", "#2AB7EF", "#88BA3F"]
print("Preparing experiment with n-back colours:", colours)

#  #D292F3 = weird lilac with a 2000s vibe
#  #F989A2 = bubblegum pink
#  #2AB7EF = Twitter blue
#  #88BA3F = medium grass green
# (#D8A244 = dark curry-ish yellow --> excluded!)

#   All colours have a luminance of 70 and a chroma of 74.

#   The colours are selected for distinguishability (is that a word?!) 
#   for people with "normal" colour vision as well as for 
#   people with protanomaly (red olour vision deficiency (CVD)), 
#   deuteranomaly (green CVD) and 
#   tritanomaly (blue CVD).

#   People with a "true" colour blindness 
#   (i.e. protanopia, deuteranopia, tritanopia)
#   shouldn't participate in this study. */


# ----------------------------------------------
### Shuffle order of texts
print("shuffle texts") 
# collect the text IDs in lists so I know which text was shown 
all_main_texts_nrs_list = ["text_01", "text_02", "text_03", "text_04", "text_05", "text_06", "text_07", "text_08", "text_09"]
# shuffle text numbers
random.shuffle(all_main_texts_nrs_list)
# append "empty" text numbers to the list where we have blocks that are not main blocks.
all_texts_nrs_list = []
for t_idx, t in enumerate(all_main_texts_nrs_list):
    # if it's the first text, it's the reading BL main block.
    # Append 1 empty text number before text (for the reading BL training) 
    # and 4 after text (for click training, 2x single task training & 1x single task main)
    if t_idx == 0: 
        all_texts_nrs_list = all_texts_nrs_list + ["", t, "", "", "", ""]
    # the second text is the first n-back block. 
    # There are 3 "empty" blocks after it (2x single task training & 1x single task main)
    elif t_idx == 1: 
        all_texts_nrs_list = all_texts_nrs_list + [t, "", "", ""]
    # all following texts are main blocks and can be appended to all_texts_nrs_list
    elif t_idx > 1:
        all_texts_nrs_list.append(t)
        
print(all_texts_nrs_list)

### Set order of blocks 
print("set block order") 
# The first blocks should be:
# - reading baseline + training
# - click training
# - in random order: 1-back (2x single task training & 1x main & 1x dual task main) & 2-back (2x single task training & 1x main & 1x dual task main)
# - in random order: 2x reading BL main, 2x 1-back main, 2x 2-back main

# this always comes first in the experiment
Reading_BL = ["Reading_Baseline_training", "Reading_Baseline_main", "click_training"]

# then you get both n-back conditions with trainings (which of them is first is randomized)
oneback = ["1back_single_training1", "1back_single_training2", "1back_single_main", "1back_dual_main"]
twoback = ["2back_single_training1", "2back_single_training2", "2back_single_main", "2back_dual_main"]

# shuffle the order of the 2 lists
main_blocks1 = [oneback, twoback]
random.shuffle(main_blocks1)

# flatten nested list
main_blocks1 = flatten_list(main_blocks1)

# now shuffle order of the last 6 main blocks:
main_blocks2 = ["Reading_Baseline_main", "Reading_Baseline_main", "1back_dual_main", 
                "1back_dual_main", "2back_dual_main", "2back_dual_main"]
random.shuffle(main_blocks2)

# put them all together:
#global all_blocks 
all_blocks = Reading_BL + main_blocks1 + main_blocks2

### Create n-back colour lists for all blocks

print("create n-back colour lists")
# The reading bl training text has 159 trials.

# The click training has 6 trials. 

# Then we also have 4 short training blocks à 20 trials each (5 targets)
# 4 * single training

# We have 2 single-task main blocks, 
# one for 1-back and 1 for 2-back à 60 trials each (10 targets):
# 2 * single main

# There are 9 dual-task main blocks à 300 stimuli each (50 targets)
# reading bl * 3
# 1-back * 3
# 2-back * 3

# --> all in all, 15 blocks

# So for every block, build a list with colour codes containing the right amount of targets.
# The function is defined in another script bc it's super long, 
# I import it at the beginning of this script.

# First, create list with length of all texts. The length of the blocks is 
# always in the same order, only the conditions change.
blocks_textlen = [159, 300, 6, # reading bl blocks + click training
                  20, 60, 60, 300, 20, 60, 60, 300, # main blocks 1 + trainings & single tasks
                  300, 300, 300, 300, 300, 300] # main blocks 2        
blocks_target_counts = [25, 50, 1, # reading bl blocks + click training
                        5, 10, 10, 50, 5, 10, 10, 50, # main blocks 1 + trainings & single tasks
                        50, 50, 50, 50, 50, 50]
# Now loop this list. Check which condition we have there and the create colour list for each text.
all_colour_lists = []
all_target_lists = []
for block_idx, block_length in enumerate(blocks_textlen):
    # get 1st letter of block name - that tells us the condition
    block_cond = all_blocks[block_idx][0]

    # for each condition, decide which n-back level we want to assign
    # For all no-n-back blocks, we use 1 (just for the colour list generation)
    #global curr_nback_level
    if block_cond == "R":
        curr_nback_level = 1
    elif block_cond == "c":
        curr_nback_level = 1
    elif block_cond == "1":
        curr_nback_level = 1
    else: curr_nback_level = 2

    # generate colour list for current block  
    #global curr_colours
    curr_colours = create_nback_stimlist(nback_level = curr_nback_level, 
                                         colour_codes = colours, 
                                         story = ["x"] * block_length, 
                                         target_abs_min = blocks_target_counts[block_idx], 
                                         target_abs_max = blocks_target_counts[block_idx], 
                                         zeroback_target = None)

    # get list of targets / non-targets
    curr_targets = get_targets(stim_list = curr_colours, 
                               nback_level = curr_nback_level)

    # add to bigger lists
    all_colour_lists.append(curr_colours)
    all_target_lists.append(curr_targets)

print("------ finished preparing stimuli! ------")


# ------------------------------------------

# init block counter for the whole experiment
exp_block_counter = 0

print("starting experiment now!")
empty_placeholder = visual.TextStim(win=win, name='empty_placeholder',
    text=None,
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "no_text_blocks"
no_text_blocksClock = core.Clock()

# Initialize components for Routine "text_blocks"
text_blocksClock = core.Clock()

# Initialize components for Routine "Q1"
Q1Clock = core.Clock()

# Initialize components for Routine "Q2"
Q2Clock = core.Clock()

# Initialize components for Routine "Q3"
Q3Clock = core.Clock()

# Initialize components for Routine "difficulty"
difficultyClock = core.Clock()

# Initialize components for Routine "vis_task"
vis_taskClock = core.Clock()

# Initialize components for Routine "motor_task"
motor_taskClock = core.Clock()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "settings_2"-------
continueRoutine = True
routineTimer.add(0.500000)
# update component parameters for each repeat
# keep track of which components have finished
settings_2Components = [empty_placeholder]
for thisComponent in settings_2Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
settings_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "settings_2"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = settings_2Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=settings_2Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *empty_placeholder* updates
    if empty_placeholder.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        empty_placeholder.frameNStart = frameN  # exact frame index
        empty_placeholder.tStart = t  # local t and not account for scr refresh
        empty_placeholder.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(empty_placeholder, 'tStartRefresh')  # time at next scr refresh
        empty_placeholder.setAutoDraw(True)
    if empty_placeholder.status == STARTED:
        # is it time to stop? (based on global clock, using actual start)
        if tThisFlipGlobal > empty_placeholder.tStartRefresh + 0.5-frameTolerance:
            # keep track of stop time/frame for later
            empty_placeholder.tStop = t  # not accounting for scr refresh
            empty_placeholder.frameNStop = frameN  # exact frame index
            win.timeOnFlip(empty_placeholder, 'tStopRefresh')  # time at next scr refresh
            empty_placeholder.setAutoDraw(False)
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in settings_2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "settings_2"-------
for thisComponent in settings_2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('empty_placeholder.started', empty_placeholder.tStartRefresh)
thisExp.addData('empty_placeholder.stopped', empty_placeholder.tStopRefresh)

# set up handler to look after randomisation of conditions etc
blocks = data.TrialHandler(nReps=30.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='blocks')
thisExp.addLoop(blocks)  # add the loop to the experiment
thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
if thisBlock != None:
    for paramName in thisBlock:
        exec('{} = thisBlock[paramName]'.format(paramName))

for thisBlock in blocks:
    currentLoop = blocks
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            exec('{} = thisBlock[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "no_text_blocks"-------
    continueRoutine = True
    # update component parameters for each repeat
    #################################################
    #                Blocks w/o text                #
    #################################################
    # this routine is for all blocks where there are 
    # coloured rectangles instead of words
    
    # the non-text blocks all come in succession, there's just 1 main block in between them.
    # So use loop here that runs the non-text blocks 
    # until we have to display a main text block (in this case we exit the routine).
    while True:
        # keep background ivory
        win.setColor(light_bg_col, colorSpace='rgb')
        win.flip()
        
        # clear buffer of all previously recorded key events:
        event.clearEvents()
        
        ### specify settings for the current block
    
        ### Prepare stimuli:
    
        # get block kind
        curr_block = all_blocks[exp_block_counter]
        print("start preparing block " + curr_block)
    
        # Check whether it's one of the non-text tasks.
        # If current block is a text block, skip this routine and go to the next.
        if curr_block not in ["click_training", "1back_single_training1", "1back_single_training2", 
                          "1back_single_main", "2back_single_training1", "2back_single_training2", "2back_single_main"]:
            print("skipping current routine")
            break
    
        # if it's one of the non-text blocks, though, prepare stimuli:
        else:
            print(curr_block + " is not a text block - preparing rect as stim now")
            
            # keep background ivory
            win.setColor(light_bg_col, colorSpace='rgb')
            win.flip()
            
            ### Show instructions
            # set instruction text
            instr_text = locals()["instr_" + curr_block]
            # create text box
            instr_text_stim = visual.TextStim(win, 
                                             text = instr_text, 
                                             height = 0.5, 
                                             pos = (0, 0),
                                             font = "Bookman Old Style",
                                             color = 'black')
            # display the text on screen
            if curr_block in ["1back_single_training2", "2back_single_training2"]:
                while True:
                    instr_text_stim.draw()
                    win.flip()
                    # skip current block (aka the second training block))
                    if event.getKeys(['space']):
                        print("start next block - skip second training block")
                        skip_curr_block = True
                        break
                    # repeat training (aka run current block, which is the second training block)
                    elif event.getKeys(['w']):
                        print("repeat training block")
                        skip_curr_block = False
                        break
            # for regular blocks that can't be repeated:
            else: 
                while True:
                    instr_text_stim.draw()
                    win.flip()
                    # start current block
                    if event.getKeys(['space']):
                        print("start current block")
                        skip_curr_block = False
                        break
                        
            # only run this if the current block shall not be skipped:
            if skip_curr_block == False:
                ### change background colour 
                # transition from white (RGB: 255, 255, 255)
                # to medium grey (RGB: 10, 10, 10)
                change_bg_colour(window = win, 
                                 start_rgb = light_bg_col, 
                                 end_rgb = dark_bg_col, 
                                 seconds = 2)
                # Wait for a brief period of time so bg is set
                core.wait(0.8)
                # keep background grey
                win.setColor(dark_bg_col, colorSpace='rgb')
                win.flip()
            
                # don't show questions
                skip_questions = True
                training_Qs = False
    
                # get n-back condition:
                curr_nback_cond = curr_block[0] # get first character of block name
                
                # if it is a 1 or a 2, set that as current n-back level:
                if curr_nback_cond in ['1', '2']:
                    curr_nback_cond == int(curr_nback_cond)
                # if it's neither 1 nor 2, it has to be a block without n-back,
                # so set curr_nback_cond to None
                else:
                    curr_nback_cond = None
                
                print("current n-back condition: " + str(curr_nback_cond))
    
                # get list with targets & list with colours
                curr_targets = all_target_lists[exp_block_counter]
                curr_colours = all_colour_lists[exp_block_counter]
                # for current text nr, get text whose name = current text nr
                curr_text = locals()[curr_text_nr]
    
                # start block loop
    
                # prepare flicker
                # hint: flicker_freq and frame_rate are set in the settings 
                # code component at the beginning of the experiment.
                
                # create flicker phase variable - start at phase = 0
                flicker_phase = 0
                # we also need the start time (let's set it as current time 
                # at this point in the script):
                start_time = core.getTime()
        
                # create empty stimulus 
                stim = visual.Rect(win = win,
                                   width = 3, # width = 3 * 1° visual angle (to make it look rectangle-ish)
                                   height = 1, # height = 1° visual angle (just like words)
                                  # colorSpace = "hex",
                                   pos = (0,0)) # center stimulus 
                
                stim.draw()
                win.flip()
    
                # clear buffer of all previously recorded key events:
                event.clearEvents()
    
                # loop colours in current text
                for trial_idx, curr_col in enumerate(curr_colours):
                    print("current idx: " + str(trial_idx) + ", curr colour:" + curr_col)
                    
                    ### prepare & show current word:
                    
                    # if it's a block with an n-back task, prepare target list
                    if curr_nback_cond != None:
                        curr_target = curr_targets[trial_idx]
                        saw_target = False
                    
                    # get trial number (start counting from 1, so add 1)
                    curr_trial_nr = trial_idx + 1
                    
                    ### ISI: wait for 100 ms
                    while core.getTime() < onset_time + 0.1:
                        # draw the stimulus during the waiting period, 
                        # but use grey as a fill colour
                        stim.Colour = dark_bg_col
                        stim.draw()
                        win.flip()
                        
                    # set current colour as colour of rectangle
                    stim.fillColor = curr_col
                    
                    # Flicker option 1: use sine-wave (gradient) flicker
                    #frame_time = core.getTime() # get current time point (in sec)
                    #flicker_intensity = np.sin(2 * np.pi * flicker_freq * (frame_time - start_time) + flicker_phase)
                    #opacity = (flicker_intensity + 1) / 2
        
                    # Flicker option 2: use square-wave (on-off) flicker
                    frame_time = core.getTime() 
                    time_passed = frame_time - start_time 
                    cycle_duration = 1 / flicker_freq
                    cycle_passed = time_passed % cycle_duration
                    
                    if cycle_passed < cycle_duration / 2:
                        opacity = 1
                    else: 
                        opacity = 0
                    
                    # set opacity
                    stim.opacity = opacity  
                    # draw stimulus on screen
                    stim.draw()
                    win.flip()
            
                    # show stimulus on screen
                    stim.draw() # draw stimulus on screen
                    win.flip() # update the window to clear the screen and display the stimulus
    
                    # send word onset trigger to LSL stream
                    marker_text = "block_" + curr_block + "_trial_" + str(curr_trial_nr) + "_" + curr_col
                    #out_marker.push_sample(["STIM_ONSET_" + marker_text])
                    
                    # record trial onset time
                    onset_time = core.getTime()
                    print("onset time: " + str(onset_time * 1000) + " ms" )
                                        
                    ### wait for key response: 
                    # In blocks with n-back task, participants can press "c" to indicate they saw a target colour and "space" to go to the next word/stimulus.
                    # In blocks without n-back task, participants can only press "space" to go to the next stimulus.
                    print("start tracking key responses")
                        
                    ### start recording responses
                    # start "endless" while loop that looks for responses
                    while True:        
                        # in each iteration, draw word on screen
                        
                                        
                        # Flicker option 1: use sine-wave (gradient) flicker
                        #frame_time = core.getTime() # get current time point (in sec)
                        #flicker_intensity = np.sin(2 * np.pi * flicker_freq * (frame_time - start_time) + flicker_phase)
                        #opacity = (flicker_intensity + 1) / 2
                    
                        # Flicker option 2: use square-wave (on-off) flicker
                        frame_time = core.getTime() 
                        time_passed = frame_time - start_time 
                        cycle_duration = 1 / flicker_freq
                        cycle_passed = time_passed % cycle_duration
                        
                        if cycle_passed < cycle_duration / 2:
                            opacity = 1
                        else: 
                            opacity = 0
                        
                        # set opacity
                        stim.opacity = opacity  
                        # draw stimulus on screen
                        stim.draw()
                        win.flip()
                
                        # show stimulus on screen
                        stim.draw() # draw stimulus on screen
                        win.flip() # update the window to clear the screen and display the stimulus
    
                        
                        # if participant presses space bar on their keyboard...
                        if event.getKeys(['space']):
                            # get reaction time
                            curr_duration = core.getTime() - onset_time
                            ### send trigger to LSL stream to indicate participant wants to go to next word
                            marker_text = "block_" + curr_block + "_trial_" + str(curr_trial_nr) + "_" + curr_col
                            #out_marker.push_sample(["REACTION_NEXT_STIM_" + marker_text])
                            print("detected space key press -- RT: " + str(curr_duration * 1000) + " ms") # *1000 to convert s to ms
                            # break while loop
                            break
    
                        # if participant pressed button "c" for the first time and it's an n-back condition 
                        # where they're actually supposed to do that (aka not a reading baseline condition)...
                        elif event.getKeys(['c']) and curr_nback_cond != None and saw_target == False:
                            # get reaction time
                            curr_nback_RT = (core.getTime() - onset_time) * 1000 # *1000 to convert s to ms  
                            ### send trigger to LSL stream to indicate n-back response
                            marker_text = "block_" + curr_block + "_trial_" + str(curr_trial_nr) + "_" + curr_col
                            #out_marker.push_sample(["NBACK_REACTION_" + marker_text])
                            # only get first target response, we don't care if they press the button more than once:
                            saw_target = True
                            print("detected C key press -- n-back RT: " + str(curr_nback_RT) + " ms")
                        # If esc is pressed, end the experiment:
                        elif event.getKeys(['escape']):
                            core.quit()
                    
                    ### end trial
                    print("end trial")
                    # stop display of current stimulus
                    win.flip()
                    
                    # check whether response was hit, miss, false alarm or correct rejection
                    # they saw a target and there was one: hit
                    if curr_nback_cond != None: 
                        if saw_target and curr_target:
                            curr_nback_response = "hit"
                        # they didn't see a target but there was one: miss
                        elif saw_target == False and curr_target:
                            curr_nback_response = "miss"
                            curr_nback_RT = None
                        # they didn't see a target and there was none: correct rejection
                        elif saw_target == False and curr_target == False:
                            curr_nback_response = "correct rejection"
                            curr_nback_RT = None
                        # they saw a target but there was none: false alarm
                        elif saw_target and curr_target == False:
                            curr_nback_response = "false alarm"
                    # if it wasn't an n-back task block:
                    else: 
                        curr_target = None
                        curr_nback_response = None
                        curr_nback_RT = None
                    
                    ### save everything in output csv
                    thisExp.addData('colour', curr_colour)
                    thisExp.addData('target', curr_target)
                    thisExp.addData('nback_response', curr_nback_response)
                    thisExp.addData('nback_RT', curr_nback_RT) # in ms
                    thisExp.addData('duration', curr_duration * 1000) # * 1000 to convert s to ms
                    thisExp.addData('trial_nr', curr_trial_nr)
                    thisExp.addData('block_nr', exp_block_counter)
                    thisExp.addData('block_name', curr_block)
                    thisExp.addData('block_kind', curr_nback_cond)
        
                    # start a new row in the csv
                    thisExp.nextEntry()
                    
                    ### IF TESTING MODE ENABLED: end loop after 4 trials
                    if expInfo['testing_mode'] == "yes":
                        if trial_idx == 3:
                            break
                    
                    ### send stimulus offset trigger to LSL stream
                    marker_text = "block_" + curr_block + "_trial_" + str(curr_trial_nr) + "_" + curr_word + "_" + curr_colour + "_" + str(curr_nback_response)
                    #out_marker.push_sample(["STIM_OFFSET_" + marker_text])
                    
                print("finished presenting trials")
                
                # change background colour from grey (RGB: 10, 10, 10)
                # to ivory (RGB: 240, 223, 204)
                change_bg_colour(window = win, 
                                 start_rgb = dark_bg_col, 
                                 end_rgb = light_bg_col, 
                                 seconds = 2)
                # Wait for a brief period of time so bg is set
                core.wait(0.8)
                # keep background ivory
                win.setColor(light_bg_col, colorSpace='rgb')
                win.flip()
            
            ### End currrent block
            # add 1 to the block counter to go load the next block
            exp_block_counter = exp_block_counter + 1
            print("Going to block " + str(exp_block_counter + 1) + "/17 now!")
    
    # go to next routine
    print("going to next routine")
    continueRoutine = False
    # keep track of which components have finished
    no_text_blocksComponents = []
    for thisComponent in no_text_blocksComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    no_text_blocksClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "no_text_blocks"-------
    while continueRoutine:
        # get current time
        t = no_text_blocksClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=no_text_blocksClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in no_text_blocksComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "no_text_blocks"-------
    for thisComponent in no_text_blocksComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "no_text_blocks" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "text_blocks"-------
    continueRoutine = True
    # update component parameters for each repeat
    #################################################
    #                Blocks with text               #
    #################################################
    # this routine is for all blocks with texts
    
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    win.flip()
    
    # clear buffer of all previously recorded key events:
    event.clearEvents()
    
    ### specify settings for the current block
    
    ### Prepare stimuli:
    
    # get block kind
    curr_block = all_blocks[exp_block_counter]
    print("start preparing block " + curr_block)
    
    # Check whether it's some of the non-text tasks, 
    # the reading bl training or one of the dual-task main blocks.
    
    # If current block it's a non-text block, skip this routine.
    if curr_block in ["click_training", "1back_single_training1", "1back_single_training2", 
                      "1back_single_main", "2back_single_training1", "2back_single_training2", "2back_single_main"]:
        print("skipping current routine")
        # skip questions & end current routine
        skip_questions = True
        continueRoutine = False
        
    # if it's the reading bl training block, prepare training stimuli:
    elif curr_block == "Reading_Baseline_training":
        # keep background ivory
        win.setColor(light_bg_col, colorSpace='rgb')
        win.flip()
        
        ### Show instructions
        # set instruction text
        instr_text = locals()["instr_" + curr_block]
        # create text box
        instr_text_stim = visual.TextStim(win, 
                                          text = instr_text, 
                                          height = 0.5, # font height: 5° visual angle
                                          font = "Bookman Old Style",
                                          pos = (0, 0),
                                          color = "black")
            
        # display the text on screen
        while True:
            # keep background ivory
            win.setColor(light_bg_col, colorSpace='rgb')
            instr_text_stim.draw()
            win.flip()
            # end showing screen if participant presses space
            if 'space' in event.getKeys():
                break 
                
        ### get training text
        curr_text = reading_bl_tr_text
        curr_text_nr = "reading_bl_training_text"
        curr_nback_cond = None
        curr_colours = all_colour_lists[0]
        # show training questions
        skip_questions = False
        training_Qs = True
        
        ### prepare flicker
        # hint: flicker_freq and frame_rate are set in the settings 
        # code component at the beginning of the experiment.
        
        # create flicker phase variable - start at phase = 0
        flicker_phase = 0
        # we also need the start time (let's set it as current time 
        # at this point in the script):
        start_time = core.getTime()
        
        
        ### change background colour 
        # transition from ivory 
        # to medium grey 
        change_bg_colour(window = win, 
                         start_rgb = light_bg_col,
                         end_rgb = dark_bg_col, 
                         seconds = 2)
        # Wait for a brief period of time so bg is set
        core.wait(0.8)
        # keep background grey
        win.setColor(dark_bg_col, colorSpace='rgb')
        win.flip()
    
    # if it's one of the "normal" main blocks, prepare main block stimuli:
    elif curr_block in ["Reading_Baseline_main", "1back_dual_main", "2back_dual_main"]:
    
        # keep background ivory
        win.setColor(light_bg_col, colorSpace='rgb')
        win.flip()
        
        ### Show instructions
        # set instruction text
        instr_text = locals()["instr_" + curr_block]
        # create text box
        instr_text_stim = visual.TextStim(win, 
                                          text = instr_text, 
                                          height = 0.5, # font height: 5° visual angle
                                          font = "Bookman Old Style",
                                          pos = (0, 0),
                                          color = "black")
        # Display the text on screen
        while True:
            instr_text_stim.draw()
            win.flip()
            # end showing screen if participant presses space
            if 'space' in event.getKeys():
                break 
        
        ### change background colour 
        # transition from ivory to medium grey
        change_bg_colour(window = win, 
                         start_rgb = light_bg_col,
                         end_rgb = dark_bg_col, 
                         seconds = 2)
        # Wait for a brief period of time so bg is set
        core.wait(0.8)
        # keep background grey
        win.setColor(dark_bg_col, colorSpace='rgb')
        win.flip()
    
        # show main block questions
        skip_questions = False
        training_Qs = False
        
        # get text nr:
        curr_text_nr = all_texts_nrs_list[exp_block_counter]
        
        # get n-back condition:
        curr_nback_cond = curr_block[0] # get first character of block name
    
        # if it is a 1 or a 2, set that as current n-back level:
        if curr_nback_cond in ['1', '2']:
            curr_nback_cond == int(curr_nback_cond)
        # if it's neither 1 nor 2, it has to be a block without n-back,
        # so set curr_nback_cond to None
        else:
            curr_nback_cond = None
        
        print("current n-back condition: " + str(curr_nback_cond))
        
        # get list with targets & list with colours
        curr_targets = all_target_lists[exp_block_counter]
        curr_colours = all_colour_lists[exp_block_counter]
        # for current text nr, get text whose name = current text nr
        curr_text = locals()[curr_text_nr]
        
        ### prepare flicker
        # hint: flicker_freq and frame_rate are set in the settings 
        # code component at the beginning of the experiment.
        
        # create flicker phase variable - start at phase = 0
        flicker_phase = 0
        # we also need the start time (let's set it as current time 
        # at this point in the script):
        start_time = core.getTime()
        
    ### Start block loop
    
    # if it's the first reading BL block, save response 
    # times in an array - we need that later for the visual task
    if exp_block_counter == 1:
        vis_task_durations = []
    
    # create empty text stimulus 
    stim = visual.TextStim(win = win, 
                           text = " ", 
                           pos = (0,0), # center stimulus
                           font = "Times New Roman",
                           height = 1) # font height = 1° visual angle
    
    # create grey rectangle that masks the text if I set opacity to 1
    # --> changing the text opacity directly isn't working: https://discourse.psychopy.org/t/opacity-of-text-stimuli-is-not-updating/11152/7    
    stim_mask = visual.Rect(win = win,
                            width = 20, # width = 20° visual angle
                            height = 3, # height = 3° visual angle 
                            pos = (0,0), # center stimulus 
                            opacity = 0, # set opacity to 0 for a start
                            fillColor = dark_bg_col,
                            colorSpace = "rgb")
    
    stim.draw()
    stim_mask.draw()
    win.flip()
    
    # clear buffer of all previously recorded key events:
    event.clearEvents()
    
    # loop words in current text
    for trial_idx, curr_word in enumerate(curr_text):
        print("current idx: " + str(trial_idx) + ", curr word:" + curr_word)
        
        ### prepare & show current word:
        
        # get current colour
        curr_colour = curr_colours[trial_idx]
        
        # if it's a block with an n-back task, prepare target list as well
        if curr_nback_cond != None:
            curr_target = curr_targets[trial_idx]
            saw_target = False
        
        # get trial number (start counting from 1, so add 1)
        curr_trial_nr = trial_idx + 1
    
        # set current word & colour as content of text stimulus
        stim.color = curr_colour
        stim.text = curr_word
        
        # Flicker option 1: use sine-wave (gradient) flicker
        # --> doesn't seem to work, I don't see the words flicker when I play this
        # create current opacity value to continue flickering the word
        frame_time = core.getTime() # get current time point (in sec)
        flicker_intensity = np.sin(2 * np.pi * flicker_freq * (frame_time - start_time) + flicker_phase)
        opacity = (flicker_intensity + 1) / 2
    
        # use square-wave (on-off) flicker
        #frame_time = core.getTime() # get current time point (in sec)
        #time_passed = frame_time - start_time # calculate time passed since start
        #cycle_duration = 1 / flicker_freq # calculate duration of one flicker cycle
        #cycle_passed = time_passed % cycle_duration # calculate time passed in current flicker cycle
        #if cycle_passed < cycle_duration / 2: # if in the first half of the cycle
        #    opacity = 1 # set opacity to 1
        #else: # if in the second half of the cycle
        #    opacity = 0 # set opacity to 0
            
        stim_mask.opacity = opacity
        
        # show word on screen
        stim.draw() # draw word on screen
        stim_mask.draw() # draw mask on screen
        win.flip() # update the window to clear the screen and display the word
    
        # send word onset trigger to LSL stream
        marker_text = "block_" + curr_block + "_trial_" + str(curr_trial_nr) + "_" + curr_word
        #out_marker.push_sample(["STIM_ONSET_" + marker_text])
        
        # record trial onset time
        onset_time = core.getTime()
        print("onset time: " + str(onset_time) + " s")
        
        ### wait for 50 ms
        while core.getTime() < onset_time + 0.05:
            # draw the stimulus during the waiting period
    
            # Flicker option 1: use sine-wave (gradient) flicker
            frame_time = core.getTime() 
            flicker_intensity = np.sin(2 * np.pi * flicker_freq * (frame_time - start_time) + flicker_phase)
            opacity = (flicker_intensity + 1) / 2
    
            # Flicker option 2: use square-wave (on-off) flicker
            #frame_time = core.getTime() 
            #time_passed = frame_time - start_time 
            #cycle_duration = 1 / flicker_freq
            #cycle_passed = time_passed % cycle_duration
            #if cycle_passed < cycle_duration / 2:
            #    opacity = 1
            #else: 
            #    opacity = 0
            
            stim_mask.opacity = opacity
        
            stim.draw() # draw text
            stim_mask.draw() # draw mask
            win.flip()
                
        ### wait for key response: 
        # In blocks with n-back task, participants can press "c" to indicate they saw a target colour and "space" to go to the next word/stimulus.
        # In blocks without n-back task, participants can only press "space" to go to the next word/stimulus.
        print("start tracking key responses")
            
        ### start recording responses
        # start "endless" while loop that looks for responses
        while True:        
            # in each iteration, draw word on screen
            # --> flicker again
    
            # Flicker option 1: use sine-wave (gradient) flicker
            frame_time = core.getTime() 
            flicker_intensity = np.sin(2 * np.pi * flicker_freq * (frame_time - start_time) + flicker_phase)
            opacity = (flicker_intensity + 1) / 2
    
            # Flicker option 2: use square-wave (on-off) flicker
            #frame_time = core.getTime() 
            #time_passed = frame_time - start_time 
            #cycle_duration = 1 / flicker_freq
            #cycle_passed = time_passed % cycle_duration
            #if cycle_passed < cycle_duration / 2:
            #    opacity = 1
            #else: 
            #    opacity = 0
                
            stim_mask.opacity = opacity
            
            stim.draw()
            stim_mask.draw()
            win.flip()
            
            # if participant presses space bar on their keyboard...
            if event.getKeys(['space']):
                # get reaction time
                curr_duration = core.getTime() - onset_time
                ### send trigger to LSL stream to indicate participant wants to go to next word
                marker_text = "block_" + curr_block + "_trial_" + str(curr_trial_nr) + "_" + curr_word
                #out_marker.push_sample(["REACTION_NEXT_STIM_" + marker_text])
                print("detected space key press -- RT: " + str(curr_duration * 1000) + " ms") #* 1000 to covert s to ms
                # break while loop
                break
    
            # if participant pressed button "c" for the first time and it's an n-back condition 
            # where they're actually supposed to do that (aka not a reading baseline condition)...
            elif event.getKeys(['c']) and curr_nback_cond != None and saw_target == False:
                # get reaction time
                curr_nback_RT = (core.getTime() - onset_time) * 1000 # *1000 to convert s to ms    
                ### send trigger to LSL stream to indicate n-back response
                marker_text = "block_" + curr_block + "_trial_" + str(curr_trial_nr) + "_" + curr_word
                #out_marker.push_sample(["NBACK_REACTION_" + marker_text])
                # only get first target response, we don't care if they press the button more than once:
                saw_target = True
                print("detected C key press -- n-back RT: " + str(curr_nback_RT) + " ms") # * 1000 to convert s to ms
            # If esc is pressed, end the experiment:
            elif event.getKeys(['escape']):
                core.quit()
        
        ### end trial
        print("end trial")
        # stop display of current word
        win.flip()
        
        # check whether response was hit, miss, false alarm or correct rejection
        # they saw a target and there was one: hit
        if curr_nback_cond != None: 
            if saw_target and curr_target:
                curr_nback_response = "hit"
            # they didn't see a target but there was one: miss
            elif saw_target == False and curr_target:
                curr_nback_response = "miss"
                curr_nback_RT = None
            # they didn't see a target and there was none: correct rejection
            elif saw_target == False and curr_target == False:
                curr_nback_response = "correct rejection"
                curr_nback_RT = None
            # they saw a target but there was none: false alarm
            elif saw_target and curr_target == False:
                curr_nback_response = "false alarm"
        # if it wasn't an n-back task block:
        else: 
            curr_target = None
            curr_nback_response = None
            curr_nback_RT = None
        
        ### save everything in output csv
        thisExp.addData('word', curr_word)
        thisExp.addData('colour', curr_colour)
        thisExp.addData('target', curr_target)
        thisExp.addData('nback_response', curr_nback_response)
        thisExp.addData('nback_RT', curr_nback_RT) # in ms
        thisExp.addData('duration', curr_duration * 1000) # *1000 to convert s to ms
        thisExp.addData('text_nr', curr_text_nr)
        thisExp.addData('trial_nr', curr_trial_nr)
        thisExp.addData('block_nr', exp_block_counter)
        thisExp.addData('block_name', curr_block)
        thisExp.addData('block_kind', curr_nback_cond)
    
        # start a new row in the csv
        thisExp.nextEntry()
        
        # if it's the first reading BL block, we need to 
        # also collect the RTs in an array
        if exp_block_counter == 1:
            vis_task_durations.append(curr_duration)
    
        ### IF TESTING MODE ENABLED: end loop after 4 trials
        if expInfo['testing_mode'] == "yes":
            if trial_idx == 3:
                break
        
        ### send word offset trigger to LSL stream
        marker_text = "block_" + curr_block + "_trial_" + str(curr_trial_nr) + "_" + curr_word + "_" + curr_colour + "_" + str(curr_nback_response)
        #out_marker.push_sample(["STIM_OFFSET_" + marker_text])
    
    print("finished presenting trials")
    
    ### Prepare questions
    
    # change background colour from grey to ivory
    change_bg_colour(window = win, 
                     start_rgb = dark_bg_col, 
                     end_rgb = light_bg_col, 
                     seconds = 2)
    # Wait for a brief period of time so bg is set
    core.wait(0.5)
    
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    win.flip()
            
    # end current routine
    continueRoutine = False
    # keep track of which components have finished
    text_blocksComponents = []
    for thisComponent in text_blocksComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    text_blocksClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "text_blocks"-------
    while continueRoutine:
        # get current time
        t = text_blocksClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=text_blocksClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in text_blocksComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "text_blocks"-------
    for thisComponent in text_blocksComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "text_blocks" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "Q1"-------
    continueRoutine = True
    # update component parameters for each repeat
    ##########################################################
    #            Text Comprehension Questions - Q1           #
    ##########################################################
    
    ### Settings:
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    win.flip()
    
    # clear buffer of all previously recorded key events:
    event.clearEvents()
    
    # check which kind of block we have
    # if there was no text before, we can skip the questions
    if skip_questions:
        continueRoutine = False
    # if we have a training text, set training questions
    elif skip_questions == False and training_Qs:
        Q1 = reading_bl_tr_Q1
        Q1_answers = reading_bl_tr_Q1_ans
        Q1_corr = reading_bl_tr_Q1_corr
        
    # if we have a main text, set regular questions
    elif skip_questions == False and training_Qs == False:
        # load first question for current text & their respective answers
        Q1 = locals()[curr_text_nr + "_Q1"]
        Q1_answers = locals()[curr_text_nr + "_Q1_ans"]
        Q1_corr = locals()[curr_text_nr + "_Q1_corr"]
    
    # Define text positions and formatting
    question_pos = (0, 3)
    answer_xpos = -7 # move questions a bit to the left 
    answer_ypos = [ 0, -2, -4, -6] # set the y axis positions of all 4 answers
    
    # Create text stim for the question:
    question = visual.TextStim(win, 
                               text = Q1, 
                               pos = question_pos,
                               color = "black",
                               height = 0.5,
                               font = "Bookman Old Style",
                               anchorHoriz = 'center',
                               alignText = 'center', 
                               wrapWidth = 10)
    # create 1 text stim for each answer option:
    answers = [visual.TextStim(win, 
                               text = Q1_answers[i], 
                               pos = (answer_xpos, answer_ypos[i]), 
                               color = "black", # set all to black as a default
                               height = 0.5, 
                               font = "Bookman Old Style",
                               wrapWidth = 15,
                               anchorHoriz = 'left', 
                               alignText = 'center') for i in range(len(Q1_answers))]
    # set up instruction text
    instr_text = visual.TextStim(win, 
                                 text = "(Bitte benutzen Sie die Pfeiltasten um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                 color = "grey",
                                 pos = (0, -10),
                                 wrapWidth = 20,
                                 height = 0.4,
                                 font = "Bookman Old Style")
    
    ### Show all on screen until I set .autoDraw = False
    question.autoDraw = True
    instr_text.autoDraw = True
    for answer in answers:
        answer.autoDraw = True
    win.flip()
    
    
    ### Record key responses:
    Q1_chosen_ans = None
    
    while True:        
        # if 1 was pressed...
        if event.getKeys(['1']):
            print('a')
            # save Q1 answer as a 
            Q1_chosen_ans = "a"
            # set font colour of the first answer (answer a) to 
            # green and the rest to black:
            answers[0].setColor("green")
            for answer in answers[1:]:
                answer.setColor("black")
                # draw updated stimulus:
                win.flip()
        # same procedure for all other answer options:
        if event.getKeys(['2']):
            print('b')
            Q1_chosen_ans = "b"
            # set font colour of the second answer (answer b) to 
            # green and the rest to black:
            answers[1].setColor("green")
            for answer in [answers[0]] + answers[2:]:
                answer.setColor("black")
                # draw updated stimulus:
                win.flip()
        if event.getKeys(['3']):
            print('c')
            Q1_chosen_ans = "c"
            # set font colour of the third answer (answer c) to 
            # green and the rest to black:
            answers[2].setColor("green")
            for answer in answers[:2] + answers[3:]:
                answer.setColor("black")
            # draw updated stimulus:
            win.flip()
        if event.getKeys(['4']):
            print('d')
            Q1_chosen_ans = "d"
            # set font colour of the fourth answer (answer d) to 
            # green and the rest to black:
            answers[3].setColor("green")
            for answer in answers[:-1]:
                answer.setColor("black")
            # draw updated stimulus 
            win.flip()
        # if participant pressed "space", check whether they chose an answer.
        # if yes, end this routine and go to next question, if not, wait for valid answer.
        elif event.getKeys(['space']) and Q1_chosen_ans != None:
            break
    
    # print chosen answer for Q1
    print("answer for Q1:" + str(Q1_chosen_ans))
    
    # check if answer was correct:
    if Q1_chosen_ans == Q1_corr: 
        print("answer correct!")
    else: 
        print("answer incorrect!")
        
    # save data:
    thisExp.addData('question', 'Q1')
    thisExp.addData('chosen_ans', Q1_chosen_ans)
    thisExp.addData('ans_correct', Q1_chosen_ans == Q1_corr)
    thisExp.addData('text_nr', curr_text_nr)
    thisExp.addData('block_nr', exp_block_counter)
    thisExp.addData('block_name', curr_block)
    thisExp.addData('block_kind', curr_nback_cond)
                    
    # start a new row in the csv
    thisExp.nextEntry()
    
    ### End Q1: Set .autoDraw = False to stop showing question & answers
    question.autoDraw = False
    instr_text.autoDraw = False
    for answer in answers:
        answer.autoDraw = False
    
    # end current routine
    continueRoutine = False
    # keep track of which components have finished
    Q1Components = []
    for thisComponent in Q1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Q1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Q1"-------
    while continueRoutine:
        # get current time
        t = Q1Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Q1Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Q1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Q1"-------
    for thisComponent in Q1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "Q1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "Q2"-------
    continueRoutine = True
    # update component parameters for each repeat
    ##########################################################
    #            Text Comprehension Questions - Q2           #
    ##########################################################
    
    ### Settings:
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    win.flip()
    
    # clear buffer of all previously recorded key events:
    event.clearEvents()
    
    # check which kind of block we have
    # if there was no text before, we can skip the questions
    if skip_questions:
        continueRoutine = False
    # if we have a training text, set training questions
    elif skip_questions == False and training_Qs:
        Q2 = reading_bl_tr_Q2
        Q2_answers = reading_bl_tr_Q2_ans
        Q2_corr = reading_bl_tr_Q2_corr
        
    # if we have a main text, set regular questions
    elif skip_questions == False and training_Qs == False:
        # load first question for current text & their respective answers
        Q2 = locals()[curr_text_nr + "_Q2"]
        Q2_answers = locals()[curr_text_nr + "_Q2_ans"]
        Q2_corr = locals()[curr_text_nr + "_Q2_corr"]
    
    # Define text positions and formatting
    question_pos = (0, 3)
    answer_xpos = -7 # move questions a bit to the left 
    answer_ypos = [ 0, -2, -4, -6] # set the y axis positions of all 4 answers
    
    # Create text stim for the question:
    question = visual.TextStim(win, 
                               text = Q2, 
                               pos = question_pos,
                               color = "black",
                               height = 0.5,
                               font = "Bookman Old Style",
                               anchorHoriz = 'center',
                               alignText = 'center', 
                               wrapWidth = 10)
    # create 1 text stim for each answer option:
    answers = [visual.TextStim(win, 
                               text = Q2_answers[i], 
                               pos = (answer_xpos, answer_ypos[i]), 
                               color = "black", # set all to black as a default
                               height = 0.5, 
                               font = "Bookman Old Style",
                               wrapWidth = 15,
                               anchorHoriz = 'left', 
                               alignText = 'center') for i in range(len(Q1_answers))]
    # set up instruction text
    instr_text = visual.TextStim(win, 
                                 text = "(Bitte benutzen Sie die Pfeiltasten um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                 color = "grey",
                                 pos = (0, -10),
                                 wrapWidth = 20,
                                 height = 0.4,
                                 font = "Bookman Old Style")
    ### Show all on screen until I set .autoDraw = False
    question.autoDraw = True
    instr_text.autoDraw = True
    for answer in answers:
        answer.autoDraw = True
    win.flip()
    
    
    ### Record key responses:
    Q2_chosen_ans = None
    
    while True:        
        # if 1 was pressed...
        if event.getKeys(['1']):
            print('a')
            # save Q2 answer as a 
            Q2_chosen_ans = "a"
            # set font colour of the first answer (answer a) to 
            # green and the rest to black:
            answers[0].setColor("green")
            for answer in answers[1:]:
                answer.setColor("black")
                # draw updated stimulus:
                win.flip()
        # same procedure for all other answer options:
        if event.getKeys(['2']):
            print('b')
            Q2_chosen_ans = "b"
            # set font colour of the second answer (answer b) to 
            # green and the rest to black:
            answers[1].setColor("green")
            for answer in [answers[0]] + answers[2:]:
                answer.setColor("black")
                # draw updated stimulus:
                win.flip()
        if event.getKeys(['3']):
            print('c')
            Q2_chosen_ans = "c"
            # set font colour of the third answer (answer c) to 
            # green and the rest to black:
            answers[2].setColor("green")
            for answer in answers[:2] + answers[3:]:
                answer.setColor("black")
            # draw updated stimulus:
            win.flip()
        if event.getKeys(['4']):
            print('d')
            Q2_chosen_ans = "d"
            # set font colour of the fourth answer (answer d) to 
            # green and the rest to black:
            answers[3].setColor("green")
            for answer in answers[:-1]:
                answer.setColor("black")
            # draw updated stimulus 
            win.flip()
        # if participant pressed "space", check whether they chose an answer.
        # if yes, end this routine and go to next question, if not, wait for valid answer.
        elif event.getKeys(['space']) and Q2_chosen_ans != None:
            break
    
    # print chosen answer for Q2
    print("answer for Q2:" + str(Q2_chosen_ans))
    
    # check if answer was correct:
    if Q2_chosen_ans == Q2_corr: 
        print("answer correct!")
    else: 
        print("answer incorrect!")
        
    # save data:
    thisExp.addData('question', 'Q2')
    thisExp.addData('chosen_ans', Q2_chosen_ans)
    thisExp.addData('ans_correct', Q2_chosen_ans == Q2_corr)
    thisExp.addData('text_nr', curr_text_nr)
    thisExp.addData('block_nr', exp_block_counter)
    thisExp.addData('block_name', curr_block)
    thisExp.addData('block_kind', curr_nback_cond)
                    
    # start a new row in the csv
    thisExp.nextEntry()
    
    ### End Q2: Set .autoDraw = False to stop showing question & answers
    question.autoDraw = False
    instr_text.autoDraw = False
    for answer in answers:
        answer.autoDraw = False
    
    # end current routine
    continueRoutine = False
    # keep track of which components have finished
    Q2Components = []
    for thisComponent in Q2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Q2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Q2"-------
    while continueRoutine:
        # get current time
        t = Q2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Q2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Q2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Q2"-------
    for thisComponent in Q2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "Q2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "Q3"-------
    continueRoutine = True
    # update component parameters for each repeat
    ##########################################################
    #            Text Comprehension Questions - Q3           #
    ##########################################################
    
    ### Settings:
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    win.flip()
    
    # clear buffer of all previously recorded key events:
    event.clearEvents()
    
    # check which kind of block we have
    # if there was no text before, we can skip the questions
    if skip_questions:
        continueRoutine = False
    # if we have a training text, set training questions
    elif skip_questions == False and training_Qs:
        Q3 = reading_bl_tr_Q3
        Q3_answers = reading_bl_tr_Q3_ans
        Q3_corr = reading_bl_tr_Q3_corr
        
    # if we have a main text, set regular questions
    elif skip_questions == False and training_Qs == False:
        # load first question for current text & their respective answers
        Q3 = locals()[curr_text_nr + "_Q3"]
        Q3_answers = locals()[curr_text_nr + "_Q3_ans"]
        Q3_corr = locals()[curr_text_nr + "_Q3_corr"]
    
    # Define text positions and formatting
    question_pos = (0, 3)
    answer_xpos = -7 # move questions a bit to the left 
    answer_ypos = [ 0, -2, -4, -6] # set the y axis positions of all 4 answers
    
    # Create text stim for the question:
    question = visual.TextStim(win, 
                               text = Q3, 
                               pos = question_pos,
                               color = "black",
                               height = 0.5,
                               font = "Bookman Old Style",
                               anchorHoriz = 'center',
                               alignText = 'center', 
                               wrapWidth = 10)
    # create 1 text stim for each answer option:
    answers = [visual.TextStim(win, 
                               text = Q3_answers[i], 
                               pos = (answer_xpos, answer_ypos[i]), 
                               color = "black", # set all to black as a default
                               height = 0.5, 
                               font = "Bookman Old Style",
                               wrapWidth = 15,
                               anchorHoriz = 'left', 
                               alignText = 'center') for i in range(len(Q1_answers))]
    # set up instruction text
    instr_text = visual.TextStim(win, 
                                 text = "(Bitte benutzen Sie die Pfeiltasten um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                 color = "grey",
                                 pos = (0, -10),
                                 wrapWidth = 20,
                                 height = 0.4,
                                 font = "Bookman Old Style")
                                 
    ### Show all on screen until I set .autoDraw = False
    question.autoDraw = True
    instr_text.autoDraw = True
    for answer in answers:
        answer.autoDraw = True
    win.flip()
    
    
    ### Record key responses:
    Q3_chosen_ans = None
    
    while True:        
        # if 1 was pressed...
        if event.getKeys(['1']):
            print('a')
            # save Q3 answer as a 
            Q3_chosen_ans = "a"
            # set font colour of the first answer (answer a) to 
            # green and the rest to black:
            answers[0].setColor("green")
            for answer in answers[1:]:
                answer.setColor("black")
                # draw updated stimulus:
                win.flip()
        # same procedure for all other answer options:
        if event.getKeys(['2']):
            print('b')
            Q3_chosen_ans = "b"
            # set font colour of the second answer (answer b) to 
            # green and the rest to black:
            answers[1].setColor("green")
            for answer in [answers[0]] + answers[2:]:
                answer.setColor("black")
                # draw updated stimulus:
                win.flip()
        if event.getKeys(['3']):
            print('c')
            Q3_chosen_ans = "c"
            # set font colour of the third answer (answer c) to 
            # green and the rest to black:
            answers[2].setColor("green")
            for answer in answers[:2] + answers[3:]:
                answer.setColor("black")
            # draw updated stimulus:
            win.flip()
        if event.getKeys(['4']):
            print('d')
            Q3_chosen_ans = "d"
            # set font colour of the fourth answer (answer d) to 
            # green and the rest to black:
            answers[3].setColor("green")
            for answer in answers[:-1]:
                answer.setColor("black")
            # draw updated stimulus 
            win.flip()
        # if participant pressed "space", check whether they chose an answer.
        # if yes, end this routine and go to next question, if not, wait for valid answer.
        elif event.getKeys(['space']) and Q3_chosen_ans != None:
            break
    
    # print chosen answer for Q3
    print("answer for Q3:" + str(Q3_chosen_ans))
    
    # check if answer was correct:
    if Q3_chosen_ans == Q3_corr: 
        print("answer correct!")
    else: 
        print("answer incorrect!")
        
    # save data:
    thisExp.addData('question', 'Q3')
    thisExp.addData('chosen_ans', Q3_chosen_ans)
    thisExp.addData('ans_correct', Q3_chosen_ans == Q3_corr)
    thisExp.addData('text_nr', curr_text_nr)
    thisExp.addData('block_nr', exp_block_counter)
    thisExp.addData('block_name', curr_block)
    thisExp.addData('block_kind', curr_nback_cond)
    
    # start a new row in the csv
    thisExp.nextEntry()
    
    ### End Q3: Set .autoDraw = False to stop showing question & answers
    question.autoDraw = False
    instr_text.autoDraw = False
    for answer in answers:
        answer.autoDraw = False
    
    # end current routine
    continueRoutine = False
    # keep track of which components have finished
    Q3Components = []
    for thisComponent in Q3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    Q3Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "Q3"-------
    while continueRoutine:
        # get current time
        t = Q3Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=Q3Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Q3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "Q3"-------
    for thisComponent in Q3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "Q3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "difficulty"-------
    continueRoutine = True
    # update component parameters for each repeat
    ##########################################################
    #            Text Difficulty Rating - Item 1            #
    ##########################################################
    
    ### Settings:
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    win.flip()
    
    # check which kind of block we have
    # skip this routine if the current block wasn't Reading BL main
    if curr_block != "Reading_Baseline_main":
        continueRoutine = False
    else:
        
        # create a keyboard object to check if key is currently pressed 
        # (not really possible with event.getKey())
        kb = keyboard.Keyboard()
        
        # set question texts, item names and labels:
        items = ["Wie anstrengend war es für Sie, dem Text zu folgen?", "Wie schwierig fanden Sie den Text inhaltlich?", "Wie verständlich war der Text für Sie?", "Wie sehr mussten Sie sich beim Lesen konzentrieren?", "Wie einfach fanden Sie die Formulierungen im Text?", "Wie interessant fanden Sie den Text?"]
        item_names = ["subj_reading_effort1", "subj_text_difficulty", "subj_text_incomprehensibility1", "subj_reading_effort2", "subj_text_incomprehensibility2", "subj_interest_in_text"]
        item_labels = [["gar nicht anstrengend", "sehr anstrengend"], ["sehr leicht", "sehr schwierig"], ["sehr verständlich", "gar nicht verständlich"], ["gar nicht", "sehr stark"], ["sehr einfach","sehr schwierig"], ["sehr langweilig","sehr interessant"]]
        
        # loop items
        for item_idx, curr_item in enumerate(items):
            print("rating text difficulty – current item: " + item_names[item_idx])
                    
            # clear buffer of all previously recorded key events:
            event.clearEvents()
            
            # get matching labels & name of current item
            curr_item_labels = item_labels[item_idx]
            print("item labels: ", curr_item_labels)
            curr_item_name = item_names[item_idx]
            
            # set up slider
            slider = visual.Slider(win = win,
                                   pos = (0, 0), # position of the slider (centered on screen)
                                   size = (10, 0.5), # size of the scale
                                   labels = curr_item_labels, # labels for the ticks
                                   ticks = [0, 100], # make ticks at 0 and 100
                                   units = "deg", # unit = viewing angle degrees
                                   color = "black", 
                                   fillColor = "green", 
                                   borderColor = "black", 
                                   granularity = 1, # scale step size
                                   labelHeight = 0.5, # font size of the labels I guess?
                                   font = "Bookman Old Style")
            slider.markerPos = 50  # initial position of slider button
    
            # set up question text
            question_text = visual.TextStim(win, 
                                            text = curr_item,
                                            color = "black",
                                            pos = (0, 2),
                                            height = 0.6,
                                            font = "Bookman Old Style")
                                            
            # set up instruction text
            instr_text = visual.TextStim(win, 
                                         text = "(Bitte benutzen Sie die Pfeiltasten um den Punkt zu bewegen. Mit der Leertaste können Sie Ihre Bewertung bestätigen.)",
                                         color = "grey",
                                         pos = (0,-3), 
                                         height = 0.4,
                                         font = "Bookman Old Style")
            # show all on screen
            question_text.draw()
            instr_text.draw()
            slider.draw()
            win.flip()                              
            core.wait(0.1)
            
            # check for key responses
            print("set question & slider - awaiting key responses now!")
            moved_slider = False
            
            while True:
                # show stimuli on screen
                question_text.draw()
                instr_text.draw()
                slider.draw()
                win.flip()
    
                # check for key events
                keys = event.getKeys()
                # if there was a key response...
                if keys:
                    # get the last key that was pressed
                    key = keys[-1]
                    
                    # if esc was pressed, end the experiment:
                    if key == "escape":
                        print("quitting experiment")
                        core.quit()
    
                    # if left arrow key was pressed, move slider button 1 unit to the left
                    elif key == "left":
                        # move slider button
                        slider.markerPos -= 1
                        
                        # keep in mind that participant moved the slider button
                        moved_slider = True
                        
                        # update slider on screen
                        #win.flip() # clear screen before drawing question & slider again
                        core.wait(0.1)
                        question_text.draw()
                        instr_text.draw()
                        slider.draw()
                        win.flip()
                        print("moving slider button to the left")
    
                    # if right arrow key is pressed, move slider button 1 unit to the right
                    elif key == "right":
                        
                        # move slider button
                        slider.markerPos += 1
                        
                        # keep in mind that participant moved the slider button
                        moved_slider = True
                        
                        # update slider on screen
                        #win.flip() # clear screen before drawing question & slider again
                        core.wait(0.1)
                        question_text.draw()
                        instr_text.draw()
                        slider.draw()
                        win.flip()
                        print("moving slider button to the right")
    
                    # if space bar is pressed and participant moved slider, save rating and go to next item
                    elif key == "space" and moved_slider == True:
                        # get slider position aka rating
                        curr_rating = slider.markerPos
                        print("Participant rated " +  curr_item_name +  " as: ", curr_rating)
                        
                        # save data:
                        thisExp.addData('question', curr_item_name)
                        thisExp.addData('chosen_ans', curr_rating)
                        thisExp.addData('text_nr', curr_text_nr)                    
                        thisExp.addData('block_nr', exp_block_counter)
                        thisExp.addData('block_name', curr_block)
                        thisExp.addData('block_kind', curr_nback_cond)
                        # start a new row in the csv
                        thisExp.nextEntry()            
                        
                        # clear window for next item
                        win.flip()
                        # wait for 500 ms before drawing the next item on screen
                        core.wait(0.5)
                        break  # end the while loop
    
    # go to next block!
    exp_block_counter += 1
    print("Going to block " + str(exp_block_counter + 1) + "/17 now!")
    continueRoutine = False
    
    # If there are still blocks left, go to next one.
    # If not, end loop here:
    if len(all_colour_lists) <= exp_block_counter + 1:
        blocks.finished = True
    
    
    # keep track of which components have finished
    difficultyComponents = []
    for thisComponent in difficultyComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    difficultyClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "difficulty"-------
    while continueRoutine:
        # get current time
        t = difficultyClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=difficultyClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in difficultyComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "difficulty"-------
    for thisComponent in difficultyComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # the Routine "difficulty" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 30.0 repeats of 'blocks'

# get names of stimulus parameters
if blocks.trialList in ([], [None], None):
    params = []
else:
    params = blocks.trialList[0].keys()
# save data for this loop
blocks.saveAsText(filename + 'blocks.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

# ------Prepare to start Routine "vis_task"-------
continueRoutine = True
# update component parameters for each repeat
# Purely visual task (-> no motor responses) 

# In this task, the first reading baseline 
# text is presented again, but this time the text proceeds 
# automatically without the participant having to press a button.
# To make sure the words are not presented too fast, 
# we take the exact time each word was presented on screen 
# from the block where the participant could control the speed.

# Put differently: We take both the text from the first reading BL 
# block AND the measured reading times for each word from this 
# block and show the text again, but this time with the previously 
# recorded duration for each word.


### INSTRUCTIONS:
# keep background ivory
win.setColor(light_bg_col, colorSpace='rgb')
win.flip()

### Show instructions
# set instruction text
instr_text = locals()["instr_visual_task"]
# create text box
instr_text_stim = visual.TextStim(win, 
                                  text = instr_text, 
                                  height = 0.5, # font height: 5° visual angle
                                  font = "Bookman Old Style",
                                  pos = (0, 0),
                                  color = "black")

# display the instructions on screen
while True:
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    instr_text_stim.draw()
    win.flip()
    # end screen if participant presses space
    if 'space' in event.getKeys():
        break 

  
### START VISUAL TASK BLOCK:

### change background colour 
# transition from ivory 
# to medium grey 
change_bg_colour(window = win, 
                 start_rgb = light_bg_col,
                 end_rgb = dark_bg_col, 
                 seconds = 2)
# Wait for a brief period of time so bg is set
core.wait(0.8)
# keep background grey
win.setColor(dark_bg_col, colorSpace='rgb')
win.flip()

### prepare stimuli
# get text from second block (where exp_counter was 1, that's the first reading BL block)
curr_text_nr = all_texts_nrs_list[1]
curr_text = locals()[curr_text_nr]
# get the same colours we used before
curr_colours = all_colour_lists[exp_block_counter]
# we don't have targets but we save this anyway
curr_targets = all_target_lists[exp_block_counter]

# the RTs are saved in the array "vis_task_durations"


### prepare flicker
# hint: flicker_freq and frame_rate are set in the settings 
# code component at the beginning of the experiment.

# create flicker phase variable - start at phase = 0
flicker_phase = 0
# we also need the start time (let's set it as current time 
# at this point in the script):
start_time = core.getTime()

### start block loop

# create empty text stimulus 
stim = visual.TextStim(win = win, 
                       text = " ", 
                       pos = (0,0), # center stimulus
                       font = "Times New Roman",
                       height = 1) # font height = 1° visual angle

# create grey rectangle that masks the text if I set opacity to 1
# --> changing the text opacity directly isn't working: https://discourse.psychopy.org/t/opacity-of-text-stimuli-is-not-updating/11152/7    
stim_mask = visual.Rect(win = win,
                        width = 20, # width = 20° visual angle
                        height = 3, # height = 3° visual angle 
                        pos = (0,0), # center stimulus 
                        opacity = 0, # set opacity to 0 for a start
                        fillColor = dark_bg_col,
                        colorSpace = "rgb")

stim.draw()
stim_mask.draw()
win.flip()


# loop words in current text
for trial_idx, curr_word in enumerate(curr_text):
    print("current idx: " + str(trial_idx) + ", curr word:" + curr_word)
    
    ### prepare & show current word:
    
    # get current colour
    curr_colour = curr_colours[trial_idx]
    
    # get duration for current word
    curr_duration = vis_task_durations[trial_idx]
    
    # get trial number (start counting from 1, so add 1)
    curr_trial_nr = trial_idx + 1

    # set current word & colour as content of text stimulus
    stim.color = curr_colour
    stim.text = curr_word
    
    # Flicker option 1: use sine-wave (gradient) flicker
    # create current opacity value to continue flickering the word
    frame_time = core.getTime() # get current time point (in sec)
    flicker_intensity = np.sin(2 * np.pi * flicker_freq * (frame_time - start_time) + flicker_phase)
    opacity = (flicker_intensity + 1) / 2

    # use square-wave (on-off) flicker
    #frame_time = core.getTime() # get current time point (in sec)
    #time_passed = frame_time - start_time # calculate time passed since start
    #cycle_duration = 1 / flicker_freq # calculate duration of one flicker cycle
    #cycle_passed = time_passed % cycle_duration # calculate time passed in current flicker cycle
    #if cycle_passed < cycle_duration / 2: # if in the first half of the cycle
    #    opacity = 1 # set opacity to 1
    #else: # if in the second half of the cycle
    #    opacity = 0 # set opacity to 0
        
    stim_mask.opacity = opacity
    
    # show word on screen
    stim.draw() # draw word on screen
    stim_mask.draw() # draw mask on screen
    win.flip() # update the window to clear the screen and display the word

    # send word onset trigger to LSL stream
    marker_text = "trial_" + str(curr_trial_nr) + "_" + curr_word + "_" + curr_colour
    #out_marker.push_sample(["STIM_ONSET_vis_task" + marker_text])
    
    # record trial onset time
    onset_time = core.getTime()
    print("word duration: " +  str(onset_time + curr_duration) + " ms")
    
    ### wait until curr_duration is over
    while core.getTime() < onset_time + curr_duration:
        # draw the stimulus during the waiting period

        # Flicker option 1: use sine-wave (gradient) flicker
        frame_time = core.getTime() 
        flicker_intensity = np.sin(2 * np.pi * flicker_freq * (frame_time - start_time) + flicker_phase)
        opacity = (flicker_intensity + 1) / 2

        # Flicker option 2: use square-wave (on-off) flicker
        #frame_time = core.getTime() 
        #time_passed = frame_time - start_time 
        #cycle_duration = 1 / flicker_freq
        #cycle_passed = time_passed % cycle_duration
        #if cycle_passed < cycle_duration / 2:
        #    opacity = 1
        #else: 
        #    opacity = 0
        
        stim_mask.opacity = opacity
    
        stim.draw() # draw text
        stim_mask.draw() # draw mask
        win.flip()

    
    ### save everything in output csv
    thisExp.addData('word', curr_word)
    thisExp.addData('colour', curr_colour)
    thisExp.addData('target', curr_target)
    thisExp.addData('duration', curr_duration * 1000) # *1000 to convert s to ms
    thisExp.addData('text_nr', curr_text_nr)
    thisExp.addData('trial_nr', curr_trial_nr)
    thisExp.addData('block_cond', 'None')
    thisExp.addData('block_nr', exp_block_counter)
    thisExp.addData('block_name', 'visual_task')
    
    # start a new row in the csv
    thisExp.nextEntry()

    ### IF TESTING MODE ENABLED: end loop after 4 trials
    if expInfo['testing_mode'] == "yes":
        if trial_idx == 3:
            break
    
    ### send word offset trigger to LSL stream
    marker_text = "trial_" + str(curr_trial_nr) + "_" + curr_word + "_" + curr_colour
    #out_marker.push_sample(["STIM_OFFSET_vis_task" + marker_text])
    
print("finished visual task block")

# change background colour from grey to ivory
change_bg_colour(window = win, 
                 start_rgb = dark_bg_col, 
                 end_rgb = light_bg_col, 
                 seconds = 2)
# Wait for a brief period of time so bg is set
core.wait(0.5)

# keep background ivory
win.setColor(light_bg_col, colorSpace='rgb')
win.flip()
        
# go to next block
exp_block_counter += 1
continueRoutine = False




# keep track of which components have finished
vis_taskComponents = []
for thisComponent in vis_taskComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
vis_taskClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "vis_task"-------
while continueRoutine:
    # get current time
    t = vis_taskClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=vis_taskClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in vis_taskComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "vis_task"-------
for thisComponent in vis_taskComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "vis_task" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "motor_task"-------
continueRoutine = True
# update component parameters for each repeat
# Purely motoric task (-> more or less no visual stimuli) 

# In this task, the participant is presented with a number of 
# squares that are separated by ISIs of a fixed duration. 
# Each square is shown for 100 ms, during the ISI nothing is shown on screen.
# The duration of the ISI is the median duration of the reading times 
# recorded in the first reading baseline block.

# compute median RT from previously recorded durations:
ISI_duration = np.median(vis_task_durations)
print("RT median:", ISI_duration)


### INSTRUCTIONS:
# keep background ivory
win.setColor(light_bg_col, colorSpace='rgb')
win.flip()

### Show instructions
# set instruction text
instr_text = locals()["instr_motor_task"]
# create text box
instr_text_stim = visual.TextStim(win, 
                                  text = instr_text, 
                                  height = 0.5, # font height: 5° visual angle
                                  font = "Bookman Old Style",
                                  pos = (0, 0),
                                  color = "black")

# display the instructions on screen
while True:
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    instr_text_stim.draw()
    win.flip()
    # end screen if participant presses space
    if 'space' in event.getKeys():
        break 

  
### START MOTOR TASK BLOCK:

### change background colour 
# transition from ivory 
# to medium grey 
change_bg_colour(window = win, 
                 start_rgb = light_bg_col,
                 end_rgb = dark_bg_col, 
                 seconds = 2)
# Wait for a brief period of time so bg is set
core.wait(0.8)
# keep background grey
win.setColor(dark_bg_col, colorSpace='rgb')
win.flip()


### start block loop

# create "empty" square stimulus:
stim = visual.Rect(win = win,
                   width = 2, # width = 2° visual angle
                   height = 2, # height = 2° visual angle 
                   pos = (0,0), # center stimulus 
                   opacity = 0, # set opacity to 0 for a start
                   fillColor = dark_bg_col,
                   colorSpace = "rgb")
# show it on screen
stim.draw()
stim_mask.draw()
win.flip()

block_onset_time = core.getTime()
# start showing 20 cues for motor response
for cue_idx in range(0,20):
    
    # show cue (white square)
    stim.color = [(x / 127.5) - 1 for x in (255, 255, 255)] # white
    stim.draw()
    win.flip()
    last_cue_onset_timestamp = core.getTime()
    # send cue onset trigger to LSL stream
    #out_marker.push_sample(["STIM_ONSET_motor_task"])
    
    # wait until 100 ms are over
    while core.getTime() < last_cue_onset_timestamp + 0.1:
        # draw the stimulus during the waiting period
        stim.draw()
        win.flip()
        
        # Check if participant pressed space - if yes, send trigger + record data:
        if event.getKeys(['space']):
            # send reaction trigger to LSL stream
            #out_marker.push_sample(["REACTION_motor_task"])
            
            # record timestamp of reaction and save in output csv
            reaction_timestamp = core.getTime()
   
            # save data in output csv:
            thisExp.addData('block_cond', 'None')
            thisExp.addData('block_nr', exp_block_counter)
            thisExp.addData('block_name', 'motor_task')
            thisExp.addData('reaction_timestamp', reaction_onset)
            thisExp.addData('last_cue_onset_timestamp', last_cue_onset_timestamp)
            
            print("participant pressed space!")
        
        # make cue disappear by making colour match the bg colour again
        stim.color = dark_bg_col
        stim.draw()
        win.flip()
    
        # send cue offset trigger to LSL stream
        #out_marker.push_sample(["STIM_OFFSET_motor_task"])
    
        # start recording motor response (space bar press)
        onset_time = core.getTime()
        
        while core.getTime() < onset_time + ISI_duration:
        
            # if space bar was pressed, record RT, send trigger & save data
            if event.getKeys(['space']):
                # send reaction trigger to LSL stream
                #out_marker.push_sample(["REACTION_motor_task"])
                
                # record timestamp of reaction and save in output csv
                reaction_timestamp = core.getTime()
       
                # save data in output csv:
                thisExp.addData('block_cond', 'None')
                thisExp.addData('block_nr', exp_block_counter)
                thisExp.addData('block_name', 'motor_task')
                thisExp.addData('reaction_timestamp', reaction_onset)
                thisExp.addData('last_cue_onset_timestamp', last_cue_onset_timestamp)
                
                print("participant pressed space!")  
                
            # --> don't save anything if participant didn't react, just show the next cue!



# If all cues were shown, start recording all motor 
# responses for a duration of 3 min:

onset_time = core.getTime()
while core.getTime() < onset_time + 3:
    
    # if there was a key response, send trigger & save data:
    if event.getKeys(['space']):
        
        # send reaction trigger to LSL stream
        #out_marker.push_sample(["REACTION_motor_task"])
        
        # record timestamp of reaction and save in output csv
        reaction_timestamp = core.getTime()
       
        # save data in output csv:
        thisExp.addData('block_cond', 'None')
        thisExp.addData('block_nr', exp_block_counter)
        thisExp.addData('block_name', 'motor_task')
        thisExp.addData('reaction_timestamp', reaction_onset)
        thisExp.addData('last_cue_onset_timestamp', last_cue_onset_timestamp)
        
    # start a new row in the csv
    thisExp.nextEntry()

    ### IF TESTING MODE ENABLED: end loop after 20 trials
    if expInfo['testing_mode'] == "yes":
        if trial_idx == 20:
            break
    
print("finished motor task block!")

# change background colour from grey to ivory
change_bg_colour(window = win, 
                 start_rgb = dark_bg_col, 
                 end_rgb = light_bg_col, 
                 seconds = 2)
# Wait for a brief period of time so bg is set
core.wait(0.5)

# keep background ivory
win.setColor(light_bg_col, colorSpace='rgb')
win.flip()
        
# go to next block (if there is one left)
exp_block_counter += 1
continueRoutine = False




# keep track of which components have finished
motor_taskComponents = []
for thisComponent in motor_taskComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
motor_taskClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "motor_task"-------
while continueRoutine:
    # get current time
    t = motor_taskClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=motor_taskClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in motor_taskComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "motor_task"-------
for thisComponent in motor_taskComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "motor_task" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
