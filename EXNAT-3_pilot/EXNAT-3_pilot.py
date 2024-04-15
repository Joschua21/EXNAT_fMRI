#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on Thu Apr 11 13:58:52 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding
import math

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'EXNAT-3_pilot'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'testing_mode': 'yes',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/sandramartin/ownCloud/EXNAT/EXNAT_fMRI/EXNAT-3 pilot/EXNAT-3_pilot.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.WARNING)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.WARNING)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1440, 900], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Settings" ---
    # Run 'Begin Experiment' code from functions
    # set screen resolution for eyetracker here:
    SCN_W, SCN_H = (1280, 800)
    
    ### import packages:
    
    # for setting the output encoding to UTF-8
    # import sys
    # --> if you don't do this, German "Umlaute" can't be displayed correctly:
    sys.stdout = open(sys.stdout.fileno(), mode = 'w', encoding = 'utf8', buffering = 1)
    # print Python environment psychopy is currently using
    print(sys.executable)
    
    # for showing pictures
    from psychopy import visual
    # for getting current date & time:
    import datetime
    # numpy for being able to calculate
    import numpy as np
    # for random number generator:
    import random
    # for saving data in csv / working with pd data frames:
    import pandas as pd
    # additional timing package (I know we have core.wait, but I also want this one)
    import time
    
    # Get functions from my custom scripts:
    # import all texts
    from EXNAT3_texts_MC_Qs import instr_pic_path, instr_Reading_Baseline_training_click, instr_pic_Reading_Baseline_training_click, instr_Reading_Baseline_main_click, instr_pic_Reading_Baseline_main_click, instr_Reading_Baseline_main_no_click, instr_click_training, instr_pic_click_training, instr_1back_single_training1, instr_pic_1back_single_training1, instr_1back_single_training2, instr_pic_1back_single_training2, instr_1back_single_main, instr_pic_1back_single_main, instr_1back_dual_main_click, instr_pic_1back_dual_main_click, instr_1back_dual_main_no_click, instr_2back_single_training1, instr_pic_2back_single_training1, instr_2back_single_training2, instr_pic_2back_single_training2, instr_2back_single_main, instr_pic_2back_single_main, instr_2back_dual_main_click, instr_pic_2back_dual_main_click, instr_2back_dual_main_no_click, instr_Reading_Baseline_training_no_click, instr_pic_1back_dual_main_no_click, instr_pic_2back_dual_main_no_click, warning_sign, reading_bl_tr_text, reading_bl_tr_Q1, reading_bl_tr_Q1_ans, reading_bl_tr_Q1_corr, reading_bl_tr_Q2, reading_bl_tr_Q2_ans, reading_bl_tr_Q2_corr, reading_bl_tr_Q3, reading_bl_tr_Q3_ans, reading_bl_tr_Q3_corr, reading_bl_tr_text_no_click, reading_bl_tr_no_click_Q1, reading_bl_tr_no_click_Q1_ans, reading_bl_tr_no_click_Q1_corr, reading_bl_tr_no_click_Q2, reading_bl_tr_no_click_Q2_ans, reading_bl_tr_no_click_Q2_corr, reading_bl_tr_no_click_Q3, reading_bl_tr_no_click_Q3_ans, reading_bl_tr_no_click_Q3_corr, text_01, text_01_Q1, text_01_Q1_ans, text_01_Q1_corr, text_01_Q2, text_01_Q2_ans, text_01_Q2_corr, text_01_Q3, text_01_Q3_ans, text_01_Q3_corr, text_02, text_02_Q1, text_02_Q1_ans, text_02_Q1_corr, text_02_Q2, text_02_Q2_ans, text_02_Q2_corr,text_02_Q3, text_02_Q3_ans, text_02_Q3_corr, text_03,text_03_Q1, text_03_Q1_ans, text_03_Q1_corr, text_03_Q2, text_03_Q2_ans, text_03_Q2_corr, text_03_Q3, text_03_Q3_ans, text_03_Q3_corr, text_04, text_04_Q1, text_04_Q1_ans, text_04_Q1_corr, text_04_Q2, text_04_Q2_ans, text_04_Q2_corr, text_04_Q3, text_04_Q3_ans, text_04_Q3_corr, text_05, text_05_Q1, text_05_Q1_ans, text_05_Q1_corr, text_05_Q2,text_05_Q2_ans, text_05_Q2_corr, text_05_Q3, text_05_Q3_ans, text_05_Q3_corr, text_06, text_06_Q1, text_06_Q1_ans, text_06_Q1_corr, text_06_Q2, text_06_Q2_ans, text_06_Q2_corr, text_06_Q3, text_06_Q3_ans, text_06_Q3_corr, text_07, text_07_Q1, text_07_Q1_ans, text_07_Q1_corr, text_07_Q2, text_07_Q2_ans, text_07_Q2_corr, text_07_Q3,text_07_Q3_ans,text_07_Q3_corr, text_08, text_08_Q1, text_08_Q1_ans, text_08_Q1_corr, text_08_Q2, text_08_Q2_ans, text_08_Q2_corr, text_08_Q3, text_08_Q3_ans, text_08_Q3_corr, text_09, text_09_Q1, text_09_Q1_ans, text_09_Q1_corr, text_09_Q2, text_09_Q2_ans, text_09_Q2_corr, text_09_Q3, text_09_Q3_ans, text_09_Q3_corr, text_10, text_10_Q1, text_10_Q1_ans, text_10_Q1_corr, text_10_Q2, text_10_Q2_ans, text_10_Q2_corr, text_10_Q3, text_10_Q3_ans, text_10_Q3_corr
    
    # import some additional functions I wrote for the experiment:
    # from EXNAT3_study_components import change_bg_colour
    from nback_colour_generator import create_nback_stimlist, draw_without_replacement, get_targets, create_0back_stimlist
    
    # build little function to flatten nested lists:
    def flatten_list(nested_list):
        flattened_list = []
        for item in nested_list:
            if isinstance(item, list):
                flattened_list.extend(flatten_list(item))
            else:
                flattened_list.append(item)
        return flattened_list
    
    # If I try to save strings containing escaped quotes in a csv file,
    # the format gets completely messed up. So we need to escape all
    # weird characters like quotes and backslashes with quotes (as odd as it sounds).
    def escape_quotes(string):
        # escape quotes with quotes instead of backslashes
        return string.replace('"', '""')
    
    # make mouse invisible during experiment
    #mouse = io.devices.mouse
    win.setMouseVisible(False)
    
    # create 10 ms timer that we can use instead of core.wait()
    my_timer = core.CountdownTimer(0.01)
    # Run 'Begin Experiment' code from stimuli
    ### Stimulus settings
    
    # measure frame rate (in Hz)
    # frame_rate = win.getActualFrameRate() # frame rate in Hz
    # print("measured frame rate:", frame_rate, "Hz")
    # set flicker frequency (in Hz)
    # flicker_freq = frame_rate/4 # 60/4 = 15 Hz
    
    # set colours you want to use for background:
    # light_bg_col_hex = "#FDFBF0" # ivory instructions background
    # dark_bg_col_hex  = "#505050" # dark grey background for stimuli
    light_bg_col = [(x / 127.5) - 1 for x in (253, 251, 240)]  # ivory instructions background (use RGB -1:1)
    dark_bg_col = [(x / 127.5) - 1 for x in (80, 80, 80)]  # dark grey background for stimuli (use RGB -1:1)
    
    # for timing test:
    # dark_bg_col = [(x / 127.5) - 1 for x in (255, 255, 255)]
    
    # make background light for a start - use rgb -1:1 colour codes
    win.setColor(light_bg_col, colorSpace='rgb')
    
    # set colours you want to use for the stimuli:
    colours = ["#D292F3", "#F989A2", "#2AB7EF", "#88BA3F"]
    print("Preparing experiment with n-back colours:", colours)
    # for timing test:
    # colours = ["#000000", "#F989A2", "#2AB7EF", "#88BA3F"]
    
    #  #D292F3 = weird lilac with a 2000s vibe
    #  #F989A2 = Barbie pink
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
    all_main_texts_nrs_list = ["text_01", "text_02", "text_03", "text_04", "text_05", "text_06", "text_07", "text_08",
                               "text_09", "text_10"]
    # shuffle text numbers
    random.shuffle(all_main_texts_nrs_list)
    
    # only get first 9 texts for the main blocks, the last one will be used for the vis task:
    #vis_task_text_nr = all_main_texts_nrs_list[-1]
    all_main_texts_nrs_list = all_main_texts_nrs_list[0:-1]
    
    # append "empty" text numbers to the list where we have blocks that are not main blocks.
    all_texts_nrs_list = []
    
    for t_idx, t in enumerate(all_main_texts_nrs_list):
        # if it's the first text, it's the reading BL main block.
        # Append 1 empty text number before text (for the reading BL training) and one after for the paced reading BL training
        if t_idx == 0:
            all_texts_nrs_list = all_texts_nrs_list + ["", t, ""]
        elif t_idx == 1:
            all_texts_nrs_list = all_texts_nrs_list + [t]
        elif t_idx == 2:
            all_texts_nrs_list = all_texts_nrs_list + [t, "", ""]
        elif t_idx == 3:
            all_texts_nrs_list = all_texts_nrs_list + [t]
        elif t_idx == 4:
            all_texts_nrs_list = all_texts_nrs_list + [t, "", ""]
        # all following texts are main blocks and can be appended to all_texts_nrs_list
        elif t_idx > 4:
            all_texts_nrs_list.append(t)
    
    print(all_texts_nrs_list)
    
    ### Set order of blocks
    print("set block order")
    
    # this always comes first in the experiment
    Reading_BL = ["Reading_Baseline_training_click", "Reading_Baseline_main_click", "Reading_Baseline_training_no_click",
                  "Reading_Baseline_main_no_click", "Reading_Baseline_main_no_click"] #"click_training"
    
    # then you get both n-back conditions with trainings (which of them is first is randomized)
    oneback = ["1back_single_training1", "1back_single_training2", "1back_dual_main_click", "1back_dual_main_no_click"]
    twoback = ["2back_single_training1", "2back_single_training2", "2back_dual_main_click", "2back_dual_main_no_click"]
    
    # shuffle the order of the 2 lists
    main_blocks1 = [oneback, twoback]
    random.shuffle(main_blocks1)
    
    # flatten nested list
    main_blocks1 = flatten_list(main_blocks1)
    
    # now shuffle order of the last 6 main blocks:
    main_blocks2 = ["1back_dual_main_no_click", "2back_dual_main_no_click"]
    random.shuffle(main_blocks2)
    
    # put them all together:
    # global all_blocks
    all_blocks = Reading_BL + main_blocks1 + main_blocks2
    print(all_blocks)
    ### Create n-back colour lists for all blocks
    
    print("create n-back colour lists")
    # The reading bl training text has 159 trials.
    
    # The click training has 6 trials.
    
    # Then we also have 4 short training blocks à 20 trials each (5 targets)
    # 4 * single training
    
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
    blocks_textlen = [159, 300, 146, 300, 300,  # reading bl blocks
                      20, 20, 300, 300, # main block 1 trainings & dual-tasks (300 trials)
                      20, 20, 300, 300, # main block 2 trainings & dual-tasks (300 trials)
                      300, 300]  # main blocks 2
    blocks_target_counts = [25, 50, 25, 50, 50,  # reading bl blocks
                            5, 5, 50, 50, # main block 1 trainings & main block
                            5, 5, 50, 50, # main block 2 trainings & main block
                            50, 50]
    # Now loop this list. Check which condition we have there and the create colour list for each text.
    all_colour_lists = []
    all_target_lists = []
    for block_idx, block_length in enumerate(blocks_textlen):
        # get 1st letter of block name - that tells us the condition
        block_cond = all_blocks[block_idx][0]
    
        # for each condition, decide which n-back level we want to assign
        # For all no-n-back blocks, we use 1 (just for the colour list generation)
        # global curr_nback_level
        if block_cond == "R":
            curr_nback_level = 1
        elif block_cond == "c":
            curr_nback_level = 1
        elif block_cond == "1":
            curr_nback_level = 1
        else:
            curr_nback_level = 2
    
        # generate colour list for current block
        # global curr_colours
        curr_colours = create_nback_stimlist(nback_level=curr_nback_level,
                                             colour_codes=colours,
                                             story=["x"] * block_length,
                                             target_abs_min=blocks_target_counts[block_idx],
                                             target_abs_max=blocks_target_counts[block_idx],
                                             zeroback_target=None)
    
        # get list of targets / non-targets
        curr_targets = get_targets(stim_list=curr_colours,
                                   nback_level=curr_nback_level)
    
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
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "no_text_training" ---
    
    # --- Initialize components for Routine "text_blocks_self_paced" ---
    
    # --- Initialize components for Routine "text_blocks_paced" ---
    
    # --- Initialize components for Routine "end" ---
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Settings" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Settings.started', globalClock.getTime())
    # keep track of which components have finished
    SettingsComponents = [empty_placeholder]
    for thisComponent in SettingsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Settings" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *empty_placeholder* updates
        
        # if empty_placeholder is starting this frame...
        if empty_placeholder.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            empty_placeholder.frameNStart = frameN  # exact frame index
            empty_placeholder.tStart = t  # local t and not account for scr refresh
            empty_placeholder.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(empty_placeholder, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'empty_placeholder.started')
            # update status
            empty_placeholder.status = STARTED
            empty_placeholder.setAutoDraw(True)
        
        # if empty_placeholder is active this frame...
        if empty_placeholder.status == STARTED:
            # update params
            pass
        
        # if empty_placeholder is stopping this frame...
        if empty_placeholder.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > empty_placeholder.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                empty_placeholder.tStop = t  # not accounting for scr refresh
                empty_placeholder.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'empty_placeholder.stopped')
                # update status
                empty_placeholder.status = FINISHED
                empty_placeholder.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in SettingsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Settings" ---
    for thisComponent in SettingsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Settings.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
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
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "no_text_training" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('no_text_training.started', globalClock.getTime())
        # Run 'Begin Routine' code from no_text_and_training_2
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
            # print("start preparing block " + curr_block)
        
            # Check whether it's one of the non-text tasks.
            # If current block is a text block, skip this routine and go to the next.
            if curr_block not in ["click_training", "1back_single_training1", "1back_single_training2",
                                  "2back_single_training1", "2back_single_training2"]:
                print(f"this is block {curr_block}")
                print(f"\tskipping n-back routine")
                break
        
            # if it's one of the non-text blocks, though, prepare stimuli:
            else:
                print(f"this is block {curr_block}")
                print(f"\tstart preparing block {curr_block}")
                # print("\t" + curr_block + " is not a text block - preparing rect as stim now")
        
                # keep background ivory
                win.setColor(light_bg_col, colorSpace='rgb')
                win.flip()
        
                ### Show instructions
                # set instruction text
                instr_text = locals()["instr_" + curr_block]
                # create text box
                instr_text_stim = visual.TextStim(win,
                                                  text=instr_text,
                                                  height=0.025,  # font height relative to height of screen
                                                  pos=(0, 0.2),  # move up a bit
                                                  color="black")
                # create ImageStim object
                curr_instr_pic = visual.ImageStim(win,
                                                  size=(0.8, 0.3),
                                                  pos=(0, -0.2),
                                                  image=locals()["instr_pic_" + curr_block])  # set path to image here
        
                # display the text & image on screen
                if curr_block in ["1back_single_training2", "2back_single_training2"]:
        
                    # draw instructions on screen
                    instr_text_stim.draw()
                    curr_instr_pic.draw()
                    win.flip()
                    core.wait(3)  # wait for 3 s before starting response window
        
                    while True:
                        instr_text_stim.draw()
                        curr_instr_pic.draw()
                        win.flip()
                        # skip current block (aka the second training block))
                        if event.getKeys(['space']):
                            print("\t\tstart next block - skip second training block")
                            skip_curr_block = True
                            break
                        # repeat training (aka run current block, which is the second training block)
                        elif event.getKeys(['w']):
                            print("\t\trepeat training block")
                            skip_curr_block = False
                            break
                # for regular blocks that can't be repeated:
                else:
                    while True:
                        instr_text_stim.draw()
                        curr_instr_pic.draw()
                        win.flip()
                        # start current block
                        if event.getKeys(['space']):
                            print("\t\tstart current block")
                            skip_curr_block = False
                            break
        
                # only run this if the current block shall not be skipped:
                if skip_curr_block == False:
                    ### change background colour
                    win.setColor(dark_bg_col, colorSpace='rgb')
                    win.flip()
        
                    # don't show questions
                    skip_questions = True
                    training_Qs = False
        
                    # get n-back condition:
                    curr_nback_cond = curr_block[0]  # get first character of block name
        
                    # if it is a 1 or a 2, set that as current n-back level:
                    if curr_nback_cond in ['1', '2']:
                        curr_nback_cond == int(curr_nback_cond)
                    # if it's neither 1 nor 2, it has to be a block without n-back,
                    # so set curr_nback_cond to None
                    else:
                        curr_nback_cond = None
        
                    print(f"\tcurrent n-back condition: {curr_nback_cond}")
        
                    # get list with targets & list with colours
                    curr_targets = all_target_lists[exp_block_counter]
                    curr_colours = all_colour_lists[exp_block_counter]
                    # for current text nr, get text whose name = current text nr
                    curr_text = locals()[curr_text_nr]
        
                    ### Start block loop
        
                    # CREATE CLOCKS:
                    my_block_clock = core.Clock()
                    my_block_clock.reset()  # start block clock
                    start_time = my_block_clock.getTime()  # get start time of block
                    # also create trial clock
                    my_trial_clock = core.Clock()
        
                    # create empty stimulus
                    stim = visual.Rect(win=win,
                                       width=0.4,  # width = 3 * 1° visual angle (to make it look rectangle-ish)
                                       height=0.15,  # height = 1° visual angle (just like words)
                                       # colorSpace = "hex",
                                       pos=(0, 0))  # center stimulus
        
                    stim.draw()
                    win.flip()
        
                    # clear buffer of all previously recorded key events:
                    event.clearEvents()
        
                    # loop colours in current text
                    for trial_idx, curr_col in enumerate(curr_colours):
                        # print("current idx: " + str(trial_idx) + ", curr colour:" + curr_col)
        
                        ### prepare & show current word:
                        my_trial_clock.reset()  # start trial clock
                        onset_time = my_trial_clock.getTime()
        
                        # if it's a block with an n-back task, prepare target list
                        if curr_nback_cond != None:
                            curr_target = curr_targets[trial_idx]
                            saw_target = False
        
                        # get trial number (start counting from 1, so add 1)
                        curr_trial_nr = trial_idx + 1
        
                        ### ISI: wait for 200 ms
                        while my_trial_clock.getTime() < 0.2:
                            win.flip()  # don't draw anything
                            core.wait(0.005)  # wait 5 ms before next iteration
        
                        # set current colour as colour of rectangle
                        stim.fillColor = curr_col
        
                        # draw stimulus on screen
                        stim.draw()
                        win.flip()
        
                        # show stimulus on screen & send trigger:
                        stim.draw()  # draw stimulus on screen
                        # update the window to clear the screen and display
                        # the stimulus
        
                        # start trial clock for measuring RTs from stimulus onset
                        my_trial_clock.reset()
        
                        # onset_time = my_trial_clock.getTime()
        
                        ### wait for key response:
                        # In blocks with n-back task, participants can press "c" to indicate they saw a target colour and "space" to go to the next word/stimulus.
                        # In blocks without n-back task, participants can only press "space" to go to the next stimulus.
                        # print("start tracking key responses")
        
                        ### start recording responses
                        # start "endless" while loop that looks for responses
                        # in each iteration, draw word on screen
                        continue_trial = True
                        while continue_trial:
        
                            # draw stimulus on screen
                            stim.draw()
                            win.flip()
        
                            # check for responses:
                            keys = event.getKeys(['space', 'c', 'escape'])
        
                            # check if there was a response. If there wasn't, we can go straight
                            # to the next iteration which will hopefully save us some dropped
                            # frames in the flicker.
                            for key in keys:
        
                                # if participant pressed the space bar on their keyboard...
                                if key == 'space':
                                    # get reaction time
                                    curr_duration = my_trial_clock.getTime() * 1000
                                    # send trigger for response:
                                    # send_trigger("response_continue")
        
                                    # break while loop to go to next trial
                                    continue_trial = False
        
                                # if participant pressed button "c" for the first time and it's an n-back condition
                                # where they're actually supposed to do that (aka not a reading baseline condition)...
                                elif key == 'c' and curr_nback_cond != None and saw_target == False:
                                    # get reaction time
                                    curr_nback_RT = my_trial_clock.getTime() * 1000
        
                                    # send trigger for response:
                                    # send_trigger("response_target")
        
                                    # only get first target response, we don't care if they press the button more than once:
                                    saw_target = True
        
                                # If esc is pressed, end the experiment:
                                elif key == 'escape':
                                    # et_abort_exp()  # shut down eyetrigger and download incremental data
                                    # make sure parallel port line is cleared
                                    # core.wait(time_after_trigger)
                                    # parallel.setData(0)
                                    core.wait(0.5)
                                    # end experiment
                                    core.quit()
        
                        ### end trial
                        # print("end trial")
        
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
                        thisExp.addData('nback_RT', curr_nback_RT)  # in ms
                        thisExp.addData('duration', curr_duration)  # in ms
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
        
                    print("\t\tfinished presenting trials")
        
                    # change background colour from grey (RGB: 10, 10, 10)
                    # to ivory (RGB: 240, 223, 204)
                    win.setColor(light_bg_col, colorSpace='rgb')
                    win.flip()
        
                # add 1 to the block counter to go load the next block
                exp_block_counter = exp_block_counter + 1
                print(f"\tGoing to block {exp_block_counter + 1}/15 now!")
        
        # go to next routine
        # print("going to next routine")
        continueRoutine = False
        # keep track of which components have finished
        no_text_trainingComponents = []
        for thisComponent in no_text_trainingComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "no_text_training" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in no_text_trainingComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "no_text_training" ---
        for thisComponent in no_text_trainingComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('no_text_training.stopped', globalClock.getTime())
        # the Routine "no_text_training" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "text_blocks_self_paced" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('text_blocks_self_paced.started', globalClock.getTime())
        # Run 'Begin Routine' code from text_blocks
        #################################################
        #           Blocks with text – self-paced       #
        #################################################
        # this routine is for all blocks with texts that are self-paced
        
        # keep background ivory
        win.setColor(light_bg_col, colorSpace='rgb')
        win.flip()
        
        # clear buffer of all previously recorded key events:
        event.clearEvents()
        
        ### specify settings for the current block
        
        ### Prepare stimuli:
        
        # get block kind
        curr_block = all_blocks[exp_block_counter]
        # print("start preparing block " + curr_block)
        
        # Check whether it's a block that isn't self-paced
        # If yes, skip this routine
        if curr_block in ["click_training", "1back_single_training1", "1back_single_training2",
                          "2back_single_training1", "2back_single_training2",
                          "Reading_Baseline_main_no_click", "1back_dual_main_no_click", "2back_dual_main_no_click",
                          "Reading_Baseline_training_no_click"]:
            print(f"this is block {curr_block}")
            print("\tskipping self-paced text routine")
            # skip questions & end current routine
            skip_questions = True
            continueRoutine = False
            # break
        
        # if it's the reading bl training block, prepare training stimuli:
        elif curr_block == "Reading_Baseline_training_click":
            print(f"this is block {curr_block}")
            print(f"start preparing block {curr_block}")
        
            # keep background ivory
            win.setColor(light_bg_col, colorSpace='rgb')
            win.flip()
        
            ### Show instructions
            # set instruction text
            instr_text = locals()["instr_" + curr_block]
            # create text box
            instr_text_stim = visual.TextStim(win,
                                              text=instr_text,
                                              height=0.025,  # font height relative to height of screen
                                              pos=(0, 0.2),  # move up a bit
                                              color="black")
            # create ImageStim object
            curr_instr_pic = visual.ImageStim(win,
                                              size=(0.6, 0.3),
                                              pos=(0, -0.2),
                                              image=locals()["instr_pic_" + curr_block])  # set path to image here
        
            # show instructions on screen
            win.setColor(light_bg_col, colorSpace='rgb')
            instr_text_stim.draw()
            curr_instr_pic.draw()
            win.flip()
            core.wait(3)  # wait for 3s before starting response window
        
            # display the text on screen
            while True:
                # keep background ivory
                win.setColor(light_bg_col, colorSpace='rgb')
                instr_text_stim.draw()
                curr_instr_pic.draw()
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
        
            # we also need the start time (let's set it as current time
            # at this point in the script):
            start_time = core.getTime()
        
            ### change background colour
            win.setColor(dark_bg_col, colorSpace='rgb')
            win.flip()
        
            print(f"\tcurrent text: {curr_text_nr}")
        
        # if it's one of the "normal" main blocks, prepare main block stimuli:
        elif curr_block in ["Reading_Baseline_main_click", "1back_dual_main_click", "2back_dual_main_click"]:
            print(f"this is block {curr_block}")
            print(f"start preparing block {curr_block}")
        
            # keep background ivory
            win.setColor(light_bg_col, colorSpace='rgb')
            win.flip()
        
            ### Show instructions
            # set instruction text
            instr_text = locals()["instr_" + curr_block]
            # create text box
            instr_text_stim = visual.TextStim(win,
                                              text=instr_text,
                                              height=0.025,  # font height relative to height of screen
                                              pos=(0, 0.2),  # move up a bit
                                              color="black")
            if curr_block == "Reading_Baseline_main_click":
                # create ImageStim object
                curr_instr_pic = visual.ImageStim(win,
                                                  size=(0.6, 0.3),
                                                  pos=(0, -0.2),
                                                  image=locals()["instr_pic_" + curr_block])  # set path to image here
            else:
                # create ImageStim object
                curr_instr_pic = visual.ImageStim(win,
                                                  size=(0.7, 0.3),
                                                  pos=(0, -0.2),
                                                  image=locals()["instr_pic_" + curr_block])  # set path to image here
        
            # show instructions
            win.setColor(light_bg_col, colorSpace='rgb')
            instr_text_stim.draw()
            curr_instr_pic.draw()
            win.flip()
            core.wait(3)  # wait for 3s before starting response window
        
            # Display the text on screen
            while True:
                instr_text_stim.draw()
                curr_instr_pic.draw()
                win.flip()
                # end showing screen if participant presses space
                if 'space' in event.getKeys():
                    break
        
            ### change background colour
            win.setColor(dark_bg_col, colorSpace='rgb')
            win.flip()
        
            # show main block questions
            skip_questions = False
            training_Qs = False
        
            # get text nr:
            curr_text_nr = all_texts_nrs_list[exp_block_counter]
        
            # get n-back condition:
            curr_nback_cond = curr_block[0]  # get first character of block name
        
            # if it is a 1 or a 2, set that as current n-back level:
            if curr_nback_cond in ['1', '2']:
                curr_nback_cond == int(curr_nback_cond)
            # if it's neither 1 nor 2, it has to be a block without n-back,
            # so set curr_nback_cond to None
            else:
                curr_nback_cond = None
        
            print(f"\tcurrent n-back condition: {curr_nback_cond}")
            print(f"\tcurrent text: {curr_text_nr}")
        
            # get list with targets & list with colours
            curr_targets = all_target_lists[exp_block_counter]
            curr_colours = all_colour_lists[exp_block_counter]
            # for current text nr, get text whose name = current text nr
            curr_text = locals()[curr_text_nr]
        
        ### Start block loop
        if curr_block in ["Reading_Baseline_training_click", "Reading_Baseline_main_click", "1back_dual_main_click", "2back_dual_main_click"]:
            # depending on condition, create arrays for saving response
            # times & words - we need that later for the paced reading tasks
            if curr_block == "Reading_Baseline_main_click":
                BL_paced_durations = []
                BL_paced_words = []
            elif curr_block == "1back_dual_main_click":
                oneback_paced_durations = []
                oneback_paced_words = []
            elif curr_block == "2back_dual_main_click":
                twoback_paced_durations = []
                twoback_paced_words = []
        
            # create empty text stimulus
            stim = visual.TextStim(win=win,
                                   text=" ",
                                   pos=(0, 0),  # center stimulus
                                   font="Times New Roman",
                                   height=0.07)
        
            stim.draw()
            win.flip()
        
            # clear buffer of all previously recorded key events:
            event.clearEvents()
        
            # CREATE CLOCKS:
            my_block_clock = core.Clock()
            my_block_clock.reset()  # start block clock
            start_time = my_block_clock.getTime()  # get start time of block
            # also create trial clock
            my_trial_clock = core.Clock()
        
            # loop words in current text
            for trial_idx, curr_word in enumerate(curr_text):
                # print("current idx: " + str(trial_idx) + ", curr word:" + curr_word)
        
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
        
                # show word on screen
                stim.draw()  # draw word on screen
        
                # start trial clock
                my_trial_clock.reset()
                onset_time = my_trial_clock.getTime()
        
                ### wait for 50 ms
                while my_trial_clock.getTime() < onset_time + 0.05:
        
                    # draw the stimulus during the waiting period
                    stim.draw()  # draw text
                    win.flip()
        
                ### wait for key response:
                # In blocks with n-back task, participants can press "c" to indicate they saw a target colour and "space" to go to the next word/stimulus.
                # In blocks without n-back task, participants can only press "space" to go to the next word/stimulus.
                # print("start tracking key responses")
        
                ### start recording responses
                # start "endless" while loop that looks for responses
                continue_trial = True
                while continue_trial:
        
                    # in each iteration, draw word on screen
                    # --> flicker again
        
                    stim.draw()
                    win.flip()
        
                    # check for key responses:
                    keys = event.getKeys(['space', 'c', 'escape'])
        
                    # if we recorded a response, check which one.
                    # If not, we go  to the next "while" iteration,
                    # so I hope this saves us a few dropped frames in the flicker.
                    for key in keys:
        
                        # if participant pressed space bar on their keyboard...
                        if key == 'space':
                            # get reaction time
                            curr_duration = my_trial_clock.getTime() * 1000
                            # send trigger for response:
                            # send_trigger("response_continue")
                            # break while loop
                            continue_trial = False
        
                        # if participant pressed button "c" for the first time and it's an n-back condition
                        # where they're actually supposed to do that (aka not a reading baseline condition)...
                        elif key == 'c' and curr_nback_cond != None and saw_target == False:
                            # get reaction time
                            curr_nback_RT = my_trial_clock.getTime() * 1000
                            # send trigger for response:
                            # send_trigger("response_target")
                            # only get first target response, we don't care if they press the button more than once:
                            saw_target = True
        
                        # If esc is pressed, end the experiment:
                        elif key == 'escape':
                            # et_abort_exp()  # shut down eyetrigger and download incremental data
                            # close trigger & close experiment
                            # core.wait(time_after_trigger)
                            # parallel.setData(0)
                            core.wait(0.5)
                            core.quit()
        
                ### end trial
                print("\tend self-paced trial")
                # stop display of current word & send trial offset trigger
                # win.callOnFlip(send_trigger, "trial_offset")
        
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
                thisExp.addData('nback_RT', curr_nback_RT)  # in ms
                thisExp.addData('duration', curr_duration)  # in ms
                thisExp.addData('text_nr', curr_text_nr)
                thisExp.addData('trial_nr', curr_trial_nr)
                thisExp.addData('block_nr', exp_block_counter)
                thisExp.addData('block_name', curr_block)
                thisExp.addData('block_kind', curr_nback_cond)
                # careful, make sure quotes in the strings are escaped using a
                # quote (weird, I know) so it's properly saved in the CSV:
                thisExp.addData('word', escape_quotes(curr_word))
        
                # start a new row in the csv
                thisExp.nextEntry()
        
                # depending on condition, save response times and words in previously created arrays
                # we need that later for the paced reading tasks
                if curr_block == "Reading_Baseline_main_click":
                    BL_paced_durations.append(curr_duration)
                    BL_paced_words.append(curr_word)
                elif curr_block == "1back_dual_main_click":
                    oneback_paced_durations.append(curr_duration)
                    oneback_paced_words.append(curr_word)
                elif curr_block == "2back_dual_main_click":
                    twoback_paced_durations.append(curr_duration)
                    twoback_paced_words.append(curr_word)
        
                ### IF TESTING MODE ENABLED: end loop after 4 trials
                if expInfo['testing_mode'] == "yes":
                    if trial_idx == 3:
                        break
        
            print("finished presenting trials")
        
            # Send end of block trigger:
            # core.wait(time_after_trigger)  # wait 3 ms
            # send block offset trigger
            # send_trigger("block_offset")
        
            # end current routine
            continueRoutine = False
        # Run 'Begin Routine' code from Q1
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
        
        if not skip_questions:
            # Define text positions and formatting
            question_pos = (0, 0.2)
            answer_xpos = -0.75 # move questions a bit to the left 
            answer_ypos = [0.1, 0.05, 0, -0.05] # set the y axis positions of all 4 answers
        
            # Create text stim for the question:
            question = visual.TextStim(win, 
                                       text = Q1, 
                                       pos = question_pos,
                                       color = "black",
                                       height = 0.025,  # font height relative to height of screen
                                       anchorHoriz = 'center',
                                       alignText = 'center', 
                                       wrapWidth = 1)
            # create 1 text stim for each answer option:
            answers = [visual.TextStim(win, 
                                       text = Q1_answers[i], 
                                       pos = (answer_xpos, answer_ypos[i]), 
                                       color = "black", # set all to black as a default
                                       height=0.025,  # font height relative to height of screen
                                       wrapWidth = 1.5,
                                       anchorHoriz = 'left', 
                                       alignText = 'center') for i in range(len(Q1_answers))]
            # set up instruction text
            instr_text = visual.TextStim(win, 
                                         text = "(Bitte benutzen Sie die Tasten 1, 2, 3 und 4, um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                         color = "grey",
                                         pos = (0, -0.3),
                                         wrapWidth = 2,
                                         height=0.018)  # font height relative to height of screen
        
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
                    # print('\ta')
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
                    # print('\tb')
                    Q1_chosen_ans = "b"
                    # set font colour of the second answer (answer b) to 
                    # green and the rest to black:
                    answers[1].setColor("green")
                    for answer in [answers[0]] + answers[2:]:
                        answer.setColor("black")
                        # draw updated stimulus:
                        win.flip()
                if event.getKeys(['3']):
                    # print('\tc')
                    Q1_chosen_ans = "c"
                    # set font colour of the third answer (answer c) to 
                    # green and the rest to black:
                    answers[2].setColor("green")
                    for answer in answers[:2] + answers[3:]:
                        answer.setColor("black")
                    # draw updated stimulus:
                    win.flip()
                if event.getKeys(['4']):
                    # print('\td')
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
            print("answer for Q1 self-paced:" + str(Q1_chosen_ans))
        
            # check if answer was correct:
            if Q1_chosen_ans == Q1_corr: 
                print("\tanswer correct!")
            else: 
                print("\tanswer incorrect!")
                
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
            # continueRoutine = False
        # Run 'Begin Routine' code from Q2
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
        
        if not skip_questions:
                # Define text positions and formatting
            question_pos = (0, 0.2)
            answer_xpos = -0.75 # move questions a bit to the left 
            answer_ypos = [ 0.1, 0.05, 0, -0.05] # set the y axis positions of all 4 answers
        
            # Create text stim for the question:
            question = visual.TextStim(win, 
                                       text = Q2, 
                                       pos = question_pos,
                                       color = "black",
                                       height = 0.025,  # font height relative to height of screen
                                       anchorHoriz = 'center',
                                       alignText = 'center', 
                                       wrapWidth = 1)
            # create 1 text stim for each answer option:
            answers = [visual.TextStim(win, 
                                       text = Q2_answers[i], 
                                       pos = (answer_xpos, answer_ypos[i]), 
                                       color = "black", # set all to black as a default
                                       height=0.025,  # font height relative to height of screen
                                       wrapWidth = 1.5,
                                       anchorHoriz = 'left', 
                                       alignText = 'center') for i in range(len(Q1_answers))]
            # set up instruction text
            instr_text = visual.TextStim(win, 
                                         text = "(Bitte benutzen Sie die Tasten 1, 2, 3 und 4, um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                         color = "grey",
                                         pos = (0, -0.3),
                                         wrapWidth = 2,
                                         height=0.018)  # font height relative to height of screen
                                         
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
                    # print('\ta')
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
                    # print('\tb')
                    Q2_chosen_ans = "b"
                    # set font colour of the second answer (answer b) to 
                    # green and the rest to black:
                    answers[1].setColor("green")
                    for answer in [answers[0]] + answers[2:]:
                        answer.setColor("black")
                        # draw updated stimulus:
                        win.flip()
                if event.getKeys(['3']):
                    # print('\tc')
                    Q2_chosen_ans = "c"
                    # set font colour of the third answer (answer c) to 
                    # green and the rest to black:
                    answers[2].setColor("green")
                    for answer in answers[:2] + answers[3:]:
                        answer.setColor("black")
                    # draw updated stimulus:
                    win.flip()
                if event.getKeys(['4']):
                    # print('\td')
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
            print("answer for Q2 self-paced:" + str(Q2_chosen_ans))
        
            # check if answer was correct:
            if Q2_chosen_ans == Q2_corr: 
                print("\tanswer correct!")
            else: 
                print("\tanswer incorrect!")
                
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
            # continueRoutine = False
        # Run 'Begin Routine' code from Q3
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
        
        if not skip_questions:
                # Define text positions and formatting
            question_pos = (0, 0.2)
            answer_xpos = -0.75 # move questions a bit to the left 
            answer_ypos = [ 0.1, 0.05, 0, -0.05] # set the y axis positions of all 4 answers
        
            # Create text stim for the question:
            question = visual.TextStim(win, 
                                       text = Q3, 
                                       pos = question_pos,
                                       color = "black",
                                       height = 0.025,  # font height relative to height of screen
                                       anchorHoriz = 'center',
                                       alignText = 'center', 
                                       wrapWidth = 1)
            # create 1 text stim for each answer option:
            answers = [visual.TextStim(win, 
                                       text = Q3_answers[i], 
                                       pos = (answer_xpos, answer_ypos[i]), 
                                       color = "black", # set all to black as a default
                                       height=0.025,  # font height relative to height of screen
                                       wrapWidth = 1.5,
                                       anchorHoriz = 'left', 
                                       alignText = 'center') for i in range(len(Q1_answers))]
            # set up instruction text
            instr_text = visual.TextStim(win, 
                                         text = "(Bitte benutzen Sie die Tasten 1, 2, 3 und 4, um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                         color = "grey",
                                         pos = (0, -0.3),
                                         wrapWidth = 2,
                                         height=0.018)  # font height relative to height of screen
                                         
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
                    # print('\ta')
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
                    # print('\tb')
                    Q3_chosen_ans = "b"
                    # set font colour of the second answer (answer b) to 
                    # green and the rest to black:
                    answers[1].setColor("green")
                    for answer in [answers[0]] + answers[2:]:
                        answer.setColor("black")
                        # draw updated stimulus:
                        win.flip()
                if event.getKeys(['3']):
                    # print('\tc')
                    Q3_chosen_ans = "c"
                    # set font colour of the third answer (answer c) to 
                    # green and the rest to black:
                    answers[2].setColor("green")
                    for answer in answers[:2] + answers[3:]:
                        answer.setColor("black")
                    # draw updated stimulus:
                    win.flip()
                if event.getKeys(['4']):
                    # print('\td')
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
            print("answer for Q3 self-paced:" + str(Q3_chosen_ans))
        
            # check if answer was correct:
            if Q3_chosen_ans == Q3_corr: 
                print("\tanswer correct!")
            else: 
                print("\tanswer incorrect!")
                
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
        
            # go to next block!
            exp_block_counter += 1
            print(f"Going to block {exp_block_counter + 1}/15 now!")
            continueRoutine = False
        
            # If there are still blocks left, go to next one.
            # If not, end loop here:
            if exp_block_counter == 15:
                blocks.finished = True
        
            # when text rating is included, only use this instead of "go to next block":
            # end current routine
            #continueRoutine = False
        # keep track of which components have finished
        text_blocks_self_pacedComponents = []
        for thisComponent in text_blocks_self_pacedComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "text_blocks_self_paced" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in text_blocks_self_pacedComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "text_blocks_self_paced" ---
        for thisComponent in text_blocks_self_pacedComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('text_blocks_self_paced.stopped', globalClock.getTime())
        # the Routine "text_blocks_self_paced" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "text_blocks_paced" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('text_blocks_paced.started', globalClock.getTime())
        # Run 'Begin Routine' code from paced_blocks
        #################################################
        #            Blocks with text – paced           #
        #################################################
        # this routine is for all blocks with texts that are paced, i.e., visually presented without space bar
        
        #---------- Calculate duration of words based on previous block ----------
        # We collected RTs & words from the self-paced block of each condition
        # for the training, we only use data from the reading BL since there is no separate training for 1- and 2-back
        
        # we calculate letter duration based on condition since participants need more time for n-back tasks than for baseline reading
        # BL reading blocks are based on duration during self-paced BL reading
        # 1- and 2-back blocks are based on their respective self-paced version
        
        # get block kind
        curr_block = all_blocks[exp_block_counter]
        
        if curr_block in ["Reading_Baseline_main_no_click", "Reading_Baseline_training_no_click"]:
        
            # exclude all RTs where participant was way too fast (< 50 ms) or
            # way too slow (> 2s), also remove the corresponding words from vis_task_words
            print("\tBL_paced_durations:", BL_paced_durations)
            print("\tBL_paced_words:", BL_paced_words)
        
            filtered_durations_BL = []
            filtered_words_BL = []
            for duration, word in zip(BL_paced_durations, BL_paced_words):
                if 50 <= duration <= 2000:
                    filtered_durations_BL.append(duration)
                    filtered_words_BL.append(word)
            print("\tfiltered_durations_BL:", filtered_durations_BL)
            print("\tfiltered_words_BL:", filtered_words_BL)
        
            # Now get number of letters (not words, I want to know how fast they read 1 letter on average!):
            letters_total_BL = sum(len(word) for word in filtered_words_BL)
            print("\tletters_total_BL:", letters_total_BL)
            # also get time it took in total to read them all:
            reading_time_total_BL = sum(filtered_durations_BL)  # in ms
        
            # Now check how many words / min they read on average.
            # reading_speed_wpm = words_total / (reading_time_total/60000)
            # print("reading speed in words / min:" + str(reading_speed_wpm))
        
            # Check average RT / letter
            RT_per_letter_baseline = reading_time_total_BL / letters_total_BL
            print("\taverage RT per letter in ms:", RT_per_letter_baseline)
        
            # save this in the output csv:
            thisExp.addData('RT_per_letter_baseline', RT_per_letter_baseline)
        
        elif curr_block in ["1back_dual_main_no_click"]:
        
            # exclude all RTs where participant was way too fast (< 50 ms) or
            # way too slow (> 2s), also remove the corresponding words from vis_task_words
            print("\t1back_paced_durations:", oneback_paced_durations)
            print("\t1back_paced_words:", oneback_paced_words)
        
            filtered_durations_1bck = []
            filtered_words_1bck = []
            for duration, word in zip(oneback_paced_durations, oneback_paced_words):
                if 50 <= duration <= 2000:
                    filtered_durations_1bck.append(duration)
                    filtered_words_1bck.append(word)
            print("\tfiltered_durations_1bck:", filtered_durations_1bck)
            print("\tfiltered_words_1bck:", filtered_words_1bck)
        
            # Now get number of letters (not words, I want to know how fast they read 1 letter on average!):
            letters_total_1bck = sum(len(word) for word in filtered_words_1bck)
            print("\tletters_total_1bck:", letters_total_1bck)
            # also get time it took in total to read them all:
            reading_time_total_1bck = sum(filtered_durations_1bck)  # in ms
        
            # Now check how many words / min they read on average.
            # reading_speed_wpm = words_total / (reading_time_total/60000)
            # print("reading speed in words / min:" + str(reading_speed_wpm))
        
            # Check average RT / letter
            RT_per_letter_1bck = reading_time_total_1bck / letters_total_1bck
            print("\taverage RT per letter in ms:", RT_per_letter_1bck)
        
            # save this in the output csv:
            thisExp.addData('RT_per_letter_1bck', RT_per_letter_1bck)
        
        elif curr_block in ["2back_dual_main_no_click"]:
        
            # exclude all RTs where participant was way too fast (< 50 ms) or
            # way too slow (> 2s), also remove the corresponding words from vis_task_words
            print("\t2back_paced_durations:", twoback_paced_durations)
            print("\t2back_paced_words:", twoback_paced_words)
        
            filtered_durations_2bck = []
            filtered_words_2bck = []
            for duration, word in zip(twoback_paced_durations, twoback_paced_words):
                if 50 <= duration <= 2000:
                    filtered_durations_2bck.append(duration)
                    filtered_words_2bck.append(word)
            print("\tfiltered_durations_2bck:", filtered_durations_2bck)
            print("\tfiltered_words_2bck:", filtered_words_2bck)
        
            # Now get number of letters (not words, I want to know how fast they read 1 letter on average!):
            letters_total_2bck = sum(len(word) for word in filtered_words_2bck)
            print("\tletters_total_2bck:", letters_total_2bck)
            # also get time it took in total to read them all:
            reading_time_total_2bck = sum(filtered_durations_2bck)  # in ms
        
            # Now check how many words / min they read on average.
            # reading_speed_wpm = words_total / (reading_time_total/60000)
            # print("reading speed in words / min:" + str(reading_speed_wpm))
        
            # Check average RT / letter
            RT_per_letter_2bck = reading_time_total_2bck / letters_total_2bck
            print("\taverage RT per letter in ms:", RT_per_letter_2bck)
        
            # save this in the output csv:
            thisExp.addData('RT_per_letter_2bck', RT_per_letter_2bck)
        
        # keep background ivory
        win.setColor(light_bg_col, colorSpace='rgb')
        win.flip()
        
        # ----------------------------------
        
        # clear buffer of all previously recorded key events:
        event.clearEvents()
        
        ### specify settings for the current block
        
        ### Prepare stimuli:
        
        # get block kind
        curr_block = all_blocks[exp_block_counter]
        # print("start preparing block " + curr_block)
        
        # Check whether it's a block that is self-paced
        # Also, if current block is a non-text block, skip this routine.
        if curr_block in ["click_training", "1back_single_training1", "1back_single_training2",
                          "2back_single_training1", "2back_single_training2",
                          "Reading_Baseline_main_click", "1back_dual_main_click", "2back_dual_main_click", "Reading_Baseline_training_click"]:
            print(f"this is block {curr_block}")
            print("\tskipping paced routine")
            # skip questions & end current routine
            skip_questions_paced = True
            continueRoutine = False
            # break
        
        # if it's the paced reading training block, prepare training stimuli:
        elif curr_block == "Reading_Baseline_training_no_click":
            print(f"start preparing block {curr_block}")
        
            # keep background ivory
            win.setColor(light_bg_col, colorSpace='rgb')
            win.flip()
        
            ### Show instructions
            # set instruction text
            instr_text = locals()["instr_" + curr_block]
            # create text box
            instr_text_stim = visual.TextStim(win,
                                              text=instr_text,
                                              height=0.025,  # font height relative to height of screen
                                              pos=(0, 0),  # move up a bit
                                              color="black")
        
            # show instructions on screen
            win.setColor(light_bg_col, colorSpace='rgb')
            instr_text_stim.draw()
            win.flip()
            core.wait(3)  # wait for 3s before starting response window
        
            # display the text on screen
            while True:
                # keep background ivory
                win.setColor(light_bg_col, colorSpace='rgb')
                instr_text_stim.draw()
                win.flip()
                # end showing screen if participant presses space
                if 'space' in event.getKeys():
                    break
        
            # clear buffer of all previously recorded key events:
            event.clearEvents()
        
            ### get training text
            curr_text_training = reading_bl_tr_text_no_click
            curr_text_nr = "Reading_Baseline_training_no_click"
            curr_text = curr_text_training
            curr_nback_cond = None
            # show training questions
            skip_questions_paced = False
            training_Qs_paced = True
        
            # get list with targets & list with colours
            curr_targets = all_target_lists[exp_block_counter]
            curr_colours = all_colour_lists[exp_block_counter]
        
            # compute RTs using participant's average reading speed / letter
            minimum_duration = 6 * RT_per_letter_baseline
            curr_durations = []
            for word in curr_text:
                duration = RT_per_letter_baseline * math.log((len(word))) + 300
                curr_durations.append(max(duration, minimum_duration))

            # print(f"\tdurations for paced task training block: {curr_durations_training}")
        
            # we also need the start time (let's set it as current time
            # at this point in the script):
            start_time = core.getTime()
        
            ### change background colour
            win.setColor(dark_bg_col, colorSpace='rgb')
            win.flip()
        
        # if it's one of the "normal" main blocks, prepare main block stimuli:
        elif curr_block in ["Reading_Baseline_main_no_click", "1back_dual_main_no_click", "2back_dual_main_no_click"]:
            print(f"start preparing block {curr_block}")
        
            # keep background ivory
            win.setColor(light_bg_col, colorSpace='rgb')
            win.flip()
        
            ### Show instructions
            # only add image, if it's a 1- or 2-back block where participants have to press "c"
            if curr_block == "Reading_Baseline_main_no_click":
        
                # set instruction text
                instr_text = locals()["instr_" + curr_block]
                # create text box
                instr_text_stim = visual.TextStim(win,
                                                  text=instr_text,
                                                  height=0.025,  # font height relative to height of screen
                                                  pos=(0, 0),  # move up a bit
                                                  color="black")
        
                # show instructions on screen
                win.setColor(light_bg_col, colorSpace='rgb')
                instr_text_stim.draw()
                win.flip()
                core.wait(3)  # wait for 3s before starting response window
        
                # Display the text on screen
                while True:
                    instr_text_stim.draw()
                    win.flip()
                    # end showing screen if participant presses space
                    if 'space' in event.getKeys():
                        break
        
                # get text nr:
                curr_text_nr = all_texts_nrs_list[exp_block_counter]
                curr_text = locals()[curr_text_nr]
                # compute RTs using participant's average reading speed / letter
                minimum_duration = 6 * RT_per_letter_baseline
                curr_durations = []
                for word in curr_text:
                    duration = RT_per_letter_baseline * math.log((len(word))) + 300
                    curr_durations.append(max(duration, minimum_duration))

                ### change background colour
                win.setColor(dark_bg_col, colorSpace='rgb')
                win.flip()
        
            elif curr_block in ["1back_dual_main_no_click", "2back_dual_main_no_click"]:
        
                # set instruction text
                instr_text = locals()["instr_" + curr_block]
                # create text box
                instr_text_stim = visual.TextStim(win,
                                                  text=instr_text,
                                                  height=0.025,  # font height relative to height of screen
                                                  pos=(0, 0.2),  # move up a bit
                                                  color="black")
                # create ImageStim object
                curr_instr_pic = visual.ImageStim(win,
                                                  size=(0.7, 0.3),
                                                  pos=(0, -0.2),
                                                  image=locals()["instr_pic_" + curr_block])  # set path to image here
        
                # show instructions on screen
                win.setColor(light_bg_col, colorSpace='rgb')
                instr_text_stim.draw()
                curr_instr_pic.draw()
                win.flip()
                core.wait(3)  # wait for 3s before starting response window
        
                # Display the text on screen
                while True:
                    instr_text_stim.draw()
                    curr_instr_pic.draw()
                    win.flip()
                    # end showing screen if participant presses space
                    if 'space' in event.getKeys():
                        break
        
                # get text nr:
                curr_text_nr = all_texts_nrs_list[exp_block_counter]
                curr_text = locals()[curr_text_nr]
                # compute RTs using participant's average reading speed / letter                
                if curr_block == "1back_dual_main_no_click":
                    minimum_duration = 6 * RT_per_letter_1bck
                    curr_durations = []
                    for word in curr_text:
                        duration = RT_per_letter_1bck * math.log(len(word)) + 490
                        curr_durations.append(max(duration, minimum_duration))
                elif curr_block == "2back_dual_main_no_click":
                    minimum_duration = 6 * RT_per_letter_2bck
                    curr_durations = []
                    for word in curr_text:
                        duration = RT_per_letter_2bck * math.log(len(word)) + 700
                        curr_durations.append(max(duration, minimum_duration))


                ### change background colour
                win.setColor(dark_bg_col, colorSpace='rgb')
                win.flip()
        
            # show main block questions
            skip_questions_paced = False
            training_Qs_paced = False
        
            # get n-back condition:
            curr_nback_cond = curr_block[0]  # get first character of block name
        
            # if it is a 1 or a 2, set that as current n-back level:
            if curr_nback_cond in ['1', '2']:
                curr_nback_cond == int(curr_nback_cond)
            # if it's neither 1 nor 2, it has to be a block without n-back,
            # so set curr_nback_cond to None
            else:
                curr_nback_cond = None
        
            print(f"\tcurrent n-back condition: {curr_nback_cond}")
            print(f"\tcurrent text: {curr_text_nr}")
        
            # get list with targets & list with colours
            curr_targets = all_target_lists[exp_block_counter]
            curr_colours = all_colour_lists[exp_block_counter]
            # for current text nr, get text whose name = current text nr
            # curr_text = locals()[curr_text_nr]
        
        ### Start block loop
        if curr_block in ["Reading_Baseline_training_no_click", "Reading_Baseline_main_no_click", "1back_dual_main_no_click",
                          "2back_dual_main_no_click"]:
        
            # create empty text stimulus
            stim = visual.TextStim(win=win,
                                   text=" ",
                                   pos=(0, 0),  # center stimulus
                                   font="Times New Roman",
                                   height=0.07)
        
            stim.draw()
            win.flip()
        
            # clear buffer of all previously recorded key events:
            event.clearEvents()
        
            # CREATE CLOCKS:
            my_block_clock = core.Clock()
            my_block_clock.reset()  # start block clock
            start_time = my_block_clock.getTime()  # get start time of block
            # also create trial clock
            my_trial_clock = core.Clock()
        
            # loop words in current text
            for trial_idx, curr_word in enumerate(curr_text):
                # print("current idx: " + str(trial_idx) + ", curr word:" + curr_word)
        
                ### prepare & show current word:
        
                # get current colour
                curr_colour = curr_colours[trial_idx]
        
                # if it's a block with an n-back task, prepare target list as well
                if curr_nback_cond != None:
                    curr_target = curr_targets[trial_idx]
                    saw_target = False
        
                # get duration for current word
                curr_duration = curr_durations[trial_idx] / 1000  # convert ms to seconds
                print("duration for current word (in s):", curr_duration)
        
                # get trial number (start counting from 1, so add 1)
                curr_trial_nr = trial_idx + 1
        
                # set current word & colour as content of text stimulus
                stim.color = curr_colour
                stim.text = curr_word
        
                # show word on screen
                stim.draw()  # draw word on screen
        
                # start trial clock & record trial onset time
                my_trial_clock.reset()
                onset_time = my_trial_clock.getTime()
        
                ### wait for key response:
                # In blocks with n-back task, participants can press "c" to indicate they saw a target colour.
        
                ### start recording responses
                # start while loop that looks for responses
                # --> end while loop only if duration for current word is over
                while my_trial_clock.getTime() < (onset_time + curr_duration):
        
                    stim.draw()
                    win.flip()
        
                    # check for key responses:
                    keys = event.getKeys(['c', 'escape'])
        
                    # if there were, check responses:
                    for key in keys:
        
                        # if participant pressed button "c" for the first time and it's an n-back condition
                        # where they're actually supposed to do that (aka not a reading baseline condition)...
                        if key == 'c' and curr_nback_cond != None and saw_target == False:
                            # get reaction time
                            curr_nback_RT = my_trial_clock.getTime() * 1000
                            # send trigger for response:
                            # send_trigger("response_target")
                            # only get first target response, we don't care if they press the button more than once:
                            saw_target = True
        
                        # If esc is pressed, end the experiment:
                        elif key == 'escape':
                            # et_abort_exp()  # shut down eyetrigger and download incremental data
                            # close trigger & close experiment
                            # core.wait(time_after_trigger)
                            # parallel.setData(0)
                            core.wait(0.5)
                            core.quit()
        
                ### end trial
                print("\tend paced trial")
                # stop display of current word & send trial offset trigger
                # win.callOnFlip(send_trigger, "trial_offset")
        
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
                thisExp.addData('nback_RT', curr_nback_RT)  # in ms
                thisExp.addData('duration', curr_duration * 1000)  # in ms
                thisExp.addData('text_nr', curr_text_nr)
                thisExp.addData('trial_nr', curr_trial_nr)
                thisExp.addData('block_nr', exp_block_counter)
                thisExp.addData('block_name', curr_block)
                thisExp.addData('block_kind', curr_nback_cond)
                # careful, make sure quotes in the strings are escaped using a
                # quote (weird, I know) so it's properly saved in the CSV:
                thisExp.addData('word', escape_quotes(curr_word))
        
                # start a new row in the csv
                thisExp.nextEntry()
        
                ### IF TESTING MODE ENABLED: end loop after 4 trials
                if expInfo['testing_mode'] == "yes":
                    if trial_idx == 3:
                        break
        
            print("finished presenting trials")
        
            # Send end of block trigger:
            # core.wait(time_after_trigger)  # wait 3 ms
            # send block offset trigger
            # send_trigger("block_offset")
        
        # end current routine
        # continueRoutine = False
        # Run 'Begin Routine' code from Q1_paced
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
        if skip_questions_paced:
            continueRoutine = False
        # if we have a training text, set training questions
        elif skip_questions_paced == False and training_Qs_paced:
            Q1 = reading_bl_tr_no_click_Q1
            Q1_answers = reading_bl_tr_no_click_Q1_ans
            Q1_corr = reading_bl_tr_no_click_Q1_corr
            
        # if we have a main text, set regular questions
        elif skip_questions_paced == False and training_Qs_paced == False:
            # load first question for current text & their respective answers
            Q1 = locals()[curr_text_nr + "_Q1"]
            Q1_answers = locals()[curr_text_nr + "_Q1_ans"]
            Q1_corr = locals()[curr_text_nr + "_Q1_corr"]
        
        if not skip_questions_paced:
            # Define text positions and formatting
            question_pos = (0, 0.2)
            answer_xpos = -0.75 # move questions a bit to the left 
            answer_ypos = [ 0.1, 0.05, 0, -0.05] # set the y axis positions of all 4 answers
        
            # Create text stim for the question:
            question = visual.TextStim(win, 
                                       text = Q1, 
                                       pos = question_pos,
                                       color = "black",
                                       height = 0.025,  # font height relative to height of screen
                                       anchorHoriz = 'center',
                                       alignText = 'center', 
                                       wrapWidth = 1)
            # create 1 text stim for each answer option:
            answers = [visual.TextStim(win, 
                                       text = Q1_answers[i], 
                                       pos = (answer_xpos, answer_ypos[i]), 
                                       color = "black", # set all to black as a default
                                       height=0.025,  # font height relative to height of screen
                                       wrapWidth = 1.5,
                                       anchorHoriz = 'left', 
                                       alignText = 'center') for i in range(len(Q1_answers))]
            # set up instruction text
            instr_text = visual.TextStim(win, 
                                         text = "(Bitte benutzen Sie die Tasten 1, 2, 3 und 4, um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                         color = "grey",
                                         pos = (0, -0.3),
                                         wrapWidth = 2,
                                         height=0.018)  # font height relative to height of screen
                                         
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
                    print('\ta')
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
                    print('\tb')
                    Q1_chosen_ans = "b"
                    # set font colour of the second answer (answer b) to 
                    # green and the rest to black:
                    answers[1].setColor("green")
                    for answer in [answers[0]] + answers[2:]:
                        answer.setColor("black")
                        # draw updated stimulus:
                        win.flip()
                if event.getKeys(['3']):
                    print('\tc')
                    Q1_chosen_ans = "c"
                    # set font colour of the third answer (answer c) to 
                    # green and the rest to black:
                    answers[2].setColor("green")
                    for answer in answers[:2] + answers[3:]:
                        answer.setColor("black")
                    # draw updated stimulus:
                    win.flip()
                if event.getKeys(['4']):
                    print('\td')
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
            print("answer for Q1 paced:" + str(Q1_chosen_ans))
        
            # check if answer was correct:
            if Q1_chosen_ans == Q1_corr: 
                print("\tanswer correct!")
            else: 
                print("\tanswer incorrect!")
                
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
            # continueRoutine = False
        # Run 'Begin Routine' code from Q2_paced
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
        if skip_questions_paced:
            continueRoutine = False
        # if we have a training text, set training questions
        elif skip_questions_paced == False and training_Qs_paced:
            Q2 = reading_bl_tr_no_click_Q2
            Q2_answers = reading_bl_tr_no_click_Q2_ans
            Q2_corr = reading_bl_tr_no_click_Q2_corr
            
        # if we have a main text, set regular questions
        elif skip_questions_paced == False and training_Qs_paced == False:
            # load first question for current text & their respective answers
            Q2 = locals()[curr_text_nr + "_Q2"]
            Q2_answers = locals()[curr_text_nr + "_Q2_ans"]
            Q2_corr = locals()[curr_text_nr + "_Q2_corr"]
        
        if not skip_questions_paced:
            # Define text positions and formatting
            question_pos = (0, 0.2)
            answer_xpos = -0.75 # move questions a bit to the left 
            answer_ypos = [ 0.1, 0.05, 0, -0.05] # set the y axis positions of all 4 answers
        
            # Create text stim for the question:
            question = visual.TextStim(win, 
                                       text = Q2, 
                                       pos = question_pos,
                                       color = "black",
                                       height = 0.025,  # font height relative to height of screen
                                       anchorHoriz = 'center',
                                       alignText = 'center', 
                                       wrapWidth = 1)
            # create 1 text stim for each answer option:
            answers = [visual.TextStim(win, 
                                       text = Q2_answers[i], 
                                       pos = (answer_xpos, answer_ypos[i]), 
                                       color = "black", # set all to black as a default
                                       height=0.025,  # font height relative to height of screen
                                       wrapWidth = 1.5,
                                       anchorHoriz = 'left', 
                                       alignText = 'center') for i in range(len(Q1_answers))]
            # set up instruction text
            instr_text = visual.TextStim(win, 
                                         text = "(Bitte benutzen Sie die Tasten 1, 2, 3 und 4, um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                         color = "grey",
                                         pos = (0, -0.3),
                                         wrapWidth = 2,
                                         height = 0.018)  # font height relative to height of screen
                                         
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
                    print('\ta')
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
                    print('\tb')
                    Q2_chosen_ans = "b"
                    # set font colour of the second answer (answer b) to 
                    # green and the rest to black:
                    answers[1].setColor("green")
                    for answer in [answers[0]] + answers[2:]:
                        answer.setColor("black")
                        # draw updated stimulus:
                        win.flip()
                if event.getKeys(['3']):
                    print('\tc')
                    Q2_chosen_ans = "c"
                    # set font colour of the third answer (answer c) to 
                    # green and the rest to black:
                    answers[2].setColor("green")
                    for answer in answers[:2] + answers[3:]:
                        answer.setColor("black")
                    # draw updated stimulus:
                    win.flip()
                if event.getKeys(['4']):
                    print('\td')
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
            print("answer for Q2 paced:" + str(Q2_chosen_ans))
        
            # check if answer was correct:
            if Q2_chosen_ans == Q2_corr: 
                print("\tanswer correct!")
            else: 
                print("\tanswer incorrect!")
                
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
            # continueRoutine = False
        # Run 'Begin Routine' code from Q3_paced
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
        if skip_questions_paced:
            continueRoutine = False
        # if we have a training text, set training questions
        elif skip_questions_paced == False and training_Qs_paced:
            Q3 = reading_bl_tr_no_click_Q3
            Q3_answers = reading_bl_tr_no_click_Q3_ans
            Q3_corr = reading_bl_tr_no_click_Q3_corr
            
        # if we have a main text, set regular questions
        elif skip_questions_paced == False and training_Qs_paced == False:
            # load first question for current text & their respective answers
            Q3 = locals()[curr_text_nr + "_Q3"]
            Q3_answers = locals()[curr_text_nr + "_Q3_ans"]
            Q3_corr = locals()[curr_text_nr + "_Q3_corr"]
        
        if not skip_questions_paced:
            # Define text positions and formatting
            question_pos = (0, 0.2)
            answer_xpos = -0.75 # move questions a bit to the left 
            answer_ypos = [ 0.1, 0.05, 0, -0.05] # set the y axis positions of all 4 answers
        
            # Create text stim for the question:
            question = visual.TextStim(win, 
                                       text = Q3, 
                                       pos = question_pos,
                                       color = "black",
                                       height = 0.025,  # font height relative to height of screen
                                       anchorHoriz = 'center',
                                       alignText = 'center', 
                                       wrapWidth = 1)
            # create 1 text stim for each answer option:
            answers = [visual.TextStim(win, 
                                       text = Q3_answers[i], 
                                       pos = (answer_xpos, answer_ypos[i]), 
                                       color = "black", # set all to black as a default
                                       height = 0.025,  # font height relative to height of screen
                                       wrapWidth = 1.5,
                                       anchorHoriz = 'left', 
                                       alignText = 'center') for i in range(len(Q1_answers))]
            # set up instruction text
            instr_text = visual.TextStim(win, 
                                         text = "(Bitte benutzen Sie die Tasten 1, 2, 3 und 4, um die richtige Antwort auszuwählen. Mit der Leertaste können Sie Ihre Auswahl bestätigen.)",
                                         color = "grey",
                                         pos = (0, -0.3),
                                         wrapWidth = 2,
                                         height = 0.018)  # font height relative to height of screen
                                         
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
                    print('\ta')
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
                    print('\tb')
                    Q3_chosen_ans = "b"
                    # set font colour of the second answer (answer b) to 
                    # green and the rest to black:
                    answers[1].setColor("green")
                    for answer in [answers[0]] + answers[2:]:
                        answer.setColor("black")
                        # draw updated stimulus:
                        win.flip()
                if event.getKeys(['3']):
                    print('\tc')
                    Q3_chosen_ans = "c"
                    # set font colour of the third answer (answer c) to 
                    # green and the rest to black:
                    answers[2].setColor("green")
                    for answer in answers[:2] + answers[3:]:
                        answer.setColor("black")
                    # draw updated stimulus:
                    win.flip()
                if event.getKeys(['4']):
                    print('\td')
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
            print("answer for Q3 paced:" + str(Q3_chosen_ans))
        
            # check if answer was correct:
            if Q3_chosen_ans == Q3_corr: 
                print("\tanswer correct!")
            else: 
                print("\tanswer incorrect!")
                
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
        
            # go to next block!
            exp_block_counter += 1
            print(f"Going to block {exp_block_counter + 1}/15 now!")
            continueRoutine = False
        
            # If there are still blocks left, go to next one.
            # If not, end loop here:
            if exp_block_counter == 15:
                blocks.finished = True
        
        # when text rating is included, only use this instead of "go to next block":
        # end current routine
        #continueRoutine = False
        # keep track of which components have finished
        text_blocks_pacedComponents = []
        for thisComponent in text_blocks_pacedComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "text_blocks_paced" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in text_blocks_pacedComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "text_blocks_paced" ---
        for thisComponent in text_blocks_pacedComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('text_blocks_paced.stopped', globalClock.getTime())
        # the Routine "text_blocks_paced" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 30.0 repeats of 'blocks'
    
    
    # --- Prepare to start Routine "end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end.started', globalClock.getTime())
    # Run 'Begin Routine' code from end
    ### END OF EXPERIMENT:
    # keep background ivory
    win.setColor(light_bg_col, colorSpace='rgb')
    win.flip()
    
    ### Show message
    # set text
    instr_text = "Hervorragend!\n\n\nVielen Dank, das Experiment ist nun zu Ende!" 
    
    # create text box
    instr_text_stim = visual.TextStim(win, 
                                      text = instr_text, 
                                      height = 0.09, 
                                      pos = (0, 0),
                                      color = "black")
    
    # display the text on screen until Space is pressed
    while True:
        # keep background ivory
        win.setColor(light_bg_col, colorSpace='rgb')
        instr_text_stim.draw()
        win.flip()
        # end screen if participant presses space
        if 'space' in event.getKeys():        
            print("ending experiment now!")
            # end experiment
            break 
    # keep track of which components have finished
    endComponents = []
    for thisComponent in endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end.stopped', globalClock.getTime())
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
