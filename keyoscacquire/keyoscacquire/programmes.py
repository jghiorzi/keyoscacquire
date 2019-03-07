#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
One programme for taking a single trace and saving it.
Two programmes for taking multiple traces, see descriptions for each function.
The run_programme function is a wrapper function for scripts and installed functions
calling the programmes with optional command line arguments.

Andreas Svela // 2019
"""


import sys
import keyoscacquire.oscacq as acq
import numpy as np

from keyoscacquire.default_options import VISA_ADDRESS, WAVEFORM_FORMAT, CH_NUMS, ACQ_TYPE, NUM_AVG, FILENAME, FILETYPE, FILE_DELIMITER, TIMEOUT # local file with default options

def get_single_trace(fname=FILENAME, ext=FILETYPE, instrument=VISA_ADDRESS, timeout=TIMEOUT, wav_format=WAVEFORM_FORMAT,
                     channel_nums=CH_NUMS, source_type='CHANnel', acq_type=ACQ_TYPE,
                     num_averages=NUM_AVG, p_mode='RAW', num_points=0):
    """This programme captures and stores a trace."""
    acq.connect_getTrace_save(fname=fname, ext=ext, instrument=instrument, timeout=timeout, wav_format=wav_format,
                          channel_nums=channel_nums, source_type=source_type, acq_type=acq_type,
                          num_averages=num_averages, p_mode=p_mode, num_points=num_points)
    print("Done")


def getTraces_connect_each_time_loop(fname=FILENAME, ext=FILETYPE, instrument=VISA_ADDRESS, timeout=TIMEOUT, wav_format=WAVEFORM_FORMAT,
                                     channel_nums=CH_NUMS, source_type='CHANnel', acq_type=ACQ_TYPE,
                                     num_averages=NUM_AVG, p_mode='RAW', num_points=0, start_num=0, file_delim=FILE_DELIMITER):
    """This program consists of a loop in which the program connects to the oscilloscope,
    a trace from the active channels are captured and stored for each loop. This permits
    the active channels to be changing thoughout the measurements, but has larger
    overhead due to establishing and closing a new connection every time.

    The loop runs each time 'enter' is hit. Alternatively one can input n-1 characters before hitting
    'enter' to capture n traces back to back. To quit press 'q'+'enter'."""
    n = start_num
    fnum = file_delim+str(n)
    fname = acq.check_file(fname, ext, num=fnum) # check that file does not exist from before, append to name if it does
    print("Running a loop where at every 'enter' oscilloscope traces will be saved as %s<n>%s," % (fname, ext))
    print("where <n> increases by one for each captured trace. Press 'q'+'enter' to quit the programme.")
    while sys.stdin.read(1) != 'q': # breaks the loop if q+enter is given as input. For any other character (incl. enter)
        fnum = file_delim+str(n)
        x, y, id, channels = acq.connect_and_getTrace(channel_nums, source_type, instrument, timeout, wav_format, acq_type, num_averages, p_mode, num_points)
        acq.plotTrace(x, y, channels, fname=fname+fnum)
        channelstring = ", ".join([channel for channel in channels]) # make string of sources
        acq.saveTrace(fname+fnum, x, y, fileheader=id+"time,"+channelstring+"\n", ext=ext)
        n += 1
    print("Quit")

def getTraces_single_connection_loop(fname=FILENAME, ext=FILETYPE, instrument=VISA_ADDRESS, timeout=TIMEOUT, wav_format=WAVEFORM_FORMAT,
                                     channel_nums=CH_NUMS, source_type='CHANnel', acq_type=ACQ_TYPE,
                                     num_averages=NUM_AVG, p_mode='RAW', num_points=0, start_num=0, file_delim=FILE_DELIMITER):
    """This program connects to the oscilloscope, sets options for the acquisition and then
    enters a loop in which the program captures and stores traces each time 'enter' is pressed.
    Alternatively one can input n-1 characters before hitting 'enter' to capture n traces
    back to back. To quit press 'q'+'enter'. This programme minimises overhead for each measurement,
    permitting measurements to be taken with quicker succession than if connecting each time
    a trace is captured. The downside is that which channels are being captured cannot be
    changing thoughout the measurements."""
    ## Initialise
    inst, id = acq.initialise(instrument, timeout, wav_format, acq_type, num_averages, p_mode, num_points)

    ## Select sources
    sourcesstring, sources, channel_nums = acq.build_sourcesstring(inst, source_type=source_type, channel_nums=channel_nums)

    n = start_num
    fnum = file_delim+str(n)
    fname = acq.check_file(fname, ext, num=fnum) # check that file does not exist from before, append to name if it does
    print("Running a loop where at every 'enter' oscilloscope traces will be saved as %s<n>%s," % (fname, ext))
    print("where <n> increases by one for each captured trace. Press 'q'+'enter' to quit the programme.")
    print("Acquire from sources", sourcesstring)
    while sys.stdin.read(1) != 'q': # breaks the loop if q+enter is given as input. For any other character (incl. enter)
        fnum = file_delim+str(n)
        x, y = acq.getTrace(inst, sources, sourcesstring, wav_format)
        acq.plotTrace(x, y, channel_nums, fname=fname+fnum)                    # plot trace and save png
        acq.saveTrace(fname+fnum, x, y, fileheader=id+"time,"+sourcesstring+"\n", ext=ext) # save trace to ext file
        n += 1

    print("Quit")
    # Set the oscilloscope running before closing the connection
    inst.write(':RUN')
    inst.close()

def run_programme(name, args):
    fname = args[1] if (len(args) >= 2 and args[1] != None) else FILENAME #if optional argument is supplied on the command line use as base filename
    ext = FILETYPE
    a_type = args[2] if (len(args) >= 3 and args[2] != None) else ACQ_TYPE #if 2nd optional argument is supplied on the command line use acquiring mode
    if a_type[:4] == 'AVER':
        try:
            num_avg = int(a_type[4:]) if len(a_type)>4 else NUM_AVG # if the type is longer than four characters, treat characters from fifth to end as number of averages
        except ValueError:
            print("\nValueError: Failed to convert \'%s\' to an integer, check that acquisition type is on the form AVER or AVER<m> where <m> is an integer (currently acq. type is \'%s\').\nExiting..\n" % (a_type[4:], a_type))
            raise
        if num_avg < 1 or num_avg > 65536: #check that num_avg is within acceptable range
            raise ValueError("\nThe number of averages {} is out of range.\nExiting..\n".format(num_avg))
            sys.exit()
        fname += " " + a_type
    else:
        num_avg = NUM_AVG # not relevant unless AVERage
    names = ["single_trace", "connect_each_time", "single_connection"]
    if name == names[0]:
        get_single_trace(fname, ext, acq_type=a_type, num_averages=num_avg)
    elif name == names[1]:
        getTraces_connect_each_time_loop(fname, ext, acq_type=a_type, num_averages=num_avg)
    elif name == names[2]:
        getTraces_single_connection_loop(fname, ext, acq_type=a_type, num_averages=num_avg)
    else:
        raise ValueError("\nUnknown name \'%s\' of program to run. Available programmes %s." % (name, str(names)))
