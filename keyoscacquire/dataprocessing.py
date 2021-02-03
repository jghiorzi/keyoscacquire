# -*- coding: utf-8 -*-
"""
This module provides functions for processing the data captured from the
oscilloscope to time and voltage values

The output from the :func:`Oscilloscope.capture_and_read` function is processed
by :func:`process_data`, a wrapper function that sends the data to the
respective binary or ascii processing functions.

This function is kept outside the Oscilloscope class as one might want to
post-process data separately from capturing it.

"""

import logging
import numpy as np

from typing import Tuple, Union, List

_log = logging.getLogger(__name__)


def process_data(
    raw: Union[List[np.ndarray], str],
    metadata: Union[List[str], Tuple[str, str]],
    wav_format: str,
    verbose_acquistion: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper function for choosing the correct _process_data function
    according to :attr:`wav_format` for the data obtained from
    :func:`~keyoscacquire.oscilloscope.Oscilloscope.capture_and_read`

    Parameters
    ----------
    raw : ~numpy.ndarray or str
        From :func:`~keyoscacquire.oscilloscope.Oscilloscope.capture_and_read`:
        Raw data, type depending on :attr:`wav_format`
    metadata : list or tuple
        From :func:`~keyoscacquire.oscilloscope.Oscilloscope.capture_and_read`:
        List of preambles or tuple of preamble and model series depending on
        :attr:`wav_format`. See :ref:`preamble`.
    wav_format : {``'WORD'``, ``'BYTE'``, ``'ASCii'``}
        Specify what waveform type was used for acquiring to choose the correct
        processing function.
    verbose_acquistion : bool
        True prints the number of points captured per channel

    Returns
    -------
    time : :class:`~numpy.ndarray`
        Time axis for the measurement
    y : :class:`~numpy.ndarray`
        Voltage values, each row represents one channel

    Raises
    ------
    ValueError
        If ``wav_format`` is not {'BYTE', 'WORD', 'ASCii'}

    See also
    --------
    :func:`keyoscacquire.oscilloscope.Oscilloscope.capture_and_read`
    """
    if wav_format[:3].upper() in ["WOR", "BYT"]:
        processing_fn = _process_data_binary
    elif wav_format[:3].upper() == "ASC":
        processing_fn = _process_data_ascii
    else:
        raise ValueError(
            f"Could not process data, waveform format '{wav_format}' is unknown."
        )
    return processing_fn(raw, metadata, verbose_acquistion)


def _process_data_binary(
    raw: List[np.ndarray], preambles: List[str], verbose_acquistion: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process raw 8/16-bit data to time values and y voltage values as received
    from :func:`Oscilloscope.capture_and_read_binary`.

    Parameters
    ----------
    raw : ~numpy.ndarray
        From :func:`~keyoscacquire.oscilloscope.Oscilloscope.capture_and_read_binary`:
        An ndarray of ints that is converted to voltage values using the preamble.
    preambles : list of str
        From :func:`~keyoscacquire.oscilloscope.Oscilloscope.capture_and_read_binary`:
        List of preamble metadata for each channel (list of comma separated
        ascii values, see :ref:`preamble`)
    verbose_acquistion : bool
        True prints the number of points captured per channel

    Returns
    -------
    time : :class:`~numpy.ndarray`
        Time axis for the measurement
    y : :class:`~numpy.ndarray`
        Voltage values, each row represents one channel
    """
    # Pick one preamble and use for calculating the time values (same for all channels)
    preamble = preambles[0].split(",")  # values separated by commas
    num_samples = int(float(preamble[2]))
    xIncr, xOrig, xRef = float(preamble[4]), float(preamble[5]), float(preamble[6])
    time = np.array(
        [(np.arange(num_samples) - xRef) * xIncr + xOrig]
    )  # compute x-values
    time = time.T  # make x values vertical
    _log.debug(f"Points captured per channel:  {num_samples:,d}")
    if verbose_acquistion:
        print(f"Points captured per channel:  {num_samples:,d}")
    y = np.empty((len(raw), num_samples))
    for i, data in enumerate(raw):  # process each channel individually
        preamble = preambles[i].split(",")
        yIncr, yOrig, yRef = float(preamble[7]), float(preamble[8]), float(preamble[9])
        y[i, :] = (data - yRef) * yIncr + yOrig
    # Convert y to np array and transpose for vertical channel columns in csv file
    y = y.T
    return time, y


def _process_data_ascii(
    raw: List[str], metadata: Tuple[str, str], verbose_acquistion: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Process raw comma separated ascii data to time values and y voltage
    values as received from :func:`Oscilloscope.capture_and_read_ascii`

    Parameters
    ----------
    raw : str
        From :func:`~keyoscacquire.oscilloscope.Oscilloscope.capture_and_read_ascii`:
        A string containing a block header and comma separated ascii values
    metadata : tuple
        From :func:`~keyoscacquire.oscilloscope.Oscilloscope.capture_and_read_ascii`:
        Tuple of the preamble for one of the channels to calculate time axis (same for
        all channels) and the model series. See :ref:`preamble`.
    verbose_acquistion : bool
        True prints the number of points captured per channel

    Returns
    -------
    time : :class:`~numpy.ndarray`
        Time axis for the measurement
    y : :class:`~numpy.ndarray`
        Voltage values, each row represents one channel
    """
    preamble, model_series = metadata
    preamble = preamble.split(",")  # Values separated by commas
    num_samples = int(float(preamble[2]))
    xIncr, xOrig, xRef = float(preamble[4]), float(preamble[5]), float(preamble[6])
    # Compute time axis and wrap in extra [] to make it 2D
    time = np.array([(np.arange(num_samples) - xRef) * xIncr + xOrig])
    time = time.T  # Make list vertical
    _log.debug(f"Points captured per channel:  {num_samples:,d}")
    if verbose_acquistion:
        print(f"Points captured per channel:  {num_samples:,d}")
    y = []
    for data in raw:
        if model_series in ["2000"]:
            # Remove first 10 characters (IEEE block header)
            data = data.split(data[:10])[1]
        elif model_series in ["9000"]:
            # Remove newline character at the end of the string
            data = data.strip().strip(",")
        data = data.split(",")  # samples separated by commas
        data = np.array([float(sample) for sample in data])
        y.append(data)  # add ascii data for this channel to y array
    y = np.transpose(np.array(y))
    return time, y
