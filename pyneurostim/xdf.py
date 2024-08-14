import numpy as np
import scipy.signal
import mne
from mne.io import get_channel_type_constants


def _resample_streams(streams, stream_ids, fs_new):
    """
    Resample multiple XDF streams to a given frequency.

    Parameters
    ----------
    streams : dict
        A dictionary mapping stream IDs to XDF streams.
    stream_ids : list[int]
        The IDs of the desired streams.
    fs_new : float
        Resampling target frequency in Hz.

    Returns
    -------
    all_time_series : np.ndarray
        Array of shape (n_samples, n_channels) containing raw data. Time intervals where a
        stream has no data contain `np.nan`.
    first_time : float
        Time of the very first sample in seconds.
    """
    start_times = []
    end_times = []
    n_total_chans = 0
    for stream_id in stream_ids:
        start_times.append(streams[stream_id]["time_stamps"][0])
        end_times.append(streams[stream_id]["time_stamps"][-1])
        n_total_chans += int(streams[stream_id]["info"]["channel_count"][0])
    first_time = min(start_times)
    last_time = max(end_times)

    n_samples = int(np.ceil((last_time - first_time) * fs_new))
    all_time_series = np.full((n_samples, n_total_chans), np.nan)

    col_start = 0
    for stream_id in stream_ids:
        start_time = streams[stream_id]["time_stamps"][0]
        end_time = streams[stream_id]["time_stamps"][-1]
        len_new = int(np.ceil((end_time - start_time) * fs_new))

        x_old = streams[stream_id]["time_series"]
        x_new = scipy.signal.resample(x_old, len_new, axis=0)

        row_start = int(
            np.floor((streams[stream_id]["time_stamps"][0] - first_time) * fs_new)
        )
        row_end = row_start + x_new.shape[0]
        col_end = col_start + x_new.shape[1]
        all_time_series[row_start:row_end, col_start:col_end] = x_new

        col_start += x_new.shape[1]

    return all_time_series, first_time


def _raw_xdf(streams, fname, stream_ids, fs_new=None):
    """LSL XDF to MNE RAW.

    Parameters
    ----------
    streams : []
        pyxdf.load(xdf)
    fname : str
        Name of the XDF file.
    stream_ids : list[int]
        IDs of streams to load. A list of available streams can be obtained with
        `pyxdf.resolve_streams(fname)`.
    fs_new : float | None
        Resampling target frequency in Hz. If only one stream_id is given, this can be
        `None`, in which case no resampling is performed.

    Returns
    -------
    raw : mne.io.Raw
        XDF file data.
    first_time : float
        Time of the very first sample in seconds.
    """

    if len(stream_ids) > 1 and fs_new is None:
        raise ValueError("Argument `fs_new` required when reading multiple streams.")

    labels_all, types_all, units_all = [], [], []
    for stream_id in stream_ids:
        stream = streams[stream_id]

        n_chans = int(stream["info"]["channel_count"][0])
        labels, types, units = [], [], []
        try:
            for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
                labels.append(str(ch["label"][0]))
                if ch["type"] and ch["type"][0].lower() in get_channel_type_constants(True):
                    types.append(ch["type"][0].lower())
                else:
                    types.append("misc")
                units.append(ch["unit"][0] if ch["unit"] else "NA")
        except (TypeError, IndexError):  # no channel labels found
            pass
        if not labels:
            labels = [f"{stream['info']['name'][0]}_{n}" for n in range(n_chans)]
        if not units:
            units = ["NA" for _ in range(n_chans)]
        if not types:
            types = ["misc" for _ in range(n_chans)]
        labels_all.extend(labels)
        types_all.extend(types)
        units_all.extend(units)

    if fs_new is not None:
        all_time_series, first_time = _resample_streams(streams, stream_ids, fs_new)
        fs = fs_new
    else:  # only possible if a single stream was selected
        all_time_series = streams[stream_ids[0]]["time_series"]
        first_time = streams[stream_ids[0]]["time_stamps"][0]
        fs = float(np.array(stream["info"]["effective_srate"]).item())

    #delete spaces
    labels_all = [s.replace(" ", "") for s in labels_all]

    info = mne.create_info(ch_names=labels_all, sfreq=fs, ch_types=types_all)

    microvolts = ("microvolt", "microvolts", "µV", "μV", "uV")
    scale = np.array([1e-6 if u in microvolts else 1 for u in units_all])
    all_time_series_scaled = (all_time_series * scale).T

    raw = mne.io.RawArray(all_time_series_scaled, info)
    raw._filenames = [fname]

    return raw, first_time

