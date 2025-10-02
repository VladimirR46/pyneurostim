import numpy as np
import scipy.signal
import mne
from mne.io import get_channel_type_constants
from pathlib import Path
from mne.transforms import _get_trans, apply_trans
from mne._fiff.constants import FIFF
from mne.io.nirx.nirx import _convert_fnirs_to_head
from mne._freesurfer import get_mni_fiducials
from mne._fiff.meas_info import _format_dig_points
from mne.utils.check import _check_fname


def _fill_nan(data):
    for i in range(data.shape[0]):
        mask = np.isnan(data[i])
        if any(mask):
            idxs = np.where(~mask)[0]
            data[i][:idxs[0]] = data[i][idxs[0]]
            data[i][idxs[-1]:] = data[i][idxs[-1]]
    return data


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
    all_time_series_scaled = _fill_nan(all_time_series_scaled)

    raw = mne.io.RawArray(all_time_series_scaled, info)
    raw._filenames = [fname]

    return raw, first_time


def _read_dig_pts(fname, coord_frame='mri'):
    #fname = str(_check_fname(fname, overwrite="read", must_exist=True))
    points = {}
    with open(fname, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            label = parts[0].strip()
            coords = np.array([float(x) for x in parts[1].strip().split()])
            points[label] = coords / 1000.0  # переводим мм → м

    ch_pos = {}
    fiducials = {}
    for key, coord in points.items():
        if key in ['nz', 'al', 'ar']:
            fiducials[key] = coord
        elif key.startswith('s') or key.startswith('d'):
            ch_pos[key.upper()] = coord  # каналы в верхнем регистре

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=fiducials['nz'],
        lpa=fiducials['al'],
        rpa=fiducials['ar'],
        coord_frame=coord_frame
    )
    return montage


def _aurora_nirs_raw_xdf(streams, fname, stream_id, dig_file=None):
    type_mapping = {
        "nirs_raw": "fnirs_cw_amplitude",
        "nirs_hbo": "hbo",
        "nirs_hbr": "hbr"
    }

    stream = streams[stream_id]
    n_chans = int(stream["info"]["channel_count"][0]) - 1 # remove frame channel
    labels, types, units = [], [], []
    try:
        for ch in stream["info"]["desc"][0]["channels"][0]["channel"][1:]:
            ch_type = ch["type"][0].lower()
            if ch_type == 'nirs_raw':
                custom_name = ch["custom_name"][0].replace("-", "_")
                wavelength = str(int(float(ch["wavelength"][0])))
                labels.append(f"{custom_name} {wavelength}")
            elif ch_type == 'nirs_hbo':
                custom_name = ch["custom_name"][0].replace("-", "_")
                labels.append(f"{custom_name} hbo")
            elif ch_type == 'nirs_hbr':
                custom_name = ch["custom_name"][0].replace("-", "_")
                labels.append(f"{custom_name} hbr")
            else:
                labels.append(str(ch["label"][0]))

            if ch["type"] and ch_type in get_channel_type_constants(True):
                types.append(ch_type)
            else:
                types.append(type_mapping.get(ch_type, "misc"))
            units.append(ch["unit"][0] if ch["unit"] else "NA")
    except (TypeError, IndexError):  # no channel labels found
        pass
    if not labels:
        labels = [f"{stream['info']['name'][0]}_{n}" for n in range(n_chans)]
    if not units:
        units = ["NA" for _ in range(n_chans)]
    if not types:
        types = ["misc" for _ in range(n_chans)]

    time_series = streams[stream_id]["time_series"][:, 1:]
    first_time = streams[stream_id]["time_stamps"][0]
    fs = float(np.array(stream["info"]["effective_srate"]).item())

    src_locs = np.full((n_chans, 3), np.nan)
    det_locs = np.full((n_chans, 3), np.nan)
    ch_locs = np.full((n_chans, 3), np.nan)
    fnirs_wavelengths = np.full(n_chans, np.nan)
    for idx, ch in enumerate(stream["info"]["desc"][0]["channels"][0]["channel"][1:]):
        if ch["type"][0].lower() in ['nirs_raw', 'nirs_hbo', 'nirs_hbr']:
            if ch["wavelength"]:
                fnirs_wavelengths[idx] = float(ch["wavelength"][0])
            label = str(ch["label"][0]).split(':')[0]
            source_idx = int(label.split("-")[0]) - 1
            detector_idx = int(label.split("-")[1]) - 1
            montage = stream["info"]["desc"][0]['montage'][0]
            src = montage['optodes'][0]['sources'][0]['source'][source_idx]['location'][0]
            det = montage['optodes'][0]['detectors'][0]['detector'][detector_idx]['location'][0]
            src_locs[idx] = np.array([float(src['x'][0]), float(src['y'][0]), float(src['z'][0])]) / 1000.
            det_locs[idx] = np.array([float(det['x'][0]), float(det['y'][0]), float(det['z'][0])]) / 1000.
            ch_locs[idx] = (src_locs[idx] + det_locs[idx]) / 2.
            #loc = ch["location"][0]
            #ch_locs[idx] = np.array([float(loc["x"][0]), float(loc["y"][0]), float(loc["z"][0])]) / 1000.

    coord_frame = 'mri'
    trans = "fsaverage"
    if dig_file:
        montage = _read_dig_pts(dig_file, coord_frame=coord_frame)
        trans = mne.channels.compute_native_head_t(montage)

    src_locs, det_locs, ch_locs, mri_head_t = _convert_fnirs_to_head(
        trans, coord_frame, "head", src_locs, det_locs, ch_locs
    )

    # Set up digitization
    dig = get_mni_fiducials("fsaverage", verbose=False)
    for fid in dig:
        fid["r"] = apply_trans(mri_head_t, fid["r"])
        fid["coord_frame"] = FIFF.FIFFV_COORD_HEAD
    for ii, ch_loc in enumerate(ch_locs, 1):
        dig.append(
            dict(
                kind=FIFF.FIFFV_POINT_EEG,  # misnomer but probably okay
                r=ch_loc,
                ident=ii,
                coord_frame=FIFF.FIFFV_COORD_HEAD,
            )
        )
    dig = _format_dig_points(dig)

    info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=types)
    with info._unlock():
        info.update(dig=dig)

    # montage
    for idx in range(src_locs.shape[0]):
        if not np.isnan(src_locs[idx]).any():
            if not np.isnan(fnirs_wavelengths[idx]):
                info["chs"][idx]["loc"][9] = fnirs_wavelengths[idx]
            info['chs'][idx]['loc'][3:6] = src_locs[idx]
            info['chs'][idx]['loc'][6:9] = det_locs[idx]
            info['chs'][idx]['loc'][:3] = ch_locs[idx] #(src_locs[idx] + det_locs[idx]) / 2.
            info["chs"][idx]["coord_frame"] = FIFF.FIFFV_COORD_HEAD

    microvolts = ("microvolt", "microvolts", "µV", "μV", "uV")
    scale = np.array([1e-6 if u in microvolts else 1 for u in units])
    time_series_scaled = (time_series * scale).T

    raw = mne.io.RawArray(time_series_scaled, info)
    raw._filenames = [Path(fname)]
    return raw, first_time

