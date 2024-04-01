import json
import numpy as np
import math
import pyxdf
from datetime import datetime
import mne
from mne.io.pick import get_channel_type_constants

from .plot import plot_design


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def load(task_file, xdf_file=None):
    return NeuroStim(task_file, xdf_file)


class NeuroStim:
    def __init__(self, task_file, xdf_file=None):
        self.events = []  # lsl json events
        self.event_stream_id = []
        self.eeg_stream_id = []
        self.streams = {}
        self.subject = []
        self.start_time = []
        self.task_name = ''
        self.samples = {}
        self.xdf_file = xdf_file

        self.load_task(task_file)
        if xdf_file:
            self.load_xdf(xdf_file)
            self._time_correction()

    def event_filter(self, event_name=None, event_source=None, trial_types=None, block_types=None, sample_types=None):
        event_args = {'event_name': event_name, 'source': event_source}
        event_args = {key: value for key, value in event_args.items() if value is not None}

        sample_args = {'trial_type': trial_types, 'block_type': block_types, 'sample_type': sample_types}
        sample_args = {key: value for key, value in sample_args.items() if value is not None}

        filtered_list = self.events
        for key, value in event_args.items():
            filtered_list = list(filter(lambda event: event.get(key, None) == value, filtered_list))

        for key, value in sample_args.items():
            filtered_list = list(filter(lambda event: self.samples[event['sample_id']].get(key, None) in value, filtered_list))

        return list(filtered_list)

    '''
    def filter_by_item(self, events, item_type, item_properties=None):
        filtered_list = filter(lambda event: self.check_item(self.samples[event['sample_id']]['items'], item_type), events)
        result = []
        if item_properties is not None:
            for event in filtered_list:
                for item in self.samples[event['sample_id']]['items']:
                    for property in item['properties']:
    '''

    def check_item(self, items, item_type):
        return len([item for item in items if item['item_type'] == item_type]) > 0

    #def get_property(self, item, name):
    #    return next(filter(lambda property: property['name'] == name, item['properties']))['value']
    def get_property(self, item, name):
        return item['properties'][name]['value']

    def get_item(self, sample, item_type):
        return next(filter(lambda item: item['item_type'] == item_type, sample['items']))

    # Setting the event time from an already synchronized lsl event stream
    def _time_correction(self):
        if len(self.streams[self.event_stream_id]['time_series']) == 0:
            print('Error: lsl events is empty')
            return

        if len(self.events) != len(self.streams[self.event_stream_id]['time_series']):
            print('Warning: lost lsl event markers, lost tokens will be recovered')
            self._repair_lsl_time()

        # set lsl time
        for i, event in enumerate(self.events):
            if event['event_id'] == self.streams[self.event_stream_id]['time_series'][i]:
                event['time'] = self.streams[self.event_stream_id]['time_stamps'][i]

    def _repair_lsl_time(self):
        lsl_events = self.streams[self.event_stream_id]
        first_id = lsl_events['time_series'][0][0]
        first_idx = list(map(lambda x: x['event_id'], self.events)).index(first_id)
        time_offset = lsl_events['time_stamps'][0] - self.events[first_idx]['time']

        for i, event in enumerate(self.events):
            id = event['event_id']

            if i < len(lsl_events['time_series']):
                if lsl_events['time_series'][i][0] != id:
                    print('insert')
                    lsl_events['time_series'] = np.insert(lsl_events['time_series'], i, id, axis=0)
                    lsl_events['time_stamps'] = np.insert(lsl_events['time_stamps'], i, event['time']+time_offset, axis=0)
                else:
                    time_offset = lsl_events['time_stamps'][i] - self.events[i]['time']
            else:
                print('append')
                lsl_events['time_series'] = np.append(lsl_events['time_series'], [[id]], axis=0)
                lsl_events['time_stamps'] = np.append(lsl_events['time_stamps'], event['time']+time_offset)

    def load_task(self, file):
        with open(file, encoding='utf-8') as json_file:
            data = json.load(json_file)
            self.events = data['events']
            self.subject = data['Subject']
            self.start_time = datetime.strptime(data['StartTime'], '%d.%m.%Y %H:%M:%S')
            self.task_name = data['TaskName']
            self.samples = data['samples']

    def load_xdf(self, xdf_file):
        self.streams, _ = pyxdf.load_xdf(xdf_file)
        self.streams = {stream["info"]["stream_id"]: stream for stream in self.streams}

        for stream_id, stream in self.streams.items():
            if 'NeuroStim-Events' in stream['info']['name'][0]:
                self.event_stream_id = stream_id
            if 'EEG' in stream['info']['type'][0]:
                self.eeg_stream_id = stream_id

    def raw_xdf(self, annotation=False):
        stream = self.streams[self.eeg_stream_id]

        n_chans = int(stream["info"]["channel_count"][0])
        labels, types, units = [], [], []
        try:
            for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
                labels.append(str(ch["label"][0]))
                if ch["type"] and ch["type"][0].lower() in get_channel_type_constants(include_defaults=True):  # noqa: E501
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

        all_time_series = stream["time_series"]
        first_time = stream["time_stamps"][0]
        fs = float(np.array(stream["info"]["effective_srate"]).item())

        #delete spaces
        labels = [s.replace(" ", "") for s in labels]

        info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=types)

        microvolts = ("microvolt", "microvolts", "µV", "μV", "uV")
        scale = np.array([1e-6 if u in microvolts else 1 for u in units])
        all_time_series_scaled = (all_time_series * scale).T

        raw = mne.io.RawArray(all_time_series_scaled, info, verbose='error')
        raw._filenames = [self.xdf_file]

        # convert events to annotations
        if annotation:
            events = self.event_filter(event_name='show', event_source='sample')
            events_time, events_name, events_duration = [], [], []
            for event in events:
                if self.samples[event['sample_id']]['sample_type'] != "":
                    events_time.append(event['time']-first_time)
                    events_name.append(self.samples[event['sample_id']]['sample_type'])
                    events_duration.append(self.samples[event['sample_id']]['duration'])

            raw.annotations.append(events_time, events_duration, events_name)

        return raw

    def plot_design(self):
        plot_design(self.samples)
