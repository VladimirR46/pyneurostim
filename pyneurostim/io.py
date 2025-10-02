import json
import numpy as np
import math
import pyxdf
from datetime import datetime
import mne
from mne.io import get_channel_type_constants
import pandas as pd

from .plot import plot_design
from .xdf import _raw_xdf, _aurora_nirs_raw_xdf


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def load(task_file, xdf_file=None):
    return NeuroStim(task_file, xdf_file)


class NeuroStim:
    def __init__(self, task_file, xdf_file=None, event_stream_name="NeuroStim-Events"):
        self.events = []  # lsl json events
        self.event_stream_id = None
        self.eeg_streams = {}
        self.streams = {}
        self.subject = []
        self.start_time = []
        self.task_name = ''
        self.samples = {}
        self.xdf_file = xdf_file
        self.event_stream_name = event_stream_name

        self.load_task(task_file)
        if xdf_file:
            self.load_xdf(xdf_file)
            self._time_correction()

    def event_filter(self, event_name=None, event_source=None, trial_types=None, block_types=None, sample_types=None, sample_id=None):

        event_args = {'event_name': event_name, 'source': event_source}
        event_args = {key: value for key, value in event_args.items() if value is not None}

        sample_args = {'trial_type': trial_types, 'block_type': block_types, 'sample_type': sample_types}
        sample_args = {key: value for key, value in sample_args.items() if value is not None}

        filtered_list = self.events
        for key, value in event_args.items():
            filtered_list = list(filter(lambda event: event.get(key, None) == value, filtered_list))

        for key, value in sample_args.items():
            filtered_list = list(filter(lambda event: self.samples[event['sample_id']].get(key, None) in value, filtered_list))

        if sample_id is not None:
            filtered_list = list(filter(lambda event: event['sample_id'] == sample_id, filtered_list))

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
            if event['event_id'] == self.streams[self.event_stream_id]['time_series'][i][0]:
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
            self.subject = data.get('Subject', 'Subject')
            self.start_time = datetime.strptime(data['StartTime'], '%d.%m.%Y %H:%M:%S')
            self.task_name = data['TaskName']
            self.samples = data['samples']

    def load_xdf(self, xdf_file):
        stream_info = pyxdf.resolve_streams(xdf_file)
        for stream in stream_info:
            if stream['name'] == self.event_stream_name:
                self.event_stream_id = stream['stream_id']
            if stream['type'] == 'EEG':
                self.eeg_streams[stream['name']] = stream['stream_id']

        if not self.event_stream_id or not self.eeg_streams:
            print(f"Warning: {self.event_stream_name} or EEG stream not found")

        self.streams, _ = pyxdf.load_xdf(xdf_file)
        self.streams = {stream["info"]["stream_id"]: stream for stream in self.streams}

    def _make_annotations(self, raw, first_time, extended_annotation=False):
        events = self.event_filter(event_name='show', event_source='sample')
        events_time, events_name, events_duration = [], [], []
        for event in events:
            sample = self.samples[event['sample_id']]
            sample_type = sample['sample_type']
            trial_type = sample['trial_type']
            block_type = sample['block_type']
            if sample_type != "":
                events_time.append(event['time'] - first_time)
                events_duration.append(self.samples[event['sample_id']]['duration'])

                # Build event name
                if extended_annotation:
                    event_name = "@".join([sample_type, block_type, trial_type])
                else:
                    event_name = sample_type

                events_name.append(event_name)

        raw.annotations.append(events_time, events_duration, events_name)

    def raw_xdf(self, annotation=False, eeg_stream_names=None, fs_new=None, extended_annotation=False):
        if len(self.eeg_streams) > 1:
            if eeg_stream_names is None:
                raise RuntimeError("It is necessary to set the names of the EEG streams")
            stream_ids = [self.eeg_streams[name] for name in eeg_stream_names]
        else:
            stream_ids = list(self.eeg_streams.values())

        if not fs_new and len(self.eeg_streams) > 1:
            fs_new = int(float(np.array(self.streams[stream_ids[0]]["info"]["nominal_srate"]).item()))

        raw, first_time = _raw_xdf(self.streams, self.xdf_file, stream_ids, fs_new)

        # convert events to annotations
        if annotation:
            self._make_annotations(raw, first_time)
        events = self.events_to_df(raw, first_time)
        return raw, events

    def get_stream_id(self, stream_name):
        stream_id = None
        for key, stream in self.streams.items():
            if stream['info']['name'][0] == stream_name:
                stream_id = key
                break
        if stream_id is None:
            raise RuntimeError(f"Stream with name '{stream_name}' not found")
        return stream_id

    def get_stream(self, stream_name=None):
        if self.xdf_file is None:
            raise RuntimeError("XDF file not loaded")
        stream_id = self.get_stream_id(stream_name)
        return self.streams[stream_id]

    def raw_stream(self, stream_name=None, stream_id=None):
        if stream_name:
            stream_id = self.get_stream_id(stream_name)
        elif stream_id is None:
            raise RuntimeError("It is necessary to set the name or ID of the stream")

        raw, first_time = _raw_xdf(self.streams, self.xdf_file, [stream_id])
        events = self.events_to_df(raw, first_time)
        return raw, events

    def raw_aurora_nirs(self, stream_name=None, annotation=False, dig_file=None):
        stream_id = self.get_stream_id(stream_name)
        raw, first_time = _aurora_nirs_raw_xdf(self.streams, self.xdf_file, stream_id, dig_file)
        if annotation:
           self._make_annotations(raw, first_time)
        events = self.events_to_df(raw, first_time)
        return raw, events

    def plot_design(self):
        return plot_design(self.samples)

    def events_to_df(self, raw=None, first_time=None):
        if raw and not first_time:
            raise ValueError("Argument `first_time` required when use raw.")

        array = []
        for event in self.events:
            if raw is not None:
                time = event['time'] - first_time
                time_idx = raw.time_as_index(time)[0]
            else:
                time = event['time']
                time_idx = None

            sample = self.samples[event['sample_id']]
            row = {'source': event['source'], 'event_name': event['event_name'], 'sample_type': sample['sample_type'],
                   'trial_type': sample['trial_type'], 'block_type': sample['block_type'],
                   'block_id': sample['block_id'], 'trial_id': sample['trial_id'], 'event_id': event['event_id'],
                   'item_id': event['item_id'], 'sample_id': event['sample_id'], 'trigger_code': sample['trigger_code'],
                   'time': time, 'index': time_idx}
            row['duration'] = sample['duration'] if row['event_name'] == 'show' else ''
            array.append(row)
        df = pd.DataFrame(array)
        return df
