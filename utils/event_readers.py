import pandas as pd
import zipfile
from os.path import splitext
from os.path import isdir
import numpy as np
from .timers import Timer
from utils.bag_reader_ros2 import BagReader
import event_camera_py
import pandas as pd


def find_sensor_size(path_to_event_file, topic):
    if isdir(path_to_event_file): # assume ROS2 bag
        bag_reader = BagReader(path_to_event_file, topic)
        while bag_reader.has_next():
            _, msg, _ = bag_reader.read_next()
            return (msg.width, msg.height)
        raise Exception("cannot find sensor size in rosbag!")
    else:
        header = pd.read_csv(path_to_event_file, sep=r'\s+',
                             header=None, names=['width', 'height'],
                             dtype={'width': np.int32, 'height': np.int32},
                             nrows=1)
        return header.values[0]


class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, topic, num_events=10000, start_index=0):
        if isdir(path_to_event_file): # assume ROS2 bag
            raise Exception("fixed event size not supported for rosbags!")
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.iterator = pd.read_csv(path_to_event_file, sep=r'\s+', header=None,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        with Timer('Reading event window from file'):
            event_window = self.iterator.__next__().values
        return event_window


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient cunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, topic, duration_ms=50.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        self.is_ros2_bag = False
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.zip', ''])
        self.is_zip_file = (file_extension == '.zip')
        if self.is_zip_file:  # '.zip'
            self.zip_file = zipfile.ZipFile(path_to_event_file)
            files_in_archive = self.zip_file.namelist()
            assert(len(files_in_archive) == 1)  # make sure there is only one text file in the archive
            self.event_file = self.zip_file.open(files_in_archive[0], 'r')
        else:
            self.is_ros2_bag = isdir(path_to_event_file)
            if self.is_ros2_bag:
                print(f'opening ROS2 bag: {path_to_event_file}')
                self.bag_reader = BagReader(path_to_event_file, topic)
                self.decoder = event_camera_py.Decoder()
            else:
                self.event_file = open(path_to_event_file, 'r')
        if not self.is_ros2_bag:
            # ignore header + the first start_index lines
            for i in range(1 + start_index):
                self.event_file.readline()

        self.last_stamp = None
        self.duration_s = duration_ms / 1000.0
        self.frame_end = None


    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        if hasattr(self, 'event_file'):
            self.event_file.close()
        self.decoder = None

    def update_frame_end(self):
        self.frame_end +=int(self.duration_s * 1000000)

    def get_first_time_stamp(self):
        while self.bag_reader.has_next() and self.frame_end is None:
            _, self.last_msg, _ = self.bag_reader.read_next()
            self.frame_end = self.decoder.find_first_sensor_time(self.last_msg)
        if self.frame_end is not None:
            self.update_frame_end()  # increment by frame delta
        return self.frame_end
    
    def fetch_events_from_bag(self):
        all_events = []
        end_of_frame_reached = False
        while self.last_msg is not None:
            has_more, t_next = self.decoder.decode_until(self.last_msg, self.frame_end)
            all_events.append(self.decoder.get_cd_events())
            if not has_more: # reached end of message, read next
                if self.bag_reader.has_next():
                    _, self.last_msg, _ = self.bag_reader.read_next()
                else:
                    self.last_msg = None
            if t_next >= self.frame_end:
                self.update_frame_end()
                break
        if all_events:
            tmp = np.concatenate(all_events, axis=0)
            s = np.column_stack((tmp['t'], tmp['x'], tmp['y'], tmp['p'])).astype(np.float64)
        else:
            s = np.zeros((4, 0))
        # print(f'frame time: {self.frame_end} shape: {s.shape}')
        # print(f'type: {s.dtype}, bytes: {s.nbytes}, msg: {self.last_msg is None}')
        return s
            

    def __next__(self):
        with Timer('Reading event window from file'):
            if self.is_ros2_bag:
                if self.frame_end is None:
                    if self.get_first_time_stamp() is None:
                        raise StopIteration
                events = self.fetch_events_from_bag()
                if self.last_msg is None and not self.bag_reader.has_next() and events.shape[1] == 0:
                    raise StopIteration
                return events
            event_list = []
            for line in self.event_file:
                if self.is_zip_file:
                    line = line.decode("utf-8")
                t, x, y, pol = line.split(' ')
                t, x, y, pol = float(t), int(x), int(y), int(pol)
                event_list.append([t, x, y, pol])
                if self.last_stamp is None:
                    self.last_stamp = t
                if t > self.last_stamp + self.duration_s:
                    self.last_stamp = t
                    event_window = np.array(event_list)
                    return event_window

        raise StopIteration
