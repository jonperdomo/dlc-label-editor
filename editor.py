#!/usr/bin/env python
"""
Label Editor
Load HDF file from DeepLabCut and edit labels. Returns updated HDF file.
"""

import os
import sys
import scipy
from scipy import io
import argparse
import cv2
import pandas as pd
import numpy as np

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

__author__ = "Jonathan Perdomo"
__license__ = "GPL-3.0"
__version__ = "0.1.0"


class Editor(object):
    def __init__(self, label_file, video_file, marker_color, marker_size, marker_thickness):
        """Initializes the editor object"""
        self.__label_file = label_file
        self.__video_file = video_file

        # Set variables
        self.__current_label = 0  # Active label index
        self.__current_frame = 0  # Active frame index
        self.__text_x, self.__text_y = (20, 20)  # Label name position
        self.__mx, self.__my = 0, 0   # Current mouse position
        self.__mouse_pressed = False
        self.__marker_size = marker_size
        self.__marker_thickness = marker_thickness
        self.__marker_color = marker_color
        self.__changes_made = False  # Whether labels were edited in the session

        # Set up the output filepath for fixed labels
        input_filename = os.path.splitext(os.path.basename(os.path.normpath(self.__label_file)))[0].rsplit('_Fixed')[0]
        folder = os.path.dirname(os.path.normpath(self.__label_file))
        output_filename = "%s_Fixed.h5" % input_filename
        self.__fixed_label_file = os.path.normpath(os.path.join(folder, output_filename))

        # Load labels
        self.__label_data = pd.read_hdf(label_file)
        self.__data_name = self.__label_data.keys()[0][0]
        self.__label_names = np.unique([k[1] for k in self.__label_data.keys()])

        # Load video
        self.__cap = cv2.VideoCapture(self.__video_file)
        self.__frame_count = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))-1

        # Set up track bar
        self.__window_name = self.__label_file
        cv2.namedWindow(self.__window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.__window_name, self._edit_label)
        cv2.createTrackbar('Label', self.__window_name, 0, len(self.__label_names)-1, self._on_label_trackbar)
        cv2.createTrackbar('Frame', self.__window_name, 0, self.__frame_count, self._on_frame_trackbar)

    def _on_label_trackbar(self, val):
        """Label trackbar value change observer"""
        # Update the label
        self.__current_label = val
        self.__cap.set(1, self.__current_frame)
        ret, frame = self.__cap.read()

        # Get the current label's position
        x, y = self._get_label_xy()

        # # Display
        self._draw_label(frame, x, y)

    def _get_label_xy(self):
        """Get the current label's coordinates"""
        x_f = self.__label_data[(self.__data_name, self.__label_names[self.__current_label], 'x')][self.__current_frame]
        y_f = self.__label_data[(self.__data_name, self.__label_names[self.__current_label], 'y')][self.__current_frame]
        x_i = int(round(x_f)) if not np.isnan(x_f) else np.nan
        y_i = int(round(y_f)) if not np.isnan(x_f) else np.nan
        return x_i, y_i

    def _display_label(self, x, y):
        """Show the current frame and corresponding label coordinates"""
        # Read the frame
        self.__cap.set(1, self.__current_frame)
        _ret, _frame = self.__cap.read()

        # Display
        self._draw_label(_frame, x, y)

    def _save_label(self, x, y):
        """Save the label's updated coordinates"""
        # Update the data frame
        self.__label_data[(self.__data_name, self.__label_names[self.__current_label], 'x')][self.__current_frame] = x
        self.__label_data[(self.__data_name, self.__label_names[self.__current_label], 'y')][self.__current_frame] = y

        # Save the updated *.H5 file
        self.__label_data.to_hdf(self.__fixed_label_file, key='df', mode='w')
        self.__changes_made = True

    def _on_frame_trackbar(self, val):
        """Frame trackbar value change observer"""
        # Update the frame index
        self.__current_frame = val
        self.__cap.set(1, self.__current_frame)
        _ret, _frame = self.__cap.read()

        # Get the current label's position
        _x, _y = self._get_label_xy()

        # Display
        self._draw_label(_frame, _x, _y)

    def _draw_label(self, frame, x, y):
        """Draw the label coordinates"""
        # Display
        if all(not np.isnan(v) for v in [x, y]):
            frame = cv2.drawMarker(frame, (x, y), self.__marker_color, cv2.MARKER_CROSS, self.__marker_size, self.__marker_thickness)
            frame = cv2.putText(frame, self.__label_names[self.__current_label], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.__marker_color, 2)

        cv2.imshow(self.__window_name, frame)

    def _edit_label(self, event, ex, ey, flags, param):
        """Mouse event observer for editing label positions"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__mx, self.__my = ex, ey
            self._save_label(self.__mx, self.__my)
            self._display_label(self.__mx, self.__my)
            self.__mouse_pressed = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.__mouse_pressed = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.__mouse_pressed:
                self.__mx, self.__my = ex, ey
                self._display_label(self.__mx, self.__my)

    def _save_matlab_file(self):
        """Convert the fixed label file to MATLAB (*.MAT) file format"""
        # Set output filepath
        input_filename = os.path.splitext(os.path.basename(os.path.normpath(self.__label_file)))[0]
        folder = os.path.dirname(os.path.normpath(self.__label_file))
        output_filename = "%s.mat" % input_filename
        output_filepath = os.path.normpath(os.path.join(folder, output_filename))

        # # Load labels
        # df = pd.read_hdf(self.__label_file)
        # data_name = df.keys()[0][0]
        # label_names = np.unique([k[1] for k in df.keys()])

        # Set up the output structure with X,Y and likelihood data
        df_out = {}
        for label in self.__label_names:
            x_ser = self.__label_data[(self.__data_name, label, 'x')]
            x_arr = x_ser.to_numpy()
            y_ser = self.__label_data[(self.__data_name, label, 'y')]
            y_arr = y_ser.to_numpy()
            ll_ser = self.__label_data[(self.__data_name, label, 'likelihood')]
            ll_arr = ll_ser.to_numpy()
            df_out[label] = {
                'x': x_arr,
                'y': y_arr,
                'likelihood': ll_arr
            }

        # Save *.MAT
        scipy.io.savemat(output_filepath, df_out)
        return output_filepath

    def run(self):
        """Run the editor"""
        # Begin editing
        running = True
        while running:
            # Read the frame
            self.__cap.set(1, self.__current_frame)
            ret, frame = self.__cap.read()

            # Update the trackbar position
            cv2.setTrackbarPos('Frame', self.__window_name, self.__current_frame)

            # Display
            if self.__mouse_pressed:
                # Update the current label's position
                x, y = self.__mx, self.__my
                self._save_label(x, y)
            else:
                # Get the current label's position
                x, y = self._get_label_xy()

            self._draw_label(frame, x, y)

            # Key event
            key = cv2.waitKey(0)
            if key == 44:
                self.__current_frame = max(0, self.__current_frame-1)  # Left
            elif key == 46:
                self.__current_frame = min(self.__frame_count, self.__current_frame+1)  # Right
            elif key == 27:
                running = False

        # When everything done, release the.... video capture object
        self.__cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

        if self.__changes_made:
            logging.info(f"Saved  edited *.H5 label file: {self.__fixed_label_file}")

            # Save in MATLAB file format
            output_filepath = self._save_matlab_file()
            logging.info(f"Saved  edited *.MAT label file: {output_filepath}")


def main():
    try:
        label_file, video_file, marker_color, marker_size, marker_thickness = read_cli_parameters()
        editor = Editor(label_file, video_file, marker_color, marker_size, marker_thickness)
        editor.run()

    except AssertionError as e:
        print(e)
        sys.exit(2)


def read_cli_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('labelfile', help="Input label file path (*.h5)")
    parser.add_argument('videofile', help="Input video file path (*.avi)")
    parser.add_argument('color', help="Marker color (red, green, blue) [Default: blue]", nargs="?", default='blue')
    parser.add_argument('size', help="Marker size [Default: 40]", nargs="?", default='40')
    parser.add_argument('thickness', help="Marker thickness [Default: 2]", nargs="?", default='2')
    args = parser.parse_args()

    # Parse arguments
    label_file = args.labelfile
    video_file = args.videofile
    marker_color = args.color.lower()
    marker_size = int(args.size)
    marker_thickness = int(args.thickness)

    # Verify file formats
    assert os.path.isfile(label_file), "Label file does not exist: {}".format(label_file)
    assert os.path.isfile(video_file), "Video file does not exist: {}".format(video_file)
    assert os.path.splitext(label_file)[-1].lower() == '.h5', "Label file is not *.H5: {}".format(label_file)
    video_ext = os.path.splitext(video_file)[-1].lower()
    assert (video_ext == '.avi' or video_ext == '.mp4'), "Video file is not *.AVI or *.MP4: {}".format(video_file)
    assert (marker_color in ['red', 'green', 'blue']), "Color is not red, green or blue: {}".format(marker_color)

    # Get color RGB
    marker_color_rgb = {
        'red': [0, 0, 255],
        'green': [0, 255, 0],
        'blue': [255, 0, 0]
    }[marker_color]

    logging.info(f"Label file is: {label_file}")
    logging.info(f"Video file is: {video_file}")

    return label_file, video_file, marker_color_rgb, marker_size, marker_thickness


if __name__ == '__main__':
    main()
