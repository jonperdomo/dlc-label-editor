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


class MatConverter(object):
    def __init__(self, label_file):
        """Initializes the editor object"""
        self.__label_file = label_file

        # Load labels
        self.__label_data = pd.read_hdf(label_file)
        self.__data_name = self.__label_data.keys()[0][0]
        self.__label_names = np.unique([k[1] for k in self.__label_data.keys()])

    def _save_matlab_file(self):
        """Convert the fixed label file to MATLAB (*.MAT) file format"""
        # Set output filepath
        input_filename = os.path.splitext(os.path.basename(os.path.normpath(self.__label_file)))[0]
        folder = os.path.dirname(os.path.normpath(self.__label_file))
        output_filename = "%s.mat" % input_filename
        output_filepath = os.path.normpath(os.path.join(folder, output_filename))

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
        """Run the *.H5 to *.MAT converter"""
        # Save in MATLAB file format
        output_filepath = self._save_matlab_file()
        logging.info(f"Saved *.MAT label file: {output_filepath}")


def main():
    try:
        label_file = read_cli_parameters()
        editor = MatConverter(label_file)
        editor.run()

    except AssertionError as e:
        print(e)
        sys.exit(2)


def read_cli_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('labelfile', help="Input label file path (*.h5)")
    args = parser.parse_args()

    # Parse arguments
    label_file = args.labelfile

    # Verify file formats
    assert os.path.isfile(label_file), "Label file does not exist: {}".format(label_file)
    assert os.path.splitext(label_file)[-1].lower() == '.h5', "Label file is not *.H5: {}".format(label_file)

    logging.info(f"Label file is: {label_file}")

    return label_file


if __name__ == '__main__':
    main()
