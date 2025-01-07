import array
import functools
import gzip
import operator
import os
import struct

import numpy as np


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


class MNISTLoader:

    @staticmethod
    def test_images():
        return MNISTLoader.vectorize(MNISTLoader._parse_mnist_file('mnist_files/test-images-idx3-ubyte.gz'))

    @staticmethod
    def test_labels():
        return MNISTLoader.one_hot(MNISTLoader._parse_mnist_file('mnist_files/test-labels-idx1-ubyte.gz'))

    @staticmethod
    def train_images():
        return MNISTLoader.vectorize(MNISTLoader._parse_mnist_file('mnist_files/train-images-idx3-ubyte.gz'))

    @staticmethod
    def train_labels():
        return MNISTLoader.one_hot(MNISTLoader._parse_mnist_file('mnist_files/train-labels-idx1-ubyte.gz'))

    @staticmethod
    def one_hot(arr):
        labels = []
        for label in arr:
            one_hot_label = np.zeros((10, 1))
            one_hot_label[label] = 1
            labels.append(one_hot_label)
        return labels

    @staticmethod
    def vectorize(arr):
        features = []
        for feature in arr:
            features.append(feature.reshape((-1, 1)))
        return features

    @staticmethod
    def _parse_idx(fd):
        """Parse an IDX file, and return it as a numpy array.

        Parameters
        ----------
        fd : file
            File descriptor of the IDX file to parse

        endian : str
            Byte order of the IDX file. See [1] for available options

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file

        1. https://docs.python.org/3/library/struct.html
            #byte-order-size-and-alignment
        """
        DATA_TYPES = {0x08: 'B',  # unsigned byte
                    0x09: 'b',  # signed byte
                    0x0b: 'h',  # short (2 bytes)
                    0x0c: 'i',  # int (4 bytes)
                    0x0d: 'f',  # float (4 bytes)
                    0x0e: 'd'}  # double (8 bytes)

        header = fd.read(4)
        if len(header) != 4:
            raise IdxDecodeError('Invalid IDX file, '
                                'file empty or does not contain a full header.')

        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

        if zeros != 0:
            raise IdxDecodeError('Invalid IDX file, '
                                'file must start with two zero bytes. '
                                'Found 0x%02x' % zeros)

        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise IdxDecodeError('Unknown data type '
                                '0x%02x in IDX file' % data_type)

        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        fd.read(4 * num_dimensions))

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(operator.mul, dimension_sizes)
        if len(data) != expected_items:
            raise IdxDecodeError('IDX file has wrong number of items. '
                                'Expected: %d. Found: %d' % (expected_items,
                                                            len(data)))

        return np.array(data).reshape(dimension_sizes)

    @staticmethod
    def _parse_mnist_file(fname):
        """Open the IDX file named fname and return it as a numpy array.

        Parameters
        ----------
        fname : str
            File name to parse

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file
        """
        fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
        with fopen(fname, 'rb') as fd:
            return MNISTLoader._parse_idx(fd)
