#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# utils.py
# ffsas: free-form inversion for small-angle scattering
# Copyright © 2021 SciML, STFC, UK. All rights reserved.


""" utilities """

import logging
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class MultiLevelLogger:
    """ Class to log multiple levels of sub-processes """

    class SimpleTimer:
        """ Simple timer class """

        def __init__(self, proc_name):
            self.start_time = time.time()
            self.proc_name = proc_name

        def elapsed(self):
            return time.time() - self.start_time

    def __init__(self, indent_char='.', indent_width=4):
        """
        Create a multi-level logger

        :param indent_char: character for indent fill (default: '.')
        :param indent_width: indent width (default: 4)
        """
        # indent
        self._indent_char = indent_char
        self._indent_width = indent_width

        # to be initialized later in activate() because
        # log activation is rank-dependent
        self._timers = []
        self._logger = None

    def activate(self, name=None, file_path=None, screen=True):
        """
        Activate this logger

        :param name: name of the logger (default: None)
        :param file_path: file to save logs (default: None)
        :param screen: show logs on screen (default: True)
        """
        # random name
        if name is None:
            name = str(uuid.uuid4())
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)
        if file_path is not None:
            self._logger.addHandler(
                logging.FileHandler(Path(file_path).expanduser(), 'w'))
        if screen:
            self._logger.addHandler(logging.StreamHandler(sys.stdout))

    @property
    def activated(self):
        return self._logger is not None

    @property
    def current_level(self):
        return len(self._timers)

    @property
    def elapsed_shallowest(self):
        """ Elapsed time of the shallowest level """
        if self.current_level == 0:
            return 0.  # not activated or not called start()
        else:
            return self._timers[-1].elapsed()

    def begin(self, proc_name: str):
        """
        Begin a sub-process

        :param proc_name: name of the sub-process
        """
        if not self.activated:
            return

        # message
        message = [self._indent_char * self._indent_width * self.current_level,
                   '<BEGIN> ', proc_name]
        self._logger.info(''.join(message))
        # timer
        self._timers.append(MultiLevelLogger.SimpleTimer(proc_name))

    def ended(self, proc_name: str = ''):
        """
        End a sub-process

        :param proc_name: name of the sub-process
        """
        if not self.activated:
            return

        # timer
        timer = self._timers.pop()
        elapsed = timer.elapsed()
        # message
        if proc_name == '':
            # calling ended() without proc_name
            proc_name = timer.proc_name
        else:
            # calling ended() with proc_name, check consistency
            assert proc_name == timer.proc_name, \
                f"Subprocess names do not match in " \
                f"begin() and ended() of a MultiLevelLogger instance." \
                f"\nSubprocess name passed to begin(): {timer.proc_name}" \
                f"\nSubprocess name passed to ended(): {proc_name}"
        message = [self._indent_char * self._indent_width * self.current_level,
                   '<ENDED> ', proc_name, f' [ELAPSED = {elapsed:f} sec]']
        self._logger.info(''.join(message))

    def message(self, what: str):
        """
        Log a message to the current sub-process

        :param what: message text to log
        """
        if not self.activated:
            return

        # handle indentation of multiple lines
        for i, line in enumerate(filter(None, what.split('\n'))):
            if i == 0:
                message = [self._indent_char * self._indent_width *
                           self.current_level, '<MESSG> ', line.strip()]
            else:
                message = [self._indent_char * self._indent_width *
                           self.current_level, ' '.rjust(8, self._indent_char),
                           line.strip()]
            self._logger.info(''.join(message))

    @contextmanager
    def subproc(self, proc_name: str):
        """
        Log a sub-process using with statement

        :param proc_name: name of the sub-process
        """
        self.begin(proc_name)
        try:
            yield self
        finally:
            self.ended(proc_name)

    def get_writer(self):
        """
        Get a writer that can redirect sys.stdout to this logger

        :return: writer of this logger
        """

        class LoggerWriter:
            def __init__(self, logger):
                self.logger = logger

            def write(self, message):
                self.logger.message(message)

            def flush(self):
                pass

        return LoggerWriter(self)


def _form_batch_ids(q_dims, batch_size):
    """ form batch ids """
    # grid axes
    batch_id_axes = []  # list of id on each axis
    batch_id_limits = []  # list of full-range slice of each axis
    for q_dim in q_dims:
        batch_id_axis = []
        for q_loc in range(0, q_dim, batch_size):
            q_id = slice(q_loc, min(q_loc + batch_size, q_dim))
            batch_id_axis.append(q_id)
        batch_id_axes.append(batch_id_axis)
        batch_id_limits.append(slice(0, len(batch_id_axis)))

    # flatten by np.mgrid
    id_of_ids = np.mgrid[batch_id_limits].reshape(len(q_dims), -1).T

    # form all batch ids
    batch_ids_all = []
    for id_of_id in id_of_ids:
        batch_id = []
        for iq, q_dim in enumerate(q_dims):
            batch_id.append(batch_id_axes[iq][id_of_id[iq]])
        batch_ids_all.append(tuple(batch_id))
    return batch_ids_all
