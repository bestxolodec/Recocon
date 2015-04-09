#!/usr/bin/python
from __future__ import division
import os
# import subprocess
import wave
import numpy as np
from recocon.logger import Logger


class Audio(Logger):
    """ Basic class for audio processing

    Contains methods for audio processing, format information and other useful
        stuff.

    Attributes:
        framerate
        signal
        path
        fft
        magnitude_spectr
        energy
        dominant_freq
        spectral_flatness
        n_of_frames: number of frames in audio track
        wav_params: parameters of a wav file. Needed for writing purposes
        rms: Root Mean Square of a signal: available only after get_rms
    """

    def __init__(self, framerate=8000, signal=None, path=None,
                 fft=None, magnitude_spectr=None, energy=None,
                 dominant_freq=None, spectral_flatness=None):

        """ Basic initialization of a signal: calc all useful parameters.
        """

        if signal is None and not path:
            self.log.error("Provide for a Audio class any information about "
                           "signal: at least path or a signal.")
            raise Exception("No sufficient information about signal")

        if signal is not None and len(signal.shape) != 1:
            self.log.error("Wrong shape of a signal! Designed to only "
                           "support one dimensional arrays!")
            raise Exception("Wrong shape of a signal! Designed to only "
                            "support one dimensional arrays!")
        self.framerate = framerate
        self.signal = signal
        self.path = path
        # if only signal was provided
        if signal is not None and not path:
            self.wav_params = None
            self.n_of_frames = len(signal)
        # write all params as attributes despite the fact that some (and maybe
        # all of theese paremeters could be None
        self.fft = fft
        self.magnitude_spectr = magnitude_spectr
        self.energy = energy
        self.dominant_freq = dominant_freq
        self.spectral_flatness = spectral_flatness

    def __getattr__(self, name):
        """ Hook to read from file only when there is need to do so  """
        self.log.debug("In __getattr__ name: {}".format(name))
        if name == 'signal' or name == 'n_of_frames' or name == 'wav_params':
            if self.__dict__['signal'] is None:
                self.log.debug("Reading audio file in __getattr__ method")
                if not os.path.isfile(self.path):
                    self.log.error("There is no audio file {}. Maybe you "
                                   "should call `video_to_audio`"
                                   " first".format(self.audio_filepath))
                    raise Exception("There is no audio file present on the FS")
                f = wave.open(self.path, 'r')
                n_of_frames = f.getnframes()
                self.signal = np.frombuffer(f.readframes(n_of_frames),
                                            np.int16)
                self.signal.setflags(write=1)
                self.log.debug(object.__getattribute__(self, "signal"))
                self.n_of_frames = n_of_frames
                self.wav_params = f.getparams()
        if name == 'rms' and not hasattr(self, 'rms'):
            self.log.debug("RMS counting")
            self.rms = np.sqrt(np.mean(np.square(self.signal, dtype=np.float)))
        return object.__getattribute__(self, name)

    def get_fft(self):
        if self.fft is None:
            self.fft = np.fft.fft(self.signal)
        return self.fft

    def get_magnitude_spectr(self):
        if self.magnitude_spectr is None:
            # normalized magnitude spectr
            self.magnitude_spectr = np.abs(self.get_fft()) / self.signal.size
        return self.magnitude_spectr

    def get_energy(self):
        if self.energy is None:
            self.energy = np.mean(np.square(self.signal, dtype=np.float))
        return self.energy

    def get_dominant_freq(self):
        if self.dominant_freq is None:
            # the next statement is valid because self.signal is always one
            # dimensional array
            freq_fft_signal = np.fft.fftfreq(self.signal.size, 1/self.framerate)
            # store for temporal purposes spectrum of amplitude
            # find the peak in the coefficients
            index_dominant_freq = np.argmax(self.get_magnitude_spectr())
            self.dominant_freq = np.abs(freq_fft_signal[index_dominant_freq])
            # if we  would need to transform max_freq_fft_signal into Herz
            # herz_max_freq_fft_signal = abs(max_freq_fft_signal * framerate)
        return self.dominant_freq

    def get_spectral_flatness(self):
        """ Calculate spectral flatness as ratio of geometric and arithmetic
            means of a magnitude_spectr

            Relys on the fact that at least self.fft_signal is present.
        """
        if self.spectral_flatness is None:
            power_spectrum = self.get_magnitude_spectr() ** 2
            # replace all zeros - they damage logarithm
            power_spectrum[power_spectrum == 0] = 1
            geom_mean_log10 = (np.sum(np.log10(power_spectrum))
                               / len(power_spectrum))
            arith_mean_log10 = np.log10(np.mean(power_spectrum))
            # FIXME: decide if absolute value is appropriate here
            self.spectral_flatness = 10 * np.abs(geom_mean_log10
                                                 - arith_mean_log10)
            self.spectral_flatness = 10 * (arith_mean_log10 - geom_mean_log10)
        return self.spectral_flatness

    def normalize_signal(self, method='peak', s=0, e=None):
        """ Normalize signal according to method passed as a `method` argument.
            This method also replaces self.signal with normalized version of
            a signal and resets all previously computed parameters of a signal.

            This method dosnt know wheter the signal was already normalized. So
            you should be careful with that.

        Args:
            method: possible methods are: peak, rms, mixed.

        """
        if not e:
            e = self.signal.size
        # perform normalization on slice
        # FIXME: find out if this copying in memory could be avoided
        signal = self.signal[s:e]
        # RootMeanSquare
        if method == 'peak':
            self.log.debug("Peak normalization choosed.")
            max_elem = np.max(np.abs(signal))
            # (-2) to accomplish normalization of both postitive and negative
            # sides of signal (theese sides are asymmetric regarding zero)
            max_int16 = (1 << 15) - 2
            norm_koef = max_int16 / max_elem
            signal = np.int16(norm_koef * signal)
        # TODO: decide if rms could be usefull and how to get rid of clipping in
        # that case
        if method == 'rms':
            self.log.debug("Rms normalization choosed.")
            rms = np.sqrt(np.mean(np.square(self.signal)))
            max_elem = rms
        # TODO: decide the best way to normalize using mixed technique
        if method == 'mixed':
            self.log.debug("Mixed normalization choosed.")
        self.signal[s:e] = signal
        # reset all params as the signal was changed
        self.fft = None
        self.magnitude_spectr = None
        self.energy = None
        self.dominant_freq = None
        self.spectral_flatness = None

    def write_to_file(self, filepath, start=0, end=None):
        self.log.debug("Writing chunk to file {} starting at frame {} "
                       "ending at frame {}".format(filepath, start, end))
        if os.path.isfile(filepath):
            self.log.warning("Rewriting existent file: {}".format(filepath))
        # if boundaries were not passed, assuming write all file down
        if not end:
            end = len(self.signal)
        assert self.wav_params, "Parameters of wav signal are unknown!"
        w = wave.open(filepath, 'w')
        w.setparams(self.wav_params)
        w.writeframes(self.signal[start:end].tostring())
