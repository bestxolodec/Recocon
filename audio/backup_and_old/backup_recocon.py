#!/usr/bin/python
from __future__ import division
import os
import subprocess
import wave
import numpy as np
import pylab as plt

"""
pocketsphinx_continuous -samprate 8000 \
    -lm zero_ru.lm \
    -dict zero_ru.dic \
    -hmm zero_ru.cd_cont_4000 \
    -infile $INFILE


/home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.cd_cont_4000
/home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.cd_ptm_4000
/home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.cd_semi_4000
/home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.dic
/home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.lm

# Fast-froward to postion in media player (using vlc)
#   http://mih.voxindeserto.de/vlc_movieoverlays.html
"""


# def logging_setup(name=__name__):
""" Globally setup logger to use it inside classes.

Args:
None
Returns:
log: logger ready for writing to it.
"""
import logging
# create logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s -'
                              ' %(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
log.addHandler(ch)

# return log


# This class could be imported from a utility module
class Logger(object):
    @property
    def log(self):
        name = '.'.join([__name__, self.__class__.__name__])
        return logging.getLogger(name)


class Audio(Logger):
    """ Basic class for audio processing

    Contains methods for audio processing, format information and other useful
        stuf.

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

    def normalize_signal(self, method='peak'):
        """ Normalize signal according to method passed as a `method` argument.
            This method also replaces self.signal with normalized version of
            a signal and resets all previously computed parameters of a signal.

            This method dosnt know wheter the signal was already normalized. So
            you should be careful with that.

        Args:
            method: possible methods are: peak, rms, mixed.

        """
        # RootMeanSquare
        if method == 'peak':
            self.log.debug("Peak normalization choosed.")
            max_elem = np.max(np.abs(self.signal))
            # (-2) to accomplish normalization of both postitive and negative
            # sides of signal (theese sides are asymmetric regarding zero)
            max_int16 = (1 << 15) - 2
            norm_koef = max_int16 / max_elem
            self.signal = np.int16(norm_koef * self.signal)
        # TODO: decide if rms could be usefull and how to get rid of clipping in
        # that case
        if method == 'rms':
            self.log.debug("Rms normalization choosed.")
            rms = np.sqrt(np.mean(np.square(self.signal)))
            max_elem = rms
        # TODO: decide the best way to normalize using mixed technique
        if method == 'mixed':
            self.log.debug("Mixed normalization choosed.")
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


class AudioChunk(Logger):
    """ This class contains information about particular audio chunk info """
    # audio_track =
    # reference_text =

    # set all data about chunk into wav file ? is it possible ?
    def __init__(self):
        pass


class Lecture(Logger):
    """Class that represents the whole lecture object: audio, video, and all
    information that is related to them.

    Attributes:
        video_filepath: video lecture path on the filesystem.
        video_filename: video lecture name of file.
        audio_filepath: audio track path on the filesystem.
        audio_filename: audio track file name.
        log: logger.
        tmpdir: directory for temporal files (audio track and chunks).
        chunks: list which contains AudioChunk objects - sequence of chunks
            that were splitted from lecture.
    """

    def __init__(self, video_filepath, tmpdir, audiofmt="wav",
                 chunks_filename="out{}"):
        """ Assignes all class attributes """

        if not os.path.isdir(tmpdir):
            self.log.warning("There are no {} directory present on the"
                             " filesystem. Creating one...".format(tmpdir))
            os.mkdir(tmpdir)
        self.tmpdir = tmpdir

        # full path to video file
        self.video_filepath = video_filepath
        self.video_filename = os.path.basename(video_filepath)

        # only wav file supported by now
        self.audio_filename = "{}.{}".format(self.video_filename.split(".")[0],
                                             audiofmt)
        self.audio_filepath = os.path.join(tmpdir, self.audio_filename)
        # an instance of an Audio class. It is equal None because we have no
        # audio file by now.
        # WARN: this might be dangerous
        # self.audio = Audio(path=self.audio_filepath)
        self.chunks_filepath = os.path.join(self.tmpdir,
                                            "{}.{}".format(chunks_filename,
                                                           audiofmt))
        self.chunks = []

    def video_to_audio(self, audio_filepath=None,  bitrate="128k",
                       channels=1, framerate=8000):
        """ Extracts audio from video to format, which is specified as a
        audio_filename extension.

        This function extracts audio from video file and stores full audio
        track to audio_filename. It executes `avconv` as a separate process:
        `avconv -i linux.mp4 -ab 160k -ac 1 -ar 16000 -vn linux_audio.wav`

        Args:
            audio_filename: filename for audio track which
                is extracted from video file.
            bitrate: Bitrate of the result audio track.
            channels: Number of channels in audio. Defaults to mono.
            framerate: Frame rate of the result audio track.
        """

        if audio_filepath is None:
            audio_filepath = self.audio_filepath
        if os.path.isfile(audio_filepath):
            self.log.warning("File {} already exists! Copying present file"
                             " with `old` extension.".format(audio_filepath))
            os.rename(audio_filepath, audio_filepath+".old")

        cmd = [str(x) for x in ['avconv', '-i', self.video_filepath,
                                '-ab', bitrate, '-ac', channels, '-ar',
                                framerate, '-vn', audio_filepath]]
        self.log.debug("Executing command: {}".format(cmd))
        avconv = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=1)
        # wait for subprocess termination
        self.log.debug("Waiting for termination of convertion process...")
        returncode = avconv.wait()

        if returncode != 0:
            self.log.error("Bad exit code from conversion "
                           "from {} to {}. Exit code: {}"
                           .format(self.video_filepath, audio_filepath,
                                   returncode))
            self.log.error(avconv.stderr.read())
            raise Exception("Convertion error: non null exit code")
        self.log.debug("Convertion completed.")
        self.audio = Audio(framerate, path=self.audio_filepath)

    def plot_params(self, filepath=None, min_length_secs=10, samplesize=10,
                    min_speech_samples=5, min_silence_samples=80,
                    n_of_startsilence_samples=30,
                    energy_threshold_of_signal_presence=3):
        assert self.audio, ("There is no Audio object associated "
                            "with {}. Maybe you should call video_to_audio "
                            "first".format(self.video_filepath))
        # FIXME: remove division by 40
        n_of_frames = self.audio.n_of_frames
        samples_per_second = 1000 // samplesize
        frames_per_sample = self.audio.framerate // samples_per_second
        number_of_samples = n_of_frames // frames_per_sample
        # initialize bool array which contains masks of whether i-th
        # sample is speech or a silence: 0 for silence and 1 for speech
        speech_mask = np.zeros(number_of_samples, np.int)
        energy_decision = np.zeros(number_of_samples, np.int)
        frequency_decision = np.zeros(number_of_samples, np.int)
        spectr_decision = np.zeros(number_of_samples, np.int)

        # set primary thresholds for: 1. energy 2. frequency
        # 3. spectral flatness. All theese values are empirically
        # founded by researchers
        energy_threshold = 40
        freq_threshold = 185
        spectr_threshold = 6

        # TODO: decide if this approach is universal for video lectures
        # The problem here is that we do not mark speech samples in first
        # `n_of_startsilence_samples` even if there are some.
        #
        # if this frames contains only zeros (appears only if audio was
        # extrnally precessed with audio software)
        min_energy = None
        min_freq = None
        min_spectr = None
        plt_energy = np.zeros(number_of_samples, np.float)
        plt_freq = np.zeros(number_of_samples, np.float)
        plt_spectr = np.zeros(number_of_samples, np.float)
        plt_min_energy = np.zeros(number_of_samples, np.float)
        # NOTE: supposing some of the first 30 samples are silence
        for sample_number in xrange(n_of_startsilence_samples):
            sample = self.audio.signal[frames_per_sample * sample_number:
                                       frames_per_sample * (sample_number + 1)]
            sample = Audio(framerate=self.audio.framerate, signal=sample)

            # Solve two problems here: 1. init `min_...` variables with
            # meaningful data if there are only zeros in signal 2. ?
            if sample.get_energy() > energy_threshold_of_signal_presence:
                # workaround for inability to compare None and float value
                min_energy = min(i for i in [sample.get_energy(), min_energy]
                                 if i is not None)
                min_freq = min(i for i in [sample.get_dominant_freq(),
                                           min_freq] if i is not None)
                min_spectr = min(i for i in [sample.get_spectral_flatness(),
                                             min_spectr] if i is not None)
                self.log.info("Processing sample_number {}. Start: {}. End: {}"
                              "".format(sample_number,
                                        frames_per_sample * sample_number,
                                        frames_per_sample *
                                        (sample_number + 1)))
                self.log.debug("First silence samples: Min energy: {}."
                               " Min freq: {}. Min spectr: {}"
                               "".format(min_energy, min_freq, min_spectr))
                self.log.debug("Current energy: {} Current freq: {} "
                               "Current spectral flatness: {}"
                               "".format(sample.get_energy(),
                                         sample.get_dominant_freq(),
                                         sample.get_spectral_flatness()))
            # within every loop remember what energy, freq and spectr
            # of a sample was
            plt_energy[sample_number] = sample.get_energy()
            plt_freq[sample_number] = sample.get_dominant_freq()
            plt_spectr[sample_number] = sample.get_spectral_flatness()
            plt_min_energy[sample_number] = min_energy

        # set decision threshold
        thr_energy = energy_threshold * np.log10(min_energy)
        thr_freq = freq_threshold
        thr_spectr = spectr_threshold

        silence_in_a_row = n_of_startsilence_samples
        for sample_number in xrange(n_of_startsilence_samples,
                                    number_of_samples):
            # set counter of speech presence to 0 if this counter exceeds 1,
            # we assume that this sample is speech sample
            counter = 0
            sample = self.audio.signal[frames_per_sample * sample_number:
                                       frames_per_sample * (sample_number + 1)]
            sample = Audio(framerate=self.audio.framerate, signal=sample)
            # within every loop remember what was energy, freq and spectr
            # of a sample
            plt_energy[sample_number] = sample.get_energy()
            plt_freq[sample_number] = sample.get_dominant_freq()
            plt_spectr[sample_number] = sample.get_spectral_flatness()
            if sample.get_energy() < energy_threshold_of_signal_presence:
                continue
            self.log.info("Processing sample_number {}. Start: {}. End: {}"
                          "".format(sample_number,
                                    frames_per_sample * sample_number,
                                    frames_per_sample * (sample_number + 1)))
            self.log.debug("Current energy: {} Current freq: {} "
                           "Current spectral flatness: {}"
                           "".format(sample.get_energy(),
                                     sample.get_dominant_freq(),
                                     sample.get_spectral_flatness()))
            self.log.debug("Sample energy : %r -- min_energy: %r "
                           " ?? threshold: %r", sample.get_energy(),
                           min_energy, thr_energy)
            if sample.get_energy() - min_energy >= thr_energy:
                self.log.info("Energy of a sample is higher then silence!")
                energy_decision[sample_number] = True
                counter += 1
            if sample.get_dominant_freq() - min_freq >= thr_freq:
                self.log.info("Dominant frequency of a sample is higher "
                              "then silence!")
                frequency_decision[sample_number] = True
                counter += 1
            # do not check spectral flatness if we have two distinct evidences
            # if counter != 2 and (sample.get_spectral_flatness() - min_spectr
            #                     >= thr_spectr):
            if (sample.get_spectral_flatness() - min_spectr >= thr_spectr):
                self.log.info("Spectral flatness is higher then that "
                              "of a silence sample!")
                spectr_decision[sample_number] = True
                counter += 1
            if counter < 2:
                self.log.debug("Detected silence sample.")
                # TODO: decide if silence_in_a_row is a proper counter to be
                # put in this recalculation formula of min_energy: maybe it
                # is more properly to write total count of silence samples here
                #
                # this is a silence and we need to update thresholds
                self.log.debug("Minimal energy is equal "
                               "to {}".format(min_energy))
                min_energy = (silence_in_a_row * min_energy +
                              sample.get_energy()) / (silence_in_a_row + 1)
                self.log.debug("Minimal energy (after adjusting) is equal "
                               "to {}".format(min_energy))
                thr_energy = energy_threshold * np.log10(min_energy)
                self.log.debug("Thr energy after adjusting:"
                               " {}".format(thr_energy))
                silence_in_a_row += 1
            else:
                silence_in_a_row = 1
                self.log.debug("Detected speech sample.")
                # speech sample
                speech_mask[sample_number] = True

            plt_min_energy[sample_number] = min_energy

        xaxis = np.arange(number_of_samples) * frames_per_sample
        plt.subplot(411)
        plt.plot(xaxis, plt_energy)
        plt.plot(xaxis, energy_decision * np.max(plt_energy))
        plt.title("Energies of samples")
        plt.subplot(412)
        plt.plot(xaxis, plt_freq)
        plt.plot(xaxis, frequency_decision * np.max(plt_freq))
        plt.title("Frequencies of samples")
        plt.subplot(413)
        plt.plot(xaxis, plt_spectr)
        plt.plot(xaxis, spectr_decision * np.max(plt_spectr))
        plt.title("Spectral flatness koefficients of samples")
        plt.subplot(414)
        plt.plot(self.audio.signal[0: number_of_samples * frames_per_sample])
        plt.title("Signal")
        plt.show()

        plt.subplot(211)
        plt.plot(xaxis, plt_spectr)
        plt.plot(xaxis, spectr_decision * np.max(plt_spectr))
        plt.title("Spectral flatness koefficients of samples")
        plt.subplot(212)
        plt.plot(self.audio.signal[0: number_of_samples * frames_per_sample])
        plt.title("Signal")
        plt.show()

        # TODO: experiment
        def movingaverage(values, window):
            weights = np.repeat(1.0, window)/window
            sma = np.convolve(values, weights, 'valid')
            return sma

        plt.subplot(411)
        plt.plot(xaxis, plt_spectr)
        # plt.plot(xaxis, spectr_decision * np.max(plt_spectr))
        plt.title("Spectral flatness koefficients of samples")
        plt.subplot(412)
        plt.plot(xaxis, plt_spectr - min_spectr)
        plt.plot(xaxis, np.ones(xaxis.shape) * thr_spectr)
        plt.title("Threshold of spectral flatness")
        plt.subplot(413)
        sma = movingaverage(plt_spectr - min_spectr, 2 * min_silence_samples)
        sma = np.r_[[0] * (2 * min_silence_samples-1), sma]
        plt.plot(xaxis, sma)
        plt.plot(xaxis, np.ones(xaxis.shape) * thr_spectr)
        plt.title("Threshold of spectral flatness averaged")
        plt.subplot(414)
        plt.plot(self.audio.signal[0: number_of_samples * frames_per_sample])
        plt.title("Signal")
        plt.show()
        # import ipdb; ipdb.set_trace()
        # plt.plot(xaxis, plt_energy)
        # plt.plot(xaxis, energy_decision * np.max(plt_energy))
        # plt.plot(xaxis, plt_min_energy)
        # plt.title("Energies of samples")
        # plt.show()

        # We need to ignore silence, that is running less then 30 samples
        # (300ms = 0.3s) and ignore speech that is running less then 10 samles
        # (100 ms = 0.1s)

        diff = np.diff(speech_mask)
        # extract array of derived values: focus on speech
        start = np.where(diff == 1)[0]
        end = np.where(diff == -1)[0]
        # filter out  all less then ..
        self.log.debug("Initial start indexies of speech: %r", start)
        self.log.debug("Initial end indexies of speech: %r", end)
        # the last frame is speech frame
        if(start.size != end.size):
            # padd `end` with highest sample number to obtain broadcastability
            # of a numpy arrays
            end = np.r_[end, speech_mask.size]
        mask = (end - start) >= min_speech_samples
        start_of_speech = start[mask]
        end_of_speech = end[mask]
        self.log.debug("Final start indexies of speech: %r", start_of_speech)
        self.log.debug("Final end indexies of speech: %r", end_of_speech)
        # find silence or  incidental noise
        # TODO: maybe `out of bounds` error, because of  end_of_speech+1
        sil_start = np.r_[0, end_of_speech+1]
        sil_end = np.r_[start_of_speech-1, len(speech_mask)]
        self.log.debug("Initial start indexies of silence: %r", sil_start)
        self.log.debug("Initial end indexies of silence: %r", sil_end)
        mask = (sil_end - sil_start) > min_silence_samples
        sil_end = sil_end[mask]
        sil_start = sil_start[mask]
        self.log.debug("Final start indexies of silence: %r", sil_start)
        self.log.debug("Final end indexies of silence: %r", sil_end)

        # calculate speech duration limits in terms of samples
        # max_speech_frames =  max_length_secs * self.framerate
        min_length_in_frames = min_length_secs * self.audio.framerate

        # adjust indicies to point on real data frames instead of samples
        start_of_speech, end_of_speech = (start_of_speech * frames_per_sample,
                                          end_of_speech * frames_per_sample)
        sil_start, sil_end = (sil_start * frames_per_sample,
                              sil_end * frames_per_sample)
        # if last chunks is whole full of speech -> add
        if start_of_speech[-1] > sil_start[-1]:
            sil_start = np.r_[sil_start, n_of_frames]
        self.log.debug("Final start indexies of speech: %r", start_of_speech)
        self.log.debug("Final start indexies of silence: %r", sil_start)

        def chunk_index(start_of_speech, sil_start):
            """ This function is a generator of a tuples:
            (s_idx, e_idx), where:
                s_idx is index of the beging of speech,
                e_idx - end of speech

            Args:
                start_of_speech: numpy array of indexes. Each of indexes
                    represents an index from which starts
                    a sequence of speech samples.
                sil_start: numpy array of indexes. Each of indexes represents
                    an idnex from which starts a sequence of silence samples

            Example:
                In [209]: start_of_speech
                Out[209]:
                60,  214,  248,  278,  329,  378,  412,  450,  461,  496,  536,
                562,  584,  620,  642,  659,  683,  774,  794,  873,  905,  986,
                1077,1097,1168, 1188, 1220, 1243, 1313, 1363, 1382, 1428, 1455,
                1481, 1504, 1528, 1585, 1637, 1659])

                In [210]: sil_start,
                Out[210]: (array([ 0, 71, 951, 1013, 1120, 1260, 1544, 1603]),)

                In [208]: np.searchsorted(start_of_speech, sil_start,
                                            side='right')
                Out[208]: array([ 0,  1, 21, 22, 24, 28, 36, 37])

                In [213]: np.searchsorted(sil_start, start_of_speech,
                                            side='left')
                array([1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2, 2, 3, 4,
                        4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8])
            """
            s_idx = e_idx = 0
            ins2speech_ind = np.searchsorted(start_of_speech, sil_start,
                                             side='right')
            ins2silence_ind = np.searchsorted(sil_start, start_of_speech,
                                              side='left')
            while (e_idx < ins2speech_ind.size and
                   s_idx < ins2silence_ind.size):
                    e_idx = ins2silence_ind[s_idx]
                    self.log.debug("Going to lookup speech index %r and silence"
                                   "index %r", s_idx, e_idx)
                    yield start_of_speech[s_idx], sil_start[e_idx]
                    s_idx = ins2speech_ind[e_idx]

        # offset for silence: used to shift an end edge of a chunk half way
        # (of a minimal silence size) right
        silence_offset = min_silence_samples // 3 * frames_per_sample
        # same for speech start, but shift will be to the left
        speech_offset = silence_offset

        # graph real edges
        plt_actual_start_of_chunk = []
        plt_actual_end_of_chunk = []

        old_s = None
        # set counter for naming wav chunks
        written_s = counter = 0
        chunks_filepath = self.chunks_filepath.format(counter)
        for s, e in chunk_index(start_of_speech, sil_start):
            # there is already some batch that dont match our criteria of
            # length (is too small)
            self.log.debug("Processing start index: %r and end index %r", s, e)
            # check that our indexies are not out of bounds of file length
            if e + silence_offset <= n_of_frames:
                e = e + silence_offset
            else:
                e = n_of_frames
            if old_s:
                s = old_s
            elif s - speech_offset >= 0:
                s = s - speech_offset
            # if the length of a speech chunk satisfy us
            if (e - s) > min_length_in_frames:
                chunks_filepath = self.chunks_filepath.format(counter)
                self.audio.write_to_file(chunks_filepath, start=s, end=e)
                plt_actual_start_of_chunk.append(s)
                plt_actual_end_of_chunk.append(e)
                # keep track of all chunks we have added
                self.chunks.append(chunks_filepath)
                # remember start of written chunk (useful in case that last
                # chunk would be too small for writing as an standalone chunk)
                written_s = s
                # increase counter of written chunks (only for naming purposes)
                counter += 1
                # everything is on the filesystem now
                old_s = None
            else:
                self.log.debug("Insufficient length %r to split chunk."
                               " It is less then min_speech_frames %r. "
                               "Accumulating.", (e - s), min_length_in_frames)
                # remember old start position
                old_s = s
        # if the last chunk's lengt is not sufficient to write it down
        if old_s:
            # concatenate it with last written chunk
            self.audio.write_to_file(chunks_filepath, start=written_s, end=e)
            plt_actual_start_of_chunk.append(s)
            plt_actual_end_of_chunk.append(e)

        xaxis = np.arange(n_of_frames)
        plt.subplot(311)
        plt.plot(xaxis, self.audio.signal[0: n_of_frames])
        plt.vlines(start * frames_per_sample, -32768, 32768, colors='r')
        plt.vlines(end * frames_per_sample, -32768, 32768, colors='g')
        plt.title("Initial split")
        plt.subplot(312)
        plt.plot(xaxis, self.audio.signal[0: n_of_frames])
        plt.vlines(start_of_speech, -32768, 32768, colors='r')
        plt.vlines(sil_start, -32768, 32768, colors='g')
        plt.title("Masked split")
        plt.subplot(313)
        plt.plot(xaxis, self.audio.signal[0: n_of_frames])
        plt.vlines(plt_actual_start_of_chunk, -32768, 32768, colors='r')
        plt.vlines(plt_actual_end_of_chunk, -32768, 32768, colors='g')
        plt.title("Actual split")
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lecture recognizer module. "
                                     "Provides interface only")
    parser.add_argument(type=str, dest="video_filepath",
                        help="Path to the video file for processing")
    parser.add_argument("-d", "--tmpdir", dest="tmpdir", action="store",
                        default="/home/ipaulo/recocon/"
                        "zero_ru_cont_8k_v2/chunks",
                        help="Path to tmp dir, where all chunks will be stored"
                        " Defaults to /home/ipaulo/recocon/"
                        "zero_ru_cont_8k_v2/chunks")
    parser.add_argument("-n", "--noconvert", dest="noconvert",
                        action="store_true", default=False, help="Do not "
                        "extract audio from video, as it was done previously.")
    args = parser.parse_args()

    print args
    lecture = Lecture(args.video_filepath, args.tmpdir)
    if not args.noconvert:
        log.debug("No option that forbids convetion!")
        lecture.video_to_audio()
    # lecture.silence_split()
    lecture.plot_params()
