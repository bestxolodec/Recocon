#!/usr/bin/env python
# encoding: utf-8

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from recocon.logger import Logger
from recocon.audio.audio import Audio
from recocon.audio.audiochunk import AudioChunk


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
        text: reference text of the lecture obtained from recognition proces.
        lm: language model for recognition
        dct: language dictionary for recognition
        hmm: hidden markov model directory fro recognition
    """

    def __init__(self, video_filepath, tmpdir, decoder, lm, dct, hmm,
                 audiofmt="wav", chunks_filename="out{}"):
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
        # set all recognition related attributes
        self.decoder = decoder
        self.lm = lm
        self.dct = dct
        self.hmm = hmm

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

    def plot_params(self, filepath=None, min_length_secs=5, samplesize=10,
                    min_speech_samples=5, min_silence_samples=30,
                    n_of_startsilence_samples=30,
                    energy_threshold_of_signal_presence=3, plot_params=False):
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
        # extrnally processed with audio software beforehand)
        min_energy = None
        min_freq = None
        min_spectr = None
        if plot_params:
            plt_energy = np.zeros(number_of_samples, np.float)
            plt_freq = np.zeros(number_of_samples, np.float)
            plt_spectr = np.zeros(number_of_samples, np.float)
            plt_min_energy = np.zeros(number_of_samples, np.float)
        # skip all samples from the begining of  data which contain only zeros
        sample_number = 0
        sample = self.audio.signal[frames_per_sample * sample_number:
                                   frames_per_sample * (sample_number + 1)]
        sample = Audio(framerate=self.audio.framerate, signal=sample)
        while sample.get_energy() < energy_threshold_of_signal_presence:
            sample_number += 1
            sample = self.audio.signal[frames_per_sample * sample_number:
                                       frames_per_sample * (sample_number + 1)]
            sample = Audio(framerate=self.audio.framerate, signal=sample)

        # NOTE: supposing some of the first 30 non-zer samples are silence
        for sample_number in range(sample_number,
                                   sample_number + n_of_startsilence_samples):
            sample = self.audio.signal[frames_per_sample * sample_number:
                                       frames_per_sample * (sample_number + 1)]
            sample = Audio(framerate=self.audio.framerate, signal=sample)

            # Solve two problems here: 1. init `min_...` variables with
            # meaningful data if there are only zeros in signal 2. ?
            self.log.debug("Energy of a frame: %r", sample.get_energy())
            if sample.get_energy() > energy_threshold_of_signal_presence:
                # workaround for inability to compare None and float value
                min_energy = min(i for i in [sample.get_energy(), min_energy]
                                 if i is not None)
                min_freq = min(i for i in [sample.get_dominant_freq(),
                                           min_freq] if i is not None)
                min_spectr = min(i for i in [sample.get_spectral_flatness(),
                                             min_spectr] if i is not None)
                self.log.debug("Processing sample_number {}. Start: {}. End: {}"
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
            if plot_params:
                plt_energy[sample_number] = sample.get_energy()
                plt_freq[sample_number] = sample.get_dominant_freq()
                plt_spectr[sample_number] = sample.get_spectral_flatness()
                plt_min_energy[sample_number] = min_energy

        # set decision threshold
        self.log.debug("min_energy: %r", min_energy)
        thr_energy = energy_threshold * np.log10(min_energy)
        thr_freq = freq_threshold
        thr_spectr = spectr_threshold

        silence_in_a_row = n_of_startsilence_samples
        for sample_number in range(n_of_startsilence_samples,
                                   number_of_samples):
            # set counter of speech presence to 0 if this counter exceeds 1,
            # we assume that this sample is speech sample
            counter = 0
            sample = self.audio.signal[frames_per_sample * sample_number:
                                       frames_per_sample * (sample_number + 1)]
            sample = Audio(framerate=self.audio.framerate, signal=sample)
            # within every loop remember what was energy, freq and spectr
            # of a sample
            if plot_params:
                plt_energy[sample_number] = sample.get_energy()
                plt_freq[sample_number] = sample.get_dominant_freq()
                plt_spectr[sample_number] = sample.get_spectral_flatness()
            if sample.get_energy() < energy_threshold_of_signal_presence:
                continue
            self.log.debug("Processing sample_number {}. Start: {}. End: {}"
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
                self.log.debug("Energy of a sample is higher then silence!")
                energy_decision[sample_number] = True
                counter += 1
            if sample.get_dominant_freq() - min_freq >= thr_freq:
                self.log.debug("Dominant frequency of a sample is higher "
                               "then silence!")
                frequency_decision[sample_number] = True
                counter += 1
            # do not check spectral flatness if we have two distinct evidences
            # if counter != 2 and (sample.get_spectral_flatness() - min_spectr
            #                     >= thr_spectr):
            if (sample.get_spectral_flatness() - min_spectr >= thr_spectr):
                self.log.debug("Spectral flatness is higher then that "
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

            if plot_params:
                plt_min_energy[sample_number] = min_energy

        if plot_params:
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

        if plot_params:
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
        silence_offset = min_silence_samples // 2 * frames_per_sample
        # same for speech start, but shift will be to the left
        speech_offset = silence_offset

        # graph real edges
        if plot_params:
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
                self.audio.normalize_signal(s=s, e=e)
                self.audio.write_to_file(chunks_filepath, start=s, end=e)
                if plot_params:
                    plt_actual_start_of_chunk.append(s)
                    plt_actual_end_of_chunk.append(e)
                # keep track of all chunks we have added
                self.chunks.append(AudioChunk(s, e, self.audio.framerate,
                                              filepath=chunks_filepath,
                                              decoder=self.decoder,
                                              lm=self.lm, dct=self.dct,
                                              hmm=self.hmm))
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

        if plot_params:
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

    def get_full_text(self):
        assert self.chunks, ("There are no available chunks to perform"
                             " concatenation of text chunks!")
        self.log.debug("Started full text assembly.")
        self.text = ' '.join([ch.text for ch in self.chunks])
        self.log.debug("Reference text: %s", self.text)
