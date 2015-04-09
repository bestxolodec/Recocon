#!/usr/bin/python
from __future__ import division
import os
import subprocess
import wave
import numpy as np

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


def logging_setup(name=__name__):
    """ Globally setup logger to use it inside classes.

    Args:
        None
    Returns:
        log: logger ready for writing to it.
    """
    import logging

    # create logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    log.addHandler(ch)

    return log


class Audio(object):
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
    """

    def __init__(self, framerate=8000, signal=None, path=None,
                 fft=None, magnitude_spectr=None, energy=None,
                 dominant_freq=None, spectral_flatness=None):

        """ Basic initialization of a signal: calc all useful parameters.
        """

        self.log = logging_setup()

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
        self.log.debug("Ended init function of an Audio object.")

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
                w = wave.open(self.path, 'r')
                n_of_frames = w.getnframes()
                import ipdb; ipdb.set_trace()
                self.signal = np.frombuffer(w.readframes(n_of_frames),
                                            np.int16)
                self.log.debug(object.__getattribute__(self, "signal"))
                self.n_of_frames = n_of_frames
                self.wav_params = w.getparams()
        return object.__getattribute__(self, name)

    def get_fft(self):
        if self.fft is None:
            self.fft = np.fft.fft(self.signal)
        return self.fft

    def get_magnitude_spectr(self):
        if self.magnitude_spectr is None:
            self.magnitude_spectr = np.abs(self.get_fft())
        return self.magnitude_spectr

    def get_energy(self):
        if self.energy is None:
            self.energy = np.mean(np.square(self.signal))
        return self.energy

    def get_dominant_freq(self):
        if self.dominant_freq is None:
            # the next statement is valid because self.signal is always one
            # dimensional array
            freq_fft_signal = np.fft.fftfreq(self.signal.size, 1/self.framerate)
            # store for temporal purposes spectrum of amplitude
            # find the peak in the coefficients
            index_dominant_freq = np.argmax(self.get_magnitude_spectr())
            self.dominant_freq = freq_fft_signal[index_dominant_freq]
            # if we  would need to transform max_freq_fft_signal into Herz
            # herz_max_freq_fft_signal = abs(max_freq_fft_signal * framerate)
        return self.dominant_freq

    def get_spectral_flatness(self):
        """ Calculate spectral flatness as ratio of geometric and arithmetic
            means of a magnitude_spectr

            Relys on the fact that at least self.fft_signal is present.
        """
        if self.spectral_flatness is None:
            # self.log.debug(self.get_magnitude_spectr())
            # self.log.debug(np.log10(self.get_magnitude_spectr()))
            # self.log.debug(np.sum(np.log10(self.get_magnitude_spectr())))
            import pdb; pdb.set_trace()
            geom_mean_log10 = (np.sum(np.log10(self.get_magnitude_spectr()))
                               / len(self.get_magnitude_spectr()))
            arith_mean_log10 = np.log10(np.mean(self.get_magnitude_spectr()))
            self.spectral_flatness = 10 * (geom_mean_log10 - arith_mean_log10)
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
        with wave.open(filepath, 'w') as w:
            w.setparams(self.wav_params)
            w.writeframes(self.signal[start:end].tostring())


class AudioChunk(Audio):
    """ This class contains information about particular audio chunk info """
    # audio_track =
    # reference_text =

    # set all data about chunk into wav file ? is it possible ?
    def __init__(self):
        pass


class Lecture(object):
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

        self.log = logging_setup()

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
                                            chunks_filename+audiofmt)

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
        self.audio = Audio(framerate, signal=None, path=self.audio_filepath)

    def silence_split(self, filepath=None, min_length_secs=5, samplesize=10,
                      max_length_secs=30, perc_of_rms_as_silence=0.1,
                      min_speech_samples=10, min_silence_samples=30):
        """ Split file `filepath` to chunks of `min_length_secs` and
        `max_length_sec`.

            Writes chunks to a filesystem in the same folder as `filepath`.
                Split is made on regions of a signal that have aplitude less
                then `perc_of_rms_as_silence * rms`.
                Length of framesize is 10ms
                160 frames in each of 20ms chunks.

        Args:
            filepath: path of the whole file that is needed to be splitted up.
            min_length_secs: minimal lenght of a chunk.
            samplesize: length of a sample (in ms). Each sample is decided to
                be either speech or silece sample. Defaults to 10ms.
            max_length_secs: maximal lenght of a chunk.
            perc_of_rms_as_silence: percent of RootMeanSquare of a signal,
                considered as a silence treshold

        Returns:
            No return

        Notes:
            # how to find out Short Term Zero Crossing Rate
            # zero_cross_indicies = np.where(np.diff(np.sign(signal)))[0]
            # find how many zero crossings has occured in every sample
            # hist, sample_edges = np.histogram(silence_indicies,
            #                                   bins=number_of_samples)
        """
        assert self.audio, ("There is no Audio object associated "
                            "with {}. Maybe you should call video_to_audio "
                            "first".format(self.video_filepath))
        n_of_frames = self.audio.n_of_frames
        frames_per_sample = self.audio.framerate * samplesize // 1000
        number_of_samples = n_of_frames // frames_per_sample
        # initialize bool array which contains masks of whether i-th
        # sample is speech or a silence: 0 for silence and 1 for speech
        speech_mask = np.zeros(number_of_samples, np.int)

        # set primary thresholds for: 1. energy 2. frequency
        # 3. spectral flatness. All theese values are empirically
        # founded by researchers
        energy_threshold = 40
        freq_threshold = 185
        spectr_threshold = 5
        # TODO: decide if this approach is universal for video lectures
        # Initial min values are calculated from first sample
        sample = Audio(self.audio.framerate,
                       self.audio.signal[0:frames_per_sample])
        min_energy = sample.get_energy()
        min_freq = sample.get_dominant_freq()
        min_spectr = sample.get_spectral_flatness()
        # NOTE: supposing first 30 samples are silence
        for sample_number in xrange(1, 30):
            sample = Audio(self.audio.framerate,
                           self.audio.signal[frames_per_sample * sample_number:
                                             frames_per_sample *
                                             (sample_number + 1)])
            min_energy = min(sample.get_energy(), min_energy)
            min_freq = min(sample.get_dominant_freq(), min_freq)
            min_spectr = min(sample.get_spectral_flatness(),
                             min_spectr)
        self.log.debug("Min energy: {}. Min freq: {}. Min spectr:"
                       " {}".format(min_energy, min_freq, min_spectr))
        # set decision threshold
        thr_energy = energy_threshold * np.log10(min_energy)
        thr_freq = freq_threshold
        thr_spectr = spectr_threshold

        silence_in_a_row = 0
        for sample_number in xrange(30, number_of_samples):
            # set counter of speech presence to 0
            counter = 0
            sample = Audio(self.audio.framerate,
                           self.audio.signal[frames_per_sample * sample_number:
                                             frames_per_sample *
                                             (sample_number+1)])
            if sample.get_energy() - min_energy >= thr_energy:
                counter += 1
            if sample.get_dominant_freq() - min_freq >= thr_freq:
                counter += 1
            if sample.get_spectral_flatness() - min_spectr >= thr_spectr:
                counter += 1
            if counter < 2:
                # self.log.debug("Detected silence sample.")
                silence_in_a_row += 1
                # TODO: decide if silence_in_a_row is a proper counter to be
                # put in this recalculation formula of min_energy: maybe there
                # is more properly to put total count of silence samples
                #
                # this is a silence and we need to update thresholds
                self.log.debug("Minimal energy is equal "
                               "to {}".format(min_energy))
                min_energy = (silence_in_a_row * min_energy -
                              sample.get_energy()) / (silence_in_a_row + 1)
                self.log.debug("Minimal energy (after adjusting) is equal "
                               "to {}".format(min_energy))
                thr_energy = energy_threshold * np.log10(min_energy)
                self.log.debug("Thr energy after adjusting:"
                               " {}".format(thr_energy))
            else:
                # silence_in_a_row = 0
                self.log.debug("Detected speech sample.")
                # speech sample
                speech_mask[sample_number] = True

        # We need to ignore silence, that is running less then 30 samples
        # (300ms = 0.3s) and ignore speech that is running less then 10 samles
        # (100 ms = 0.1s)

        diff = np.diff(speech_mask)
        # extract array of derived values: focus on speech
        start = np.where(diff == 1)[0]
        end = np.where(diff == -1)[0]
        # filter out  all less then ..
        mask = (end - start) >= min_speech_samples
        start_of_speech = start[mask]
        end_of_speech = end[mask]
        # find silence or  incidental noise
        # TODO: maybe `out of bounds` error, because of  end_of_speech+1
        sil_start = np.r_[0, end_of_speech+1]
        sil_end = np.r_[start_of_speech-1, len(speech_mask)]
        mask = (sil_end - sil_start) > min_silence_samples
        sil_end = sil_end[mask]
        sil_start = sil_start[mask]

        # calculate speech duration limits in terms of samples
        # max_speech_frames =  max_length_secs * self.framerate
        min_speech_frames = min_length_secs * self.audio.framerate

        # adjust indicies to point on real data frames instead of samples
        start_of_speech, end_of_speech = (start_of_speech * frames_per_sample,
                                          end_of_speech * frames_per_sample)

        # set counter for naming wav chunks
        counter = 0
        old_s = None
        for s, e in zip(start_of_speech, end_of_speech):
            # if there are already some batch that dont match our criteria of
            # length (is too small)
            if old_s:
                s = old_s
            # if the length of a speech chunk satisfy us
            if (e - s) > min_speech_frames:
                chunks_filepath = self.chunks_filepath.format(counter)
                self.audio.write_to_file(chunks_filepath, start=s, end=e)
                # keep track of all chunks we have added
                self.chunks.append(chunks_filepath)
                # increase counter of written chunks (only for naming purposes)
                counter += 1
                # everything is on the filesystem now
                old_s = None
            else:
                # remember old start position
                old_s = s


"""
a = signal[10000:11000]
fa = np.fft.fft(a) / a.size
nu = np.fft.fftfreq(a.size, 1)
Fk = np.fft.fftshift(fa)
nu = np.fft.fftshift(nu)
f, ax = plt.subplots(3,1,sharex=True)
ax[0].plot(nu, np.real(Fk))
ax[0].set_ylabel(r'$Re[F_k]$', size='x-large')
ax[1].plot(nu, np.imag(Fk))
ax[1].set_ylabel(r'$Im[F_k]$', size='x-large')
ax[2].plot(nu, np.absolute(Fk)**2)
ax[2].set_ylabel(r'$\vert F_k \vert^2$, size='x-large')
ax[2].set_ylabel(r'$\vert F_k \vert^2$', size='x-large')
ax[2].set_xlabel(r'$\widetilde{\nu}$', size = 'x-large')
plt.show()



        files = []
        count = 0
        for location in indices:
            start = location['start']
            end = location['end']
            print str(start) + ' to ' + str(end)
            w.setpos(start)  # Set position on the original wav file
            chunkData = w.readframes(end-start)  # And read to where we need

            chunkAudio = wave.open('file_'+str(count)+".wav", 'w')
            chunkAudio.setnchannels(w.getnchannels())
            chunkAudio.setsampwidth(w.getsampwidth())
            chunkAudio.setframerate(w.getframerate())
            chunkAudio.writeframes(chunkData)
            chunkAudio.close()
            files.append('file_'+str(count)+".wav")
            count += 1

        return files

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to `path`"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def add_silence(L, seconds):
    "Add silence to the start and end of `L` of length `seconds` (float)"
    LRtn = array('h', [0 for i in xrange(int(seconds*RATE))])
    LRtn.extend(L)
    LRtn.extend([0 for i in xrange(int(seconds*RATE))])
    return LRtn
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lecture recognizer module. "
                                     "Provides interface only")
    parser.add_argument(type=str, dest="video_filepath",
                        help="Path to the video file for processing")
    parser.add_argument("-d", "--tmpdir", dest="tmpdir", action="store",
                        default="/home/ipaulo/recocon/zero_ru_cont_8k_v2/chunks",
                        help="Path to tmp dir, where all chunks will be stored."
                        " Defaults to /home/ipaulo/recocon/zero_ru_cont_8k_v2/chunks")
    parser.add_argument("-n", "--noconvert", dest="noconvert",
                        action="store_true", default=False, help="Do not "
                        "extract audio from video, as it was done previously.")
    args = parser.parse_args()
    log = logging_setup()
    print args
    lecture = Lecture(args.video_filepath, args.tmpdir)
    if not args.noconvert:
        log.debug("No option that forbids convetion!")
        lecture.video_to_audio()
    lecture.silence_split()
