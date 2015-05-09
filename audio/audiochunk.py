#!/usr/bin/python

from recocon.logger import Logger
# import multiprocessing as m
import subprocess
import re


class AudioChunk(Logger):
    """ This class contains information about particular audio chunk

    Attrs:
        audio: audio track, an instance of Audio class, associated with
            filepath.
        text: text associated with this chunk
        start: second in the initial video file from which this track starts.
        end: second in the initial video file that this track lasts to.
        framerate: framerate of a parent audio track
        lm:
        dct:
        hmm:
        raw_text: raw result of a speech recognition process.

     """

    def __init__(self, start, end, framerate, filepath=None, signal=None,
                 decoder="pocketsphinx_continuous",
                 lm="/home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.lm",
                 dct="/home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.dic",
                 hmm="/home/ipaulo/recocon/zero_ru_cont_8k_v2/"
                 "zero_ru.cd_semi_4000",  ctl_filepath="/tmp/ctl",
                 outputfile="/tmp/hyp"):

        assert signal or filepath, ("No data to recognize! Provide "
                                    "filename or signal!")
        self.signal = signal
        self.filepath = filepath
        self.framerate = framerate
        self.start = start // framerate
        self.end = end // framerate
        self.decoder = decoder
        self.lm = lm
        self.dct = dct
        self.hmm = hmm
        self.ctl_filepath = ctl_filepath
        self.outputfile = outputfile
        self.raw_text = None

    def _get_decoder_cmd(self):
        ''' Constructs  decoder command according to attr `self.decoder` '''
        assert self.decoder, "No decoder option passed!"
        if self.decoder == 'pocketsphinx_continuous':
            cmd = [str(x) for x in [self.decoder, '-samprate',
                                    self.framerate, '-lm', self.lm, '-dict',
                                    self.dct, '-hmm', self.hmm, '-infile',
                                    self.filepath]]
        if self.decoder == 'pocketsphinx_batch':
            """
            pocketsphinx_batch \
                -samprate 8000 \
                -adcin yes \
                -lm /home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.lm \
                -dict /home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.dic \
                -hmm n/zero_ru_cont_8k_v2/zero_ru.cd_semi_4000 \
                -cepext .wav \
                -cepdir / \
                -ctl /tmp/bla
            """
            # cut extension of audio files
            filepathsplit = self.filepath.split('.')
            ext = "." + filepathsplit[-1]
            filepath_withoutext = ".".join(filepathsplit[:-1])
            # write out control file with list of audio files to process
            with open(self.ctl_filepath, 'wt') as ctl:
                ctl.write(filepath_withoutext)
            cmd = [str(x) for x in [self.decoder, '-samprate', self.framerate,
                                    '-adcin', 'yes', '-lm', self.lm, '-dict',
                                    self.dct, '-hmm', self.hmm, '-cepext',
                                    ext, '-cepdir', '/', '-ctl',
                                    self.ctl_filepath, '-hyp',
                                    self.outputfile]]
        return cmd

    @property
    def text(self):
        if not self.raw_text:
            # get decoder command to execute
            cmd = self._get_decoder_cmd()
            self.log.info("Executing command: {}".format(cmd))
            pock_cont = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, bufsize=1)
            # wait for subprocess termination
            self.log.debug("Waiting for termination of recognition process...")
            returncode = pock_cont.wait()

            if returncode:
                self.log.error("Bad exit code from recognition process. "
                               "Exit code: {}".format(returncode))
                self.log.error(pock_cont.stderr.read())
                # FIXME: decide if following is useful
                # raise Exception("Recognition error: non null exit code")
            self.log.info("Recognition completed. Stderror follows:")
            self.log.info(pock_cont.stderr.read())
            # get only the meaningful part of a recognition data
            # depending on a type of decoder
            if self.decoder == 'pocketsphinx_continuous':
                reg = '\d+: (?P<text>.*)'
                lines = pock_cont.stdout.read().splitlines()
                try:
                    self.raw_text = ' '.join([re.search(reg, line).group('text')
                                              for line in lines])
                except AttributeError:
                    self.log.error("There was an error while parsing results"
                                   " from decoder.See raw result %s", lines)
            if self.decoder == 'pocketsphinx_batch':
                raw_text = open(self.outputfile, 'rt').read()
                # delete final () construction from decoder output
                raw_text = re.sub(' \(.*', '', raw_text)
                self.raw_text = raw_text.replace("\n", " ")
        return self.raw_text
