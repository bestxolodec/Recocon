#!/usr/bin/python
from __future__ import division
from audio.Lecture import Lecture


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
    parser.add_argument("-lm", dest="lm", action="store",
                        default="/home/ipaulo/recocon/"
                        "zero_ru_cont_8k_v2/zero_ru.lm",
                        help="Path to language model file. Required for"
                        " recognition process. Defaults to /home/ipaulo/"
                        "recocon/zero_ru_cont_8k_v2/zero_ru.lm")
    parser.add_argument("-dct", dest="dct", action="store",
                        default="/home/ipaulo/recocon/"
                        "zero_ru_cont_8k_v2/zero_ru.dic",
                        help="Path to language dictionary file. "
                        " Defaults to /home/ipaulo/recocon/"
                        "zero_ru_cont_8k_v2/zero_ru.dic")
    parser.add_argument("-hmm", dest="hmm", action="store",
                        default="/home/ipaulo/recocon/"
                        "zero_ru_cont_8k_v2/zero_ru.cd_semi_4000",
                        help="Path to directory, where parameters of Hidden"
                        "Markov Model are stored. "
                        "Defaults to /home/ipaulo/recocon/"
                        "zero_ru_cont_8k_v2/zero_ru.cd_semi_4000")
    parser.add_argument("-decoder", dest="decoder", action="store",
                        default="pocketsphinx_continuous",
                        help="Program name of decoder, that is found in $PATH"
                        ". Defaults to `pocketsphinx_continuous`.")
    parser.add_argument("-fr", dest="framerate", action="store",
                        default=8000, type=int,
                        help="Framerate to which convert the video_filepath"
                        " file. Defaults to 8000")
    args = parser.parse_args()

    print args
    lecture = Lecture(args.video_filepath, args.tmpdir, decoder=args.decoder,
                      lm=args.lm, dct=args.dct, hmm=args.hmm)
    if not args.noconvert:
        lecture.video_to_audio(framerate=args.framerate, bitrate="256k")
    # lecture.silence_split()
    lecture.plot_params()
    print lecture.get_full_text()
    import cPickle as pic
    pic.dump(lecture.chunks, open("./chunks.pickle", 'wb'))
