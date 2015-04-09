
DIRECTORY="/home/ipaulo/recocon/batches"
CODE="/home/ipaulo/recocon/code/recocon.py"

# для каждого отрезка проводим эксперименты
# 	- для режимов распознавнаия pocketsphinx_batch, pocketsphinx_continuous
# 		- с разными моделями (-hmm)
# 		- с базой в 16кГц


# optional arguments:
#   -h, --help            show this help message and exit
#   -d TMPDIR, --tmpdir TMPDIR
#                         Path to tmp dir, where all chunks will be stored
#                         Defaults to
#                         /home/ipaulo/recocon/zero_ru_cont_8k_v2/chunks
#   -n, --noconvert       Do not extract audio from video, as it was done
#                         previously.
#   -lm LM                Path to language model file. Required for recognition
#                         process. Defaults to
#                         /home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.lm
#   -dct DCT              Path to language dictionary file. Defaults to
#                         /home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.dic
#   -hmm HMM              Path to directory, where parameters of HiddenMarkov
#                         Model are stored. Defaults to /home/ipaulo/recocon/zer
#                         o_ru_cont_8k_v2/zero_ru.cd_semi_4000
#   -decoder DECODER      Program name of decoder, that is found in $PATH.
#                         Defaults to `pocketsphinx_continuous`.

# create `logs` directory in the batches directory
mkdir -p $DIRECTORY/logs

# for b in $(ls $DIRECTORY | grep wav$) ; do
for b in batch3x.wav batch4x.wav batch5x.wav ; do
		echo $b
		for program in pocketsphinx_continuous; do
				echo $program

			 	#8000 kHz
			 	for hmm in zero_ru.cd_cont_4000 zero_ru.cd_ptm_4000 zero_ru.cd_semi_4000; do
			 		$CODE -d $DIRECTORY/${b}_chunks \
			 			-decoder $program \
			 			-lm /home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.lm \
			 			-dct /home/ipaulo/recocon/zero_ru_cont_8k_v2/zero_ru.dic \
			 			-hmm /home/ipaulo/recocon/zero_ru_cont_8k_v2/$hmm \
			 			$DIRECTORY/$b &> $DIRECTORY/logs/${b}_${program}_zero_ru_cont_${hmm}.log
			 	done

				# 16 kHz
				# $CODE -d $DIRECTORY/${b}_chunks \
				# 		-fr 16000 \
				# 		-decoder $program \
				# 		-lm /home/ipaulo/recocon/voxforge-ru-0.2/etc/msu_ru_nsh.lm.dmp \
				# 		-dct /home/ipaulo/recocon/voxforge-ru-0.2/etc/msu_ru_nsh.dic \
				# 		-hmm /home/ipaulo/recocon/voxforge-ru-0.2/model_parameters/msu_ru_nsh.cd_cont_1000_8gau_16000 \
				# 		$DIRECTORY/$b	&> $DIRECTORY/logs/${b}_${program}_voxforge.log
		done
done




