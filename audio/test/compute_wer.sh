program="perl /home/ipaulo/recocon/sphinxtrain/scripts/decode/word_align.pl"
program="python /home/ipaulo/recocon/sphinxtrain/scripts/decode/word_align.py"
ref="/home/ipaulo/recocon/batches/logs/text/ref"
gypot="/home/ipaulo/recocon/batches/logs/text/batch0x.wav_pocketsphinx_batch_zero_ru_cont_zero_ru.cd_cont_4000.log"
dir="/home/ipaulo/recocon/batches/logs/text/"

for decoder in pocketsphinx_continuous pocketsphinx_batch; do
    mkdir  -p $dir/$decoder
    for model in  voxforge cd_cont_4000 cd_ptm_4000 cd_semi_4000 ; do
        mkdir  -p $dir/$decoder/$model
        for speed in 0x 1x 2x 3x 4x 5x; do
            for i in $(ls $dir | grep $decoder | grep $model | grep $speed ); do
                gypot=$dir/$i
                result=$dir/$decoder/$model/result
                # construct string '1x WERVALUE'
                paste <(echo $speed) <($program $ref $gypot | awk '/WER/ {print $2}') >> $result
            done
        done
    done
done
