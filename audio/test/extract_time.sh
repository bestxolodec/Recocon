logdir="/home/ipaulo/recocon/batches/logs/"
outdir=$logdir/time

mkdir -p $outdir
for decoder in pocketsphinx_continuous pocketsphinx_batch; do
    mkdir  -p $outdir/$decoder
    for model in  voxforge cd_cont_4000 cd_ptm_4000 cd_semi_4000 ; do
        mkdir  -p $outdir/$decoder/$model
        for speed in 0x 1x 2x 3x 4x 5x; do
            for i in $(ls $logdir | grep log$ | grep $decoder | grep $model | grep $speed ); do
                filename=$logdir/$i
                output_file=$outdir/$decoder/$model/result
                # construct string '1x WERVALUE'
                paste <(echo $speed) <(awk '/wall.*xRT/ { sum += $4  } END {print sum }'  $filename)  >> $outdir/$i
            done
        done
    done
done
