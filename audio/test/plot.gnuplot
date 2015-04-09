#!/usr/bin/env gnuplot

cont = "/home/ipaulo/recocon/batches/logs/text/pocketsphinx_batch/cd_cont_4000/result"
semi = "/home/ipaulo/recocon/batches/logs/text/pocketsphinx_batch/cd_semi_4000/result"
ptm  = "/home/ipaulo/recocon/batches/logs/text/pocketsphinx_batch/cd_ptm_4000/result"
vox = "/home/ipaulo/recocon/batches/logs/text/pocketsphinx_batch/voxforge/result"

set terminal pngcairo enhanced size 800,   600
set key left top
set key outside
set xlabel "Скорость речи"
set ylabel "WER, %"
set grid
set encoding utf8

set title "CMU Sphinx, batch mode"
set output 'wer_batch.png'
plot cont using 2:xtic(1)  with lines  lw 2 title "Морфемная, непрерывная", \
     semi using 2:xtic(1)  with lines  lw 2 title "Морфемная, полу-непрерывная", \
     ptm using 2:xtic(1)   with lines  lw 2 title "Морфемная, фонетически связанная", \
     vox using 2:xtic(1)   with lines  lw 2 title "Словесная, малая база"




cont = "/home/ipaulo/recocon/batches/logs/text/pocketsphinx_continuous/cd_cont_4000/result"
semi = "/home/ipaulo/recocon/batches/logs/text/pocketsphinx_continuous/cd_semi_4000/result"
ptm  = "/home/ipaulo/recocon/batches/logs/text/pocketsphinx_continuous/cd_ptm_4000/result"
vox = "/home/ipaulo/recocon/batches/logs/text/pocketsphinx_continuous/voxforge/result"


set title "CMU Sphinx, continuous mode"
set output 'wer_continuous.png'
plot cont using 2:xtic(1)  with lines  lw 2 title "Морфемная, непрерывная", \
     semi using 2:xtic(1)  with lines  lw 2 title "Морфемная, полу-непрерывная", \
     ptm using 2:xtic(1)   with lines  lw 2 title "Морфемная, фонетически связанная", \
     vox using 2:xtic(1)   with lines  lw 2 title "Словесная, малая база"
