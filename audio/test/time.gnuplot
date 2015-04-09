#!/usr/bin/env gnuplot

cont = "/home/ipaulo/recocon/batches/logs/time/pocketsphinx_batch/cd_cont_4000/result"
semi = "/home/ipaulo/recocon/batches/logs/time/pocketsphinx_batch/cd_semi_4000/result"
ptm  = "/home/ipaulo/recocon/batches/logs/time/pocketsphinx_batch/cd_ptm_4000/result"
vox = "/home/ipaulo/recocon/batches/logs/time/pocketsphinx_batch/voxforge/result"

set terminal pngcairo enhanced size 800,   600
set key out vert
set key bot center
set xlabel "Скорость речи"
set ylabel "Затраченное время на распознавание, сек"
set grid
set encoding utf8

set title "CMU Sphinx, batch mode"
set output 'time_batch.png'
plot cont using 2:xtic(1)  with lines  lw 2 title "Морфемная, непрерывная", \
     semi using 2:xtic(1)  with lines  lw 2 title "Морфемная, полу-непрерывная", \
     ptm using 2:xtic(1)   with lines  lw 2 title "Морфемная, фонетически связанная", \
     vox using 2:xtic(1)   with lines  lw 2 title "Словесная, малая база, 16 kHz"

set title "CMU Sphinx, batch mode"
set output 'time_batch_zoomed.png'
plot ptm using 2:xtic(1)   with lines  lw 2 title "Морфемная, фонетически связанная", \
     vox using 2:xtic(1)   with lines  lw 2 title "Словесная, малая база, 16 kHz"




cont = "/home/ipaulo/recocon/batches/logs/time/pocketsphinx_continuous/cd_cont_4000/result"
semi = "/home/ipaulo/recocon/batches/logs/time/pocketsphinx_continuous/cd_semi_4000/result"
ptm  = "/home/ipaulo/recocon/batches/logs/time/pocketsphinx_continuous/cd_ptm_4000/result"
vox = "/home/ipaulo/recocon/batches/logs/time/pocketsphinx_continuous/voxforge/result"


set title "CMU Sphinx, continuous mode"
set output 'time_continuous.png'
plot cont using 2:xtic(1)  with lines  lw 2 title "Морфемная, непрерывная", \
     semi using 2:xtic(1)  with lines  lw 2 title "Морфемная, полу-непрерывная", \
     ptm using 2:xtic(1)   with lines  lw 2 title "Морфемная, фонетически связанная", \
     vox using 2:xtic(1)   with lines  lw 2 title "Словесная, малая база, 16 kHz"
