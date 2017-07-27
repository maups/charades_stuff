set term png size 2500,1500
set title "TRAINING RESULTS" font ",20"
set key right
set xrange [ 1 : 3000 ]
set yrange [ 0 : 70 ]
set grid

set output "show.png"

plot	'train.dat' using 1:2 t 'TRAINING LOSS' with lines lw 1 lc 1, \
	'train.dat' using 1:($3*100) t 'TRAINING RANK-1' with lines lw 1 lc 2, \
	'train.dat' using 1:2 notitle with lines smooth bezier lw 3 lc 8, \
	'train.dat' using 1:($3*100) notitle with lines smooth bezier lw 3 lc 8, \
	'val.dat' using 1:($2/5) t 'VALIDATION LOSS' with lines lw 1 lc 3, \
	'val.dat' using 1:3 t 'VALIDATION RANK-1' with lines lw 1 lc 4, \
	'val.dat' using 1:4 t 'VALIDATION RANK-5' with lines lw 1 lc 5, \
	'val.dat' using 1:($2/5) notitle with lines smooth bezier lw 3 lc 8, \
	'val.dat' using 1:3 notitle with lines smooth bezier lw 3 lc 8, \
	'val.dat' using 1:4 notitle with lines smooth bezier lw 3 lc 8
