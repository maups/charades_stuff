#!/bin/bash

mkdir Charades_v1_rgb_scaled
jump=24

for i in $(ls Charades_v1_rgb)
do
	mkdir Charades_v1_rgb_scaled/$i
	j=1
	name=$(printf "%s-%06d.jpg" $i $j)
	while [ -f Charades_v1_rgb/$i/$name ]
	do
		#echo $name

		convert -resize 128x128! Charades_v1_rgb/$i/$name Charades_v1_rgb_scaled/$i/$name

		j=$((j+jump))
		name=$(printf "%s-%06d.jpg" $i $j)
	done

#	break
done

