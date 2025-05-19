#!/bin/bash

levs="850 800 750 700 650 600 500"

date="20170715_00"

for ilev in $levs; do

    if [ $ilev -eq 850 ]; then
       cat label_list2d_${date}_${ilev}.csv > label_list2d_${date}.csv
    else
       tail -n +2 label_list2d_${date}_${ilev}.csv >> label_list2d_${date}.csv
    fi 

done


