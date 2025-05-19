#!/bin/bash
#module load anaconda3/2021.05

#cyclones="TALAS SONCA ROKE NESAT HAITANG"
#dates=(20170715_00 20170720_18 20170721_18 20170725_18 20170728_00)

#cyclones="ROKE NESAT HAITANG"
#dates=(20170721_18 20170725_18 20170728_00)

#cyclones="HAITANG"
#dates=(20170728_00)

cyclones="ROKE"
dates=(20170721_06)

count=0
for cyclone in $cyclones; do
   echo "$count $cyclone" 
   date=${dates[$((count++))]}
   python traject_clustering.py -c $cyclone -i $date 
done

