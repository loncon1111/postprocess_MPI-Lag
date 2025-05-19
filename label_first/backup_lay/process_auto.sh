#!/bin/bash

module load anaconda3/5.1.0_gnu_64

sdate="20170725_12"
edate="20170729_00"

date=$sdate

until [ ${date} == ${edate} ]; do 
   date=`./newtime ${date} 6`
   echo $date
   python more_lay_filtered.py -i ${date}
done
