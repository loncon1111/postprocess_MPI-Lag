#!/bin/bash

module load anaconda3/5.1.0_gnu_64

sdate="20170721_06"
edate="20170722_12"

date=$sdate

until [ ${date} == ${edate} ]; do 
   date=`./newtime ${date} 6`
   echo $date
   python more_lay_filtered.py -i ${date}
done
