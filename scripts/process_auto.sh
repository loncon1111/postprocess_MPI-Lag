#!/bin/bash

sdate="20170728_00"
edate="20170728_06"

date=$sdate
echo $date
until [ ${date} == ${edate} ]; do 
   echo "python new_lcs.py -i $date"
   python new_lcs.py -i ${date}
   date=`./newtime ${date} 6`
done
