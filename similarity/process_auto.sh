#!/bin/bash
#module load anaconda3/2021.05


sdate="20170720_12"
edate="20170721_06"

date=$sdate
echo $date
until [ ${date} == ${edate} ]; do 
   echo "python portion_trajclus.py -i $date"
   python portion_trajclus.py -i ${date}
   
   date=`./newtime ${date} 6`
done
