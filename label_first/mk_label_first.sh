#!/bin/bash

sdate="$1"
edate="$2"
tcname="SONCA"

S_DAY=`echo ${sdate} | cut -c7-8`
S_MON=`echo ${sdate} | cut -c5-6`
S_YEAR=`echo ${sdate} | cut -c1-4`
S_HOUR=`echo ${sdate} | cut -c9-10`

E_DAY=`echo ${edate} | cut -c7-8`
E_MON=`echo ${edate} | cut -c5-6`
E_YEAR=`echo ${edate} | cut -c1-4`
E_HOUR=`echo ${edate} | cut -c9-10`

hh_incre=6

echo `date -j -f "%Y%m%d%H" ${sdate} +%s` - `date  -j -f "%Y%m%d%H" ${edate} +%s`
S_DATE=`date -j -f "%Y%m%d%H%M" ${sdate}00 +%s`
E_DATE=`date  -j -f "%Y%m%d%H%M" ${edate}00 +%s`

diff=$(((E_DATE-$S_DATE)/(60*60))) 
echo $diff
ndate=$(($diff/$hh_incre + 1))
echo $ndate

HH_INC=0

for (( idate=0;idate<=ndate;idate++ )); do
    HH_INC=$((idate*hh_incre))
    I_DATE=`date -j -v+${HH_INC}H -f "%Y%m%d%H" "$sdate" +"%Y%m%d_%H"`
    echo $I_DATE
    cp label_first_template.xlsx label_first_${I_DATE}_${tcname}.xlsx
done

