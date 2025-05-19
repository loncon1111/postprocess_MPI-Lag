
for istart in {60..640..20}; do
    rm dtw_groups_1.py
    sed "s/xxxx/${istart}/g" dtw_groups.py > dtw_groups_1.py
    python dtw_groups_1.py 
done
