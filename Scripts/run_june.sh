set -e
#declare -a weeks=('202014' '202016' '202018' '202020' '202022' '202024' '202026' '202028' '202030')
declare -a weeks=('202014')
declare -a params=()

END=10
for ((i=1;i<=END;i++)); do
    echo $i
    python -u main.py -d cpu -ew 202014 --seed 15234 -m june -di COVID -id 2020-03-16 -nw 8
done 
