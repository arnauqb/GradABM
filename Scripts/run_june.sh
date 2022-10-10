set -e
#declare -a weeks=('202014' '202016' '202018' '202020' '202022' '202024' '202026' '202028' '202030')
declare -a weeks=('202014')
declare -a params=()

for w in "${weeks[@]}"
do
    python -u main.py -d 6 -ew "$w" --seed 1234 -m june -di COVID -id 2020-03-16 -nw 9
done 
