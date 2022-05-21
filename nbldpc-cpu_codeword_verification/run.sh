#/bin/sh

rm data/times.txt

for i in {1..21}
do	
	echo "run"
	echo $((i)) 

	taskset --cpu-list 1 ./min_max
done
