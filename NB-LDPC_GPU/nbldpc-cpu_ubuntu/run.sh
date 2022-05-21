#/bin/sh

rm data/times.txt

for i in {1..20}
do	
	echo "run"
	echo $((i)) 

	taskset --cpu-list 2 ./min_max
done
