#/bin/sh

rm data/times.txt

for i in {1..21}
do	
	echo "run"
	echo $((i)) 

	taskset --cpu-list 1 ./min_max2
done

mv data/times.txt data/times_denver_25.txt

rm data/times.txt

for i in {1..21}
do	
	echo "run"
	echo $((i)) 

	taskset --cpu-list 1 ./min_max
done

mv data/times.txt data/times_denver_50.txt


