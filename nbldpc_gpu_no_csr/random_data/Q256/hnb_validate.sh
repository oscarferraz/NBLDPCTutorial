#!/bin/bash
lines=0

#==========================================================================
while IFS= read -r line
do
	for one_thing in $line; do
    		echo $one_thing >> symbol_llr_vertical.txt
	done

	
done < symbol_llr.txt
#==========================================================================
#lines=0
#while IFS= read -r line
#do
#	NUMBER=$(echo "$line" | grep -o -E '[1-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5]')
#	lines2=$(echo "$NUMBER" | wc -w)
#	echo "$lines"
#	let "lines=lines+lines2"
#
#	
#done < hnb.txt
#
#if [ $lines -ne $nnz ]
#then
#	echo "ERROR: number of symbols not equal in H matrix "$lines
#	exit 1
#else
#	echo "number of symbols equal to H matrix"
#fi

#==========================================================================
#rm temp_val.txt
#while IFS= read -r line
#do
#	NUMBER=$(echo "$line" | grep -o -E '[1-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5]')
#	#echo $NUMBER
#	for word in $NUMBER
#	do
#		echo -e "$word"  >> temp_val.txt
#	done 
#done < hnb.txt

lines=0
paste temp_val.txt val.txt | while IFS="$(printf '\t')" read -r f1 f2
do

	#echo ${f1//[[:blank:]]/}
	#echo ${f2//[[:blank:]]/}
	#echo $f1
	#echo $f2

	if [[ $f1 -eq $f2 ]]
	then
		let "lines=lines+1"
	else
		echo "ERROR: val not equal in H matrix "$lines
		exit 1
	fi
	
done
echo "val is equal to H matrix"
