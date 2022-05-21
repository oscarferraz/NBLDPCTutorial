#!/bin/bash
lines=0
while IFS= read -r line
do
	if [ $lines -eq 0 ]
	then
		M=$line
	fi

	if [ $lines -eq 1 ]
	then
		N=$line
	fi

	if [ $lines -eq 2 ]
	then
		col_weight=$line
	fi

	if [ $lines -eq 3 ]
	then
		row_weight=$line
	fi

	if [ $lines -eq 4 ]
	then
		nnz=$line
	fi

	let "lines=lines+1"
done < params.txt

echo "rows="$M
echo "columns="$N
echo "symbols per row="$row_weight
echo "symbols per column="$col_weight
echo "symbols="$nnz

#==========================================================================
#lines=0
#while IFS= read -r line
#do
#	let "lines=lines+1"
#done < hnb.txt
#
#if [ $lines -ne $M ]
#then
#	echo "ERROR: number of lines not equal in H matrix "$lines
#	exit 1
#else
#	echo "number of lines equal to H matrix"
#fi

#==========================================================================
#while IFS= read -r line
#do
#	NUMBER=$(echo "$line" | grep -o -E '[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5]')
#	lines=$(echo "$NUMBER" | wc -w)
	#echo "$lines"
#	if [ $lines -ne $N ]
#	then
#		echo "ERROR: number of columns not equal in H matrix"
#		exit 1	
#	fi
#	
#done < hnb.txt
#echo "number of columns equal to H matrix"
#==========================================================================
#while IFS= read -r line
#do
#	NUMBER=$(echo "$line" | grep -o -E '[1-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5]')
#	lines=$(echo "$NUMBER" | wc -w)
#	if [ $lines -ne $row_weight ]
#	then
#		echo "ERROR: number of symbols per row not equal in H matrix"
#		exit 1
#	fi
#done < hnb.txt
#echo "symbols per row equal to H matrix"
#==========================================================================
#for (( c=1; c<=$N; c++ ))
#do
#	echo "$c"
#	lines=$(cat hnb.txt | awk -v x=$c '{print $x}' | grep -o -E '[1-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5]'|wc -w)
#	if [ $lines -ne $col_weight ]
#	then
#		echo "ERROR: number of symbols per column not equal in H matrix"
#		exit 1
#	fi
#done
#echo "symbols per column equal to H matrix"
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
rm temp_val.txt
