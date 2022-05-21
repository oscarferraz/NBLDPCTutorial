#!/bin/bash
lines=0
index=0
counter=0
zero="0"
while IFS= read -r line
do
	a=( $line )
	for i in {0..384}
	do
	   	if [[ ${a[$i]} -ne $zero ]]
		then
 			echo $i >> col_ind
			
		fi
	done
done < GF4_H





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


