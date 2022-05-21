#!/bin/bash

rm tempos.txt

for i in {1..20}
do


	sudo /usr/local/cuda-9.0/bin/nvprof --log-file output.log ./min_max

	grep -F "GPU_FB_metrics" output.log| grep -q "GPU activities" && found1=1 || found1=0 
	grep -F "GPU_CN" output.log| grep -q "GPU activities" && found2=1 || found2=0 
	grep -F "GPU_VN" output.log| grep -q "GPU activities" && found3=1 || found3=0 
	grep -F "[CUDA memset]" output.log| grep -q "GPU activities" && found4=1 || found4=0 
	grep -F "[CUDA memcpy HtoD]" output.log| grep -q "GPU activities" && found5=1 || found5=0 
	grep -F "[CUDA memcpy DtoH]" output.log| grep -q "GPU activities" && found6=1 || found6=0 

	string=`grep -F "GPU_FB_metrics" output.log|awk '{print $2}'`
	if [ "$found1" -eq "1" ];
	then
		string=`grep -F "GPU_FB_metrics" output.log|awk '{print $4}'`
 	fi
	echo -n ${string::-2} >> tempos.txt


	string=`grep -F "GPU_CN" output.log|awk '{print $2}'`
	if [ "$found2" -eq "1" ];
	then
		string=`grep -F "GPU_CN" output.log|awk '{print $4}'`
 	fi
	echo -n -e "\t" >> tempos.txt
	echo -n ${string::-2} >> tempos.txt

	string=`grep -F "GPU_VN" output.log|awk '{print $2}'`
	if [ "$found3" -eq "1" ];
	then
		string=`grep -F "GPU_VN" output.log|awk '{print $4}'`
 	fi
	echo -n -e "\t" >> tempos.txt
	echo -n ${string::-2} >> tempos.txt


	string=`grep -F "[CUDA memset]" output.log|awk '{print $2}'`
	if [ "$found4" -eq "1" ];
	then
		string=`grep -F "[CUDA memset]" output.log|awk '{print $4}'`
 	fi
	echo -n -e "\t" >> tempos.txt
	echo -n ${string::-2} >> tempos.txt



	string=`grep -F "[CUDA memcpy HtoD]" output.log|awk '{print $2}'`
	if [ "$found5" -eq "1" ];
	then
		string=`grep -F "[CUDA memcpy HtoD]" output.log|awk '{print $4}'`
 	fi
	echo -n -e "\t" >> tempos.txt
	echo -n ${string::-2} >> tempos.txt


	string=`grep -F "[CUDA memcpy DtoH]" output.log|awk '{print $2}'`
	if [ "$found6" -eq "1" ];
	then
		string=`grep -F "[CUDA memcpy DtoH]" output.log|awk '{print $4}'`
 	fi
	echo -n -e "\t" >> tempos.txt
	echo ${string::-2} >> tempos.txt


done

