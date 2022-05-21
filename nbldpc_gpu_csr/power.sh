#!/bin/bash

rm power.txt

while true
do


	cat /sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input

	sleep 0.5


done

