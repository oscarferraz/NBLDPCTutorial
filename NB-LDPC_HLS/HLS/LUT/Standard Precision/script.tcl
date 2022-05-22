############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project Q4
set_top minmax
add_files Q4/header.h
add_files Q4/main.cpp
add_files -tb Q4/test.cpp
open_solution "solution1" -flow_target vivado
set_part {xcvu13p-fsga2577-1-i}
create_clock -period 3 -name default
config_export -format ip_catalog -rtl verilog -vivado_optimization_level 0 -vivado_phys_opt none -vivado_report_level 0
source "./Q4/solution1/directives.tcl"
csim_design
csynth_design
cosim_design -trace_level port
export_design -flow syn -rtl verilog -format ip_catalog
