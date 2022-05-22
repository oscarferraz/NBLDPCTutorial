############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_top -name minmax "minmax"
set_directive_dependence -type inter -dependent true "VN"
set_directive_pipeline -II 4 "VN/VN_label2"
set_directive_shared "minmax" BETAMn
set_directive_shared "minmax" ALPHAm1
set_directive_array_partition -type complete -dim 0 "CN" add
set_directive_array_partition -type complete -dim 0 "CN" mult
set_directive_array_partition -type complete -dim 1 "CN" inv
set_directive_array_partition -type block -factor 16 -dim 1 "CN" val
set_directive_array_partition -type complete -dim 1 "VN" ALPHA_t
set_directive_array_partition -type complete -dim 0 "CN" B
set_directive_array_partition -type complete -dim 0 "CN" F
set_directive_pipeline -II 8 -enable_flush "CN/CN_label1"
set_directive_bind_storage -type rom_np "CN" row_ptr
set_directive_dependence -variable B -class array -type inter -dependent true "CN"
set_directive_dependence -variable BETAMn -class array -type inter -dependent false "CN"
set_directive_inline -recursive "minmax/MINMAX"
set_directive_array_partition -type complete -dim 2 "minmax" BETAMn
set_directive_array_partition -type block -factor 8 -dim 1 "minmax" BETAMn
set_directive_array_partition -type complete -dim 2 "CN" ALPHAm
set_directive_array_partition -type block -factor 8 -dim 1 "CN" ALPHAm
set_directive_interface -mode s_axilite "minmax" GAMMAN
set_directive_interface -mode s_axilite -depth 11250 "minmax" ALPHAm
set_directive_array_partition -dim 2 -type complete "minmax" ALPHAm1
set_directive_array_partition -dim 1 -factor 8 -type block "minmax" ALPHAm1
