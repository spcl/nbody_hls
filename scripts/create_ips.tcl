# Configuration variables
source config.tcl
set prjPath $workspace/${kernelName}_ex/${kernelName}_ex

open_project $prjPath.xpr

# 512-bit to 96-bit memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name memory_to_kernel_converter
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {64} CONFIG.M_TDATA_NUM_BYTES {12}] [get_ips memory_to_kernel_converter]
generate_target {instantiation_template} [get_files $prjPath.srcs/sources_1/ip/memory_to_kernel_converter/memory_to_kernel_converter.xci]

# 96-bit to 512-bit memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name kernel_to_memory_converter
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {12} CONFIG.M_TDATA_NUM_BYTES {64}] [get_ips kernel_to_memory_converter]
generate_target {instantiation_template} [get_files /home/definelj/dev/nbody_hls/build/nbody_ex/nbody_ex.srcs/sources_1/ip/kernel_to_memory_converter/kernel_to_memory_converter.xci]

close_project
