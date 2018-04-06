# Configuration variables
source config.tcl
set prjPath $workspace/${kernelName}_ex/${kernelName}_ex

open_project $prjPath.xpr

# 512-bit to 96-bit memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name velocity_memory_to_kernel_converter
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {64} CONFIG.M_TDATA_NUM_BYTES {12}] [get_ips velocity_memory_to_kernel_converter]
generate_target {instantiation_template} [get_files $prjPath.srcs/sources_1/ip/velocity_memory_to_kernel_converter/velocity_memory_to_kernel_converter.xci]

# 516-bit to 128-bit memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name position_memory_to_kernel_converter
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {64} CONFIG.M_TDATA_NUM_BYTES {16}] [get_ips position_memory_to_kernel_converter]
generate_target {instantiation_template} [get_files $prjPath.srcs/sources_1/ip/position_memory_to_kernel_converter/position_memory_to_kernel_converter.xci]

# 96-bit to 512-bit memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name velocity_kernel_to_memory_converter
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {12} CONFIG.M_TDATA_NUM_BYTES {64}] [get_ips velocity_kernel_to_memory_converter]
generate_target {instantiation_template} [get_files $prjPath.srcs/sources_1/ip/velocity_kernel_to_memory_converter/velocity_kernel_to_memory_converter.xci]

# 128-bit to 516-bit memory width converter
create_ip -name axis_dwidth_converter -vendor xilinx.com -library ip -module_name position_kernel_to_memory_converter
set_property -dict [list CONFIG.S_TDATA_NUM_BYTES {16} CONFIG.M_TDATA_NUM_BYTES {64}] [get_ips position_kernel_to_memory_converter]
generate_target {instantiation_template} [get_files $prjPath.srcs/sources_1/ip/position_kernel_to_memory_converter/position_kernel_to_memory_converter.xci]

close_project
