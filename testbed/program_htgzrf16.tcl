open_hw_manager
connect_hw_server -allow_non_jtag

open_hw_target {localhost:3121/xilinx_tcf/Xilinx/00001ced7db201}
set_property PROBES.FILE {} [get_hw_devices xczu49dr_0]
set_property FULL_PROBES.FILE {} [get_hw_devices xczu49dr_0]
set_property PROGRAM.FILE {/home/zhizhenzhong/Documents/Bitstreams/HTG_ZRF16.bit} [get_hw_devices xczu49dr_0]
program_hw_devices [get_hw_devices xczu49dr_0]
refresh_hw_device [lindex [get_hw_devices xczu49dr_0] 0]

puts "FPGA Programmed!"
exit