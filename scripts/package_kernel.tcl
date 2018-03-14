source config.tcl
set prjDir $workspace/${kernelName}_ex

open_project $prjDir/${kernelName}_ex.xpr

source $prjDir/imports/package_kernel.tcl

close_project
