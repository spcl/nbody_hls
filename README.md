This project implements the N^2 formulation of N-body simulation.

Since the project requires reading sets of 3-dimensional coordinates (12
bytes), a Vivado IP is used for the data width conversion. This is done by using
the Vitis kernel wizard, then importing the HLS sources into the project, and
packaging this as an RTL kernel.

Steps to build kernel:
- `make synthesize_nbody`
- `make setup_project`
- `make package_kernel`
- `make build_kernel`

_This project will only work with Vitis (i.e., not SDx or SDAccel), and has
only been tested with Vitis 2019.2._
