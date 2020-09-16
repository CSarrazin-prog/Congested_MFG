# Congested_MFG
A code to approximate solutions to variational mean field games with congestion penalization, using discrete probability measures on the space of continuous trajectories. 

To run, this code requires the following libraries, all available with anaconda:

  -the pysdot library for the integration of radial functions over Laguerre cells. See:https://github.com/sd-ot/pysdot.git
  -the agd library for Hamiltonian Fast Marching, only for the non-convex example. See:https://github.com/Mirebeau/HamiltonFastMarching.git
  -the scipy library for approximating the minimizers of a functional, using an L-BFGS algorithm.
