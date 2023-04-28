# Density-Evolution-for-relay-flat-fading-channel-
The GitHub repository contains a tool for generating the weight distribution of Low-Density Parity-Check (LDPC) codes using Density Evolution for a relay flat fading channel. This tool was originally developed by Alexandre de Baynast and can be found at https://www.ece.rice.edu/~debaynas/codes.html.

Please note that due to changes in the implementation of the "lsqlin" function, the tool only works until MATLAB R2016 (active-set algorithm removed). If you are using a newer version of MATLAB, you will need to either use an older version or rewrite the code to use other solvers such as CVX, CPLEX, GUROBI, or others.
