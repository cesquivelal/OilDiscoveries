# OilDiscoveries

Replication code for "The Sovereign Default Risk of Giant Oil Discoveries" by Carlos Esquivel

January, 2023:

https://cesquivelal.github.io/EsquivelGOFD.pdf

# Data

The folder Data contains the panel data used in Section 2 and STATA code for all regression exercises.

# Quantitative solution of model

The code is written in the Julia language, version 1.7.2 and uses the following packages:
      Distributed, Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots, NLsolve, Plots

The file Primitives.jl defines all objects and functions that are used to solve and simulate the model.

The file mainSolve.jl solves the model and creates .csv files with all the information that characterizes the solution.

The file ResultsForPaper.jl generates the results in the quantitative analysis in Section 4.

The folders Best, Best_highRho, Best_patient, and Best_sameVol contain .csv files with the model solution for the benchmark calibration as well as each of the counter-factual exercises. The file ResultsForPaper.jl can generate results using any of these exercises as a source.
