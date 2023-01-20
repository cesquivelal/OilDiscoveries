# OilDiscoveries

Replication code for "The Sovereign Default Risk of Giant Oil Discoveries" by Carlos Esquivel

January, 2023:

https://cesquivelal.github.io/EsquivelGOFD.pdf

# Data

The folder Data contains the panel data used in Section 2 in file Oil_Spreads_Macro.csv and STATA code for all regression exercises.

# Quantitative solution of model

The code is written in the Julia language, version 1.7.2 and uses the following packages:
      Distributed, Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots, NLsolve, Plots

The file Primitives.jl defines all objects and functions that are used to solve and simulate the model.

The file mainSolve.jl uses Primitives.jl to solve the model and create .csv files with all the information and data that characterize the solution.

The file ResultsForPaper.jl generates the results in the quantitative analysis in Section 4. This file uses the .csv solution files created by mainSolve.jl and requires the user to specify the folder in which these files are.
