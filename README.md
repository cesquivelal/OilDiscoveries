# OilDiscoveries

Replication code for "The Sovereign Default Risk of Giant Oil Discoveries" by Carlos Esquivel

March, 2025:

https://cesquivelal.github.io/EsquivelGOFD.pdf

# Data

The file EsquivelOilDiscoveries_data.csv contains the panel data used in Section 2.
The file EmpiricalResults.do runs all regressions in STATA and saves the results in the file Regressions_Benchmark.xls (this file has to be then manually saved as a .xlsx file)
The file EmpiricalResults.jl has functions to access the results saved in Regressions_Benchmark.xlsx (note the file extension is .xslx) and create Figures 2, 3, and 4.

# Model

The code is written in the Julia language, version 1.7.2 and uses the following packages:
      Distributed, Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, QuadGK, Sobol, Roots, NLsolve, Plots

The file ModelPrimitives.jl defines all objects and functions that are used to solve and simulate the model.
The file ModelResults.jl has functions to use the model solution and simulations to create Tables 2, 3 and 4, and Figures 5 through 10.

# Paper results

The file PaperResults.jl uses ModelPrimitives.jl, EmpiricalResults.jl, and ModelResults.jl to solve all versions of the model and generate all the results in the paper.
