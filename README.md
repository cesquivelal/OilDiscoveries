# The Sovereign Default Risk of Giant Oil Discoveries — replication package

Replication code for "The Sovereign Default Risk of Giant Oil Discoveries" by Carlos Esquivel

August, 2026: https://cesquivelal.github.io/EsquivelGOFD.pdf

## Run order

The two steps are not interchangeable. Stata writes files that Julia reads, so **Stata must
run first.**

**1. Stata.** Start Stata in this folder, or `cd` to it, and run

```stata
do Code_Data/EmpiricalResults.do
do Code_Data/FalsificationTests.do
```

Every path in the `.do` files is relative to this folder, so the working directory has to be
this folder and not `Code_Data`. `EmpiricalResults.do` writes
`Code_Data/Regressions_Benchmark.txt`; `FalsificationTests.do` estimates 1,000 replications,
takes a few minutes, and writes `Code_Data/Falsification_Draws.csv`. Both are read by the
Julia driver, and neither is shipped with the package.

`Code_Data/UnitRootTests.do` reports the panel unit root tests quoted in Section 2. It feeds
nothing else and can be run at any point.

**2. Julia.** Then, from any working directory,

```bash
julia PaperResults.jl
```

`PaperResults.jl` runs everything and writes every figure and table. It creates `Graphs/` and
`Tables/` itself.

## Requirements

**Stata**, with two user-written commands:

```stata
ssc install xtscc
ssc install outreg2
```

**Julia 1.6 or later.** The code loads, from the general registry:

```julia
using Pkg
Pkg.add(["Parameters","Interpolations","Optim","Distributions","FastGaussQuadrature",
         "Sobol","Roots","Plots","PythonPlot"])
```

and, from the standard library, `DelimitedFiles`, `Distributed`, `LinearAlgebra`, `Printf`,
`Random`, `SharedArrays`, `SparseArrays` and `Statistics`. Plots draws through the PythonPlot
backend, which needs a working matplotlib; PythonPlot installs one on first use if none is
found.

## How long it takes, and the `UseSavedFile` switch

Section I is fast.

Section III is not. `UseSavedFile` is the first line under its banner and is **`false`**, which
re-solves every model from scratch. One model takes roughly 45 minutes, and the driver solves
nine of them, so a full run from scratch is the better part of a day. Each solved model is
serialized to `Code_Model/Model_*.csv` as it finishes, about 4.4 MB each. They are not shipped.

Setting `UseSavedFile=true` reloads those files instead of re-solving, which is how to re-draw a
figure or re-write a table after a first full run. It fails if the files are not there.

## Files with code

```
PaperResults.jl              driver: runs everything, writes every figure and table
Code_Model/
  Primitives.jl              the model: parameters, grids, VFI solver, simulation
  ModelResults.jl            results and plotting functions, Tables 2, 3, 5 and 7
  Sensitivity_Technology.jl  Figures 8, 9 and 20-23, and the Table 3 variants
  Setup_Calibrated.csv       one column per model variant; calibration lives here
  CalibrationData.xlsx       source of the data targets in Table 2. Not read by the code
Code_Data/
  EsquivelOilDiscoveries_data.csv   the panel: 1993-2012, discoveries, spreads, macro series
  EmpiricalResults.do        Driscoll-Kraay panel regressions -> Regressions_Benchmark.txt
  FalsificationTests.do      1,000 reassignment draws -> Falsification_Draws.csv
  UnitRootTests.do           panel unit root tests quoted in Section 2
  EmpiricalResults.jl        reads the Stata output and draws Figures 2-4 and 11-19
  Evidence_of_Mechanism/
    ICIO_Technology.jl              builds Table 4 from the OECD ICIO
    ICIO_Technology_Panel.csv       the tidy panel Table 4 is built from
    ICIO_ImportUse_Availability.csv which country-years the OECD actually measures
    Input-Output analysis - INEGI.xlsx  source of the calibrated input-output shares.
                                        Not read by the code
```

`Build_ICIO_Panel` rebuilds `ICIO_Technology_Panel.csv` from the OECD source archives. It
downloads about 600 MB and takes some forty minutes, so the call is left commented out in the
driver and the panel it writes is included with the package instead.

## Output

`Graphs/` gets Figures 2 to 9 and 11 to 24, one PDF each, named for the figure number in the
paper. Figures 1 and 10 are not produced by code and are not in this package.

`Tables/` gets the rows of Tables 2, 3, 4, 5 and 7 as `TableN_rows.tex`, which the manuscript
reads with `\input`, and the same numbers as `TableN.csv`. Tables 1 and 6 are typeset in the
manuscript and have no file here. Two further files are written and are not printed tables:
`Table_ICIO_B0609_*`, which repeats Table 4 with oilfield services inside the oil sector, and
`Table_Decomposition_*`, the decomposition of the gains from selling oil rents.

Numbers quoted in the text but not in any table are printed to the console as the driver runs.
