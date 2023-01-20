using Distributed
addprocs(39)

@everywhere include("Primitives.jl")

β=0.86447265625; d0=-0.4177734375; d1=0.5822784810126582
α0=17.255859375; φ=2.0
par, GRIDS=Setup_MomentMatching(β,d0,d1,α0,φ)
SolveAndSaveModel_VFI(GRIDS,par)
