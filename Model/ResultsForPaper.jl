
using Distributed
using Plots; pyplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                    background_color_legend=nothing,foreground_color_legend=nothing,
                    legendfontsize=18,guidefontsize=18,titlefontsize=20,tickfontsize=18,
                    markersize=9,size=(650,500))

FOLDER="Best"
include("$FOLDER\\Primitives.jl")
FOLDER_GRAPHS="Graphs"
β=0.86447265625; d0=-0.4177734375; d1=0.5822784810126582
α0=17.255859375; φ=2.0
par, GRIDS=Setup_MomentMatching(β,d0,d1,α0,φ)
SOL=Unpack_Solution(FOLDER,GRIDS,par)
MOMENTS=AverageMomentsManySamples(SOL,GRIDS,par)

##################################################################
### Default probabilities 10 years after discovery vs 10 years ###
##################################################################
#BenchAlt3: 2.17 - 2.20
#BenchAlt: 3.00 - 3.26
#benchmark: 2.08 - 2.19
#same volatility: 4.38 - 5.22
#patient: 0.40 - 0.45
#options: 2.88 - 3.02
#high cost: 0.00 - 0.00
U_PrDef10, D_PrDef10=DefaultProbabilitiesAfterDisc(SOL,GRIDS,par)

##################################################################
######### Plots of responses to oil discoveries vs data ##########
##################################################################
par=Pars(par,NSamplesPaths=10000)
PATHS=GetAverageDifferenceInPaths(SOL,GRIDS,par)
FOLDER_DATA=" "
PlotResponsesVsData(FOLDER_GRAPHS,FOLDER_DATA,PATHS,SOL,GRIDS,par)

##################################################################
######### Figure 7: Transition of size of oil field ##########
##################################################################
nTS=Array{Float64,1}(undef,length(PATHS.n))
t=-1
for i in 1:length(PATHS.n)
    if t<=par.Twait
        nTS[i]=par.nL
    else
        nTS[i]=par.nH
    end
    t=t+1
end
t0=-1
t1=par.Tpaths-2
plt_n=plot([t0:t1],nTS[1:end-2],legend=false,marker="-o",
     xticks = t0:1:t1,
     ylabel="n(t)",xlabel="t")
savefig(plt_n,"$FOLDER_GRAPHS\\Figure7.pdf")

##################################################################
####### Figure 8: Transition of borrowing and investment #########
##################################################################
PATHS_LEV=GetAveragePathsAfterDiscovery(SOL,GRIDS,par)
PlotFigure8_Transition_kb(FOLDER_GRAPHS,PATHS_LEV,GRIDS,par)

##################################################################
####### Figure 9: Price schedule of bonds #########
##################################################################
PlotFigure9_PriceSchedule(FOLDER_GRAPHS,SOL,GRIDS,par)

##################################################################
###### Figure 10: Gap between value of repayment and def. ########
##################################################################
PlotFigure10_GapVP_VD(FOLDER_GRAPHS,SOL,GRIDS,par)

##################################################################
###### Figure 11: Spreads and risk premium ########
##################################################################
Plot_SigmaT_and_RP(FOLDER_GRAPHS,PATHS,PATHS_LEV,GRIDS,par)

##################################################################
###### Welfare analysis ########
##################################################################
#Welfare gains of an oil discovery
#benchmark: 0.436
#same volatility: 0.445
#patient: 0.655
#options: 0.604
#high cost: 0.269
AvWG_Discovery(SOL,GRIDS,par)

#Volatility of consumption and traded income
using Distributed
FOLDER="Best_patient"
include("$FOLDER\\Primitives.jl")
β=0.86447265625; d0=-0.4177734375; d1=0.5822784810126582
α0=17.255859375; φ=2.0
par, GRIDS=Setup_MomentMatching(β,d0,d1,α0,φ)
SOL=Unpack_Solution(FOLDER,GRIDS,par)
Table4Calculations(SOL,GRIDS,par)

##################################################################
###### Compare same-volatility vs. benchmark spreads ########
##################################################################
using Distributed
FOLDER_BEN="Best"
include("$FOLDER_BEN\\Primitives.jl")
β=0.86447265625; d0=-0.4177734375; d1=0.5822784810126582
α0=17.255859375; φ=2.0
par_BEN, GRIDS_BEN=Setup_MomentMatching(β,d0,d1,α0,φ)
SOL_BEN=Unpack_Solution(FOLDER_BEN,GRIDS_BEN,par_BEN)
par_BEN=Pars(par_BEN,NSamplesPaths=1000)
PATHS_BEN=GetAverageDifferenceInPaths(SOL_BEN,GRIDS_BEN,par_BEN)

FOLDER_VOL="Best_sameVol"
include("$FOLDER_VOL\\Primitives.jl")
par_VOL, GRIDS_VOL=Setup_MomentMatching(β,d0,d1,α0,φ)
SOL_VOL=Unpack_Solution(FOLDER_VOL,GRIDS_VOL,par_VOL)
par_VOL=Pars(par_VOL,NSamplesPaths=1000)
PATHS_VOL=GetAverageDifferenceInPaths(SOL_VOL,GRIDS_VOL,par_VOL)

#Plot Spreads
tend=15
t0=0
t1=tend
#Details for plots
size_width=600
size_height=400
SIZE_PLOTS=(size_width,size_height)
LW=3.0
#Make plot
BEN=PATHS_BEN.Spreads[2:t1+2] .- PATHS_BEN.Spreads[1]
VOL=PATHS_VOL.Spreads[2:t1+2] .- PATHS_VOL.Spreads[1]
plt_spreads_vol=plot([t0:t1],[BEN, VOL],label=["benchmark" "same-volatility"],
    linestyle=[:solid :dash],
    ylabel="percentage points",xlabel="t",ylims=[-0.05,0.8],
    legend=:bottomright,size=SIZE_PLOTS,linewidth=LW)
FOLDER_GRAPHS="GraphsForPaper"
savefig(plt_spreads_vol,"$FOLDER_GRAPHS\\Figure12.pdf")
