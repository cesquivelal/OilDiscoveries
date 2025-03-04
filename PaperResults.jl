
using Plots; pyplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                    background_color_legend=nothing,foreground_color_legend=nothing,
                    legendfontsize=18,guidefontsize=18,titlefontsize=20,tickfontsize=18,
                    markersize=9,size=(650,500))


include("ModelPrimitives.jl")
##################################################
############# I. Empirical results ###############
##################################################
include("EmpiricalResults.jl")
FOLDER_GRAPHS="Graphs"
RegressionsFile="Regressions_Benchmark.xlsx"
npv=4.5; T=16
fig_21, fig_22=Plot_Figure_2(RegressionsFile,npv,T)
fig_21
fig_22
savefig(fig_21,"$FOLDER_GRAPHS\\Figure2.pdf")
savefig(fig_22,"$FOLDER_GRAPHS\\Figure3.pdf")

include("EmpiricalResults.jl")
plt_con=Plot_Consumption(RegressionsFile,npv,T)
savefig(plt_con,"$FOLDER_GRAPHS\\Figure4.pdf")

RegressionsOnlyEMBI="Regressions_OnlyEMBI.xlsx"
fig_2A, fig_22A=Plot_Figure_2(RegressionsOnlyEMBI,npv,T)
fig_22A
savefig(fig_22A,"$FOLDER_GRAPHS\\Figure3A.pdf")

plt_con2=Plot_Consumption(RegressionsOnlyEMBI,npv,T)
savefig(plt_con2,"$FOLDER_GRAPHS\\Figure4A.pdf")

##################################################
############# II. Model Results ##################
##################################################
include("ModelResults.jl")
country_column=2
MOD_Quant=SolveQuantitativeModel(country_column)

N=100; T_LongRun=10000
Conditional=true; Only_nL=true
MOM=AverageMomentsManySamples_LongRun(N,T_LongRun,Conditional,Only_nL,MOD_Quant)
MOM
MOM.σ_con/MOM.σ_GDP

#Parameters for simulation of paths
N=10000           #10,000 paths
Tbefore=1         #Return one period before discovery (will be benchmark for changes)
Tafter=16         #Periods after discovery, make it more than 15 for graphs to work
DropDefaults=true
TS_Quant=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_Quant)
WithData=false; Sell=false
fig_5=Plot_Figure_5(Sell,WithData,RegressionsFile,npv,T,TS_Quant)
savefig(fig_5,"$FOLDER_GRAPHS\\Figure5.pdf")

DropDefaults=false
TS_Quant_withDef=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_Quant)
fig_5_withDef=Plot_Figure_5(Sell,WithData,RegressionsFile,npv,T,TS_Quant_withDef)
savefig(fig_5_withDef,"$FOLDER_GRAPHS\\Figure5_withDefaults.pdf")

fig_6=PlotFigure_6(Tafter,TS_Quant_withDef)
savefig(fig_6,"$FOLDER_GRAPHS\\Figure6.pdf")

country_column=3
MOD_DomField=SolveQuantitativeModel(country_column)

country_column=4
MOD_OilEmbargo=SolveQuantitativeModel(country_column)

DropDefaults=true
TS_DomField=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_DomField)
TS_OilEmbargo=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_OilEmbargo)
fig_8=Plot_Figure_8(Tafter,TS_DomField,TS_OilEmbargo)
savefig(fig_8,"$FOLDER_GRAPHS\\Figure8.pdf")

DropDefaults=false
TS_DomField_dd=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_DomField)
TS_OilEmbargo_dd=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_OilEmbargo)
fig_8_dd=Plot_Figure_8(Tafter,TS_DomField_dd,TS_OilEmbargo_dd)
savefig(fig_8_dd,"$FOLDER_GRAPHS\\Figure8_withDefaults.pdf")

fig_9=PlotFigure_9(Tafter,TS_DomField_dd,TS_OilEmbargo_dd)
savefig(fig_9,"$FOLDER_GRAPHS\\Figure9.pdf")

country_column=5
MOD_Sell=SolveQuantitativeModel(country_column)
country_column=6
MOD_Sell_D=SolveQuantitativeModel(country_column)

TS_Sell_D=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_Sell_D)
include("ModelResults.jl")
DropDefaults=true; Sell=true
fig_10=Plot_Figure_5(Sell,WithData,RegressionsFile,npv,T,TS_Sell_D)
savefig(fig_10,"$FOLDER_GRAPHS\\Figure10.pdf")


Nwg=1000
wg_Ben=AverageWelfareGains(Nwg,MOD_Quant)
wg_DomField=AverageWelfareGains(Nwg,MOD_DomField)
wg_OilEmbargo=AverageWelfareGains(Nwg,MOD_OilEmbargo)
wg_Sell_D=AverageWelfareGains(Nwg,MOD_Sell_D)

npv=ValueOfField(MOD_Sell_D.par)
