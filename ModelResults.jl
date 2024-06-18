
using Distributed
addprocs(39) #add all available additional processors,
             #solution of model is slow and exploits parallelization

################################################################################
#### Solve model with different parameterizations and save solutions in csv ####
################################################################################
#The file "ModelPrimitives.jl" contains all functions that are used to solve
#the model and generate model simulations
@everywhere include("ModelPrimitives.jl")
@everywhere XX=readdlm("Setup.csv",',')
@everywhere VEC=XX[2:end,2]*1.0
@everywhere nHfactor=1.22
@everywhere par=UnpackParameters_Vector(nHfactor,VEC)
@everywhere GRIDS=CreateGrids(par)

@everywhere NAME="OilDiscoveries.csv"
@everywhere Parallel=true
@everywhere PrintProgress=true
@everywhere SaveProgress=true
SolveAndSaveModel_VFI(Parallel,PrintProgress,SaveProgress,NAME,GRIDS,par)

@everywhere NAME="OilDiscoveries_higherAlfa0.csv"
@everywhere VEC=XX[2:end,3]*1.0
@everywhere par=UnpackParameters_Vector(nHfactor,VEC)
@everywhere GRIDS=CreateGrids(par)
SolveAndSaveModel_VFI(Parallel,PrintProgress,SaveProgress,NAME,GRIDS,par)

@everywhere NAME="OilDiscoveries_impatient.csv"
@everywhere VEC=XX[2:end,4]*1.0
@everywhere par=UnpackParameters_Vector(nHfactor,VEC)
@everywhere GRIDS=CreateGrids(par)
SolveAndSaveModel_VFI(Parallel,PrintProgress,SaveProgress,NAME,GRIDS,par)

@everywhere NAME="OilDiscoveries_fix.csv"
@everywhere VEC=XX[2:end,5]*1.0
@everywhere par=UnpackParameters_Vector(nHfactor,VEC)
@everywhere GRIDS=CreateGrids(par)
SolveAndSaveModel_VFI(Parallel,PrintProgress,SaveProgress,NAME,GRIDS,par)

@everywhere NAME="OilDiscoveries_put.csv"
@everywhere VEC=XX[2:end,6]*1.0
@everywhere par=UnpackParameters_Vector(nHfactor,VEC)
@everywhere GRIDS=CreateGrids(par)
SolveAndSaveModel_VFI(Parallel,PrintProgress,SaveProgress,NAME,GRIDS,par)

@everywhere NAME="OilDiscoveries_riskNeutral.csv"
@everywhere VEC=XX[2:end,7]*1.0
@everywhere par=UnpackParameters_Vector(nHfactor,VEC)
@everywhere GRIDS=CreateGrids(par)
SolveAndSaveModel_VFI(Parallel,PrintProgress,SaveProgress,NAME,GRIDS,par)

@everywhere NAME="OilDiscoveries_Patient.csv"
@everywhere VEC=XX[2:end,8]*1.0
@everywhere par=UnpackParameters_Vector(nHfactor,VEC)
@everywhere GRIDS=CreateGrids(par)
SolveAndSaveModel_VFI(Parallel,PrintProgress,SaveProgress,NAME,GRIDS,par)

################################################################################
############### Compute moments from models, Table 2 and Table 3 ###############
################################################################################
#Benchmark:
FOLDER=" " #Folder where csv files with solutions are saved
NAME="OilDiscoveries.csv"
MODEL=UnpackModel_Vector(nHfactor,NAME,FOLDER)
N=1000; Tmom=300 #1000 samples with 300 periods
MOM=AverageMomentsManySamples(N,Tmom,MODEL) #Object with all moments
#Moments in Tables 2 and 3:
MOM.DefaultPr
MOM.Av_Spreads
MOM.Av_RP_Spr
MOM.σ_Spreads
MOM.Debt_GDP
MOM.σ_con/MOM.σ_GDP
MOM.σ_GDP
MOM.σ_TB
MOM.σ_CA
MOM.σ_RER
MOM.Corr_Spreads_GDP
MOM.Corr_TB_GDP
MOM.Corr_CA_GDP
MOM.Corr_RER_GDP

NAME="OilDiscoveries_impatient.csv"
MODEL_impatient=UnpackModel_Vector(nHfactor,NAME,FOLDER)
MOM_impatient=AverageMomentsManySamples(N,Tmom,MODEL_impatient)

NAME="OilDiscoveries_higherAlfa0.csv"
MODEL_higher=UnpackModel_Vector(nHfactor,NAME,FOLDER)
MOM_higher=AverageMomentsManySamples(N,Tmom,MODEL_higher)

################################################################################
################### Simulate discovery paths, Figure 6 and 7 ###################
################################################################################
FOLDER_GRAPHS="Graphs" #Folder to store graphs
FILE_REGRESSIONS="Regressions_Benchmark.xlsx" #Excel file with STATA output
                                              #make sure file extension is xlsx
#Parameters for simulation of paths
FixedShocks=false #Do not fix shocks to their mean
N=10000           #10,000 paths
Tbefore=1         #Return one period before discovery (will be benchmark for changes)
Tafter=16         #Periods after discovery, make it more than 15 for graphs to work

#Unpack risk neutral version
NAME="OilDiscoveries_riskNeutral.csv"
MODEL_RN=UnpackModel_Vector(nHfactor,NAME,FOLDER)

#Path series with benchmark
TS_D, TS_ND=AverageDiscoveryPaths(FixedShocks,N,Tbefore,Tafter,MODEL)
#Path series with higher α0
TS_D_higher, TS_ND_higher=AverageDiscoveryPaths(FixedShocks,N,Tbefore,Tafter,MODEL_higher)
#Path series with risk neutral
TS_D_RN, TS_ND_RN=AverageDiscoveryPaths(FixedShocks,N,Tbefore,Tafter,MODEL_RN)

#Create and save Figure 6 and 7
plt_6=Plot_Figure_6_with3Mods(FOLDER_GRAPHS,FILE_REGRESSIONS,TS_D,TS_D_higher,TS_D_RN,MODEL)
plt_7=Plot_Figure_7(FOLDER_GRAPHS,TS_D,MODEL)

################################################################################
########################### Welfare gains, Table 4 #############################
################################################################################
N=1000 #Average over 1,000 draws from the ergodic distribution
#Welfare gains in benchmark model
wg=AverageWelfareGains(N,MODEL)

#Welfare gains in model with high risk aversion

wg_highα=AverageWelfareGains(N,MODEL_higher)

#Welfare gains in model with fixed oil price
NAME="OilDiscoveries_fix.csv"
MODEL_fix=UnpackModel_Vector(nHfactor,NAME,FOLDER)
wg_fix=AverageWelfareGains(N,MODEL_fix)

#Welfare gains in model with put options
NAME="OilDiscoveries_put.csv"
MODEL_put=UnpackModel_Vector(nHfactor,NAME,FOLDER)
wg_put=AverageWelfareGains(N,MODEL_put)
