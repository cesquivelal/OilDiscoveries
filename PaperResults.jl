
##################################################
############# I. Empirical results ###############
##################################################
include("Code_Data\\EmpiricalResults.jl")
FOLDER_GRAPHS="Graphs"
RegressionsFile="Code_Data\\Regressions_Benchmark.xlsx"
npv=4.5; T=16
fig_2, fig_3, fig_4=Figures_Empirical(RegressionsFile,npv,T)
fig_2
savefig(fig_2,"$FOLDER_GRAPHS\\Figure2.pdf")
savefig(fig_3,"$FOLDER_GRAPHS\\Figure3.pdf")
savefig(fig_4,"$FOLDER_GRAPHS\\Figure4.pdf")

fig_2A, fig_3A, fig_4A=Figures_Empirical_Appendix(RegressionsFile,npv,T)
savefig(fig_2A,"$FOLDER_GRAPHS\\Figure2A.pdf")
savefig(fig_3A,"$FOLDER_GRAPHS\\Figure3A.pdf")
savefig(fig_4A,"$FOLDER_GRAPHS\\Figure4A.pdf")

##################################################
######### II. Simple Model Results ###############
##################################################
using Distributed
include("Code_Model\\Primitives.jl")
include("Code_Model\\ModelResults.jl")
SETUP_FILE="Code_Model\\Setup_Calibrated.csv"
col_Canonical=2
col_OilExempt=3
Simple_Model_Result(col_Canonical,col_OilExempt,SETUP_FILE,FOLDER_GRAPHS)

##################################################
####### III. Benchmark Model Results #############
##################################################
UseSavedFile=false
MOD_BEN, MOM_BEN, fig_6, fig_7, fig_8=Results_Benchmark_Calibration(UseSavedFile,SETUP_FILE)
SaveModel_Vector("Code_Model\\Model_Benchmark.csv",MOD_BEN)
MOD_BEN.par.nL
MOD_BEN.par.nH
MOD_BEN.par.σ_ϵz
MOD_BEN.par.A
MOD_BEN.par.β
MOD_BEN.par.θf
##########################
#Table 2: Targeted moments
##########################
MOM_BEN.σ_GDP
MOM_BEN.GDP_drop_DefEv
MOM_BEN.MeanSpreads
MOM_BEN.WK_GDP

################################
#Table 3: Business cycle moments
################################
MOM_BEN.DefaultPr
MOM_BEN.Debt_GDP
MOM_BEN.StdSpreads
MOM_BEN.σ_con/MOM_BEN.σ_GDP
MOM_BEN.σ_G/MOM_BEN.σ_GDP
MOM_BEN.σ_TB_y
MOM_BEN.σ_CA_y
MOM_BEN.Corr_Spreads_GDP
MOM_BEN.Corr_TB_GDP
MOM_BEN.Corr_CA_GDP

MOD_no_g, MOD_no_g_identical, MOM_no_g, MOM_no_g_identical, fig_9, fig_10=Results_Two_Alternatives(UseSavedFile,SETUP_FILE)
SaveModel_Vector("Code_Model\\Model_No_g.csv",MOD_no_g)
SaveModel_Vector("Code_Model\\Model_No_g_identical.csv",MOD_no_g_identical)
MOM_no_g.σ_GDP
MOM_no_g.GDP_drop_DefEv
MOM_no_g.MeanSpreads
MOM_no_g.WK_GDP

MOM_no_g.DefaultPr
MOM_no_g.Debt_GDP
MOM_no_g.StdSpreads
MOM_no_g.σ_con/MOM_BEN.σ_GDP
MOM_no_g.σ_G/MOM_BEN.σ_GDP
MOM_no_g.σ_TB_y
MOM_no_g.σ_CA_y
MOM_no_g.Corr_Spreads_GDP
MOM_no_g.Corr_TB_GDP
MOM_no_g.Corr_CA_GDP

MOM_no_g_identical.σ_GDP
MOM_no_g_identical.GDP_drop_DefEv
MOM_no_g_identical.MeanSpreads
MOM_no_g_identical.WK_GDP

MOM_no_g_identical.DefaultPr
MOM_no_g_identical.Debt_GDP
MOM_no_g_identical.StdSpreads
MOM_no_g_identical.σ_con/MOM_BEN.σ_GDP
MOM_no_g_identical.σ_G/MOM_BEN.σ_GDP
MOM_no_g_identical.σ_TB_y
MOM_no_g_identical.σ_CA_y
MOM_no_g_identical.Corr_Spreads_GDP
MOM_no_g_identical.Corr_TB_GDP
MOM_no_g_identical.Corr_CA_GDP


##########################
#Figure 6: Model responses
##########################
fig_6
savefig(fig_6,"$FOLDER_GRAPHS\\Figure6.pdf")

##########################
#Figure 7: Cost of default
##########################
fig_7
savefig(fig_7,"$FOLDER_GRAPHS\\Figure7.pdf")

##############################
#Figure 8: Fraction in default
##############################
fig_8
savefig(fig_8,"$FOLDER_GRAPHS\\Figure8.pdf")

#######################################
#Figure 9: Alternative models responses
#######################################
fig_9
savefig(fig_9,"$FOLDER_GRAPHS\\Figure9.pdf")

#############################
#Figure 10: Spreads schedules
#############################
fig_10
savefig(fig_10,"$FOLDER_GRAPHS\\Figure10.pdf")

#######################
#Table 4: Welfare gains
#######################
coulumn_Benchmark=4
MOD_priv=Model_Private_Field(coulumn_Benchmark,SETUP_FILE)
MOD_sell=Model_Sell_Giant_Field(coulumn_Benchmark,SETUP_FILE)
SaveModel_Vector("Code_Model\\Model_Private_Field.csv",MOD_priv)
SaveModel_Vector("Code_Model\\Model_Sell_Giant_Field.csv",MOD_sell)

N=1000
wg=Average_Welfare_Gains(N,MOD_BEN)
wg_priv=Average_Welfare_Gains(N,MOD_priv)
wg_no_g=Average_Welfare_Gains(N,MOD_no_g)
wg_sell=Average_Welfare_Gains(N,MOD_sell)

MOM_priv=AverageMomentsManySamples(MOD_priv.par.Tmom,MOD_priv.par.NSamplesMoments,MOD_priv)
MOM_priv
