##################################################
################ Paths and setup #################
##################################################
#Every path is built from the location of this file, so the package runs from
#wherever it is copied to and from any working directory
const ROOT=@__DIR__
const FOLDER_MODEL=joinpath(ROOT,"Code_Model")
const FOLDER_DATA=joinpath(ROOT,"Code_Data")
const FOLDER_ICIO=joinpath(FOLDER_DATA,"Evidence_of_Mechanism")
const FOLDER_GRAPHS=joinpath(ROOT,"Graphs")
const FOLDER_TABLES=joinpath(ROOT,"Tables")
const SETUP_FILE=joinpath(FOLDER_MODEL,"Setup_Calibrated.csv")
mkpath(FOLDER_GRAPHS)
mkpath(FOLDER_TABLES)

##################################################
############# I. Empirical results ###############
##################################################
#Reads Regressions_Benchmark.txt and Falsification_Draws.csv, written by
#Code_Data/EmpiricalResults.do and Code_Data/FalsificationTests.do. Run both in
#Stata before this script
include(joinpath(FOLDER_DATA,"EmpiricalResults.jl"))
RegressionsFile=joinpath(FOLDER_DATA,"Regressions_Benchmark.txt")
npv=4.5; T=16

#Figures 2, 3 and 4: responses of spreads, of GDP, the current account and
#government debt, and of consumption
fig_2, fig_3, fig_4=Figures_Empirical(RegressionsFile,npv,T)
savefig(fig_2,joinpath(FOLDER_GRAPHS,"Figure2.pdf"))
savefig(fig_3,joinpath(FOLDER_GRAPHS,"Figure3.pdf"))
savefig(fig_4,joinpath(FOLDER_GRAPHS,"Figure4.pdf"))

#Figures 11, 15 and 16: the same three on the EMBI sample
fig_11, fig_15, fig_16=Figures_Empirical_Appendix(RegressionsFile,npv,T)
savefig(fig_11,joinpath(FOLDER_GRAPHS,"Figure11.pdf"))
savefig(fig_15,joinpath(FOLDER_GRAPHS,"Figure15.pdf"))
savefig(fig_16,joinpath(FOLDER_GRAPHS,"Figure16.pdf"))

#Figure 12: response of spreads with discovery histories reassigned
fig_12, ndraws_12=Figures_Empirical_Falsification(FOLDER_DATA,RegressionsFile,npv,T)
savefig(fig_12,joinpath(FOLDER_GRAPHS,"Figure12.pdf"))

#Figure 13: estimated coefficients with leads of discovery size
fig_13=Figures_Empirical_E1a(RegressionsFile,npv)
savefig(fig_13,joinpath(FOLDER_GRAPHS,"Figure13.pdf"))

#Figure 14: response of spreads under alternative measures of discovery size
fig_14=Figures_Empirical_E1b(RegressionsFile,T)
savefig(fig_14,joinpath(FOLDER_GRAPHS,"Figure14.pdf"))

#Figure 17: response of spreads with the discoveries after 2008 removed
fig_17=Figures_Empirical_E2c(RegressionsFile,npv,T)
savefig(fig_17,joinpath(FOLDER_GRAPHS,"Figure17.pdf"))

#Figure 18: the six responses with and without the oil-price interactions
fig_18=Figures_Empirical_R24(RegressionsFile,npv,T)
savefig(fig_18,joinpath(FOLDER_GRAPHS,"Figure18.pdf"))

#Figure 19: the same comparison with undiscounted reserves in place of the NPV
fig_19=Figures_Empirical_R24_URR(RegressionsFile,T)
savefig(fig_19,joinpath(FOLDER_GRAPHS,"Figure19.pdf"))

#Cross-country evidence on oil technology, from the OECD ICIO.
#Build_ICIO_Panel downloads 600 MB and takes about forty minutes; the panel it
#writes is shipped with the package, so the line stays commented out
include(joinpath(FOLDER_ICIO,"ICIO_Technology.jl"))
#Build_ICIO_Panel(FOLDER_ICIO)

#Table 4: technology of oil extraction and of the rest of the economy.
#Table_ICIO_B0609 is the same table with oilfield services inside the oil
#sector, a robustness check rather than a printed exhibit
Write_ICIO_Table(FOLDER_ICIO,FOLDER_TABLES;SETUP_FILE=SETUP_FILE)
Write_ICIO_Table(FOLDER_ICIO,FOLDER_TABLES;SETUP_FILE=SETUP_FILE,WithB09=true)

#Counts quoted in the text
Summarize_Gamma(FOLDER_ICIO;SETUP_FILE=SETUP_FILE)
Summarize_ICIO(FOLDER_ICIO)

##################################################
######### II. Simple model results ###############
##################################################
include(joinpath(FOLDER_MODEL,"Primitives.jl"))
include(joinpath(FOLDER_MODEL,"ModelResults.jl"))
col_Canonical=2
col_OilExempt=3

#Figure 5: average responses to oil discoveries in the canonical model
Simple_Model_Result(col_Canonical,col_OilExempt,SETUP_FILE,FOLDER_GRAPHS)

##################################################
####### III. Benchmark model results #############
##################################################
#false re-solves every model from scratch, which is what the package does by
#default: the solved models are not shipped. One solve takes roughly 45 minutes
#and nine are solved in all. Set to true to reload Code_Model/Model_*.csv from a
#previous run instead
UseSavedFile=false

MOD_BEN, MOM_BEN, fig_6, fig_7, _=Results_Benchmark_Calibration(UseSavedFile,SETUP_FILE,
                                                                FOLDER_MODEL)
MOD_no_g, MOM_no_g, fig_24=Results_No_g_Alternative(UseSavedFile,SETUP_FILE,FOLDER_MODEL)
if !UseSavedFile
    SaveModel_Vector(joinpath(FOLDER_MODEL,"Model_Benchmark.csv"),MOD_BEN)
    SaveModel_Vector(joinpath(FOLDER_MODEL,"Model_No_g.csv"),MOD_no_g)
end

#Values quoted in the text, not in any table
MOD_BEN.par.nL
MOD_BEN.par.nH

#The technology variants are rows of Table 3, so they are solved here
include(joinpath(FOLDER_MODEL,"Sensitivity_Technology.jl"))
if !UseSavedFile
    Solve_Technology_Variants(SETUP_FILE,FOLDER_MODEL)
end
MOM_tech=Moments_Technology_Variants(FOLDER_MODEL)

#Tables 2, 3 and 7: parameters set with SMM, business cycle moments, and the
#same moments for the model with only aggregate consumption
Write_Model_Tables(MOD_BEN,MOM_BEN,MOM_tech,VARIANT_LABELS_TABLE[2:end],
                   MOM_no_g,FOLDER_TABLES)

#Figure 6: average responses to oil discoveries
savefig(fig_6,joinpath(FOLDER_GRAPHS,"Figure6.pdf"))

#Figure 7: cost of default
savefig(fig_7,joinpath(FOLDER_GRAPHS,"Figure7.pdf"))

#Figure 24: responses in the model with only aggregate consumption
savefig(fig_24,joinpath(FOLDER_GRAPHS,"Figure24.pdf"))

##################################################
##### IV. Sensitivity to the technology gaps #####
##################################################
#These functions write their own PDFs, so there is no savefig here

#Figure 8: relative exposure of oil to the cost of default over the three
#technology pairs. Static, so nothing is solved
fig_exposure=Figure_Technology_Exposure(SETUP_FILE,FOLDER_GRAPHS;FOLDER_ICIO=FOLDER_ICIO)

#Figures 20 and 21: the same sweeps with the Armington elasticity at 5.1, in
#both sectors and in the final sector only
fig_mu_both, fig_mu_final=Figures_Armington_Sensitivity(SETUP_FILE,FOLDER_GRAPHS;
                                                       FOLDER_ICIO=FOLDER_ICIO)

#Figures 22 and 23: the same pair for the Dixit-Stiglitz curvature across
#imported varieties
fig_nu_both, fig_nu_final=Figures_DixitStiglitz_Sensitivity(SETUP_FILE,FOLDER_GRAPHS;
                                                           FOLDER_ICIO=FOLDER_ICIO)

#Figure 9: responses to a discovery under the alternative technologies
fig_responses, TS_tech, SPR_CF_tech=Figure_Technology_Responses(FOLDER_MODEL,FOLDER_GRAPHS)

#Table 5: average welfare gains of giant oil discoveries
coulumn_Benchmark=4
if UseSavedFile
    MOD_priv=UnpackModel_File("Model_Private_Field.csv",FOLDER_MODEL)
    #SellOilDiscovery is not serialized, restore it before using the model
    MOD_sell=Restore_Sell_Parameters(UnpackModel_File("Model_Sell_Giant_Field.csv",FOLDER_MODEL))
else
    MOD_priv=Model_Private_Field(coulumn_Benchmark,SETUP_FILE)
    MOD_sell=Model_Sell_Giant_Field(coulumn_Benchmark,SETUP_FILE)
    SaveModel_Vector(joinpath(FOLDER_MODEL,"Model_Private_Field.csv"),MOD_priv)
    SaveModel_Vector(joinpath(FOLDER_MODEL,"Model_Sell_Giant_Field.csv"),MOD_sell)
end

Write_Welfare_Table(MOD_BEN,MOD_priv,MOD_sell,FOLDER_TABLES)

#Decomposition of the gains from selling oil rents. Writes
#Table_Decomposition_rows.tex, which is not a printed table
if UseSavedFile
    MOD_R, MOD_F=Load_Decomposition_Models(FOLDER_MODEL)
else
    Solve_Decomposition_Models(SETUP_FILE,FOLDER_MODEL;coulumn_Benchmark=coulumn_Benchmark)
    MOD_R, MOD_F=Load_Decomposition_Models(FOLDER_MODEL)
end

Write_Decomposition_Table(MOD_BEN,MOD_R,MOD_F,MOD_sell,FOLDER_TABLES)
