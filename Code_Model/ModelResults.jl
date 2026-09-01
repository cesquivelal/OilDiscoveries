
using Printf
using Plots; pythonplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                        background_color_legend=nothing,foreground_color_legend=nothing,
                        legendfontsize=20,guidefontsize=20,titlefontsize=20,tickfontsize=20,
                        markersize=9,size=(650,500))

################################################################################
### Results from simple model with endowments
################################################################################
function Plot_Figure_Simple_Model(Tbefore::Int64,Tafter::Int64,TS_Canonical::Paths,TS_OilExempt::Paths)
    #Details for graphs
    npv=4.5
    tend=Tafter+1+Tbefore#15
    t0=-Tbefore
    t1=Tafter

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["penalty to total income" "oil income immune"]
    LINESTYLES=[:solid :dash]
    COLORS=[:blue :green]

    #Plot Spreads
    TITLE="spreads"
    mod=TS_Canonical.Spreads[1:tend] .- TS_Canonical.Spreads[1]
    mod2=TS_OilExempt.Spreads[1:tend] .- TS_OilExempt.Spreads[1]
    plt_spreads=plot([t0:t1],[mod mod2],label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",#ylims=[-0.05,1.3],
        legend=:right,size=SIZE_PLOTS,linewidth=LW)

    #Plot discovery
    TITLE="oil rents"
    mod=100*TS_Canonical.n ./ (1*TS_Canonical.GDP[1])
    mod2=100*TS_OilExempt.n ./ (1*TS_OilExempt.GDP[1])
    plt_n=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot GDP
    TITLE="GDP"
    mod=100*(log.(TS_Canonical.GDP[1:tend]) .- log.(TS_Canonical.GDP[1]))
    mod2=100*(log.(TS_OilExempt.GDP[1:tend]) .- log.(TS_OilExempt.GDP[1]))
    plt_gdp=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot current account
    TITLE="current account"
    mod01=100*TS_Canonical.CA ./ (1*TS_Canonical.GDP[1])
    mod=mod01[1:tend] .- mod01[1]
    mod02=100*TS_OilExempt.CA ./ (1*TS_OilExempt.GDP[1])
    mod2=mod02[1:tend] .- mod02[1]
    plt_CA=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot government debt
    TITLE="government debt"
    mod01=100*TS_Canonical.B ./ (1*TS_Canonical.GDP[1])
    mod=mod01[1:tend] .- mod01[1]
    mod02=100*TS_OilExempt.B ./ (1*TS_OilExempt.GDP[1])
    mod2=mod02[1:tend] .- mod02[1]
    plt_B=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot consumption
    TITLE="consumption"
    mod=100*(log.(TS_Canonical.C[1:tend]) .- log.(TS_Canonical.C[1]))
    mod2=100*(log.(TS_OilExempt.C[1:tend]) .- log.(TS_OilExempt.C[1]))
    plt_c=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Create plot array
    l = @layout([a b; c d])
    plt=plot(plt_spreads,plt_n,
             plt_CA,plt_B,
             layout=l,size=(size_width*2,size_height*2))
    return plt
end

function Simple_Model_Result(col_Canonical::Int64,col_OilExempt::Int64,SETUP_FILE::String,FOLDER_GRAPHS::String)
    MOD_Canonical, NAME=Model_FromSetup(col_Canonical,SETUP_FILE)
    MOD_OilExempt, NAME=Model_FromSetup(col_OilExempt,SETUP_FILE)

    DropDefaults=false; N=10000
    Tbefore=2; Tafter=15
    TS_Canonical=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_Canonical)
    TS_OilExempt=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_OilExempt)
    fig_5=Plot_Figure_Simple_Model(Tbefore,Tafter,TS_Canonical,TS_OilExempt)
    savefig(fig_5,joinpath(FOLDER_GRAPHS,"Figure5.pdf"))
    return nothing
end

################################################################################
### Results from quantitative model
################################################################################
function Plot_Responses_Full_Model(Tbefore::Int64,Tafter::Int64,TS_Mod::Paths)
    #Details for graphs
    tend=Tafter#15
    t0=-Tbefore
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["data" "model" "" ""]
    LINESTYLES=[:solid :dash :dot :dot]
    COLORS=[:black :green :black :black]

    #Plot fraction in default
    TITLE=""
    mod=TS_Mod.Def
    plt_def=plot([t0:t1],mod,label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="fraction in default",xlabel="t",#ylims=[-0.05,1.3],
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot Spreads
    TITLE="spreads"
    mod=TS_Mod.Spreads .- TS_Mod.Spreads[1]
    plt_spreads=plot([t0:t1],mod,label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",#ylims=[-0.05,1.3],
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot GDP
    TITLE="GDP"
    mod=100*(log.(TS_Mod.GDP) .- log.(TS_Mod.GDP[1]))
    plt_gdp=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot current account
    TITLE="current account"
    mod01=100*TS_Mod.CA ./ (1*TS_Mod.GDP)
    mod=mod01# .- mod01[1]
    plt_CA=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot government debt
    TITLE="government debt"
    mod01=100*TS_Mod.B ./ (1*TS_Mod.GDP[1])
    # mod01=TS_Mod.B
    mod=mod01
    plt_B=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of Av(GDP)",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot private consumption
    mod_c=100*(log.(TS_Mod.C) .- log.(TS_Mod.C[1]))
    mod_g=100*(log.(TS_Mod.G) .- log.(TS_Mod.G[1]))
    cg_min=minimum(vcat(mod_c,mod_g))-0.1
    cg_max=maximum(vcat(mod_c,mod_g))+0.1
    YLIMS=[cg_min,cg_max]
    TITLE="private consumption"
    plt_c=plot([t0:t1],mod_c,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,ylims=YLIMS)

    #Plot government consumption
    TITLE="government consumption"
    plt_g=plot([t0:t1],mod_g,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,ylims=YLIMS)

    #Plot labor supply
    TITLE="labor supply"
    mod=100*(log.(TS_Mod.L) .- log.(TS_Mod.L[1]))
    plt_l=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)#,ylims=YLIMS_C)

    #Create plot array
    l = @layout([a b; c d; e f])
    plt=plot(plt_spreads,plt_gdp,
             plt_CA,plt_B,plt_c,plt_g,
             layout=l,size=(size_width*2,size_height*3))
    return plt, plt_def
end

function Plot_Default_Costs(MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack Y_Matrix, YO_Matrix, GDP_Final_Matrix, GDP_Oil_Matrix = GRIDS
    @unpack GR_z = GRIDS
    @unpack τ = par

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["small field" "large field"]
    LINESTYLES=[:solid :dash :dot :dot]
    COLORS=[:blue :green :black :black]
    YLABEL="percentage drop"
    XLABEL="productivity shock z"

    COST_NL_y=100*((Y_Matrix[:,1,1] ./ Y_Matrix[:,1,2]) .- 1)
    COST_NH_y=100*((Y_Matrix[:,end,1] ./ Y_Matrix[:,end,2]) .- 1)
    COST_NL_o=100*((YO_Matrix[:,1,1] ./ YO_Matrix[:,1,2]) .- 1)
    COST_NH_o=100*((YO_Matrix[:,end,1] ./ YO_Matrix[:,end,2]) .- 1)
    ylow_y=minimum([COST_NL_y COST_NH_y])
    ylow_o=minimum([COST_NL_o COST_NH_o])
    ylow=min(ylow_y,ylow_o)
    yhigh_y=maximum([COST_NL_y COST_NH_y])
    yhigh_o=maximum([COST_NL_o COST_NH_o])
    yhigh=max(yhigh_y,yhigh_o)
    YLIMS_Y=[ylow,yhigh]

    #Final sector output
    TITLE="final good output"
    plt_yf=plot(GR_z,[COST_NL_y COST_NH_y],
                title=TITLE,
                xlabel=XLABEL,
                label=LABELS,
                linecolor=COLORS,
                linestyle=LINESTYLES,
                ylims=YLIMS_Y,
                ylabel=YLABEL)

    #Oil sector output
    TITLE="oil output"
    plt_yo=plot(GR_z,[COST_NL_o COST_NH_o],
                title=TITLE,
                xlabel=XLABEL,
                label=LABELS,
                linecolor=COLORS,
                linestyle=LINESTYLES,
                ylims=YLIMS_Y,
                legend=:bottomright,
                ylabel=YLABEL)

    #Private consumption
    C=(1-τ)*GDP_Final_Matrix
    COST_NL_c=100*((C[:,1,1] ./ C[:,1,2]) .- 1)
    COST_NH_c=100*((C[:,end,1] ./ C[:,end,2]) .- 1)
    ylow_c=minimum([COST_NL_c COST_NH_c])
    yhigh_c=maximum([COST_NL_c COST_NH_c])

    G=τ*GDP_Final_Matrix+GDP_Oil_Matrix
    COST_NL_g=100*((G[:,1,1] ./ G[:,1,2]) .- 1)
    COST_NH_g=100*((G[:,end,1] ./ G[:,end,2]) .- 1)
    ylow_g=minimum([COST_NL_g COST_NH_g])
    yhigh_g=maximum([COST_NL_g COST_NH_g])

    ylow=min(ylow_c,ylow_g)
    yhigh=max(yhigh_c,yhigh_g)
    YLIMS_C=[ylow,yhigh]

    TITLE="private consumption"
    plt_pc=plot(GR_z,[COST_NL_c COST_NH_c],
                title=TITLE,
                xlabel=XLABEL,
                label=LABELS,
                linecolor=COLORS,
                linestyle=LINESTYLES,
                ylims=YLIMS_C,
                ylabel=YLABEL)

    #Government consumption
    TITLE="government consumption"
    plt_gc=plot(GR_z,[COST_NL_g COST_NH_g],
                title=TITLE,
                xlabel=XLABEL,
                label=LABELS,
                linecolor=COLORS,
                linestyle=LINESTYLES,
                ylims=YLIMS_C,
                ylabel=YLABEL)

    #Create plot array
    l = @layout([a b; c d])
    plt=plot(plt_yf,plt_yo,
             plt_pc,plt_gc,
             layout=l,size=(size_width*2,size_height*2))
    return plt
end

function Results_Benchmark_Calibration(UseSavedFile::Bool,SETUP_FILE::String,FOLDER_MODEL::String)
    #Solve model
    if UseSavedFile
        MOD_BEN=UnpackModel_File("Model_Benchmark.csv",FOLDER_MODEL)
    else
        col_Benchmark=4
        MOD_BEN, NAME=Model_FromSetup(col_Benchmark,SETUP_FILE)
    end

    #Compute moments
    MOM_BEN=AverageMomentsManySamples(MOD_BEN.par.Tmom,MOD_BEN.par.NSamplesMoments,MOD_BEN)

    #Compute average responses
    DropDefaults=false
    N=10000; Tbefore=2; Tafter=15
    TS_Mod=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_BEN)
    fig_6, fig_8=Plot_Responses_Full_Model(Tbefore,Tafter,TS_Mod)

    #Plot default cost
    fig_7=Plot_Default_Costs(MOD_BEN)

    return MOD_BEN, MOM_BEN, fig_6, fig_7, fig_8
end

function Plot_Responses_Alt_Model(Tbefore::Int64,Tafter::Int64,TS_Mod_no_g::Paths)
    #Details for graphs
    tend=Tafter#15
    t0=-Tbefore
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["no g" "" "" ""]
    LINESTYLES=[:solid :dash :dot :dot]
    COLORS=[:blue :green :black :black]

    #Plot Spreads
    TITLE="spreads"
    mod=TS_Mod_no_g.Spreads .- TS_Mod_no_g.Spreads[1]
    plt_spreads=plot([t0:t1],mod,label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",#ylims=[-0.05,1.3],
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot GDP
    TITLE="GDP"
    mod=100*(log.(TS_Mod_no_g.GDP) .- log.(TS_Mod_no_g.GDP[1]))
    plt_gdp=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot current account
    TITLE="current account"
    mod01=100*TS_Mod_no_g.CA ./ (1*TS_Mod_no_g.GDP)
    mod=mod01# .- mod01[1]
    plt_CA=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot government debt
    TITLE="government debt"
    mod01=100*TS_Mod_no_g.B ./ (1*TS_Mod_no_g.GDP[1])
    # mod01=TS_Mod.B
    mod=mod01
    plt_B=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of Av(GDP)",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot private consumption
    mod_c=100*(log.(TS_Mod_no_g.C) .- log.(TS_Mod_no_g.C[1]))
    TITLE="total consumption"
    plt_c=plot([t0:t1],mod_c,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot fraction in default
    TITLE="fraction in default"
    mod=TS_Mod_no_g.Def
    plt_def=plot([t0:t1],mod,label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="fraction in default",xlabel="t",#ylims=[-0.05,1.3],
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Create plot array
    l = @layout([a b; c d; e f])
    plt=plot(plt_spreads,plt_gdp,
             plt_CA,plt_B,plt_c,plt_def,
             layout=l,size=(size_width*2,size_height*3))
    return plt
end

function Results_No_g_Alternative(UseSavedFile::Bool,SETUP_FILE::String,FOLDER_MODEL::String)
    #Solve model
    if UseSavedFile
        MOD_no_g=UnpackModel_File("Model_No_g.csv",FOLDER_MODEL)
    else
        col_no_g=5
        MOD_no_g, NAME=Model_FromSetup(col_no_g,SETUP_FILE)
    end

    #Compute moments
    MOM_no_g=AverageMomentsManySamples(MOD_no_g.par.Tmom,MOD_no_g.par.NSamplesMoments,MOD_no_g)

    #Compute average responses
    DropDefaults=false
    N=10000; Tbefore=2; Tafter=15
    TS_Mod_no_g=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_no_g)
    fig_9=Plot_Responses_Alt_Model(Tbefore,Tafter,TS_Mod_no_g)

    return MOD_no_g, MOM_no_g, fig_9
end

################################################################################
### Write the rows of Tables 2, 3 and 7 as LaTeX, to be read with \input
###
### MOM_VAR and LABELS_VAR are the technology variants other than the benchmark,
### from Moments_Technology_Variants, in the order Table 3 prints them
################################################################################
function Write_Model_Tables(MOD_BEN::Model,MOM_BEN::Moments,MOM_VAR::Array{Moments,1},
                            LABELS_VAR::Array{String,1},MOM_no_g::Moments,
                            FOLDER_TABLES::String)
    @unpack par = MOD_BEN
    f2(x::Float64)=@sprintf("%.2f",x)
    f3(x::Float64)=@sprintf("%.3f",x)

    #Table 2, parameters set with SMM and the moments they target
    #Data targets from CalibrationData.xlsx, sheet "Calibration summary"
    #The four moments are in percent, as they are in Table 3, which reads DATA_T2
    NAME_T2=["Standard deviation of \$z_{t}\$","TFP of domestic intermediates",
             "Discount factor","Working capital requirement for final sector"]
    SYMBOL_T2=["\$\\sigma_{z}\$","\$A\$","\$\\beta\$","\$\\theta_{f}\$"]
    VALUE_T2=[par.σ_ϵz,par.A,par.β,par.θf]
    MOMENT_T2=["GDP standard deviation","Output drop in default","Average spread",
               "Private working capital / GDP"]
    DATA_T2=[3.11,-13.28,2.90,8.09]
    MODEL_T2=[MOM_BEN.σ_GDP,MOM_BEN.GDP_dropAv_DefEv,MOM_BEN.MeanSpreads,MOM_BEN.WK_GDP]

    ROWS_T2=Array{String,1}(undef,length(NAME_T2))
    for i in 1:length(NAME_T2)
        CELLS=[NAME_T2[i],SYMBOL_T2[i],f3(VALUE_T2[i]),MOMENT_T2[i],
               f2(DATA_T2[i]),f2(MODEL_T2[i])]
        ROWS_T2[i]=join(["{\\footnotesize{}"*c*"}" for c in CELLS]," & ")*"\\tabularnewline"
    end

    #Table 3, the four moments targeted in Table 2 followed by the untargeted ones.
    #The variants of Figure 9 are not recalibrated, so the targeted moments are
    #reported for them too. Everything is in percent except the three ratios
    #sigma_g/sigma_y is not defined without separate government consumption
    DATA_T3=vcat([f2(d) for d in DATA_T2],
                 ["2.45","0.21","1.36","1.45","1.05","2.57","2.14","-0.22","-0.19","-0.49"])
    function Cells_T3(MOM::Moments,With_g_Model::Bool)
        return [f2(MOM.σ_GDP),f2(MOM.GDP_dropAv_DefEv),f2(MOM.MeanSpreads),f2(MOM.WK_GDP),
                f2(MOM.DefaultPr),f2(MOM.Debt_GDP/100),f2(MOM.StdSpreads),
                f2(MOM.σ_con/MOM.σ_GDP),
                With_g_Model ? f2(MOM.σ_G/MOM.σ_GDP) : "n.a.",
                f2(MOM.σ_TB_y),f2(MOM.σ_CA_y),
                f2(MOM.Corr_Spreads_GDP),f2(MOM.Corr_TB_GDP),f2(MOM.Corr_CA_GDP)]
    end
    #Fourteen moments fit the text width at footnotesize, one step below Table 2
    function Rows_T3(LABEL::Array{String,1},CELLS::Array{Array{String,1},1})
        ROWS=Array{String,1}(undef,length(LABEL))
        for i in 1:length(LABEL)
            ROWS[i]=join(["{\\footnotesize{}"*c*"}" for c in vcat(LABEL[i],CELLS[i])]," & ")*
                    "\\tabularnewline"
        end
        return ROWS
    end

    #Table 3 in the text, the benchmark and the technology variants of Figure 9,
    #all of which keep separate government consumption
    LABEL_T3=vcat(["data","benchmark"],LABELS_VAR)
    CELLS_T3=vcat([DATA_T3,Cells_T3(MOM_BEN,true)],[Cells_T3(M,true) for M in MOM_VAR])
    ROWS_T3=Rows_T3(LABEL_T3,CELLS_T3)

    #The same table in the appendix for the model with only aggregate consumption
    LABEL_T3A=["data","benchmark","no g"]
    CELLS_T3A=[DATA_T3,Cells_T3(MOM_BEN,true),Cells_T3(MOM_no_g,false)]
    ROWS_T3A=Rows_T3(LABEL_T3A,CELLS_T3A)

    #LaTeX rows to be read with \input, and the same numbers as csv
    for (NAME,ROWS) in (("Table2_rows.tex",ROWS_T2),("Table3_rows.tex",ROWS_T3),
                        ("Table7_rows.tex",ROWS_T3A))
        open(joinpath(FOLDER_TABLES,NAME),"w") do io
            for r in ROWS
                println(io,r)
            end
        end
    end
    MAT_T2=hcat(NAME_T2,SYMBOL_T2,VALUE_T2,MOMENT_T2,DATA_T2,MODEL_T2)
    writedlm(joinpath(FOLDER_TABLES,"Table2.csv"),vcat(["parameter" "symbol" "value" "moment" "data" "model"],MAT_T2),',')
    HEAD_T3=["model" "sigma_y" "GDPdrop_default" "MeanSpreads" "WK_GDP" "DefaultPr" "Debt_GDP" "StdSpreads" "sigma_c_sigma_y" "sigma_g_sigma_y" "sigma_tb_y" "sigma_ca_y" "corr_spread_y" "corr_tb_y" "corr_ca_y"]
    MAT_T3=hcat(LABEL_T3,permutedims(hcat(CELLS_T3...)))
    writedlm(joinpath(FOLDER_TABLES,"Table3.csv"),vcat(HEAD_T3,MAT_T3),',')
    MAT_T3A=hcat(LABEL_T3A,permutedims(hcat(CELLS_T3A...)))
    writedlm(joinpath(FOLDER_TABLES,"Table7.csv"),vcat(HEAD_T3,MAT_T3A),',')

    return nothing
end

function Restore_Sell_Parameters(MOD::Model;WithWindfall::Bool=true,
                                 KeepRentsInDefault::Bool=false)
    #SellOilDiscovery, OilRents_Discovery, NPV_Discovery and KeepRentsInDefault are not in
    #the saved parameter vector. They are recomputed here from parameters that are saved,
    #exactly as Model_Sell_Giant_Field computes them. The keywords match that function's,
    #so which of V^S, V^F and V^R a saved file holds is carried by the call, not the file
    @unpack SOLUTION, GRIDS, par, ITP_OUTPUT = MOD
    OilRents_Discovery=OilRents_FromDiscovery_ss(par)
    if WithWindfall
        NPV_Discovery=NPV_giant_field(par)
    else
        NPV_Discovery=0.0
    end
    par_Sell=Pars(par,SellOilDiscovery=true,OilRents_Discovery=OilRents_Discovery,
                  NPV_Discovery=NPV_Discovery,KeepRentsInDefault=KeepRentsInDefault)
    return Model(SOLUTION,GRIDS,par_Sell,ITP_OUTPUT)
end

function Write_Welfare_Table(MOD_BEN::Model,MOD_priv::Model,MOD_sell::Model,
                             FOLDER_TABLES::String)
    #Averages of N discoveries drawn from the ergodic distribution, as in the table note
    N=1000
    WG=[Average_Welfare_Gains(N,MOD_BEN),Average_Welfare_Gains(N,MOD_priv),
        Average_Welfare_Gains(N,MOD_sell)]

    ROW="welfare gains & "*join([@sprintf("%.1f",w) for w in WG]," & ")*"\\tabularnewline"
    open(joinpath(FOLDER_TABLES,"Table5_rows.tex"),"w") do io
        println(io,ROW)
    end

    COLUMN=["benchmark","oil firm operated by domestic private sector",
            "sell oil rents from giant field to foreigners"]
    writedlm(joinpath(FOLDER_TABLES,"Table5.csv"),vcat(["model" "welfare_gains"],hcat(COLUMN,WG)),',')

    return nothing
end

#############################################################
#Decomposition of the gains from selling the giant field's
#rents, following Aguiar, Amador and Fourakis (2020). V^R
#gives up the flows only when the government repays, V^F in
#every period, both receiving nothing
#############################################################
const DECOMPOSITION_FILES=["Model_Sell_Repay_Only.csv","Model_Sell_No_Windfall.csv"]

function Solve_Decomposition_Models(SETUP_FILE::String,FOLDER::String;
                                    coulumn_Benchmark::Int64=4)
    #Both are the benchmark calibration with the sell counterfactual's budget changes
    #turned on one at a time, so neither needs a column of Setup_Calibrated.csv
    MOD_R=Model_Sell_Giant_Field(coulumn_Benchmark,SETUP_FILE;WithWindfall=false,
                                 KeepRentsInDefault=true)
    SaveModel_Vector(joinpath(FOLDER,DECOMPOSITION_FILES[1]),MOD_R)
    MOD_F=Model_Sell_Giant_Field(coulumn_Benchmark,SETUP_FILE;WithWindfall=false)
    SaveModel_Vector(joinpath(FOLDER,DECOMPOSITION_FILES[2]),MOD_F)
    return nothing
end

function Load_Decomposition_Models(FOLDER::String)
    #The sell parameters are not serialized, so which economy a file holds is carried
    #by the call, exactly as it is for Model_Sell_Giant_Field.csv
    MOD_R=Restore_Sell_Parameters(UnpackModel_File(DECOMPOSITION_FILES[1],FOLDER);
                                  WithWindfall=false,KeepRentsInDefault=true)
    MOD_F=Restore_Sell_Parameters(UnpackModel_File(DECOMPOSITION_FILES[2],FOLDER);
                                  WithWindfall=false)
    return MOD_R, MOD_F
end

function Write_Decomposition_Table(MOD_BEN::Model,MOD_R::Model,MOD_F::Model,
                                   MOD_sell::Model,FOLDER_TABLES::String)
    #Same averaging as the welfare table: N discoveries drawn from the ergodic
    #distribution, the same draws across all four economies
    N=1000
    ζR, ζD, ζW, ζ, ζProduct, MaxResidual=Average_Welfare_Decomposition(N,MOD_BEN,MOD_R,
                                                                       MOD_F,MOD_sell)

    ROW="welfare gains & "*join([@sprintf("%.1f",w) for w in [ζR,ζD,ζW,ζ]]," & ")*"\\tabularnewline"
    open(joinpath(FOLDER_TABLES,"Table_Decomposition_rows.tex"),"w") do io
        println(io,ROW)
    end

    COLUMN=["foregone flows","default incentives","windfall","total",
            "product of the three factors","max identity residual across draws"]
    VALUES=[ζR,ζD,ζW,ζ,ζProduct,MaxResidual]
    writedlm(joinpath(FOLDER_TABLES,"Table_Decomposition.csv"),
             vcat(["term" "value"],hcat(COLUMN,VALUES)),',')

    return ζR, ζD, ζW, ζ, ζProduct, MaxResidual
end

#############################################################
#Decomposition of the welfare gain of a discovery across the
#three technology variants of Figure 9, ordered by Gamma.
#Not called by PaperResults.jl: no table in the paper uses it
#############################################################
const MECHANISM_FILES=["Model_Identical.csv","Model_SameTheta.csv","Model_Benchmark.csv"]
const MECHANISM_LABELS=["identical","technology gaps only","benchmark"]

function Average_Welfare_Decomposition_Mechanism(N::Int64,MOD_ident::Model,MOD_theta::Model,
                                                 MOD_bench::Model)
    #The N discoveries are drawn from the ergodic distribution of the benchmark and the
    #three value functions are evaluated at the same states, as in the selling
    #decomposition. Drawing from the benchmark is also what makes the last term equal
    #the number Write_Welfare_Table already reports
    @unpack par = MOD_bench
    @unpack drp = par
    T=drp+N; ForMoments=true
    TS=Simulate_Paths(ForMoments,T,MOD_bench)
    χI=0.0; χT=0.0; χB=0.0
    fTech=0.0; fWk=0.0; MaxResidual=0.0
    for i in 1:N
        t=drp+i
        wI=Welfare_Gains_t(t,TS,MOD_ident)
        wT=Welfare_Gains_t(t,TS,MOD_theta)
        wB=Welfare_Gains_t(t,TS,MOD_bench)
        #Per-draw factors, whose product is the total by construction
        rTech=(1+wT/100)/(1+wI/100)
        rWk=(1+wB/100)/(1+wT/100)
        MaxResidual=max(MaxResidual,abs((1+wI/100)*rTech*rWk-(1+wB/100)))
        χI=χI+wI/N; χT=χT+wT/N; χB=χB+wB/N
        fTech=fTech+100*(rTech-1)/N
        fWk=fWk+100*(rWk-1)/N
    end
    #The two ratios of the averaged levels, which is how the paper would state them
    rTech_lev=100*(((1+χT/100)/(1+χI/100))-1)
    rWk_lev=100*(((1+χB/100)/(1+χT/100))-1)
    #Product of the averaged per-draw factors, which differs from the averaged total
    #only by the covariance of the factors across draws
    χProduct=100*(((1+χI/100)*(1+fTech/100)*(1+fWk/100))-1)
    return χI, χT, χB, fTech, fWk, rTech_lev, rWk_lev, χProduct, MaxResidual
end

function Targeted_Moments_Mechanism(FOLDER::String;FILES::Array{String,1}=MECHANISM_FILES)
    #The four moments the calibration targets, for the three variants, none of them
    #recalibrated. Targets are sigma_GDP=3.11, GDP drop in default=-13.28,
    #mean spreads=2.9, working capital over GDP=8.09
    MOM=Array{Moments,1}(undef,length(FILES))
    for i in 1:length(FILES)
        MOD=UnpackModel_File(FILES[i],FOLDER)
        MOM[i]=AverageMomentsManySamples(MOD.par.Tmom,MOD.par.NSamplesMoments,MOD)
    end
    return MOM
end

function Write_Mechanism_Decomposition(MOD_ident::Model,MOD_theta::Model,MOD_bench::Model,
                                       MOM::Array{Moments,1},FOLDER_TABLES::String)
    N=1000
    χI, χT, χB, fTech, fWk, rTech_lev, rWk_lev, χProduct,
        MaxResidual=Average_Welfare_Decomposition_Mechanism(N,MOD_ident,MOD_theta,MOD_bench)

    COLUMN=["chi identical","chi technology gaps only","chi benchmark",
            "technology gaps term","working capital gap term",
            "technology gaps term, ratio of averages","working capital term, ratio of averages",
            "product of the averaged factors","max identity residual across draws"]
    VALUES=[χI,χT,χB,fTech,fWk,rTech_lev,rWk_lev,χProduct,MaxResidual]
    writedlm(joinpath(FOLDER_TABLES,"Table_Mechanism_Decomposition.csv"),
             vcat(["term" "value"],hcat(COLUMN,VALUES)),',')

    #Drift of the four targeted moments across the same three economies
    DRIFT=Array{Any,2}(undef,length(MOM)+1,5)
    DRIFT[1,:]=["model" "sigma_GDP" "GDP_drop_default" "mean_spreads" "WK_GDP"]
    for i in 1:length(MOM)
        DRIFT[i+1,:]=[MECHANISM_LABELS[i] MOM[i].σ_GDP MOM[i].GDP_dropAv_DefEv MOM[i].MeanSpreads MOM[i].WK_GDP]
    end
    writedlm(joinpath(FOLDER_TABLES,"Table_Mechanism_Drift.csv"),DRIFT,',')

    return χI, χT, χB, fTech, fWk, rTech_lev, rWk_lev, χProduct, MaxResidual
end

#The same three economies as the chain above, reported as levels on the Gamma axis.
#Gamma runs 0.160, 0.282, 1.000 over the three
const MECHANISM_LEVEL_FILES=["Model_Benchmark.csv","Model_SameTheta.csv","Model_Identical.csv"]
const MECHANISM_LEVEL_LABELS=["benchmark","technology gaps only","identical"]

function Write_Mechanism_Levels(FOLDER::String,FOLDER_TABLES::String;
                                FILES::Array{String,1}=MECHANISM_LEVEL_FILES,
                                LABELS::Array{String,1}=MECHANISM_LEVEL_LABELS,
                                N::Int64=1000)
    #chi at each point of the Gamma axis, on the same draws from the benchmark ergodic
    #distribution, split by whether the draw is a default state
    MODS=[UnpackModel_File(FILES[i],FOLDER) for i in 1:length(FILES)]
    MOD_ref=MODS[1]
    @unpack drp = MOD_ref.par
    TS=Simulate_Paths(true,drp+N,MOD_ref)
    OUT=Array{Any,2}(undef,length(FILES)+1,9)
    OUT[1,:]=["model" "Gamma" "chi" "chi_repayment" "chi_default" "sigma_GDP" "GDP_drop_default" "mean_spreads" "WK_GDP"]
    for k in 1:length(MODS)
        c=zeros(N); d=zeros(N)
        for i in 1:N
            t=drp+i; d[i]=TS.Def[t]
            c[i]=Welfare_Gains_t(t,TS,MODS[k])
        end
        MOM=AverageMomentsManySamples(MODS[k].par.Tmom,MODS[k].par.NSamplesMoments,MODS[k])
        OUT[k+1,:]=[LABELS[k] Gamma_SufficientStatistic(MODS[k].par) mean(c) mean(c[d .== 0.0]) mean(c[d .== 1.0]) MOM.σ_GDP MOM.GDP_dropAv_DefEv MOM.MeanSpreads MOM.WK_GDP]
    end
    writedlm(joinpath(FOLDER_TABLES,"Table_Mechanism_Levels.csv"),OUT,',')
    return OUT
end
