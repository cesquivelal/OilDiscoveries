
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
    savefig(fig_5,"$FOLDER_GRAPHS\\Figure5.pdf")
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

function Results_Benchmark_Calibration(UseSavedFile::Bool,SETUP_FILE::String)
    #Solve model
    if UseSavedFile
        MOD_BEN=UnpackModel_File("Model_Benchmark.csv","Code_Model")
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

function Plot_Two_Responses_Alt_Models(Tbefore::Int64,Tafter::Int64,TS_Mod_no_g::Paths,TS_Mod_no_g_identical::Paths)
    #Details for graphs
    tend=Tafter#15
    t0=-Tbefore
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["no g" "no g, same technology" "" ""]
    LINESTYLES=[:solid :dash :dot :dot]
    COLORS=[:blue :green :black :black]

    #Plot Spreads
    TITLE="spreads"
    mod=TS_Mod_no_g.Spreads .- TS_Mod_no_g.Spreads[1]
    mod2=TS_Mod_no_g_identical.Spreads .- TS_Mod_no_g_identical.Spreads[1]
    plt_spreads=plot([t0:t1],[mod mod2],label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",#ylims=[-0.05,1.3],
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot GDP
    TITLE="GDP"
    mod=100*(log.(TS_Mod_no_g.GDP) .- log.(TS_Mod_no_g.GDP[1]))
    mod2=100*(log.(TS_Mod_no_g_identical.GDP) .- log.(TS_Mod_no_g_identical.GDP[1]))
    plt_gdp=plot([t0:t1],[mod mod2],legend=:bottomright,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot current account
    TITLE="current account"
    mod01=100*TS_Mod_no_g.CA ./ (1*TS_Mod_no_g.GDP)
    mod02=100*TS_Mod_no_g_identical.CA ./ (1*TS_Mod_no_g_identical.GDP)
    mod=mod01# .- mod01[1]
    mod2=mod02# .- mod01[1]
    plt_CA=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot government debt
    TITLE="government debt"
    mod01=100*TS_Mod_no_g.B ./ (1*TS_Mod_no_g.GDP[1])
    mod02=100*TS_Mod_no_g_identical.B ./ (1*TS_Mod_no_g_identical.GDP[1])
    # mod01=TS_Mod.B
    mod=mod01
    mod2=mod02
    plt_B=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of Av(GDP)",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot private consumption
    mod_c=100*(log.(TS_Mod_no_g.C) .- log.(TS_Mod_no_g.C[1]))
    mod_c2=100*(log.(TS_Mod_no_g_identical.C) .- log.(TS_Mod_no_g_identical.C[1]))
    TITLE="total consumption"
    plt_c=plot([t0:t1],[mod_c mod_c2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot fraction in default
    TITLE="fraction in default"
    mod=TS_Mod_no_g.Def
    mod2=TS_Mod_no_g_identical.Def
    plt_def=plot([t0:t1],[mod mod2],label=LABELS,
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

function Plot_Three_Spread_Schedules(DoLevels::Bool,MOD::Model,MOD_no_g::Model,MOD_no_g_identical::Model)
    #Compute long-run average debt
    T=10000; ForMoments=true
    TS0=Simulate_Paths(ForMoments,T,MOD)
    TS1=Simulate_Paths(ForMoments,T,MOD_no_g)
    TS2=Simulate_Paths(ForMoments,T,MOD_no_g_identical)
    B0=sum(TS0.B .* (1 .- TS0.Def))/sum((1 .- TS0.Def))
    B1=sum(TS1.B .* (1 .- TS1.Def))/sum((1 .- TS1.Def))
    B2=sum(TS2.B .* (1 .- TS2.Def))/sum((1 .- TS2.Def))

    z0=1.0

    foo_0(n_ind::Int64)=ComputeSpreadWithQ(MOD.SOLUTION.itp_q1(B0,z0,n_ind),MOD.par)
    foo_1(n_ind::Int64)=ComputeSpreadWithQ(MOD_no_g.SOLUTION.itp_q1(B1,z0,n_ind),MOD_no_g.par)
    foo_2(n_ind::Int64)=ComputeSpreadWithQ(MOD_no_g_identical.SOLUTION.itp_q1(B2,z0,n_ind),MOD_no_g_identical.par)

    nn=collect(range(-1,stop=length(MOD.GRIDS.GR_n)-2,length=length(MOD.GRIDS.GR_n)))
    yy0=Array{Float64,1}(undef,length(nn))
    yy1=Array{Float64,1}(undef,length(nn))
    yy2=Array{Float64,1}(undef,length(nn))
    for i in 1:length(nn)
        if DoLevels
            yy0[i]=foo_0(i)
            yy1[i]=foo_1(i)
            yy2[i]=foo_2(i)
        else
            if i==1
                yy0[i]=0.0
                yy1[i]=0.0
                yy2[i]=0.0
            else
                yy0[i]=foo_0(i)-foo_0(1)
                yy1[i]=foo_1(i)-foo_1(1)
                yy2[i]=foo_2(i)-foo_2(1)
            end
        end
    end
    #Details for plot
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["benchmark" "no g" "no g, same technology"]
    LINESTYLES=[:solid :dash :dot]
    COLORS=[:blue :green :black]
    YLABEL="percentage points"
    XLAEL="years since discovery"
    plt=plot(nn,[yy0 yy1 yy2],
             label=LABELS,
             linestyle=LINESTYLES,
             linecolor=COLORS,
             legend=:best,
             size=SIZE_PLOTS)
    return plt
end

function Plot_Two_Spread_Schedules_n(DoLevels::Bool,MOD_no_g::Model,MOD_no_g_identical::Model)
    #Compute long-run average debt
    T=10000; ForMoments=true
    TS1=Simulate_Paths(ForMoments,T,MOD_no_g)
    TS2=Simulate_Paths(ForMoments,T,MOD_no_g_identical)
    B1=1.0*sum(TS1.B .* (1 .- TS1.Def))/sum((1 .- TS1.Def))
    B2=1.0*sum(TS2.B .* (1 .- TS2.Def))/sum((1 .- TS2.Def))

    z0=1.0

    foo_1(n_ind::Int64)=ComputeSpreadWithQ(MOD_no_g.SOLUTION.itp_q1(B1,z0,n_ind),MOD_no_g.par)
    foo_2(n_ind::Int64)=ComputeSpreadWithQ(MOD_no_g_identical.SOLUTION.itp_q1(B2,z0,n_ind),MOD_no_g_identical.par)

    nn=collect(range(-1,stop=length(MOD_no_g.GRIDS.GR_n)-2,length=length(MOD_no_g.GRIDS.GR_n)))
    yy1=Array{Float64,1}(undef,length(nn))
    yy2=Array{Float64,1}(undef,length(nn))
    for i in 1:length(nn)
        if DoLevels
            yy1[i]=foo_1(i)
            yy2[i]=foo_2(i)
        else
            if i==1
                yy1[i]=0.0
                yy2[i]=0.0
            else
                yy1[i]=foo_1(i)-foo_1(1)
                yy2[i]=foo_2(i)-foo_2(1)
            end
        end
    end
    #Details for plot
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["different technologies" "same technology"]
    LINESTYLES=[:solid :dash :dot]
    COLORS=[:blue :green :black]
    YLABEL="percentage points"
    XLABEL="years since discovery"
    if DoLevels
        TITLE="spreads, z and b fixed"
    else
        TITLE="change in spreads, z and b fixed"
    end
    plt=plot(nn,[yy1 yy2],
             label=LABELS,
             title=TITLE,
             linestyle=LINESTYLES,
             linecolor=COLORS,
             ylabel=YLABEL,
             xlabel=XLABEL,
             legend=:bottomleft,
             size=SIZE_PLOTS)
    return plt
end

function Plot_One_Spread_Schedule_b(TITLE::String,MOD::Model)
    @unpack SOLUTION, GRIDS, par = MOD
    @unpack GR_b, GR_n = GRIDS
    @unpack Nz = par
    @unpack itp_q1 = SOLUTION
    bb=GR_b; z0=1.0
    foo_L(b::Float64)=ComputeSpreadWithQ(itp_q1(b,z0,1),par)
    foo_disc(b::Float64)=ComputeSpreadWithQ(itp_q1(b,z0,2),par)
    foo_H(b::Float64)=ComputeSpreadWithQ(itp_q1(b,z0,length(GR_n)),par)
    yy_L=foo_L.(bb)
    yy_disc=foo_disc.(bb)
    yy_H=foo_H.(bb)

    T=10000; ForMoments=true
    TS=Simulate_Paths(ForMoments,T,MOD)
    bb_y=100*bb ./ mean(TS.GDP)

    #Details for plot
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["N=NL" "discovery" "N=NH"]
    LINESTYLES=[:solid :dash :dot]
    COLORS=[:blue :green :black]
    YLABEL="percentage points"
    XLABEL="debt, percentage of Av(GDP)"
    YLIMS=[0.0,10.0]
    XLIMS=[0.0,75.0]
    plt=plot(bb_y,[yy_L yy_disc yy_H],
             label=LABELS,
             title=TITLE,
             linestyle=LINESTYLES,
             linecolor=COLORS,
             legend=:topleft,
             ylims=YLIMS,
             xlims=XLIMS,
             ylabel=YLABEL,
             xlabel=XLABEL,
             size=SIZE_PLOTS)
    return plt
end

function Plot_Compare_SpreadSchedules(MOD_no_g::Model,MOD_no_g_identical::Model)
    DoLevels=false
    plt_1=Plot_Two_Spread_Schedules_n(DoLevels,MOD_no_g,MOD_no_g_identical)
    TITLE="no g, different technologies"
    plt_2=Plot_One_Spread_Schedule_b(TITLE,MOD_no_g)
    TITLE="no g, identical technologies"
    plt_3=Plot_One_Spread_Schedule_b(TITLE,MOD_no_g_identical)
    #Create plot array
    #Details for plot
    size_width=600
    size_height=400
    l = @layout([a b c])
    plt=plot(plt_1,plt_2,plt_3,
             layout=l,size=(size_width*3,size_height*1))
    return plt
end

function Results_Two_Alternatives(UseSavedFile::Bool,SETUP_FILE::String)
    #Solve models
    if UseSavedFile
        MOD_no_g=UnpackModel_File("Model_No_g.csv","Code_Model")
        MOD_no_g_identical=UnpackModel_File("Model_No_g_identical.csv","Code_Model")
    else
        col_no_g=5; col_no_g_identical=6
        MOD_no_g, NAME=Model_FromSetup(col_no_g,SETUP_FILE)
        MOD_no_g_identical, NAME=Model_FromSetup(col_no_g_identical,SETUP_FILE)
    end

    #Compute moments
    MOM_no_g=AverageMomentsManySamples(MOD_no_g.par.Tmom,MOD_no_g.par.NSamplesMoments,MOD_no_g)
    MOM_no_g_identical=AverageMomentsManySamples(MOD_no_g_identical.par.Tmom,MOD_no_g_identical.par.NSamplesMoments,MOD_no_g_identical)

    #Compute average responses
    DropDefaults=false
    N=10000; Tbefore=2; Tafter=15
    TS_Mod_no_g=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_no_g)
    TS_Mod_no_g_identical=AverageDiscoveryPaths(DropDefaults,N,Tbefore,Tafter,MOD_no_g_identical)
    fig_9=Plot_Two_Responses_Alt_Models(Tbefore,Tafter,TS_Mod_no_g,TS_Mod_no_g_identical)

    fig_10=Plot_Compare_SpreadSchedules(MOD_no_g,MOD_no_g_identical)

    return MOD_no_g, MOD_no_g_identical, MOM_no_g, MOM_no_g_identical, fig_9, fig_10
end
