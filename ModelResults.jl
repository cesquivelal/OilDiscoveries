#Assumes that ModelPrimitives.jl is already called

function SolveTwoCanonicalModels(country_column::Int64)
    XX=readdlm("Setup.csv",',')
    VEC=XX[2:end,country_column]*1.0
    par=UnpackParameters_Vector(VEC)
    GRIDS=CreateGrids(par)

    PrintProgress=true
    par=Pars(par,OilImmune=false)
    MOD_Cost_to_total=SolveModel_VFI(PrintProgress,GRIDS,par)

    par=Pars(par,OilImmune=true)
    MOD_No_Cost_to_oil=SolveModel_VFI(PrintProgress,GRIDS,par)

    return MOD_Cost_to_total, MOD_No_Cost_to_oil
end

function SolveQuantitativeModel(country_column::Int64)
    XX=readdlm("Setup.csv",',')
    VEC=XX[2:end,country_column]*1.0
    par=UnpackParameters_Vector(VEC)
    GRIDS=CreateGrids(par)

    PrintProgress=true
    MOD_Quant=SolveModel_VFI(PrintProgress,GRIDS,par)

    return MOD_Quant
end

################################################################################
### Results for paper
################################################################################
#Simple model, oil penalty vs oil immune
function Plot_Figure_3(Tafter::Int64,TS_Ben::Paths,TS_Oi::Paths)
    #Details for graphs
    npv=4.5
    tend=Tafter-1#15
    t0=0
    t1=tend

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
    mod=TS_Ben.Spreads[2:t1+2] .- TS_Ben.Spreads[1]
    mod2=TS_Oi.Spreads[2:t1+2] .- TS_Oi.Spreads[1]
    plt_spreads=plot([t0:t1],[mod mod2],label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",#ylims=[-0.05,1.3],
        legend=:topright,size=SIZE_PLOTS,linewidth=LW)

    #Plot GDP
    TITLE="GDP"
    mod=100*(log.(TS_Ben.GDP[2:t1+2]) .- log.(TS_Ben.GDP[1]))
    mod2=100*(log.(TS_Oi.GDP[2:t1+2]) .- log.(TS_Oi.GDP[1]))
    plt_gdp=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot current account
    TITLE="current account"
    mod01=100*TS_Ben.CA ./ (1*TS_Ben.GDP)
    mod=mod01[2:t1+2] .- mod01[1]
    mod02=100*TS_Oi.CA ./ (1*TS_Oi.GDP)
    mod2=mod02[2:t1+2] .- mod02[1]
    plt_CA=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot government debt
    TITLE="government debt"
    mod01=100*TS_Ben.B ./ (1*TS_Ben.GDP)
    mod=mod01[2:t1+2] .- mod01[1]
    mod02=100*TS_Oi.B ./ (1*TS_Oi.GDP)
    mod2=mod02[2:t1+2] .- mod02[1]
    plt_B=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot consumption
    TITLE="consumption"
    mod=100*(log.(TS_Ben.Cons[2:t1+2]) .- log.(TS_Ben.Cons[1]))
    mod2=100*(log.(TS_Oi.Cons[2:t1+2]) .- log.(TS_Oi.Cons[1]))
    plt_c=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Create plot array
    l = @layout([a b])
    plt=plot(plt_spreads,plt_gdp,
             layout=l,size=(size_width*2,size_height*1))
    return plt
end

function Plot_Figure_5_WithData(FileRegs::String,npv::Float64,Tafter::Int64,TS_Mod::Paths)
    #Details for graphs
    tend=Tafter-1#15
    t0=0
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["data" "model" "" ""]
    LINESTYLES=[:solid :dash :dot :dot]
    COLORS=[:black :green :black :black]

    #Plot Spreads
    TITLE="spreads"
    COLUMN=2
    reg=Read_Regression_Objects(FileRegs,COLUMN)
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,Tafter,reg)
    mod=TS_Mod.Spreads[2:t1+2] .- TS_Mod.Spreads[1]
    plt_spreads=plot([t0:t1],[Δy[2:end] mod cilow[2:end] cihigh[2:end]],label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",#ylims=[-0.05,1.3],
        legend=:topright,size=SIZE_PLOTS,linewidth=LW)

    #Plot GDP
    TITLE="GDP"
    COLUMN=5
    reg=Read_Regression_Objects(FileRegs,COLUMN)
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,Tafter,reg)
    mod=100*(log.(TS_Mod.GDP[2:t1+2]) .- log.(TS_Mod.GDP[1]))
    plt_gdp=plot([t0:t1],[Δy[2:end] mod cilow[2:end] cihigh[2:end]],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot current account
    TITLE="current account"
    COLUMN=8
    reg=Read_Regression_Objects(FileRegs,COLUMN)
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,Tafter,reg)
    mod01=100*TS_Mod.CA ./ (1*TS_Mod.GDP)
    mod=mod01[2:t1+2] .- mod01[1]
    plt_CA=plot([t0:t1],[Δy[2:end] mod cilow[2:end] cihigh[2:end]],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot government debt
    TITLE="government debt"
    COLUMN=17
    reg=Read_Regression_Objects(FileRegs,COLUMN)
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,Tafter,reg)
    mod01=100*TS_Mod.B ./ (1*TS_Mod.GDP)
    mod=mod01[2:t1+2] .- mod01[1]
    plt_B=plot([t0:t1],[Δy[2:end] mod cilow[2:end] cihigh[2:end]],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot private consumption
    TITLE="private consumption"
    COLUMN=23
    reg=Read_Regression_Objects(FileRegs,COLUMN)
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,Tafter,reg)
    mod=100*(log.(TS_Mod.Cons[2:t1+2]) .- log.(TS_Mod.Cons[1]))
    plt_c=plot([t0:t1],[Δy[2:end] mod cilow[2:end] cihigh[2:end]],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot government consumption
    TITLE="government consumption"
    COLUMN=26
    reg=Read_Regression_Objects(FileRegs,COLUMN)
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,Tafter,reg)
    mod=100*(log.(TS_Mod.G[2:t1+2]) .- log.(TS_Mod.G[1]))
    plt_g=plot([t0:t1],[Δy[2:end] mod cilow[2:end] cihigh[2:end]],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Create plot array
    l = @layout([a b; c d; e f])
    plt=plot(plt_spreads,plt_gdp,
             plt_CA,plt_B,plt_c,plt_g,
             layout=l,size=(size_width*2,size_height*3))
    return plt
end

function Plot_Figure_5_JustModel(Sell::Bool,npv::Float64,Tafter::Int64,TS_Mod::Paths)
    #Details for graphs
    tend=Tafter-1#15
    t0=0
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["data" "model" "" ""]
    LINESTYLES=[:solid :dash :dot :dot]
    COLORS=[:black :green :black :black]

    #Plot Spreads
    TITLE="spreads"
    mod=TS_Mod.Spreads[2:t1+2] .- TS_Mod.Spreads[1]
    plt_spreads=plot([t0:t1],mod,label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",#ylims=[-0.05,1.3],
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot GDP
    TITLE="GDP"
    mod=100*(log.(TS_Mod.GDP[2:t1+2]) .- log.(TS_Mod.GDP[1]))
    plt_gdp=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot current account
    TITLE="current account"
    mod01=100*TS_Mod.CA ./ (1*TS_Mod.GDP)
    mod=mod01[2:t1+2] .- mod01[1]
    plt_CA=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot government debt
    TITLE="government debt"
    mod01=100*TS_Mod.B ./ (1*TS_Mod.GDP)
    mod=mod01[2:t1+2] .- mod01[1]
    plt_B=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot private consumption
    if Sell
        YLIMS_C=[-1.0,35.0]
    else
        YLIMS_C=[-1.0,13.0]
    end
    TITLE="private consumption"
    mod=100*(log.(TS_Mod.Cons[2:t1+2]) .- log.(TS_Mod.Cons[1]))
    plt_c=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,ylims=YLIMS_C)

    #Plot government consumption
    TITLE="government consumption"
    mod=100*(log.(TS_Mod.G[2:t1+2]) .- log.(TS_Mod.G[1]))
    plt_g=plot([t0:t1],mod,legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)#,ylims=YLIMS_C)

    #Create plot array
    l = @layout([a b; c d; e f])
    plt=plot(plt_spreads,plt_gdp,
             plt_CA,plt_B,plt_c,plt_g,
             layout=l,size=(size_width*2,size_height*3))
    return plt
end

function Plot_Figure_5(Sell::Bool,WithData::Bool,FileRegs::String,npv::Float64,Tafter::Int64,TS_Mod::Paths)
    if WithData
        return Plot_Figure_5_WithData(FileRegs,npv,Tafter,TS_Mod)
    else
        return Plot_Figure_5_JustModel(Sell,npv,Tafter,TS_Mod)
    end
end

function PlotFigure_6(Tafter::Int64,TS_Mod::Paths)
    #Details for graphs
    tend=Tafter-1#15
    t0=0
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["data" "model" "" ""]
    LINESTYLES=[:solid :dash :dot :dot]
    COLORS=[:black :green :black :black]

    #Plot in default
    mod=TS_Mod.Def[2:t1+2]
    plt_Def=plot([t0:t1],mod,label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,
        ylabel="fraction in default",xlabel="t",
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    return plt_Def
end

function Plot_Figure_8(Tafter::Int64,TS_DomField::Paths,TS_OilEmbargo::Paths)
    #Details for graphs
    tend=Tafter-1#15
    t0=0
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["domestic field" "oil embargo"]
    LINESTYLES=[:solid :dash]
    COLORS=[:green :orange]

    #Plot Spreads
    TITLE="spreads"
    mod=TS_DomField.Spreads[2:t1+2] .- TS_DomField.Spreads[1]
    mod2=TS_OilEmbargo.Spreads[2:t1+2] .- TS_OilEmbargo.Spreads[1]
    plt_spreads=plot([t0:t1],[mod mod2],label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",
        legend=:topright,size=SIZE_PLOTS,linewidth=LW)

    #Plot GDP
    TITLE="GDP"
    mod=100*(log.(TS_DomField.GDP[2:t1+2]) .- log.(TS_DomField.GDP[1]))
    mod2=100*(log.(TS_OilEmbargo.GDP[2:t1+2]) .- log.(TS_OilEmbargo.GDP[1]))
    plt_gdp=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot current account
    TITLE="current account"
    mod01=100*TS_DomField.CA ./ (1*TS_DomField.GDP)
    mod02=100*TS_OilEmbargo.CA ./ (1*TS_OilEmbargo.GDP)
    mod=mod01[2:t1+2] .- mod01[1]
    mod2=mod02[2:t1+2] .- mod02[1]
    plt_CA=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot government debt
    TITLE="government debt"
    mod01=100*TS_DomField.B ./ (1*TS_DomField.GDP)
    mod02=100*TS_OilEmbargo.B ./ (1*TS_OilEmbargo.GDP)
    mod=mod01[2:t1+2] .- mod01[1]
    mod2=mod02[2:t1+2] .- mod02[1]
    plt_B=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot private consumption
    YLIMS_C=[-1.0,16.0]
    TITLE="private consumption"
    mod=100*(log.(TS_DomField.Cons[2:t1+2]) .- log.(TS_DomField.Cons[1]))
    mod2=100*(log.(TS_OilEmbargo.Cons[2:t1+2]) .- log.(TS_OilEmbargo.Cons[1]))
    plt_c=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,ylims=YLIMS_C)

    #Plot government consumption
    TITLE="government consumption"
    mod=100*(log.(TS_DomField.G[2:t1+2]) .- log.(TS_DomField.G[1]))
    mod2=100*(log.(TS_OilEmbargo.G[2:t1+2]) .- log.(TS_OilEmbargo.G[1]))
    plt_g=plot([t0:t1],[mod mod2],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,ylims=YLIMS_C)

    #Create plot array
    l = @layout([a b; c d; e f])
    plt=plot(plt_spreads,plt_gdp,
             plt_CA,plt_B,plt_c,plt_g,
             layout=l,size=(size_width*2,size_height*3))
    return plt
end

function PlotFigure_9(Tafter::Int64,TS_DomField::Paths,TS_OilEmbargo::Paths)
    #Details for graphs
    tend=Tafter-1#15
    t0=0
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["domestic field" "oil embargo"]
    LINESTYLES=[:solid :dash]
    COLORS=[:green :orange]

    #Plot in default
    mod=TS_DomField.Def[2:t1+2]
    mod2=TS_OilEmbargo.Def[2:t1+2]
    plt_Def=plot([t0:t1],[mod mod2],label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,
        ylabel="fraction in default",xlabel="t",
        legend=:bottomright,size=SIZE_PLOTS,linewidth=LW)

    return plt_Def
end
