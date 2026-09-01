
using Plots; pythonplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                        background_color_legend=nothing,foreground_color_legend=nothing,
                        legendfontsize=18,guidefontsize=18,titlefontsize=20,tickfontsize=18,
                        markersize=9,size=(650,400))

using Parameters, Distributions

@with_kw mutable struct RegressionObjects
    #Persistence coefficient
    ρ::Float64
    #Coefficients of interest
    psi::Vector{Float64}
    #Confidence intervals
    CIρ::Vector{Float64}
    CIpsi::Array{Float64,2}
end

#Read the tab-delimited file that outreg2 writes alongside the .xls. Regressions are
#located by the ctitle given in EmpiricalResults.do, not by column number, so adding or
#reordering regressions in the .do file does not change which one each figure reads.
#Each regression occupies three columns: coefficient, ci_low, ci_high.
function Read_Regression_Table(FILE::String)
    return [String.(split(chomp(line),'\t')) for line in eachline(FILE)]
end

#TERM is the name of the discovery-size regressor in the .do file. It is "npv" for every
#regression in the paper, but the alternative-size exercise runs the same specification with
#alternative measures, whose coefficients carry their own names. NLEADS>0 also reads that
#many leads, which come first in psi so that it runs from the earliest lead to the last lag.
function Read_Regression_Objects(FILE::String,SPEC::String,TERM::String="npv",NLEADS::Int64=0)
    tab=Read_Regression_Table(FILE)

    #Locate the regression by name in the header row, and the coefficients by row label
    hrow=findfirst(row -> length(row)>0 && row[1]=="VARIABLES",tab)
    hrow===nothing && error("no VARIABLES header row in $FILE")
    col=findfirst(==(SPEC),tab[hrow])
    col===nothing && error("no regression named '$SPEC' in $FILE")

    function Read_Row(LABEL::String)
        r=findfirst(row -> length(row)>0 && row[1]==LABEL,tab)
        r===nothing && error("no row '$LABEL' in $FILE")
        return parse.(Float64,tab[r][col:col+2])
    end

    ar1=Read_Row("ar1")
    ρ=ar1[1]
    CIρ=ar1[2:3]

    LABELS=[["$(TERM)_f$i" for i in NLEADS:-1:1];[TERM];["$(TERM)_$i" for i in 1:10]]
    psi=Vector{Float64}(undef,length(LABELS))
    CIpsi=Array{Float64,2}(undef,length(LABELS),2)
    for i in 1:length(LABELS)
        row=Read_Row(LABELS[i])
        psi[i]=row[1]
        CIpsi[i,1]=row[2]
        CIpsi[i,2]=row[3]
    end
    return RegressionObjects(ρ,psi,CIρ,CIpsi)
end

function ImpulseResponse_TS(npv::Float64,T::Int64,reg::RegressionObjects)
    @unpack ρ, CIρ, psi, CIpsi = reg
    Δy=Vector{Float64}(undef,T+1)
    cilow=Vector{Float64}(undef,T+1)
    cihigh=Vector{Float64}(undef,T+1)

    #Impact
    Δy[1]=psi[1]*npv
    cilow[1]=CIpsi[1,1]*npv
    cihigh[1]=CIpsi[1,2]*npv

    #Following years
    for t in 2:T+1
        if t<=length(psi)
            Δy[t]=ρ*Δy[t-1]+psi[t]*npv
            cilow[t]=CIρ[1]*Δy[t-1]+CIpsi[t,1]*npv
            cihigh[t]=CIρ[2]*Δy[t-1]+CIpsi[t,2]*npv
        else
            Δy[t]=ρ*Δy[t-1]
            cilow[t]=CIρ[1]*Δy[t-1]
            cihigh[t]=CIρ[2]*Δy[t-1]
        end
    end
    return Δy, cilow, cihigh
end

function PlotImpulseResponse(name::String,units::String,npv::Float64,T::Int64,reg::RegressionObjects,IsConsumption::Bool,YLIMS_C::Array{Float64,1})
    #Data for plot
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,T,reg)
    xx=collect(range(0,stop=T-1,length=length(Δy)))
    z0=zeros(Float64,length(Δy))

    #Plot parameters
    XLABEL="t"
    XTICKS=0:5:15
    COLORS=[:black :black :black :black]
    LINESTYLES=[:solid :solid :dot :dot]

    if IsConsumption
        YLIMS=deepcopy(YLIMS_C)
        #Make plot
        plt=plot(xx,[z0 Δy cilow cihigh],title=name,
                 xlabel=XLABEL,ylabel=units,legend=false,
                 linestyle=LINESTYLES,linecolor=COLORS,
                 xticks=XTICKS,ylims=YLIMS)

        #Return
        return plt
    else
        #Make plot
        plt=plot(xx,[z0 Δy cilow cihigh],title=name,
                 xlabel=XLABEL,ylabel=units,legend=false,
                 linestyle=LINESTYLES,linecolor=COLORS,
                 xticks=XTICKS)

        #Return
        return plt
    end
end

############################################
###### Plots in main text
############################################
function Figures_Empirical(File::String,npv::Float64,T::Int64)
    size_width=550
    size_height=400
    FILE=File
    YLIMS_C=[0.0,0.0]

    #Spreads
    name=""
    SPEC="spreads"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage points"
    plt_spr=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #GDP
    name="GDP"
    SPEC="gdp"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage change"
    plt_gdp=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Current account
    name="current account"
    SPEC="ca"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage of GDP"
    plt_ca=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Net government debt
    name="net government debt"
    SPEC="net debt"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage of GDP"
    plt_gde=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #private consumption
    YLIMS_C=[-0.4,0.6]
    name="private"
    SPEC="private"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage change"
    plt_c=PlotImpulseResponse(name,units,npv,T,reg,true,YLIMS_C)

    #government consumption
    name="government"
    SPEC="government"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage change"
    plt_g=PlotImpulseResponse(name,units,npv,T,reg,true,YLIMS_C)

    #Create plot array
    l = @layout([a b])
    plt_cons=plot(plt_c,plt_g,
             layout=l,size=(size_width*2,size_height*1))

    #Create plot array
    l = @layout([a b c])
    plt_mac=plot(plt_gdp,plt_ca,plt_gde,
             layout=l,size=(size_width*3,size_height*1))
    return plt_spr, plt_mac, plt_cons
end

function Figures_Empirical_Appendix(File::String,npv::Float64,T::Int64)
    size_width=550
    size_height=400
    FILE=File
    YLIMS_C=[0.0,0.0]

    #Spreads, control for reserves
    name="control for log(reserves)"
    SPEC="spreads_res"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage points"
    plt_spr_res=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Spreads, control for reserves and lags
    name="control for 10 lags of log(reserves)"
    SPEC="spreads_res_lags"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage points"
    plt_spr_res_lag=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #GDP
    name="GDP"
    SPEC="gdp_embi"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage change"
    plt_gdp=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Current account
    name="current account"
    SPEC="ca_embi"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage of GDP"
    plt_ca=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Net government debt
    name="net government debt"
    SPEC="net debt_embi"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage of GDP"
    plt_gde=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #private consumption
    YLIMS_C=[-0.4,1.0]
    name="private"
    SPEC="private_embi"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage change"
    plt_c=PlotImpulseResponse(name,units,npv,T,reg,true,YLIMS_C)

    #government consumption
    name="government"
    SPEC="government_embi"
    reg=Read_Regression_Objects(FILE,SPEC)
    units="percentage change"
    plt_g=PlotImpulseResponse(name,units,npv,T,reg,true,YLIMS_C)

    #Create plot array
    l = @layout([a b])
    plt_spr_res=plot(plt_spr_res,plt_spr_res_lag,
             layout=l,size=(size_width*2,size_height*1))

    #Create plot array
    l = @layout([a b])
    plt_cons=plot(plt_c,plt_g,
             layout=l,size=(size_width*2,size_height*1))

    #Create plot array
    l = @layout([a b c])
    plt_mac=plot(plt_gdp,plt_ca,plt_gde,
             layout=l,size=(size_width*3,size_height*1))

    return plt_spr_res, plt_mac, plt_cons
end

############################################
###### Figure 14, alternative measures of discovery size
############################################
#The response of spreads to the same discovery measured three ways, with the paper's
#benchmark drawn in red inside each panel. Each series is scaled by the median discovery
#under its own measure, so they are comparable even though the units are not: the first
#three are percent of GDP and the last is millions of barrels. Medians are taken over the
#61 giant discoveries in the 37 EMBI countries between 1993 and 2012.
const E1B_BENCHMARK=("spreads","npv",4.4926)
const E1B_COMPARE=[("e1b_risk","npv_risk",   3.1494,"country r, constant profile"),
                   ("e1b_com5","npv_com5",   4.6995,"r = 5%, constant profile"),
                   ("e1b_urr", "npv_urr", 1000.0000,"reserves, no discounting")]

function Figures_Empirical_E1b(File::String,T::Int64)
    size_width=550
    size_height=400

    #The paper's benchmark, repeated in every panel. This is the Figure 2 regression itself,
    #so it is read from the spreads column rather than re-estimated in the do file.
    SPEC_B, TERM_B, SCALE_B = E1B_BENCHMARK
    REG_B=Read_Regression_Objects(File,SPEC_B,TERM_B)
    Δy_B, cilow_B, cihigh_B=ImpulseResponse_TS(SCALE_B,T,REG_B)

    REGS=[Read_Regression_Objects(File,SPEC,TERM) for (SPEC,TERM,SCALE,NAME) in E1B_COMPARE]
    IRFS=[ImpulseResponse_TS(E1B_COMPARE[i][3],T,REGS[i]) for i in 1:length(E1B_COMPARE)]

    lo=minimum([minimum(cilow_B);[minimum(irf[2]) for irf in IRFS]])
    hi=maximum([maximum(cihigh_B);[maximum(irf[3]) for irf in IRFS]])
    pad=0.05*(hi-lo)
    YLIMS=[lo-pad,hi+pad]

    xx=collect(range(0,stop=T-1,length=length(Δy_B)))
    z0=zeros(Float64,length(Δy_B))

    COLORS=[:black :red :red :red :black :black :black]
    LINESTYLES=[:solid :solid :dot :dot :solid :dot :dot]

    PLTS=Vector{Any}(undef,length(E1B_COMPARE))
    for i in 1:length(E1B_COMPARE)
        SPEC, TERM, SCALE, NAME = E1B_COMPARE[i]
        Δy, cilow, cihigh = IRFS[i]
        LABELS= i==1 ? ["" "benchmark" "" "" "alternative" "" ""] : ["" "" "" "" "" "" ""]
        PLTS[i]=plot(xx,[z0 Δy_B cilow_B cihigh_B Δy cilow cihigh],title=NAME,
                     xlabel="t",ylabel="percentage points",
                     linestyle=LINESTYLES,linecolor=COLORS,
                     label=LABELS,legend=(i==1 ? :topleft : false),legendfontsize=14,
                     xticks=0:5:15,ylims=YLIMS)
    end

    l = @layout([a b c])
    plt=plot(PLTS...,layout=l,size=(size_width*3,size_height*1))
    return plt
end

############################################
###### Figure 17, dropping the discoveries from 2009 on
############################################
#The response of spreads with the nine discoveries of 2009 and later set to zero. The
#estimation window and the controls are the benchmark's, so every surviving discovery
#keeps all ten horizons. Both series are scaled by the same median discovery.
#Regression e2c_obs, which truncates the estimation window at 2008 instead of removing
#the treatments, is estimated by EmpiricalResults.do but is not drawn.
const E2C_SPEC=("e2c_treat","npv_pre")

function Figures_Empirical_E2c(File::String,npv::Float64,T::Int64)
    REG_B=Read_Regression_Objects(File,"spreads","npv")
    Δy_B, cilow_B, cihigh_B=ImpulseResponse_TS(npv,T,REG_B)

    SPEC, TERM = E2C_SPEC
    REG=Read_Regression_Objects(File,SPEC,TERM)
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,T,REG)

    xx=collect(range(0,stop=T-1,length=length(Δy_B)))
    z0=zeros(Float64,length(Δy_B))

    plt=plot(xx,[z0 Δy_B cilow_B cihigh_B Δy cilow cihigh],
             xlabel="t",ylabel="percentage points",
             linestyle=[:solid :solid :dot :dot :solid :dot :dot],
             linecolor=[:black :red :red :red :black :black :black],
             label=["" "benchmark" "" "" "restricted" "" ""],
             legend=:topleft,legendfontsize=14,xticks=0:5:15)
    return plt
end

############################################
###### Figure 18, the six responses with and without the oil-price interactions
############################################
#Every outcome of Figures 2, 3 and 4 estimated both with and without the ten
#discovery-by-oil-price interactions. The specification in the paper is in red and the one
#without them in black. Everything else is each outcome's own benchmark. Sample sizes are
#equal within a panel and differ across panels, as they do between Figures 2, 3 and 4.
const R24_PAIRS=[("spreads",               "percentage points","spreads", "r24_noint"),
                 ("GDP",                   "percentage change","gdp",     "r24_gdp_noint"),
                 ("current account",       "percentage of GDP","ca",      "r24_ca_noint"),
                 ("net government debt",   "percentage of GDP","net debt","r24_nd_noint"),
                 ("private consumption",   "percentage change","private", "r24_priv_noint"),
                 ("government consumption","percentage change","government","r24_gov_noint")]

function Figures_Empirical_R24(File::String,npv::Float64,T::Int64)
    size_width=550
    size_height=400
    PANELS=[]
    for (name,units,SPEC_W,SPEC_N) in R24_PAIRS
        REG_W=Read_Regression_Objects(File,SPEC_W,"npv")
        REG_N=Read_Regression_Objects(File,SPEC_N,"npv")
        Δy_W, cilow_W, cihigh_W=ImpulseResponse_TS(npv,T,REG_W)
        Δy_N, cilow_N, cihigh_N=ImpulseResponse_TS(npv,T,REG_N)

        xx=collect(range(0,stop=T-1,length=length(Δy_W)))
        z0=zeros(Float64,length(Δy_W))

        #Only the first panel carries the legend, the rest share its convention
        ShowLegend=(name=="spreads")
        LABELS=ShowLegend ? ["" "with interactions" "" "" "without" "" ""] : ["" "" "" "" "" "" ""]
        plt=plot(xx,[z0 Δy_W cilow_W cihigh_W Δy_N cilow_N cihigh_N],title=name,
                 xlabel="t",ylabel=units,
                 linestyle=[:solid :solid :dot :dot :solid :dot :dot],
                 linecolor=[:black :red :red :red :black :black :black],
                 label=LABELS,legend=ShowLegend ? :topleft : false,
                 legendfontsize=13,xticks=0:5:15)
        push!(PANELS,plt)
    end

    l = @layout([a b; c d; e f])
    return plot(PANELS...,layout=l,size=(size_width*2,size_height*3))
end

############################################
###### Figure 19, the URR specification with and without the oil-price interactions
############################################
#The same comparison repeated with undiscounted reserves in place of the net present value,
#which is the third panel of Figure 14. The specification in the paper is in red and the one
#without the ten interactions in black. Both are estimated on the URR regression's own 437
#observations. Scaled by the median discovery under this measure, 1,000 million barrels, so
#the level is not comparable with Figure 2's.
const R24_URR=("e1b_urr","r24_urr_noint","npv_urr",1000.0)

function Figures_Empirical_R24_URR(File::String,T::Int64)
    SPEC_W, SPEC_N, TERM, SCALE = R24_URR
    REG_W=Read_Regression_Objects(File,SPEC_W,TERM)
    REG_N=Read_Regression_Objects(File,SPEC_N,TERM)
    Δy_W, cilow_W, cihigh_W=ImpulseResponse_TS(SCALE,T,REG_W)
    Δy_N, cilow_N, cihigh_N=ImpulseResponse_TS(SCALE,T,REG_N)

    xx=collect(range(0,stop=T-1,length=length(Δy_W)))
    z0=zeros(Float64,length(Δy_W))

    plt=plot(xx,[z0 Δy_W cilow_W cihigh_W Δy_N cilow_N cihigh_N],
             xlabel="t",ylabel="percentage points",
             linestyle=[:solid :solid :dot :dot :solid :dot :dot],
             linecolor=[:black :red :red :red :black :black :black],
             label=["" "with interactions" "" "" "without" "" ""],
             legend=:topleft,legendfontsize=14,xticks=0:5:15)
    return plt
end

############################################
###### Figure 13, pre-trends with leads
############################################
#The coefficients themselves rather than the impulse response they imply, which compounds
#every coefficient through ρ. They are scaled by the size of the discovery, as the
#responses are. The first panel is the Figure 2 regression itself, so it is read from the
#spreads column.
const E1A_SPECS=[("spreads",  0,"no leads"),
                 ("e1a_lead1",1,"one lead"),
                 ("e1a_lead3",3,"three leads")]

function Figures_Empirical_E1a(File::String,npv::Float64)
    size_width=550
    size_height=400

    REGS=[Read_Regression_Objects(File,SPEC,"npv",NLEADS) for (SPEC,NLEADS,NAME) in E1A_SPECS]

    #One vertical scale for all four, so the panels can be read against each other
    lo=minimum([minimum(r.CIpsi[:,1]) for r in REGS])*npv
    hi=maximum([maximum(r.CIpsi[:,2]) for r in REGS])*npv
    pad=0.05*(hi-lo)
    YLIMS=[lo-pad,hi+pad]

    PLTS=Vector{Any}(undef,length(E1A_SPECS))
    for i in 1:length(E1A_SPECS)
        SPEC, NLEADS, NAME = E1A_SPECS[i]
        xx=collect(-NLEADS:10)
        yy=REGS[i].psi*npv
        ERRLOW=yy.-REGS[i].CIpsi[:,1]*npv
        ERRHIGH=REGS[i].CIpsi[:,2]*npv.-yy
        PLTS[i]=scatter(xx,yy,yerror=(ERRLOW,ERRHIGH),title=NAME,
                        xlabel="t",ylabel="percentage points",legend=false,
                        markercolor=:black,markerstrokecolor=:black,markersize=5,
                        linecolor=:black,xticks=[-3,0,5,10],xlims=(-3.5,10.5),ylims=YLIMS)
        hline!(PLTS[i],[0.0],linecolor=:black,linestyle=:solid,label="")
        vline!(PLTS[i],[0.0],linecolor=:gray,linestyle=:dash,label="")
    end

    l = @layout([a b c])
    plt=plot(PLTS...,layout=l,size=(size_width*3,size_height*1))
    return plt
end

############################################
###### Figure 12, falsification
############################################
#FalsificationTests.do writes one row per replication with the persistence coefficient and
#the eleven coefficients on discovery size. The impulse responses and the bands are
#computed here rather than in Stata.
const E1A_DRAWFILE="Falsification_Draws.csv"

#The response implied by one draw, following the recursion in ImpulseResponse_TS
function IRF_Path(npv::Float64,T::Int64,ρ::Float64,psi::Vector{Float64})
    Δy=Vector{Float64}(undef,T+1)
    Δy[1]=psi[1]*npv
    for t in 2:T+1
        Δy[t]= t<=length(psi) ? ρ*Δy[t-1]+psi[t]*npv : ρ*Δy[t-1]
    end
    return Δy
end

function Read_Draws(FILE::String)
    rows=[split(chomp(l),',') for l in eachline(FILE)]
    hdr=String.(rows[1])
    col(nm)=findfirst(==(nm),hdr)
    ρ=[parse(Float64,r[col("rho")]) for r in rows[2:end]]
    psi=[[parse(Float64,r[col("psi$i")]) for i in 0:10] for r in rows[2:end]]
    return ρ, psi
end

#Pointwise percentiles of the simulated responses
function Simulated_Band(FILE::String,npv::Float64,T::Int64)
    ρ, psi = Read_Draws(FILE)
    PATHS=[IRF_Path(npv,T,ρ[i],psi[i]) for i in 1:length(ρ)]
    LO=[quantile([p[t] for p in PATHS],0.05) for t in 1:T+1]
    MED=[quantile([p[t] for p in PATHS],0.50) for t in 1:T+1]
    HI=[quantile([p[t] for p in PATHS],0.95) for t in 1:T+1]
    return LO, MED, HI, length(ρ)
end

function Figures_Empirical_Falsification(FOLDER::String,File::String,npv::Float64,T::Int64)
    #The benchmark response, drawn in red
    REG_B=Read_Regression_Objects(File,"spreads","npv")
    Δy_B, cilow_B, cihigh_B=ImpulseResponse_TS(npv,T,REG_B)

    LO, MED, HI, NDRAWS = Simulated_Band(joinpath(FOLDER,E1A_DRAWFILE),npv,T)

    xx=collect(range(0,stop=T-1,length=T+1))

    #No zero line here, unlike the other figures: the median of the reassignment draws sits
    #on zero and a line there hides it
    plt=plot(xx,[Δy_B cilow_B cihigh_B MED LO HI],
             xlabel="t",ylabel="percentage points",
             linestyle=[:solid :dot :dot :solid :dot :dot],
             linecolor=[:red :red :red :black :black :black],
             label=["benchmark" "" "" "reassigned" "" ""],
             legend=:topleft,legendfontsize=14,xticks=0:5:15)
    return plt, NDRAWS
end
