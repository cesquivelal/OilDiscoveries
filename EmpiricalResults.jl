
using Plots; pythonplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
                        background_color_legend=nothing,foreground_color_legend=nothing,
                        legendfontsize=18,guidefontsize=18,titlefontsize=20,tickfontsize=18,
                        markersize=9,size=(650,400))

using XLSX, Parameters, Distributions

@with_kw mutable struct RegressionObjects
    #Persistence coefficient
    ρ::Float64
    #Coefficients of interest
    psi::Vector{Float64}
    #Confidence intervals
    CIρ::Vector{Float64}
    CIpsi::Array{Float64,2}
end

function Read_Regression_Objects(FILE::String,COLUMN::Int64,ROWpsi::Int64=6)
    xf=XLSX.readxlsx(FILE)
    sh=xf["Sheet1"]
    ρ=parse(Float64,sh[5,COLUMN])
    psi=Vector{Float64}(undef,11)
    for i in 0:10
        psi[i+1]=parse(Float64,sh[ROWpsi+i,COLUMN])
    end
    CIρ=Vector{Float64}(undef,2)
    CIρ[1]=parse(Float64,sh[5,COLUMN+1])
    CIρ[2]=parse(Float64,sh[5,COLUMN+2])

    CIpsi=Array{Float64,2}(undef,11,2)
    for i in 0:10
        CIpsi[i+1,1]=parse(Float64,sh[ROWpsi+i,COLUMN+1])
        CIpsi[i+1,2]=parse(Float64,sh[ROWpsi+i,COLUMN+2])
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
    COLUMN=2
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage points"
    plt_spr=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #GDP
    name="GDP"
    COLUMN=5
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_gdp=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Current account
    name="current account"
    COLUMN=8
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_ca=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Net government debt
    name="net government debt"
    COLUMN=11
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_gde=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #private consumption
    YLIMS_C=[-0.4,0.6]
    name="private"
    COLUMN=14
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_c=PlotImpulseResponse(name,units,npv,T,reg,true,YLIMS_C)

    #government consumption
    name="government"
    COLUMN=17
    reg=Read_Regression_Objects(FILE,COLUMN)
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
    COLUMN=20
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage points"
    plt_spr_res=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Spreads, control for reserves and lags
    name="control for 10 lags of log(reserves)"
    COLUMN=23
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage points"
    plt_spr_res_lag=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #GDP
    name="GDP"
    COLUMN=26
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_gdp=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Current account
    name="current account"
    COLUMN=29
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_ca=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #Net government debt
    name="net government debt"
    COLUMN=32
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_gde=PlotImpulseResponse(name,units,npv,T,reg,false,YLIMS_C)

    #private consumption
    YLIMS_C=[-0.4,1.0]
    name="private"
    COLUMN=35
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_c=PlotImpulseResponse(name,units,npv,T,reg,true,YLIMS_C)

    #government consumption
    name="government"
    COLUMN=38
    reg=Read_Regression_Objects(FILE,COLUMN)
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
