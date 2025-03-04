
using Plots; pyplot(fontfamily="serif",linewidth=2.0,grid=false,legend=true,
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

function PlotImpulseResponse(name::String,units::String,npv::Float64,T::Int64,reg::RegressionObjects)
    #Data for plot
    Δy, cilow, cihigh=ImpulseResponse_TS(npv,T,reg)
    xx=collect(range(0,stop=T-1,length=length(Δy)))
    z0=zeros(Float64,length(Δy))

    #Plot parameters
    XLABEL="t"
    XTICKS=0:5:15
    COLORS=[:black :black :black :black]
    LINESTYLES=[:solid :solid :dot :dot]

    #Make plot
    plt=plot(xx,[z0 Δy cilow cihigh],title=name,
             xlabel=XLABEL,ylabel=units,legend=false,
             linestyle=LINESTYLES,linecolor=COLORS,
             xticks=XTICKS)

    #Return
    return plt
end


############################################
###### Plots in main text
############################################
function Plot_Figure_2(File::String,npv::Float64,T::Int64)
    size_width=550
    size_height=400
    FILE=File

    #Spreads
    name=""
    COLUMN=2
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage points"
    plt_spr=PlotImpulseResponse(name,units,npv,T,reg)

    #GDP
    name="GDP"
    COLUMN=5
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_gdp=PlotImpulseResponse(name,units,npv,T,reg)

    #Current account
    name="current account"
    COLUMN=8
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_ca=PlotImpulseResponse(name,units,npv,T,reg)

    #Net government debt
    name="net government debt"
    COLUMN=17
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_gde=PlotImpulseResponse(name,units,npv,T,reg)

    #Create plot array
    l = @layout([a b c])
    plt=plot(plt_gdp,plt_ca,plt_gde,
             layout=l,size=(size_width*3,size_height*1))
    return plt_spr, plt
end

function Plot_Consumption(File::String,npv::Float64,T::Int64)
    size_width=550
    size_height=400
    FILE=File

    #Spreads
    name="private"
    COLUMN=23
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_c=PlotImpulseResponse(name,units,npv,T,reg)

    #GDP
    name="public"
    COLUMN=26
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_g=PlotImpulseResponse(name,units,npv,T,reg)

    #Create plot array
    l = @layout([a b])
    plt=plot(plt_c,plt_g,
             layout=l,size=(size_width*2,size_height*1))
    return plt
end
