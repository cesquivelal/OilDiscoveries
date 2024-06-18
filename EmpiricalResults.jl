
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
###### Plot in main text
############################################
function Plot_Figure_2(File::String,npv::Float64,T::Int64)
    size_width=650
    size_height=400
    FILE=File

    #Spreads
    name="spreads"
    COLUMN=14
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage points"
    plt_spreads=PlotImpulseResponse(name,units,npv,T,reg)

    #GDP
    name="GDP"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_gdp=PlotImpulseResponse(name,units,npv,T,reg)

    #Investment
    name="investment"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_inv=PlotImpulseResponse(name,units,npv,T,reg)

    #Current account
    name="current account"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_ca=PlotImpulseResponse(name,units,npv,T,reg)

    #Create plot array
    l = @layout([a b; c d])
    plt=plot(plt_spreads,plt_gdp,plt_inv,plt_ca,
             layout=l,size=(size_width*2,size_height*2))
    FOLDER_GRAPHS="Graphs"
    savefig(plt,"$FOLDER_GRAPHS\\Figure2.pdf")
    return plt
end
npv=4.5; T=16
RegressionsFile="Regressions_Benchmark.xlsx"
fig_2=Plot_Figure_2(RegressionsFile,npv,T)

function Plot_Figure_3(File::String,npv::Float64,T::Int64)
    size_width=430
    size_height=350
    FILE=File

    #Total consumption
    name="total"
    COLUMN=38
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_t=PlotImpulseResponse(name,units,npv,T,reg)

    #Private consumption
    name="private"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_p=PlotImpulseResponse(name,units,npv,T,reg)

    #Government consumption
    name="government"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_g=PlotImpulseResponse(name,units,npv,T,reg)

    #Create plot array
    l = @layout([a b c])
    plt=plot(plt_t,plt_p,plt_g,
             layout=l,size=(size_width*3,size_height*1))
    FOLDER_GRAPHS="Graphs"
    savefig(plt,"$FOLDER_GRAPHS\\Figure3.pdf")
    return plt
end
RegressionsFile="Regressions_Benchmark.xlsx"
fig_3=Plot_Figure_3(RegressionsFile,npv,T)

function Plot_Figure_4(File::String,npv::Float64,T::Int64)
    size_width=650
    size_height=400
    FILE=File

    #FDI
    name="net FDI position"
    COLUMN=26
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_fdi=PlotImpulseResponse(name,units,npv,T,reg)

    #Net IIP
    name="net IIP"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_iip=PlotImpulseResponse(name,units,npv,T,reg)

    #Primary balance
    name="primary balance"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_pba=PlotImpulseResponse(name,units,npv,T,reg)

    #Net government debt
    name="net government debt"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage of GDP"
    plt_gde=PlotImpulseResponse(name,units,npv,T,reg)

    #Create plot array
    l = @layout([a b; c d])
    plt=plot(plt_fdi,plt_iip,plt_pba,plt_gde,
             layout=l,size=(size_width*2,size_height*2))
    FOLDER_GRAPHS="Graphs"
    savefig(plt,"$FOLDER_GRAPHS\\Figure4.pdf")
    return plt
end
RegressionsFile="Regressions_Benchmark.xlsx"
fig_4=Plot_Figure_4(RegressionsFile,npv,T)

function Plot_Figure_5(File::String,npv::Float64,T::Int64)
    size_width=650
    size_height=400
    FILE=File

    #Share of investment in non-traded
    name="share in non-traded"
    COLUMN=2
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percent of total"
    plt_N=PlotImpulseResponse(name,units,npv,T,reg)

    #Share of investment in manufacturing
    name="share in manufacturing"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percent of total"
    plt_M=PlotImpulseResponse(name,units,npv,T,reg)

    #Share of investment in commodities
    name="share in commodities"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percent of total"
    plt_C=PlotImpulseResponse(name,units,npv,T,reg)

    #Real exchange rate
    name="real exchange rate"
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE,COLUMN)
    units="percentage change"
    plt_rer=PlotImpulseResponse(name,units,npv,T,reg)

    #Create plot array
    l = @layout([a b; c d])
    plt=plot(plt_N,plt_M,plt_C,plt_rer,
             layout=l,size=(size_width*2,size_height*2))
    FOLDER_GRAPHS="Graphs"
    savefig(plt,"$FOLDER_GRAPHS\\Figure5.pdf")
    return plt
end
RegressionsFile="Regressions_Benchmark.xlsx"
fig_5=Plot_Figure_5(RegressionsFile,npv,T)
