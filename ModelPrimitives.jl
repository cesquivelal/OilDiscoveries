
using Parameters, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, Sobol, Roots

################################################################
#### Defining parameters and other structures for the model ####
################################################################
#Define parameter and grid structure, use quarterly calibration
@with_kw struct Pars
    #Frequency of calibration
    Quarterly::Bool = true    #This will affect r_star and the computation of spreads and DefPr
    #Preferences
    σ::Float64 = 2.0          #CRRA parameter, private consumption
    σG::Float64 = 2.0         #CRRA parameter, government consumption
    ω::Float64 = 2.2          #Frisch wage elasticity (Mendoza and Yue(2012))
    β::Float64 = 0.98         #Discount factor
    r_star::Float64 = 0.01    #Risk-free interest rate
    With_g::Bool = false      #Separate government consumption
    τ::Float64 = 0.0          #Fixed income tax

    #Debt parameters
    γ::Float64 = 0.05         #Reciprocal of average maturity
    ξ::Float64 = 0.0          #Exogenous capital outflows, ξ=-0.67 in Mendoza and Yue (2012)

    #Autarky in default
    φ::Float64 = 0.25        #Probability of re-admission (Mendoza and Yue(2012))

    #Endowment case for illustration
    IsEndowmentCase::Bool = false
    OilExempt::Bool = false
    knk::Float64 = 0.969      #Arellano (2008)

    #Production
    #Oil production and discoveries
    SellOilDiscovery::Bool = false
    NPV_Discovery::Float64 = 0.0
    OilRents_Discovery::Float64 = 0.0
    PrivateOilField::Bool = false
    IdenticalTechnology::Bool = false
    WithDiscoveries::Bool = false
    yoil_GDP_target_L::Float64 = 0.0     #Target size of oil sector, relative to GDP
    yoil_GDP_target_H::Float64 = 0.0     #Target size of oil sector, relative to GDP
    nL::Float64 = 1.0
    nH::Float64 = 2.0
    πLH::Float64 = 0.01#0.01           #Probability of discovery
    πHL::Float64 = 0.02#1/50           #Probability of exhaustion
    Twait::Int64 = 0                   #Periods between discovery and production
    #Final good production
    αMf::Float64 = 0.46        #Share of intermediates (Mendoza and Yue(2012) was 0.43)
    αLf::Float64 = 0.14        #Share of labor (Mendoza and Yue(2012) was 0.40)
    αK::Float64 = 0.40         #Share of capital (Mendoza and Yue(2012) was 0.17)
    k::Float64 = 1.00          #Fixed capital stock (Mendoza and Yue(2012)?)
    #Oil production
    αMo::Float64 = 0.30        #Share of intermediates from I-O matrix
    αLo::Float64 = 0.04        #Share of labor from I-O matrix
    αN::Float64 = 1-αMo-αLo    #Share of capital and field from I-O matrix
    #Armington aggregators
    μ::Float64 = 2.9           #Armington elasticity (Mendoza and Yue(2012))
    λf::Float64 = 0.62         #Weight of domestic inputs (Mendoza and Yue(2012) was 0.62)
    λo::Float64 = 0.72         #Weight of domestic inputs from I-O matrix
    #Aggregators of foreign materials
    ν::Float64 = 2.44         #Elasticity across imported varieties (Mendoza and Yue(2012))
    θf::Float64 = 0.70        #Share of imports that require working capital (Mendoza and Yue(2012))
    θo::Float64 = 0.44        #Directly from PEMEX income statement (debt to suppliers)/(cost of sales)
    pj::Float64 = 1.0         #International price of intermediate varieties
    #Domestic intermediates
    A::Float64 = 0.31         #Intermediate goods TFP (Mendoza and Yue(2012))
    η::Float64 = αLf/(αLf+αK) #Labor share in intermediate value added,
                              #same as for final goods (Mendoza and Yue(2012) was 0.7)

    #Stochastic process
    #parameters for productivity shock
    σ_ϵz::Float64 = 0.017           #Standard deviation
    μ_ϵz::Float64 = 0.0-0.5*(σ_ϵz^2)
    dist_ϵz::UnivariateDistribution = Normal(μ_ϵz,σ_ϵz)
    ρ_z::Float64 = 0.95
    μ_z::Float64 = exp(μ_ϵz+0.5*(σ_ϵz^2))
    GR_z_Width::Float64 = 3.0 #Numer of standard deviations
    zlow::Float64 = exp(log(μ_z)-GR_z_Width*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    zhigh::Float64 = exp(log(μ_z)+GR_z_Width*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))

    #Parameters for price of oil
    poil::Float64 = 1.0

    #Quadrature parameters
    N_GL::Int64 = 21
    #Grids
    Nz::Int64 = 11
    Nb::Int64 = 21
    blow::Float64 = 0.0
    bhigh::Float64 = 4.0#15.0
    b_y_high::Float64 = 1.0     #To choose bhigh given an appropriate high b/y

    #Parameters for solution algorithm
    cmin::Float64 = sqrt(eps(Float64))
    Tol_V::Float64 = 1e-6       #Tolerance for absolute distance for value functions
    Tol_q::Float64 = 1e-6       #Tolerance for absolute distance for q
    cnt_max::Int64 = 200           #Maximum number of iterations on VFI
    blowOpt::Float64 = blow-0.01             #Minimum level of debt for optimization
    bhighOpt::Float64 = bhigh+0.01            #Maximum level of debt for optimization

    #Simulation parameters
    Tlong::Int64 = 20000
    drp::Int64 = 100
    Tmom::Int64 = 400
    NSamplesMoments::Int64 = 2000
    HPFilter_Par::Float64 = 1600.0
end

@with_kw mutable struct OutputInterpolations{T1}
    itp_gdp_final::T1
    itp_gdp_oil::T1
    itp_L::T1
    itp_y::T1
    itp_yo::T1
    itp_WorkingCapital::T1
end

@with_kw struct Grids
    #Grids of states
    GR_n::Array{Float64,1}
    GR_z::Array{Float64,1}
    GR_b::Array{Float64,1}

    #Quadrature vectors for integrals
    ϵz_weights::Vector{Float64}
    ZPRIME::Array{Float64,2}
    PDFz::Array{Float64,1}
    FacQz::Float64

    #Matrix for discoveries
    PI_n::Array{Float64,2}

    #Pre-computed values
    GDP_Final_Matrix::Array{Float64,3}
    GDP_Oil_Matrix::Array{Float64,3}
    L_Matrix::Array{Float64,3}
    Y_Matrix::Array{Float64,3}
    YO_Matrix::Array{Float64,3}
    WORKING_CAPITAL_Matrix::Array{Float64,3}
end

function CreateOilFieldGrids(par::Pars)
    @unpack WithDiscoveries = par
    if WithDiscoveries
        @unpack πLH, πHL, Twait, nL, nH = par
        if Twait==0
            #No waiting, 2x2 matrix
            Nn=2
        else
            #Waiting, Twait+1 is periods in between
            Nn=2+Twait
        end
        GR_n=ones(Float64,Nn)*nL
        GR_n[end]=nH
        #Probability of discovery and no discovery
        PI_n=zeros(Float64,Nn,Nn)
        PI_n[1,1]=1.0-πLH
        PI_n[1,2]=πLH
        PI_n[end,1]=πHL
        PI_n[end,end]=1.0-πHL
        if Twait>0
            for t=1:Twait
                PI_n[t+1,t+2]=1.0
            end
        end
        return GR_n, PI_n
    else
        Nn=2
        GR_n=zeros(Float64,Nn)
        PI_n=zeros(Float64,Nn,Nn)
        PI_n[1,1]=1.0
        PI_n[2,2]=1.0
        return GR_n, PI_n
    end
end

function Create_GL_objects(GR_z::Array{Float64,1},N_GL::Int64,σϵ::Float64,ρ::Float64,μ_z::Float64,dist_ϵ::UnivariateDistribution,GR_z_Width::Float64)
    #Gauss-Legendre vectors for y'
    GL_nodes, GL_weights = gausslegendre(N_GL)
    ϵlow=-GR_z_Width*σϵ
    ϵhigh=GR_z_Width*σϵ
    ϵ_nodes=0.5*(ϵhigh-ϵlow).*GL_nodes .+ 0.5*(ϵhigh+ϵlow)
    ϵ_weights=GL_weights .* 0.5*(ϵhigh-ϵlow)
    #Matrices for integration over shock y
    N=length(GR_z)
    ZPRIME=Array{Float64,2}(undef,N,N_GL)
    PDFz=pdf.(dist_ϵ,ϵ_nodes)
    for z_ind in 1:N
        z=GR_z[z_ind]
        ZPRIME[z_ind,:]=exp.((1.0-ρ)*log(μ_z) + ρ*log(z) .+ ϵ_nodes)
    end
    FacQz=dot(ϵ_weights,PDFz)
    return ϵ_weights, ZPRIME, PDFz, FacQz
end

@with_kw mutable struct Solution{T1,T2,T3,T4,T5,T6}
    ### Arrays
    #Value Functions
    VD::T1
    VP::T2
    V::T2
    #Expectations and price
    EVD::T1
    EV::T2
    q1::T2
    #Policy function
    bprime::T2
    ### Interpolation objects
    #Value Functions
    itp_VD::T3
    itp_VP::T4
    itp_V::T4
    #Expectations and price
    itp_EVD::T3
    itp_EV::T4
    itp_q1::T5
    #Policy functions
    itp_bprime::T6
end

@with_kw struct Model
    SOLUTION::Solution
    GRIDS::Grids
    par::Pars
    ITP_OUTPUT::OutputInterpolations
end

@with_kw mutable struct State
    Default::Bool
    n_ind::Int64
    n::Float64
    z::Float64
    b::Float64
end

################################################################
#################### Auxiliary functions #######################
################################################################
function MyBisection(foo,a::Float64,b::Float64;xatol::Float64=1e-8)
    s=sign(foo(b))
    x=(a+b)/2.0
    d=(b-a)/2.0
    while d>xatol
        d=d/2.0
        if s==sign(foo(x))
            x=x-d
        else
            x=x+d
        end
    end
    return x
end

function TransformIntoBounds(x::Float64,min::Float64,max::Float64)
    (max - min) * (1.0/(1.0 + exp(-x))) + min
end

function TransformIntoReals(x::Float64,min::Float64,max::Float64)
    log((x - min)/(max - x))
end

################################################################
################# Preferences and technology ###################
################################################################
function CRRA(x::Float64,σ::Float64)
    return ((x^(1-σ))-1)/(1-σ)
end

function Utility(c::Float64,g::Float64,L::Float64,par::Pars)
    @unpack cmin, With_g = par
    @unpack σ, ω = par
    if With_g
        @unpack σG = par
        if c<=0.0 || g<=0.0
            x=cmin
            return CRRA(x,σ)
        else
            labor_disutility=(L^(1+(1/ω)))/(1+(1/ω))
            x=c+labor_disutility
            if x<=0.0
                x=cmin
                return CRRA(x,σ)
            else
                return CRRA(x,σ)+CRRA(g,σG)
            end
        end
    else
        if c<=0.0
            x=cmin
            return CRRA(x,σ)
        else
            labor_disutility=(L^(1+(1/ω)))/(1+(1/ω))
            x=c+labor_disutility
            if x<=0.0
                x=cmin
            end
            return CRRA(x,σ)
        end
    end
end

function ComputeSpreadWithQ(qq::Float64,par::Pars)
    @unpack r_star, γ, Quarterly = par
    if Quarterly
        #Quarterly calibration
        ib=(((γ+(1-γ)*qq)/qq)^4)-1
        rf=((1+r_star)^4)-1
    else
        #Yearly calibration
        ib=(((γ+(1-γ)*qq)/qq)^1)-1
        rf=((1+r_star)^1)-1
    end
    return min(100*(ib-rf),100.0)
end

function Final_Good_Output(z::Float64,Mf::Float64,Lf::Float64,par::Pars)
    @unpack αMf, αLf, αK, k = par
    return z*(Mf^αMf)*(Lf^αLf)*(k^αK)
end

function Oil_Output(z::Float64,Mo::Float64,Lo::Float64,n::Float64,par::Pars)
    @unpack αMo, αLo, αN = par
    return z*(Mo^αMo)*(Lo^αLo)*(n^αN)
end

function Domestic_Intermediates_Output(Ld::Float64,par::Pars)
    @unpack A, η = par
    return A*(Ld^η)
end

function Price_star_h(IsFinalSector::Bool,x::State,par::Pars)
    @unpack Default = x
    @unpack r_star, pj, ν = par
    if IsFinalSector
        θ=par.θf
    else
        θ=par.θo
    end

    NO_WORKING_CAPITAL=(1-θ)*(pj^(1-ν))
    if Default
        return (NO_WORKING_CAPITAL)^(1/(1-ν))
    else
        WORKING_CAPITAL=θ*(((1+r_star)*pj)^(1-ν))
        return (WORKING_CAPITAL+NO_WORKING_CAPITAL)^(1/(1-ν))
    end
end

function Price_M_h(IsFinalSector::Bool,pd::Float64,x::State,par::Pars)
    @unpack μ = par
    if IsFinalSector
        λ=par.λf
    else
        λ=par.λo
    end
    Pstar=Price_star_h(IsFinalSector,x,par)
    return ((λ^μ)*(pd^(1-μ))+((1-λ)^μ)*(Pstar^(1-μ)))^(1/(1-μ))
end

function LaborSupply(w::Float64,par::Pars)
    @unpack ω, With_g, τ = par
    if With_g
        return ((1-τ)*w)^ω
    else
        return w^ω
    end
end

function LaborDemand_h(IsFinalSector::Bool,x::State,w::Float64,pd::Float64,par::Pars)
    if IsFinalSector
        @unpack αLf, αMf, αK, k = par
        @unpack z = x
        C1=z*(k^αK)
        C2=(αLf/w)^(1-αMf)
        PMf=Price_M_h(IsFinalSector,pd,x,par)
        C3=(αMf/PMf)^αMf
        EXP=1/(1-αLf-αMf)
        return (C1*C2*C3)^EXP
    else
        @unpack αLo, αMo, αN, poil = par
        @unpack z, n = x
        C1=poil*z*(n^αN)
        C2=(αLo/w)^(1-αMo)
        PMo=Price_M_h(IsFinalSector,pd,x,par)
        C3=(αMo/PMo)^αMo
        EXP=1/(1-αLo-αMo)
        return (C1*C2*C3)^EXP
    end
end

function LaborDemand_d(w::Float64,pd::Float64,par::Pars)
    @unpack η, A = par
    return ((η*pd*A)/w)^(1/(1-η))
end

function MaterialsDemand_h(IsFinalSector::Bool,x::State,w::Float64,pd::Float64,par::Pars)
    if IsFinalSector
        αL=par.αLf
        αM=par.αMf
    else
        αL=par.αLo
        αM=par.αMo
    end
    PMh=Price_M_h(IsFinalSector,pd,x,par)
    Lh=LaborDemand_h(IsFinalSector,x,w,pd,par)
    return (αM/αL)*(w/PMh)*Lh
end

function DomesticMaterialsDemand_h(IsFinalSector::Bool,x::State,w::Float64,pd::Float64,par::Pars)
    @unpack μ = par
    if IsFinalSector
        λ=par.λf
    else
        λ=par.λo
    end
    Mh=MaterialsDemand_h(IsFinalSector,x,w,pd,par)
    PMh=Price_M_h(IsFinalSector,pd,x,par)
    return ((λ*PMh/pd)^μ)*Mh
end

function Intermediates_Excess_Demand(w::Float64,pd::Float64,x::State,par::Pars)
    mdf=DomesticMaterialsDemand_h(true,x,w,pd,par)
    mdo=DomesticMaterialsDemand_h(false,x,w,pd,par)
    DEMAND=mdf+mdo
    Ld=LaborDemand_d(w,pd,par)
    SUPPLY=Domestic_Intermediates_Output(Ld,par)
    return DEMAND-SUPPLY
end

function Labor_Excess_Demand(w::Float64,pd::Float64,x::State,par::Pars)
    Lf=LaborDemand_h(true,x,w,pd,par)
    Lo=LaborDemand_h(false,x,w,pd,par)
    Ld=LaborDemand_d(w,pd,par)
    DEMAND=Lf+Lo+Ld
    SUPPLY=LaborSupply(w,par)
    return DEMAND-SUPPLY
end

function Wage_Given_pd(pd::Float64,x::State,par::Pars)
    foo(w::Float64)=Labor_Excess_Demand(w,pd,x,par)
    wlow=eps(Float64)
    whigh=1.0
    cnt=0
    while sign(foo(whigh))==sign(foo(wlow))
        whigh=whigh*2
        cnt=cnt+1
        if cnt>10
            break
        end
    end
    return MyBisection(foo,wlow,whigh)
end

function pd_Given_State(x::State,par::Pars)
    foo(pd::Float64)=Intermediates_Excess_Demand(Wage_Given_pd(pd,x,par),pd,x,par)
    pdlow=eps(Float64)
    pdhigh=10.0
    cnt=1
    while sign(foo(pdhigh))==sign(foo(pdlow))
        pdhigh=2*pdhigh
        cnt=cnt+1
        if cnt>10
            break
        end
    end
    return MyBisection(foo,pdlow,pdhigh)
end

function ImportedMaterialsDemand_h(IsFinalSector::Bool,pd::Float64,x::State,Mh::Float64,par::Pars)
    @unpack μ = par
    if IsFinalSector
        λ=par.λf
    else
        λ=par.λo
    end
    Pstarh=Price_star_h(IsFinalSector,x,par)
    PMh=Price_M_h(IsFinalSector,pd,x,par)
    return (((1-λ)*PMh/Pstarh)^μ)*Mh
end

function DefaultCost(y::Float64,Ey::Float64,par::Pars)
    #Just for simple endowment case
    #Arellano (2008)
    @unpack knk = par
    return max(0.0,y-knk*Ey)
end

function Compute_Quantities_GivenState(x::State,par::Pars)
    @unpack IsEndowmentCase = par
    if IsEndowmentCase
        @unpack nL, OilExempt, μ_z = par
        @unpack Default, z, n = x
        L=0.0
        WorkingCapital=0.0
        if Default
            zD=z-DefaultCost(z,μ_z,par)
            if OilExempt
                nD=n
            else
                nD=n-DefaultCost(n,nL,par)
            end
            return zD, nD, L, zD, nD, WorkingCapital
        else
            return z, n, L, z, n, WorkingCapital
        end
    else
        @unpack poil, r_star, pj, θo, θf, ν = par
        @unpack z, n = x
        pd=pd_Given_State(x,par)
        w=Wage_Given_pd(pd,x,par)

        Lf=LaborDemand_h(true,x,w,pd,par)
        Lo=LaborDemand_h(false,x,w,pd,par)
        Ld=LaborDemand_d(w,pd,par)
        L=LaborSupply(w,par)

        Mf=MaterialsDemand_h(true,x,w,pd,par)
        Mo=MaterialsDemand_h(false,x,w,pd,par)
        mstar_f=ImportedMaterialsDemand_h(true,pd,x,Mf,par)
        mstar_o=ImportedMaterialsDemand_h(false,pd,x,Mo,par)

        Pstar_f=Price_star_h(true,x,par)
        Pstar_o=Price_star_h(false,x,par)

        y=Final_Good_Output(z,Mf,Lf,par)
        yo=Oil_Output(z,Mo,Lo,n,par)

        gdp_final=y-Pstar_f*mstar_f+w*Lo
        gdp_oil=poil*yo-Pstar_o*mstar_o-w*Lo

        #Also compute working capital
        mj_θf=((Pstar_f/((1+r_star)*pj))^ν)*mstar_f
        mj_θo=((Pstar_o/((1+r_star)*pj))^ν)*mstar_o

        wkf=(1+r_star)*θf*pj*mj_θf
        wko=(1+r_star)*θo*pj*mj_θo
        WorkingCapital=wkf#+wko

        return gdp_final, gdp_oil, L, y, yo, WorkingCapital
    end
end

function Fill_GDP_L_Matrices(GR_n::Array{Float64,1},GR_z::Array{Float64,1},par::Pars)
    @unpack Nz = par
    Nn=length(GR_n)
    GDP_Final_Matrix=Array{Float64,3}(undef,Nz,Nn,2)
    GDP_Oil_Matrix=Array{Float64,3}(undef,Nz,Nn,2)
    L_Matrix=Array{Float64,3}(undef,Nz,Nn,2)
    Y_Matrix=Array{Float64,3}(undef,Nz,Nn,2)
    YO_Matrix=Array{Float64,3}(undef,Nz,Nn,2)
    WORKING_CAPITAL_Matrix=Array{Float64,3}(undef,Nz,Nn,2)
    for I in CartesianIndices(GDP_Final_Matrix)
        (z_ind,n_ind,d_ind)=Tuple(I)
        if d_ind==1
            Default=true
        else
            Default=false
        end
        n=GR_n[n_ind]; z=GR_z[z_ind]; b=0.0
        x=State(Default,n_ind,n,z,b)
        gdp_final, gdp_oil, L, y, yo, WorkingCapital=Compute_Quantities_GivenState(x,par)
        GDP_Final_Matrix[I]=gdp_final
        GDP_Oil_Matrix[I]=gdp_oil
        L_Matrix[I]=L
        Y_Matrix[I]=y
        YO_Matrix[I]=yo
        WORKING_CAPITAL_Matrix[I]=WorkingCapital

    end
    return GDP_Final_Matrix, GDP_Oil_Matrix, L_Matrix, Y_Matrix, YO_Matrix, WORKING_CAPITAL_Matrix
end

function CreateGrids(par::Pars)
    #Grid for z
    @unpack Nz, zlow, zhigh = par
    GR_z=collect(range(zlow,stop=zhigh,length=Nz))

    #Grid of oil discoveries
    GR_n, PI_n=CreateOilFieldGrids(par)

    #Pre-computed values
    GDP_Final_Matrix, GDP_Oil_Matrix, L_Matrix, Y_Matrix, YO_Matrix, WORKING_CAPITAL_Matrix=Fill_GDP_L_Matrices(GR_n,GR_z,par)

    #Gauss-Legendre objects
    @unpack N_GL, σ_ϵz, ρ_z, μ_z, dist_ϵz, GR_z_Width = par
    ϵz_weights, ZPRIME, PDFz, FacQz=Create_GL_objects(GR_z,N_GL,σ_ϵz,ρ_z,μ_z,dist_ϵz,GR_z_Width)

    #Grid of debt
    @unpack Nb, b_y_high, Quarterly = par
    xTrend=State(false,1,par.nL,μ_z,0.0)
    gdp_final, gdp_oil, L, y, yo, WorkingCapital=Compute_Quantities_GivenState(xTrend,par)
    avGDP=gdp_final+gdp_oil
    blow=0.0
    if Quarterly
        bhigh=b_y_high*4*avGDP
    else
        bhigh=b_y_high*1*avGDP
    end
    GR_b=collect(range(blow,stop=bhigh,length=Nb))

    return Grids(GR_n,GR_z,GR_b,ϵz_weights,ZPRIME,PDFz,FacQz,PI_n,GDP_Final_Matrix,GDP_Oil_Matrix,L_Matrix,Y_Matrix,YO_Matrix,WORKING_CAPITAL_Matrix)
end

function Oil_Share_In_GDP_Average(n::Float64,par::Pars)
    @unpack μ_z = par
    Default=false; n_ind=1; z=μ_z; b=0.0
    x=State(Default,n_ind,n,z,b)
    gdp_final, gdp_oil, L, y, yo, WorkingCapital=Compute_Quantities_GivenState(x,par)
    return gdp_oil/(gdp_final+gdp_oil)
end

function Pick_n_Given_Target(yoil_GDP_target::Float64,par::Pars)
    foo(n::Float64)=Oil_Share_In_GDP_Average(n,par)-yoil_GDP_target
    nlow=0.0
    nhigh=2.0
    cnt=0
    while sign(foo(nlow))==sign(foo(nhigh))
        nhigh=2*nhigh
        cnt=cnt+1
        if cnt>5
            break
        end
    end
    return MyBisection(foo,nlow,nhigh)
end

function OilRents_FromDiscovery_ss(par::Pars)
    @unpack μ_z, nL, nH, αN = par
    Default=false; n_ind=1; z=μ_z; b=0.0
    x=State(Default,n_ind,nH,z,b)
    pd=pd_Given_State(x,par)
    w=Wage_Given_pd(pd,x,par)

    Lo=LaborDemand_h(false,x,w,pd,par)
    Mo=MaterialsDemand_h(false,x,w,pd,par)
    yo=Oil_Output(z,Mo,Lo,nH,par)
    return αN*(yo/nH)*(nH-nL)
end

function NPV_giant_field(par::Pars)
    @unpack r_star, πHL = par
    Ro=OilRents_FromDiscovery_ss(par)
    npv=Ro*(1/((1+r_star)^5))*((1+r_star)/(r_star+πHL))
    return npv
end

################################################################
############# Functions to interpolate matrices ################
################################################################
function CreateInterpolation_ValueFunctions(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_n = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    ORDER_SHOCKS=Linear()
    ORDER_K_STATES=Linear()#Cubic(Line(OnGrid()))
    ORDER_B_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Zs,Ns),Interpolations.Flat())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_B_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Zs,Ns),Interpolations.Flat())
    end
end

function CreateInterpolation_Price(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_n, GR_z, GR_b = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Zs,Ns),Interpolations.Flat())
end

function CreateInterpolation_Policy(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_n, GR_z, GR_b = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Zs,Ns),Interpolations.Flat())
end

function CreateInterpolation_ForExpectations(MAT::Array{Float64,1},GRIDS::Grids)
    @unpack GR_z = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    ORDER_SHOCKS=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_SHOCKS))
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Zs),Interpolations.Flat())
end

function CreateInterpolationObject_Output(MAT::Array{Float64,3},GRIDS::Grids)
    @unpack GR_z, GR_n = GRIDS
    Ds=range(1,stop=2,length=2)
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    ORDER_SHOCKS=Linear()

    INT_DIMENSIONS=(BSpline(ORDER_SHOCKS),NoInterp(),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Zs,Ns,Ds),Interpolations.Line())
end

function Pack_OutputInterpolations(GRIDS::Grids)
    itp_gdp_final=CreateInterpolationObject_Output(GRIDS.GDP_Final_Matrix,GRIDS)
    itp_gdp_oil=CreateInterpolationObject_Output(GRIDS.GDP_Oil_Matrix,GRIDS)
    itp_L=CreateInterpolationObject_Output(GRIDS.L_Matrix,GRIDS)
    itp_y=CreateInterpolationObject_Output(GRIDS.Y_Matrix,GRIDS)
    itp_yo=CreateInterpolationObject_Output(GRIDS.YO_Matrix,GRIDS)
    itp_WorkingCapital=CreateInterpolationObject_Output(GRIDS.WORKING_CAPITAL_Matrix,GRIDS)

    return OutputInterpolations(itp_gdp_final,itp_gdp_oil,itp_L,itp_y,itp_yo,itp_WorkingCapital)
end

################################################################################
### Functions to pack models in vectors and save to CSV
################################################################################
function InitiateSolution(GRIDS::Grids,par::Pars)
    @unpack Nz, Nb = par
    Nn=length(GRIDS.GR_n)
    ### Allocate all values to object
    VD=zeros(Float64,Nz,Nn)
    VP=zeros(Float64,Nb,Nz,Nn)
    V=zeros(Float64,Nb,Nz,Nn)
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)
    #Expectations and price
    EVD=zeros(Float64,Nz,Nn)
    EV=zeros(Float64,Nb,Nz,Nn)
    q1=zeros(Float64,Nb,Nz,Nn)

    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)

    #Policy function
    bprime=zeros(Float64,Nb,Nz,Nn)

    #Policy function
    itp_bprime=CreateInterpolation_Policy(bprime,GRIDS)
    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,bprime,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_bprime)
end

function StackSolution_Vector(SOLUTION::Solution)
    #Stack vectors of repayment first
    @unpack VP, V, EV, q1 = SOLUTION
    @unpack bprime = SOLUTION
    VEC=reshape(VP,(:))
    VEC=vcat(VEC,reshape(V,(:)))
    VEC=vcat(VEC,reshape(EV,(:)))
    VEC=vcat(VEC,reshape(q1,(:)))
    VEC=vcat(VEC,reshape(bprime,(:)))

    #Then stack vectors of default
    @unpack VD, EVD = SOLUTION
    VEC=vcat(VEC,reshape(VD,(:)))
    VEC=vcat(VEC,reshape(EVD,(:)))

    return VEC
end

function VectorOfRelevantParameters(par::Pars)
    #Stack important values from parameters
    #Parameters that will only change manually
    VEC=par.N_GL                #1
    VEC=vcat(VEC,par.Nz)        #2
    VEC=vcat(VEC,par.Nb)        #3
    VEC=vcat(VEC,par.GR_z_Width)#4
    VEC=vcat(VEC,par.cnt_max)   #5
    VEC=vcat(VEC,par.b_y_high)  #6
    VEC=vcat(VEC,par.γ)         #7
    VEC=vcat(VEC,par.φ)         #8

    if par.Quarterly            #9
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end

    if par.WithDiscoveries      #10
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end
    VEC=vcat(VEC,par.Twait)     #11
    VEC=vcat(VEC,par.yoil_GDP_target_L)  #12
    VEC=vcat(VEC,par.yoil_GDP_target_H)  #13

    if par.IdenticalTechnology  #14
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end

    if par.IsEndowmentCase  #15
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end

    if par.OilExempt  #16
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end
    VEC=vcat(VEC,par.knk)         #17

    #Parameters that may change with calibration exercise
    VEC=vcat(VEC,par.A)         #18
    VEC=vcat(VEC,par.β)         #19
    VEC=vcat(VEC,par.θf)        #20
    VEC=vcat(VEC,par.ρ_z)       #21
    VEC=vcat(VEC,par.σ_ϵz)      #22

    if par.With_g  #23
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end
    VEC=vcat(VEC,par.τ)      #24

    if par.PrivateOilField  #25
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end
    VEC=vcat(VEC,par.θo)      #26

    return VEC
end

function Create_Model_Vector(MODEL::Model)
    @unpack SOLUTION, par = MODEL
    VEC_PAR=VectorOfRelevantParameters(par)
    N_parameters=length(VEC_PAR)
    VEC=vcat(N_parameters,VEC_PAR)

    #Stack SOLUTION in one vector
    VEC_SOL=StackSolution_Vector(SOLUTION)

    return vcat(VEC,VEC_SOL)
end

function SaveModel_Vector(NAME::String,MODEL::Model)
    VEC=Create_Model_Vector(MODEL)
    writedlm(NAME,VEC,',')
    return nothing
end

function ExtractMatrixFromSolutionVector(start::Int64,size::Int64,IsDefault::Bool,VEC::Vector{Float64},GRIDS::Grids,par::Pars)
    @unpack Nz, Nb = par
    @unpack GR_n = GRIDS
    Nn=length(GR_n)
    finish=start+size-1
    vec=VEC[start:finish]
    if IsDefault
        I=(Nz,Nn)
        return reshape(vec,I)
    else
        I=(Nb,Nz,Nn)
        return reshape(vec,I)
    end
end

function TransformVectorToSolution(VEC::Array{Float64},GRIDS::Grids,par::Pars)
    #The file SolutionVector.csv must be in FOLDER
    #for this function to work
    @unpack Nz, Nb = par
    @unpack GR_n = GRIDS
    Nn=length(GR_n)
    size_repayment=Nn*Nz*Nb
    size_default=Nn*Nz

    #Allocate vectors into matrices
    #Repayment
    start=1
    VP=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,GRIDS,par)
    start=start+size_repayment
    V=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,GRIDS,par)
    start=start+size_repayment
    EV=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,GRIDS,par)
    start=start+size_repayment
    q1=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,GRIDS,par)
    start=start+size_repayment
    bprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,GRIDS,par)

    #Default
    start=start+size_repayment
    VD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,GRIDS,par)
    start=start+size_default
    EVD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,GRIDS,par)
    #Create interpolation objects
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)

    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)

    itp_bprime=CreateInterpolation_Policy(bprime,GRIDS)

    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,bprime,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_bprime)
end

function UnpackParameters_Vector(VEC::Array{Float64})
    par=Pars()

    #Parameters that will only change manually
    par=Pars(par,N_GL=convert(Int64,VEC[1]))
    par=Pars(par,Nz=convert(Int64,VEC[2]))
    par=Pars(par,Nb=convert(Int64,VEC[3]))
    par=Pars(par,GR_z_Width=convert(Int64,VEC[4]))
    par=Pars(par,cnt_max=VEC[5])
    par=Pars(par,b_y_high=VEC[6])
    par=Pars(par,γ=VEC[7])
    par=Pars(par,φ=VEC[8])

    if VEC[9]==1.0
        par=Pars(par,Quarterly=true)
        par=Pars(par,r_star=0.017,HPFilter_Par=1600.0)
    else
        par=Pars(par,Quarterly=false)
        par=Pars(par,r_star=0.04,HPFilter_Par=100.0)
    end

    if VEC[10]==1.0
        par=Pars(par,WithDiscoveries=true)
    else
        par=Pars(par,WithDiscoveries=false)
    end
    par=Pars(par,Twait=VEC[11])
    par=Pars(par,yoil_GDP_target_L=VEC[12])
    par=Pars(par,yoil_GDP_target_H=VEC[13])

    if VEC[14]==1.0
        par=Pars(par,IdenticalTechnology=true)
        αMo=par.αMf
        αLo=par.αLf
        αN=par.αK
        λo=par.λf
        θo=VEC[20]
        par=Pars(par,αMo=αMo,αLo=αLo,αN=αN,λo=λo,θo=θo)
    else
        θo=VEC[26]
        par=Pars(par,IdenticalTechnology=false,θo=θo)
    end

    if VEC[15]==1.0
        par=Pars(par,IsEndowmentCase=true)
    else
        par=Pars(par,IsEndowmentCase=false)
    end

    if VEC[16]==1.0
        par=Pars(par,OilExempt=true)
    else
        par=Pars(par,OilExempt=false)
    end

    par=Pars(par,knk=VEC[17])

    #Parameters that may change with calibration exercise
    par=Pars(par,A=VEC[18])
    par=Pars(par,β=VEC[19])
    par=Pars(par,θf=VEC[20])
    par=Pars(par,ρ_z=VEC[21])
    par=Pars(par,σ_ϵz=VEC[22])

    if VEC[23]==1.0
        par=Pars(par,With_g=true)
    else
        par=Pars(par,With_g=false)
    end
    par=Pars(par,τ=VEC[24])

    if VEC[25]==1.0
        par=Pars(par,PrivateOilField=true)
    else
        par=Pars(par,PrivateOilField=false)
    end

    #Parameters that depend on inputed parameters
    μ_ϵz=0.0-0.5*(par.σ_ϵz^2)
    dist_ϵz=Normal(μ_ϵz,par.σ_ϵz)
    μ_z=exp(μ_ϵz+0.5*(par.σ_ϵz^2))
    par=Pars(par,μ_ϵz=μ_ϵz,dist_ϵz=dist_ϵz,μ_z=μ_z)

    zlow=exp(log(μ_z)-par.GR_z_Width*sqrt((par.σ_ϵz^2.0)/(1.0-(par.ρ_z^2.0))))
    zhigh=exp(log(μ_z)+par.GR_z_Width*sqrt((par.σ_ϵz^2.0)/(1.0-(par.ρ_z^2.0))))
    par=Pars(par,zlow=zlow,zhigh=zhigh)

    nL=Pick_n_Given_Target(par.yoil_GDP_target_L,par)
    nH=Pick_n_Given_Target(par.yoil_GDP_target_H,par)
    par=Pars(par,nL=nL,nH=nH)

    return par
end

function Setup_From_Vector(VEC_PAR::Array{Float64})
    #Vector of parameters has to have the correct structure
    par=UnpackParameters_Vector(VEC_PAR)
    GRIDS=CreateGrids(par)
    par=Pars(par,blow=GRIDS.GR_b[1],bhigh=GRIDS.GR_b[end])
    par=Pars(par,blowOpt=par.blow-0.001,bhighOpt=par.bhigh+0.001)
    return par, GRIDS
end

function Setup_From_File(setup_coulumn::Int64,SETUP_FILE::String)
    XX=readdlm(SETUP_FILE,',')
    VEC_PAR=XX[2:end,setup_coulumn]*1.0
    return Setup_From_Vector(VEC_PAR)
end

function UnpackModel_Vector(VEC)
    #Extract parameters and create grids
    N_parameters=convert(Int64,VEC[1])
    VEC_PAR=1.0*VEC[2:N_parameters+1]
    par, GRIDS=Setup_From_Vector(VEC_PAR)
    ITP_OUTPUT=Pack_OutputInterpolations(GRIDS)

    #Extract solution object
    VEC_SOL=VEC[N_parameters+2:end]
    SOL=TransformVectorToSolution(VEC_SOL,GRIDS,par)

    return Model(SOL,GRIDS,par,ITP_OUTPUT)
end

function UnpackModel_File(NAME::String,FOLDER::String)
    #Unpack Vector with data
    if FOLDER==" "
        VEC=readdlm(NAME,',')
    else
        VEC=readdlm("$FOLDER\\$NAME",',')
    end

    return UnpackModel_Vector(VEC)
end

################################################################
########## Functions to compute expectations ###################
################################################################
function Expectation_over_zprime(foo,z_ind::Int64,GRIDS::Grids)
    #foo is a function of floats for z'
    #kN', kT', and b' are given
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    int=0.0
    for j in 1:length(ϵz_weights)
        int=int+ϵz_weights[j]*PDFz[j]*foo(ZPRIME[z_ind,j])
    end
    return int/FacQz
end

function Expectation_over_zprime_v_and_q(foo,z_ind::Int64,GRIDS::Grids)
    #foo is a function of floats for z' that returns a
    #tuple (v,q)
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    int_v=0.0
    int_q=0.0
    for j in 1:length(ϵz_weights)
        v, q=foo(ZPRIME[z_ind,j])
        int_v=int_v+ϵz_weights[j]*PDFz[j]*v
        int_q=int_q+ϵz_weights[j]*PDFz[j]*q
    end
    return int_v/FacQz, int_q/FacQz
end

function SDF_Lenders(par::Pars)
    @unpack r_star = par
    return 1/(1+r_star)
end

function ValueAndBondsPayoff(nprime_ind::Int64,zprime::Float64,bprime::Float64,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    @unpack itp_VP, itp_VD = SOLUTION
    vd=itp_VD(zprime,nprime_ind)
    vp=itp_VP(bprime,zprime,nprime_ind)
    if vd>vp
        return vd, 0.0
    else
        @unpack γ = par
        SDF=SDF_Lenders(par)
        if γ==1.0
            return vp, SDF
        else
            @unpack itp_q1, itp_bprime = SOLUTION
            bb=itp_bprime(bprime,zprime,nprime_ind)
            qq=itp_q1(bb,zprime,nprime_ind)
            return vp, SDF*(γ+(1-γ)*qq)
        end
    end
end

function UpdateExpectations!(MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack PI_n, GR_n, GR_b = GRIDS
    Nn=length(GR_n)
    #Update default
    for I in CartesianIndices(MODEL.SOLUTION.EVD)
        (z_ind,n_ind)=Tuple(I)
        int=0.0
        for nprime_ind in 1:Nn
            if PI_n[n_ind,nprime_ind]>0.0
                foo_mat=CreateInterpolation_ForExpectations(MODEL.SOLUTION.VD[:,nprime_ind],GRIDS)
                EVD=Expectation_over_zprime(foo_mat,z_ind,GRIDS)
                int=int+PI_n[n_ind,nprime_ind]*EVD
            end
        end
        MODEL.SOLUTION.EVD[I]=int
    end

    #Update repayment
    for I in CartesianIndices(MODEL.SOLUTION.q1)
        (b_ind,z_ind,n_ind)=Tuple(I)
        int_v=0.0
        int_q=0.0
        bprime=GR_b[b_ind]
        for nprime_ind in 1:Nn
            if PI_n[n_ind,nprime_ind]>0.0
                foo_mat_vq(zprime::Float64)=ValueAndBondsPayoff(nprime_ind,zprime,bprime,MODEL)
                EVz, q1z=Expectation_over_zprime_v_and_q(foo_mat_vq,z_ind,GRIDS)
                int_v=int_v+PI_n[n_ind,nprime_ind]*EVz
                int_q=int_q+PI_n[n_ind,nprime_ind]*q1z
            end
        end
        MODEL.SOLUTION.EV[I]=int_v
        MODEL.SOLUTION.q1[I]=int_q
    end
    return nothing
end

###############################################################################
#Function to compute consumption and value given the state and policies
###############################################################################
function ExogenousCapitalFlow(z::Float64,MODEL::Model)
    @unpack par = MODEL
    @unpack ξ = par
    #These are outflows
    return ξ*log(z)
end

function Calculate_Tr(x::State,bprime::Float64,MODEL::Model)
    @unpack par = MODEL
    @unpack Default, n_ind, z, b = x
    XT=ExogenousCapitalFlow(z,MODEL)
    if Default
        return 0.0-XT
    else
        @unpack SOLUTION, par = MODEL
        @unpack γ = par
        @unpack itp_q1 = SOLUTION
        #Compute net borrowing from the rest of the world
        qq=itp_q1(bprime,z,n_ind)
        return qq*(bprime-(1-γ)*b)-γ*b-XT
    end
end

function Evaluate_ValueFunction(x::State,I::CartesianIndex,bprime::Float64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack β, With_g, SellOilDiscovery = par
    @unpack GDP_Final_Matrix, GDP_Oil_Matrix, L_Matrix = GRIDS
    @unpack Default, z = x
    if Default
        @unpack φ = par
        @unpack itp_EV, itp_EVD = SOLUTION
        (z_ind,n_ind)=Tuple(I)
        if With_g
            @unpack τ, PrivateOilField = par
            if PrivateOilField
                Y=GDP_Final_Matrix[z_ind,n_ind,1]+GDP_Oil_Matrix[z_ind,n_ind,1]
                cons=(1-τ)*Y
                g=τ*Y
            else
                cons=(1-τ)*GDP_Final_Matrix[z_ind,n_ind,1]
                g=τ*GDP_Final_Matrix[z_ind,n_ind,1]+GDP_Oil_Matrix[z_ind,n_ind,1]
                if SellOilDiscovery
                    if n_ind==2
                        #Discovery today, sell it
                        @unpack NPV_Discovery  = par
                        g=τ*GDP_Final_Matrix[z_ind,n_ind,1]+GDP_Oil_Matrix[z_ind,n_ind,1]+NPV_Discovery
                    else
                        if n_ind==length(GRIDS.GR_n)
                            #Large field today, export extra rents
                            @unpack OilRents_Discovery = par
                            g=τ*GDP_Final_Matrix[z_ind,n_ind,1]+GDP_Oil_Matrix[z_ind,n_ind,1]-OilRents_Discovery
                        end
                    end
                end
            end
        else
            cons=GDP_Final_Matrix[z_ind,n_ind,1]+GDP_Oil_Matrix[z_ind,n_ind,1]
            g=0.0
        end
        L=L_Matrix[z_ind,n_ind,1]
        return Utility(cons,g,L,par)+β*φ*itp_EV(0.0,z,n_ind)+β*(1.0-φ)*itp_EVD(z,n_ind)
    else
        @unpack itp_EV, itp_q1 = SOLUTION
        (b_ind,z_ind,n_ind)=Tuple(I)
        Tr=Calculate_Tr(x,bprime,MODEL)
        if With_g
            @unpack τ, PrivateOilField = par
            if PrivateOilField
                Y=GDP_Final_Matrix[z_ind,n_ind,2]+GDP_Oil_Matrix[z_ind,n_ind,2]
                cons=(1-τ)*Y
                g=τ*Y+Tr
            else
                cons=(1-τ)*GDP_Final_Matrix[z_ind,n_ind,2]
                g=τ*GDP_Final_Matrix[z_ind,n_ind,2]+GDP_Oil_Matrix[z_ind,n_ind,2]+Tr
                if SellOilDiscovery
                    if n_ind==2
                        #Discovery today, sell it
                        @unpack NPV_Discovery  = par
                        g=τ*GDP_Final_Matrix[z_ind,n_ind,2]+GDP_Oil_Matrix[z_ind,n_ind,2]+Tr+NPV_Discovery
                    else
                        if n_ind==length(GRIDS.GR_n)
                            #Large field today, export extra rents
                            @unpack OilRents_Discovery = par
                            g=τ*GDP_Final_Matrix[z_ind,n_ind,2]+GDP_Oil_Matrix[z_ind,n_ind,2]+Tr-OilRents_Discovery
                        end
                    end
                end
            end
        else
            cons=GDP_Final_Matrix[z_ind,n_ind,2]+GDP_Oil_Matrix[z_ind,n_ind,2]+Tr
            g=0.0
        end
        L=L_Matrix[z_ind,n_ind,2]
        qq=itp_q1(bprime,z,n_ind)
        if bprime>0.0 && qq==0.0
            #Small penalty for larger debt positions
            #wrong side of laffer curve, it is decreasing
            return Utility(cons,g,L,par)+β*itp_EV(bprime,z,n_ind)-abs(bprime)*sqrt(eps(Float64))
        else
            return Utility(cons,g,L,par)+β*itp_EV(bprime,z,n_ind)
        end
    end
end

###############################################################################
#Functions to optimize given guesses and state
###############################################################################
function Map_From_I_to_State(Default::Bool,I::CartesianIndex,MODEL::Model)
    @unpack GRIDS = MODEL
    @unpack GR_n, GR_z, GR_b = GRIDS
    if Default
        (z_ind,n_ind)=Tuple(I)
        n=GR_n[n_ind]; z=GR_z[z_ind]; b=0.0
        return State(Default,n_ind,n,z,b)
    else
        (b_ind,z_ind,n_ind)=Tuple(I)
        n=GR_n[n_ind]; z=GR_z[z_ind]; b=GR_b[b_ind]
        return State(Default,n_ind,n,z,b)
    end
end

function GridSearch_bprime(I::CartesianIndex,MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack GR_b = GRIDS
    (b_ind,z_ind,n_ind)=Tuple(I)
    Default=false
    x=Map_From_I_to_State(Default,I,MODEL)
    val=-Inf
    bpol=0
    bprime=0.0
    for btry in 1:length(GR_b)
        bprime=GR_b[btry]
        vv=Evaluate_ValueFunction(x,I,bprime,MODEL)
        if vv>val
            val=vv
            bpol=btry
        end
    end
    if bpol<=1
        return par.blowOpt, GR_b[2]
    else
        if bpol>=length(GR_b)
            return GR_b[length(GR_b)-1], par.bhighOpt
        else
            return GR_b[bpol-1], GR_b[bpol+1]
        end
    end
end

function Optimizer_bprime(I::CartesianIndex,MODEL::Model)
    #Do grid search for bounds
    blowOpt, bhighOpt = GridSearch_bprime(I,MODEL)

    Default=false
    x=Map_From_I_to_State(Default,I,MODEL)
    foo(bprime::Float64)=-Evaluate_ValueFunction(x,I,bprime,MODEL)
    res=optimize(foo,blowOpt,bhighOpt,GoldenSection())

    bprime=Optim.minimizer(res)
    vp=Evaluate_ValueFunction(x,I,bprime,MODEL)

    return vp, bprime
end

###############################################################################
#Update solution
###############################################################################
function DefaultUpdater!(I::CartesianIndex,MODEL::Model)
    Default=true; bprime=0.0
    x=Map_From_I_to_State(Default,I,MODEL)
    MODEL.SOLUTION.VD[I]=Evaluate_ValueFunction(x,I,bprime,MODEL)
    return nothing
end

function RepaymentUpdater!(I::CartesianIndex,MODEL::Model)
    MODEL.SOLUTION.VP[I], MODEL.SOLUTION.bprime[I]=Optimizer_bprime(I,MODEL)

    (b_ind,z_ind)=Tuple(I)
    if MODEL.SOLUTION.VP[I]<MODEL.SOLUTION.VD[z_ind]
        MODEL.SOLUTION.V[I]=MODEL.SOLUTION.VD[z_ind]
    else
        MODEL.SOLUTION.V[I]=MODEL.SOLUTION.VP[I]
    end
    return nothing
end

function UpdateDefault!(MODEL::Model)
    @unpack par = MODEL
    @unpack WithDiscoveries = par
    #Loop over all states to fill array of VD
    for I in CartesianIndices(MODEL.SOLUTION.VD)
        if WithDiscoveries
            DefaultUpdater!(I,MODEL)
        else
            (z_ind,n_ind)=Tuple(I)
            if n_ind==1
                DefaultUpdater!(I,MODEL)
            end
        end
    end

    IsDefault=true
    MODEL.SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.VD,IsDefault,MODEL.GRIDS)

    return nothing
end

function UpdateRepayment!(MODEL::Model)
    @unpack par = MODEL
    @unpack WithDiscoveries = par
    #Loop over all states to fill array of VD
    for I in CartesianIndices(MODEL.SOLUTION.VP)
        if WithDiscoveries
            RepaymentUpdater!(I,MODEL)
        else
            (b_ind,z_ind,n_ind)=Tuple(I)
            if n_ind==1
                RepaymentUpdater!(I,MODEL)
            end
        end
    end

    IsDefault=false
    MODEL.SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.VP,IsDefault,MODEL.GRIDS)
    MODEL.SOLUTION.itp_V=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.V,IsDefault,MODEL.GRIDS)
    MODEL.SOLUTION.itp_bprime=CreateInterpolation_Policy(MODEL.SOLUTION.bprime,MODEL.GRIDS)

    return nothing
end

function UpdateSolution!(MODEL::Model)
    UpdateDefault!(MODEL)
    UpdateRepayment!(MODEL)
    UpdateExpectations!(MODEL)

    #Compute expectation interpolations
    IsDefault=true
    MODEL.SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EVD,IsDefault,MODEL.GRIDS)
    IsDefault=false
    MODEL.SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EV,IsDefault,MODEL.GRIDS)
    MODEL.SOLUTION.itp_q1=CreateInterpolation_Price(MODEL.SOLUTION.q1,MODEL.GRIDS)

    return nothing
end

###############################################################################
#Value Function Iteration
###############################################################################
function ComputeDistance_q(MODEL_CURRENT::Model,MODEL_NEXT::Model)
    @unpack par = MODEL_CURRENT
    @unpack Tol_q = par
    dst_q, Ix=findmax(abs.(MODEL_CURRENT.SOLUTION.q1 .- MODEL_NEXT.SOLUTION.q1))
    NotConv=sum(abs.(MODEL_CURRENT.SOLUTION.q1 .- MODEL_NEXT.SOLUTION.q1) .> Tol_q)
    NotConvPct=100.0*NotConv/length(MODEL_CURRENT.SOLUTION.q1)
    return round(dst_q,digits=7), round(NotConvPct,digits=2), Ix
end

function ComputeDistanceV(MODEL_CURRENT::Model,MODEL_NEXT::Model)
    @unpack par = MODEL_CURRENT
    dst_D=maximum(abs.(MODEL_CURRENT.SOLUTION.VD .- MODEL_NEXT.SOLUTION.VD))
    dst_V, Iv=findmax(abs.(MODEL_CURRENT.SOLUTION.V .- MODEL_NEXT.SOLUTION.V))

    NotConv=sum(abs.(MODEL_CURRENT.SOLUTION.V .- MODEL_NEXT.SOLUTION.V) .> par.Tol_V)
    NotConvPct=100.0*NotConv/length(MODEL_CURRENT.SOLUTION.V)
    return round(abs(dst_D),digits=7), round(abs(dst_V),digits=7), Iv, round(NotConvPct,digits=2)
end

function RelevantDistances(PrintProg::Bool,cnt::Int64,MODEL_CURRENT::Model,MODEL_NEXT::Model)
    dst_q, NotConvPct, Ix=ComputeDistance_q(MODEL_CURRENT,MODEL_NEXT)
    dst_D, dst_P, Iv, NotConvPct_P=ComputeDistanceV(MODEL_CURRENT,MODEL_NEXT)
    dst_V=max(dst_D,dst_P)

    if PrintProg
        println("cnt=$cnt, dst_D=$dst_D, dst_P=$dst_P at $Iv, $NotConvPct_P% of V not converged, dst_q=$dst_q")
    end

    return dst_V, dst_q
end

function InitiateModel(GRIDS::Grids,par::Pars)
    SOLUTION=InitiateSolution(GRIDS,par)
    ITP_OUTPUT=Pack_OutputInterpolations(GRIDS)
    return Model(SOLUTION,GRIDS,par,ITP_OUTPUT)
end

function SolveModel_VFI(PrintProg::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, cnt_max = par
    if PrintProg
        println("Preparing solution guess")
    end
    MODEL_CURRENT=InitiateModel(GRIDS,par)
    MODEL_NEXT=deepcopy(MODEL_CURRENT)
    dst_V=1.0; dst_D=1.0; dst_P=1.0; NotConvPct_P=1.0; dst_q=1.0; NotConvPct=100.0
    cnt=0
    if PrintProg
        println("Starting VFI")
    end
    while cnt<cnt_max && (dst_V>Tol_V || dst_q>Tol_q)
        UpdateSolution!(MODEL_NEXT)
        cnt=cnt+1
        dst_V, dst_q=RelevantDistances(PrintProg,cnt,MODEL_CURRENT,MODEL_NEXT)
        MODEL_CURRENT=deepcopy(MODEL_NEXT)
    end

    return MODEL_NEXT
end

function Model_FromSetup(setup_coulumn::Int64,SETUP_FILE::String)
    XX=readdlm(SETUP_FILE,',')
    NAME=convert(String,XX[1,setup_coulumn])
    VEC_PAR=XX[2:end,setup_coulumn]*1.0
    par, GRIDS=Setup_From_Vector(VEC_PAR)

    PrintProg=true; PrintAll=true
    MOD=SolveModel_VFI(PrintProg,GRIDS,par)

    return MOD, NAME
end

function Save_Model_FromSetup(setup_coulumn::Int64,SETUP_FILE::String)
    MOD, NAME=Model_FromSetup(setup_coulumn,SETUP_FILE)
    SaveModel_Vector(NAME,MOD)
    return nothing
end

function Model_FromSetup_And_Counterfactual(setup_coulumn::Int64,SETUP_FILE::String)
    XX=readdlm(SETUP_FILE,',')
    NAME=convert(String,XX[1,setup_coulumn])
    VEC_PAR=XX[2:end,setup_coulumn]*1.0

    VEC_PAR_COUNTERFACTUAL=deepcopy(VEC_PAR)
    #Set identical technology dummy
    VEC_PAR_COUNTERFACTUAL[14]=1.0

    par, GRIDS=Setup_From_Vector(VEC_PAR)
    par_Counterfactual, GRIDS_Counterfactual=Setup_From_Vector(VEC_PAR_COUNTERFACTUAL)

    PrintProg=true; PrintAll=true
    MOD=SolveModel_VFI(PrintProg,GRIDS,par)
    MOD_Counterfactual=SolveModel_VFI(PrintProg,GRIDS_Counterfactual,par_Counterfactual)

    return MOD, MOD_Counterfactual, NAME
end

################################################################################
### Functions for simulations
################################################################################
@with_kw mutable struct Paths
    #Paths of shocks
    n_ind::Array{Int64,1}
    n::Array{Float64,1}
    z::Array{Float64,1}

    #Paths of chosen states
    Def::Array{Float64,1}
    B::Array{Float64,1}

    #Path of prices
    Spreads::Array{Float64,1}
    Pstar_f::Array{Float64,1}
    Pstar_o::Array{Float64,1}

    #Path of quantities
    GDP::Array{Float64,1}
    C::Array{Float64,1}
    G::Array{Float64,1}
    Y::Array{Float64,1}
    Yo::Array{Float64,1}
    L::Array{Float64,1}
    TB::Array{Float64,1}
    CA::Array{Float64,1}
    WorkingCapital::Array{Float64,1}
end

function InitiateEmptyPaths(T::Int64)
    #Initiate with zeros to facilitate averages
    #Paths of shocks
    i1=zeros(Int64,T)
    f2=zeros(Float64,T)
    f3=zeros(Float64,T)
    f4=zeros(Float64,T)
    f5=zeros(Float64,T)
    f6=zeros(Float64,T)
    f7=zeros(Float64,T)
    f8=zeros(Float64,T)
    f9=zeros(Float64,T)
    f10=zeros(Float64,T)
    f11=zeros(Float64,T)
    f12=zeros(Float64,T)
    f13=zeros(Float64,T)
    f14=zeros(Float64,T)
    f15=zeros(Float64,T)
    f16=zeros(Float64,T)
    f17=zeros(Float64,T)
    return Paths(i1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17)
end

function Get_CDF(PDF::Array{Float64,1})
    N,=size(PDF)
    CDF=zeros(Float64,N)
    CDF[1]=PDF[1]
    for i in 2:N
        CDF[i]=CDF[i-1]+PDF[i]
    end
    return CDF
end

function Draw_New_discreteShock(x_ind::Int64,PI_x::Array{Float64,2})
    PDF=PI_x[x_ind,:]
    CDF=Get_CDF(PDF)
    x=rand()
    x_prime=0
    for i in 1:length(CDF)
        if x<=CDF[i]
            x_prime=i
            break
        else
        end
    end
    return x_prime
end

function TS_Discrete_nShocks(n_ind0::Int64,T::Int64,PI_n::Array{Float64,2})
    n_ind_TS=Array{Int64,1}(undef,T)
    #initial μ state is defined outside this function
    n_ind_TS[1]=n_ind0
    for t in 2:T
        n_ind_TS[t]=Draw_New_discreteShock(n_ind_TS[t-1],PI_n)
    end
    return n_ind_TS
end

function Simulate_n_shocks!(ForMoments::Bool,PATHS::Paths,MODEL::Model)
    @unpack GRIDS = MODEL
    @unpack GR_n, PI_n = GRIDS
    T=length(PATHS.n)
    PATHS.n_ind[1]=1
    for t in 2:T
        if ForMoments
            PATHS.n_ind[t]=1
        else
            PATHS.n_ind[t]=Draw_New_discreteShock(PATHS.n_ind[t-1],PI_n)
        end
    end
    for t in 1:T
        PATHS.n[t]=GR_n[PATHS.n_ind[t]]
    end
    return nothing
end

function Simulate_z_shocks!(PATHS::Paths,MODEL::Model)
    @unpack par = MODEL
    @unpack μ_z, ρ_z, dist_ϵz = par
    T=length(PATHS.z)
    ϵz_TS=rand(dist_ϵz,T)

    PATHS.z[1]=1.0
    for t in 2:T
        PATHS.z[t]=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(PATHS.z[t-1])+ϵz_TS[t])
    end
    return nothing
end

function ComputeSpreads(n_ind::Int64,z::Float64,bprime::Float64,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    @unpack itp_q1 = SOLUTION
    qq=max(itp_q1(bprime,z,n_ind),1e-2)
    return ComputeSpreadWithQ(qq,par)
end

function Quantities_Into_Paths!(t::Int64,x::State,bprime::Float64,PATHS::Paths,MODEL::Model)
    @unpack par, ITP_OUTPUT = MODEL
    @unpack With_g = par
    @unpack Default, n_ind, z, b = x

    #Path of quantities
    if Default
        d=1
    else
        d=2
    end
    gdp_final=ITP_OUTPUT.itp_gdp_final(z,n_ind,d)
    gdp_oil=ITP_OUTPUT.itp_gdp_oil(z,n_ind,d)
    L=ITP_OUTPUT.itp_L(z,n_ind,d)
    y=ITP_OUTPUT.itp_y(z,n_ind,d)
    yo=ITP_OUTPUT.itp_yo(z,n_ind,d)
    WorkingCapital=ITP_OUTPUT.itp_WorkingCapital(z,n_ind,d)

    PATHS.Pstar_f[t]=Price_star_h(true,x,par)
    PATHS.Pstar_o[t]=Price_star_h(false,x,par)

    #Path of quantities
    PATHS.GDP[t]=gdp_final+gdp_oil
    Tr=Calculate_Tr(x,bprime,MODEL)
    if With_g
        @unpack τ, PrivateOilField = par
        if PrivateOilField
            Y=gdp_final+gdp_oil
            PATHS.C[t]=(1-τ)*Y
            PATHS.G[t]=τ*Y+Tr
        else
            PATHS.C[t]=(1-τ)*gdp_final
            PATHS.G[t]=τ*gdp_final+gdp_oil+Tr
        end
    else
        PATHS.C[t]=gdp_final+gdp_oil+Tr
        PATHS.G[t]=0.0
    end
    PATHS.Y[t]=y
    PATHS.Yo[t]=yo
    PATHS.L[t]=L
    PATHS.TB[t]=PATHS.GDP[t]-PATHS.C[t]
    PATHS.CA[t]=-(bprime-b)
    PATHS.WorkingCapital[t]=WorkingCapital

    return nothing
end

function Simulate_EndogenousVariables!(Def0::Float64,B0::Float64,PATHS::Paths,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    @unpack itp_VP, itp_VD, itp_bprime = SOLUTION
    #Allocate floats once
    n_ind=0; n=0.0; z=0.0; b=0.0; bprime=0.0; gdp=0.0; L=0.0

    #Must have already simulated productivity and readmission shocks
    T=length(PATHS.B)
    PATHS.B[1]=B0
    Def_1=Def0 #Default state from previous period
    for t in 1:T
        n_ind=PATHS.n_ind[t]
        n=PATHS.n[t]
        z=PATHS.z[t]
        b=PATHS.B[t]

        if Def_1==1.0
            #Coming from default, check if reentry
            # if PATHS.Readmission[t]<=par.θ
            if rand()<=par.φ
                #Reentry
                Default=false
                x=State(Default,n_ind,n,z,0.0)
                PATHS.B[t]=0.0
                bprime=itp_bprime(b,z,n_ind)
                PATHS.Def[t]=0.0
                PATHS.Spreads[t]=ComputeSpreads(n_ind,z,bprime,MODEL)
            else
                #Remain in default
                Default=true
                x=State(Default,n_ind,n,z,0.0)
                bprime=0.0
                PATHS.Def[t]=1.0
                PATHS.Spreads[t]=0.0
            end
        else
            #Coming from repayment, check if would default today
            if itp_VD(z,n_ind)>itp_VP(b,z,n_ind)
                #Default
                Default=true
                x=State(Default,n_ind,n,z,0.0)
                bprime=0.0
                PATHS.Def[t]=1.0
                PATHS.Spreads[t]=0.0
            else
                #Repayment
                Default=false
                x=State(Default,n_ind,n,z,b)
                bprime=itp_bprime(b,z,n_ind)
                PATHS.Def[t]=0.0
                PATHS.Spreads[t]=ComputeSpreads(n_ind,z,bprime,MODEL)
            end
        end

        Quantities_Into_Paths!(t,x,bprime,PATHS,MODEL)

        Def_1=PATHS.Def[t]
        if t<T
            PATHS.B[t+1]=bprime
        end
    end

    return nothing
end

function Simulate_Paths(ForMoments::Bool,T::Int64,MODEL::Model)
    Random.seed!(1234)
    PATHS=InitiateEmptyPaths(T)
    Simulate_z_shocks!(PATHS,MODEL)
    Simulate_n_shocks!(ForMoments,PATHS,MODEL)
    Def0=0.0; B0=0.0
    Simulate_EndogenousVariables!(Def0,B0,PATHS,MODEL)
    return PATHS
end

function Fill_Path_Simulation!(PATHS::Paths,MODEL::Model)
    #This is only used to compute moments
    Simulate_z_shocks!(PATHS,MODEL)
    #Compute all moments conditional on no discovery
    ForMoments=true
    Simulate_n_shocks!(ForMoments,PATHS,MODEL)

    Def0=0.0; B0=0.0
    Simulate_EndogenousVariables!(Def0,B0,PATHS,MODEL)
    return nothing
end

function ExtractFromLongPaths!(t0::Int64,t1::Int64,PATHS::Paths,PATHS_long::Paths)
    PATHS.n_ind .= PATHS_long.n_ind[t0:t1]
    PATHS.n .= PATHS_long.n[t0:t1]
    PATHS.z .= PATHS_long.z[t0:t1]

    #Paths of chosen states
    PATHS.Def .= PATHS_long.Def[t0:t1]
    PATHS.B .= PATHS_long.B[t0:t1]

    PATHS.Spreads .= PATHS_long.Spreads[t0:t1]
    PATHS.Pstar_f .= PATHS_long.Pstar_f[t0:t1]
    PATHS.Pstar_o .= PATHS_long.Pstar_o[t0:t1]

    #Path of quantities
    PATHS.GDP .= PATHS_long.GDP[t0:t1]
    PATHS.C .= PATHS_long.C[t0:t1]
    PATHS.G .= PATHS_long.G[t0:t1]
    PATHS.Y .= PATHS_long.Y[t0:t1]
    PATHS.Yo .= PATHS_long.Yo[t0:t1]
    PATHS.L .= PATHS_long.L[t0:t1]
    PATHS.TB .= PATHS_long.TB[t0:t1]
    PATHS.CA .= PATHS_long.CA[t0:t1]
    PATHS.WorkingCapital .= PATHS_long.WorkingCapital[t0:t1]
    return nothing
end

@with_kw mutable struct Moments
    #Initiate them at 0.0 to facilitate average across samples
    #Default, spreads, and Debt
    DefaultPr::Float64 = 0.0
    MeanSpreads::Float64 = 0.0
    StdSpreads::Float64 = 0.0
    #Stocks
    Debt_GDP::Float64 = 0.0
    #GDP process
    ρ_GDP::Float64 = 0.0
    σ_GDP::Float64 = 0.0
    #Volatilities
    σ_con::Float64 = 0.0
    σ_G::Float64 = 0.0
    σ_L::Float64 = 0.0
    σ_TB_y::Float64 = 0.0
    σ_CA_y::Float64 = 0.0
    #Cyclicality
    Corr_con_GDP::Float64 = 0.0
    Corr_L_GDP::Float64 = 0.0
    Corr_Spreads_GDP::Float64 = 0.0
    Corr_TB_GDP::Float64 = 0.0
    Corr_CA_GDP::Float64 = 0.0
    #Other
    GDP_drop_DefEv::Float64 = 0.0
    WK_GDP::Float64 = 0.0
    yoil_GDP_target_L::Float64 = 0.0
    yoil_GDP_target_H::Float64 = 0.0
end

function hp_filter(y::Vector{Float64}, lambda::Float64)
    #Returns trend component
    n = length(y)
    @assert n >= 4

    diag2 = lambda*ones(n-2)
    diag1 = [ -2lambda; -4lambda*ones(n-3); -2lambda ]
    diag0 = [ 1+lambda; 1+5lambda; (1+6lambda)*ones(n-4); 1+5lambda; 1+lambda ]

    #D = spdiagm((diag2, diag1, diag0, diag1, diag2), (-2,-1,0,1,2))
    D = spdiagm(-2 => diag2, -1 => diag1, 0 => diag0, 1 => diag1, 2 => diag2)

    D\y
end

function MomentsIntoStructure!(t0::Int64,t1::Int64,PATHS::Paths,PATHS_long::Paths,MOM::Moments,MOD::Model)
    @unpack par = MOD
    #Will fill values into structures PATHS_long, PATHS, and MOM
    Fill_Path_Simulation!(PATHS_long,MOD)
    ExtractFromLongPaths!(t0,t1,PATHS,PATHS_long)

    #Compute other easy moments
    if sum(PATHS.Def .== 0.0)==0.0
        MOM.MeanSpreads=mean(PATHS.Spreads)
        MOM.StdSpreads=std(PATHS.Spreads)
        if par.Quarterly
            MOM.Debt_GDP=mean(100 .* (PATHS.B ./ (4 .* PATHS.GDP)))
        else
            MOM.Debt_GDP=mean(100 .* (PATHS.B ./ (1 .* PATHS.GDP)))
        end
        MOM.WK_GDP=mean(100*(PATHS.WorkingCapital ./ PATHS.GDP))
    else
        MOM.MeanSpreads=sum((PATHS.Spreads) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
        VarSpr=sum(((PATHS.Spreads .- MOM.MeanSpreads) .^ 2) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
        MOM.StdSpreads=sqrt(VarSpr)
        if par.Quarterly
            MOM.Debt_GDP=sum((100 .* (PATHS.B ./ (4 .* PATHS.GDP))) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
        else
            MOM.Debt_GDP=sum((100 .* (PATHS.B ./ (1 .* PATHS.GDP))) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
        end
        MOM.WK_GDP=sum((100 .* (PATHS.WorkingCapital ./ PATHS.GDP)) .* (PATHS.Def .== 0.0))/sum(PATHS.Def .== 0.0)
    end

    # MOM.OilRents_GDP=mean(100*(PATHS.n ./ PATHS.GDP))
    Nn=length(MOD.GRIDS.GR_n)
    if sum(PATHS.n_ind .== 1)>0.0
        MOM.yoil_GDP_target_L=100*sum((PATHS.Yo ./ PATHS.GDP) .* (PATHS.n_ind .== 1))/sum(PATHS.n_ind .== 1)
    else
        MOM.yoil_GDP_target_L=0.0
    end
    if sum(PATHS.n_ind .== Nn)>0.0
        MOM.yoil_GDP_target_H=100*sum((PATHS.Yo ./ PATHS.GDP) .* (PATHS.n_ind .== Nn))/sum(PATHS.n_ind .== Nn)
    else
        MOM.yoil_GDP_target_H=0.0
    end

    ###Hpfiltering
    #GDP
    log_GDP=log.(abs.(PATHS.GDP))
    GDP_trend=hp_filter(log_GDP,par.HPFilter_Par)
    GDP_cyc=100.0*(log_GDP .- GDP_trend)
    #Consumption
    log_con=log.(abs.(PATHS.C))
    con_trend=hp_filter(log_con,par.HPFilter_Par)
    con_cyc=100.0*(log_con .- con_trend)
    #Government consumption
    if par.With_g
        log_g=log.(abs.(PATHS.G))
        g_trend=hp_filter(log_g,par.HPFilter_Par)
        g_cyc=100.0*(log_g .- g_trend)
    else
        g_cyc=zeros(Float64,length(PATHS.G))
    end
    #Labor
    L_TS=PATHS.L
    log_L=log.(abs.(L_TS))
    L_trend=hp_filter(log_L,par.HPFilter_Par)
    L_cyc=100.0*(log_L .- L_trend)
    #Trade balance and current account
    TB_y_TS=100 * PATHS.TB ./ PATHS.GDP
    CA_y_TS=100 * PATHS.CA ./ PATHS.GDP
    #GDP process
    MOM.ρ_GDP=cor(GDP_cyc[2:end],GDP_cyc[1:end-1])
    MOM.σ_GDP=std(GDP_cyc)
    #Volatilities
    MOM.σ_con=std(con_cyc)
    MOM.σ_G=std(g_cyc)
    MOM.σ_L=std(L_cyc)
    MOM.σ_TB_y=std(TB_y_TS)
    MOM.σ_CA_y=std(CA_y_TS)
    #Correlations with GDP
    MOM.Corr_con_GDP=cor(GDP_cyc,con_cyc)
    MOM.Corr_L_GDP=cor(GDP_cyc,L_cyc)
    MOM.Corr_Spreads_GDP=cor(GDP_cyc .* (PATHS.Def .== 0.0),PATHS.Spreads .* (PATHS.Def .== 0.0))
    MOM.Corr_TB_GDP=cor(GDP_cyc .* (PATHS.Def .== 0.0),TB_y_TS .* (PATHS.Def .== 0.0))
    MOM.Corr_CA_GDP=cor(GDP_cyc .* (PATHS.Def .== 0.0),CA_y_TS .* (PATHS.Def .== 0.0))

    return nothing
end

function DefaultMomentsIntoStructure!(MOM::Moments,MOD::Model)
    @unpack par = MOD
    @unpack drp, Tlong, Quarterly, μ_z = par
    T=drp+Tlong
    PATHS_long=InitiateEmptyPaths(T)
    Fill_Path_Simulation!(PATHS_long,MOD)
    #Compute default probability and GDP_drop_Def
    Def_Ev=0.0
    for t in 2:length(PATHS_long.Def)
        if PATHS_long.Def[t]==1.0 && PATHS_long.Def[t-1]==0.0
            Def_Ev=Def_Ev+1
        end
    end
    if Quarterly
        #Transform to annual default probability
        pr_q=Def_Ev/sum(1 .- PATHS_long.Def)
        pr_ndq=1-pr_q
        pr_ndy=pr_ndq^4
        MOM.DefaultPr=100*(1-pr_ndy)
    else
        #Default frequency is default probability
        MOM.DefaultPr=100*Def_Ev/sum(1 .- PATHS_long.Def)
    end

    xTrend=State(false,1,par.nL,μ_z,0.0)
    gdp_final_trend, gdp_oil_trend, =Compute_Quantities_GivenState(xTrend,par)
    gdpTrend=gdp_final_trend+gdp_oil_trend
    drp_DefEv=0.0
    if Def_Ev>0.0
        for t in 2:length(PATHS_long.Def)
            if PATHS_long.Def[t]==1.0 && PATHS_long.Def[t-1]==0.0
                # DROP=100*((PATHS_long.GDP[t]/gdpTrend)-1)
                DROP=100*((PATHS_long.GDP[t]/PATHS_long.GDP[t-1])-1)
                drp_DefEv=drp_DefEv+DROP/Def_Ev
            end
        end
    end
    MOM.GDP_drop_DefEv=drp_DefEv

    return nothing
end

function AverageMomentsManySamples(Tmom::Int64,NSamplesMoments::Int64,MOD::Model)
    @unpack par = MOD
    @unpack drp = par
    T=drp+Tmom
    PATHS_long=InitiateEmptyPaths(T)
    PATHS=InitiateEmptyPaths(Tmom)
    t0=drp+1; t1=T

    Random.seed!(1234)
    MOMENTS=Moments(); MOMS=Moments()
    DefaultMomentsIntoStructure!(MOMENTS,MOD)
    for i in 1:NSamplesMoments
        # println(i)
        MomentsIntoStructure!(t0,t1,PATHS,PATHS_long,MOMS,MOD)
        #Default, spreads, and Debt
        # MOMENTS.DefaultPr=MOMENTS.DefaultPr+MOMS.DefaultPr/NSamplesMoments
        MOMENTS.MeanSpreads=MOMENTS.MeanSpreads+MOMS.MeanSpreads/NSamplesMoments
        MOMENTS.StdSpreads=MOMENTS.StdSpreads+MOMS.StdSpreads/NSamplesMoments
        #Stocks
        MOMENTS.Debt_GDP=MOMENTS.Debt_GDP+MOMS.Debt_GDP/NSamplesMoments
        #GDP process
        MOMENTS.ρ_GDP=MOMENTS.ρ_GDP+MOMS.ρ_GDP/NSamplesMoments
        MOMENTS.σ_GDP=MOMENTS.σ_GDP+MOMS.σ_GDP/NSamplesMoments
        #Volatilities
        MOMENTS.σ_con=MOMENTS.σ_con+MOMS.σ_con/NSamplesMoments
        MOMENTS.σ_G=MOMENTS.σ_G+MOMS.σ_G/NSamplesMoments
        MOMENTS.σ_L=MOMENTS.σ_L+MOMS.σ_L/NSamplesMoments
        MOMENTS.σ_TB_y=MOMENTS.σ_TB_y+MOMS.σ_TB_y/NSamplesMoments
        MOMENTS.σ_CA_y=MOMENTS.σ_CA_y+MOMS.σ_CA_y/NSamplesMoments
        #Cyclicality
        MOMENTS.Corr_con_GDP=MOMENTS.Corr_con_GDP+MOMS.Corr_con_GDP/NSamplesMoments
        MOMENTS.Corr_L_GDP=MOMENTS.Corr_L_GDP+MOMS.Corr_L_GDP/NSamplesMoments
        MOMENTS.Corr_Spreads_GDP=MOMENTS.Corr_Spreads_GDP+MOMS.Corr_Spreads_GDP/NSamplesMoments
        MOMENTS.Corr_TB_GDP=MOMENTS.Corr_TB_GDP+MOMS.Corr_TB_GDP/NSamplesMoments
        MOMENTS.Corr_CA_GDP=MOMENTS.Corr_CA_GDP+MOMS.Corr_CA_GDP/NSamplesMoments
        #Other
        # MOMENTS.GDP_drop_DefEv=MOMENTS.GDP_drop_DefEv+MOMS.GDP_drop_DefEv/NSamplesMoments
        MOMENTS.WK_GDP=MOMENTS.WK_GDP+MOMS.WK_GDP/NSamplesMoments
        MOMENTS.yoil_GDP_target_L=MOMENTS.yoil_GDP_target_L+MOMS.yoil_GDP_target_L/NSamplesMoments
        MOMENTS.yoil_GDP_target_H=MOMENTS.yoil_GDP_target_H+MOMS.yoil_GDP_target_H/NSamplesMoments
    end
    return MOMENTS
end

################################################################################
### Functions to create average paths of discovery
################################################################################
function Extract_TS(t1::Int64,tT::Int64,PATHS::Paths)
    return Paths(PATHS.n_ind[t1:tT],
                 PATHS.n[t1:tT],
                 PATHS.z[t1:tT],
                 PATHS.Def[t1:tT],
                 PATHS.B[t1:tT],
                 PATHS.Spreads[t1:tT],
                 PATHS.Pstar_f[t1:tT],
                 PATHS.Pstar_o[t1:tT],
                 PATHS.GDP[t1:tT],
                 PATHS.C[t1:tT],
                 PATHS.G[t1:tT],
                 PATHS.Y[t1:tT],
                 PATHS.Yo[t1:tT],
                 PATHS.L[t1:tT],
                 PATHS.TB[t1:tT],
                 PATHS.CA[t1:tT],
                 PATHS.WorkingCapital[t1:tT])
end

function SimulatePathsOfDiscovery_BeforeExtracting(Tbefore::Int64,Tafter::Int64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack GR_n = GRIDS
    @unpack drp = par
    T=drp+Tbefore+1+Tafter
    PATHS=InitiateEmptyPaths(T)

    #Simulate discovery
    PATHS.n_ind .= ones(Int64,T)
    PATHS.n_ind[drp+Tbefore+1]=2

    n_ind_TS=TS_Discrete_nShocks(2,Tafter+1,GRIDS.PI_n)
    PATHS.n_ind[drp+Tbefore+1:end] .= n_ind_TS

    for t in 1:T
        PATHS.n[t]=GR_n[PATHS.n_ind[t]]
    end

    #Simulate other shocks
    Simulate_z_shocks!(PATHS,MODEL)
    Def0=0.0; B0=0.0
    Simulate_EndogenousVariables!(Def0,B0,PATHS,MODEL)

    return PATHS
end

function SimulatePathsOfDiscovery(DropDefaults::Bool,Tbefore::Int64,Tafter::Int64,MODEL::Model)
    @unpack par = MODEL
    @unpack drp = par
    PATHS=SimulatePathsOfDiscovery_BeforeExtracting(Tbefore,Tafter,MODEL)
    if DropDefaults
        cnt=1
        while sum(PATHS.Def[drp+Tbefore-25:drp+Tbefore+1+Tafter])>0.0
            PATHS=SimulatePathsOfDiscovery_BeforeExtracting(Tbefore,Tafter,MODEL)
        end
    end
    return Extract_TS(drp+1,drp+Tbefore+1+Tafter,PATHS)
end

function SumPathForAverage!(N::Int64,PATHS_AV::Paths,PATHS::Paths)
    #Paths of shocks
    PATHS_AV.n=PATHS_AV.n .+ (PATHS.n ./ N)
    PATHS_AV.z=PATHS_AV.z .+ (PATHS.z ./ N)

    #Paths of chosen states
    PATHS_AV.Def=PATHS_AV.Def .+ (PATHS.Def ./ N)
    PATHS_AV.B=PATHS_AV.B .+ (PATHS.B ./ N)

    #Path of prices
    PATHS_AV.Spreads=PATHS_AV.Spreads .+ (PATHS.Spreads ./ N)
    PATHS_AV.Pstar_f=PATHS_AV.Pstar_f .+ (PATHS.Pstar_f ./ N)
    PATHS_AV.Pstar_o=PATHS_AV.Pstar_o .+ (PATHS.Pstar_o ./ N)

    #Path of quantities
    PATHS_AV.GDP=PATHS_AV.GDP .+ (PATHS.GDP ./ N)
    PATHS_AV.C=PATHS_AV.C .+ (PATHS.C ./ N)
    PATHS_AV.G=PATHS_AV.G .+ (PATHS.G ./ N)
    PATHS_AV.Y=PATHS_AV.Y .+ (PATHS.Y ./ N)
    PATHS_AV.Yo=PATHS_AV.Yo .+ (PATHS.Yo ./ N)
    PATHS_AV.L=PATHS_AV.L .+ (PATHS.L ./ N)
    PATHS_AV.TB=PATHS_AV.TB .+ (PATHS.TB ./ N)
    PATHS_AV.CA=PATHS_AV.CA .+ (PATHS.CA ./ N)
    PATHS_AV.WorkingCapital=PATHS_AV.WorkingCapital .+ (PATHS.WorkingCapital ./ N)
    return nothing
end

function AverageDiscoveryPaths(DropDefaults::Bool,N::Int64,Tbefore::Int64,Tafter::Int64,MODEL::Model)
    @unpack par = MODEL
    @unpack drp = par
    Random.seed!(1234)
    PATHS_AV=InitiateEmptyPaths(Tbefore+1+Tafter)
    for i in 1:N
        println(i)
        PATHS=SimulatePathsOfDiscovery(DropDefaults,Tbefore,Tafter,MODEL)
        SumPathForAverage!(N,PATHS_AV,PATHS)
    end
    return PATHS_AV
end

################################################################################
### Functions to test a sample of parameters
################################################################################
function MomentsNames_RowVector()
    VEC=hcat("DefaultPr","MeanSpreads")
    VEC=hcat(VEC,"StdSpreads")
    VEC=hcat(VEC,"Debt_GDP")
    VEC=hcat(VEC,"rho_GDP")
    VEC=hcat(VEC,"std_GDP")
    VEC=hcat(VEC,"std_con")
    VEC=hcat(VEC,"std_L")
    VEC=hcat(VEC,"std_TB_y")
    VEC=hcat(VEC,"Corr_con_GDP")
    VEC=hcat(VEC,"Corr_L_GDP")
    VEC=hcat(VEC,"Corr_Spreads_GDP")
    VEC=hcat(VEC,"Corr_TB_GDP")
    VEC=hcat(VEC,"GDP_drop_DefEv")
    VEC=hcat(VEC,"WK_GDP")
    VEC=hcat(VEC,"yoil_GDP_target_L")
    VEC=hcat(VEC,"yoil_GDP_target_H")
    return VEC
end

function Moments_To_RowVector(MOM::Moments)
    VEC=hcat(MOM.DefaultPr,MOM.MeanSpreads)
    VEC=hcat(VEC,MOM.StdSpreads)
    VEC=hcat(VEC,MOM.Debt_GDP)
    VEC=hcat(VEC,MOM.ρ_GDP)
    VEC=hcat(VEC,MOM.σ_GDP)
    VEC=hcat(VEC,MOM.σ_con)
    VEC=hcat(VEC,MOM.σ_L)
    VEC=hcat(VEC,MOM.σ_TB_y)
    VEC=hcat(VEC,MOM.Corr_con_GDP)
    VEC=hcat(VEC,MOM.Corr_L_GDP)
    VEC=hcat(VEC,MOM.Corr_Spreads_GDP)
    VEC=hcat(VEC,MOM.Corr_TB_GDP)
    VEC=hcat(VEC,MOM.GDP_drop_DefEv)
    VEC=hcat(VEC,MOM.WK_GDP)
    VEC=hcat(VEC,MOM.yoil_GDP_target_L)
    VEC=hcat(VEC,MOM.yoil_GDP_target_H)
    return VEC
end

function Get_VEC_PAR_TRY(PARS_TRY::Array{Float64,1},VEC_PAR::Array{Float64})
    A=PARS_TRY[1]      #18 in VEC_PAR
    β=PARS_TRY[2]      #19 in VEC_PAR
    θf=PARS_TRY[3]      #20 in VEC_PAR
    # ρ_z=PARS_TRY[4]    #21 in VEC_PAR
    σ_ϵz=PARS_TRY[4]   #22 in VEC_PAR
    PAR_VEC=hcat(A,β,θf,σ_ϵz)

    VEC_PAR_TRY=deepcopy(VEC_PAR)
    VEC_PAR_TRY[18]=A
    VEC_PAR_TRY[19]=β
    VEC_PAR_TRY[20]=θf
    # VEC_PAR_TRY[21]=ρ_z
    VEC_PAR_TRY[22]=σ_ϵz
    return VEC_PAR_TRY, PAR_VEC
end

function CheckMomentsForTry(PARS_TRY::Array{Float64,1},VEC_PAR::Array{Float64})
    VEC_PAR_TRY, PAR_VEC=Get_VEC_PAR_TRY(PARS_TRY,VEC_PAR)
    par_Try, GRIDS_Try=Setup_From_Vector(VEC_PAR_TRY)
    PrintProgress=false
    MODEL_Try=SolveModel_VFI(PrintProgress,GRIDS_Try,par_Try)
    MOMENTS=AverageMomentsManySamples(par_Try.Tmom,par_Try.NSamplesMoments,MODEL_Try)
    MOM_VEC=Moments_To_RowVector(MOMENTS)
    return hcat(PAR_VEC,MOM_VEC)
end

function VectorsWithBounds(SETUP_PARAMETER_BOUNDS::String)
    XX=readdlm(SETUP_PARAMETER_BOUNDS,',')
    PAR_BOUNDS=1.0*XX[2:end,2:end]
    NAMES_VEC=XX[2:end,1]
    PAR_NAMES=hcat(NAMES_VEC[1],NAMES_VEC[2])
    for i in 3:length(NAMES_VEC)
        PAR_NAMES=hcat(PAR_NAMES,NAMES_VEC[i])
    end
    #Vectors with bounds
    lb=PAR_BOUNDS[:,1]
    ub=PAR_BOUNDS[:,2]
    return lb, ub, PAR_NAMES
end

function GenerateMatrixForTries(N::Int64,SETUP_PARAMETER_BOUNDS::String)
    lb, ub, PAR_NAMES=VectorsWithBounds(SETUP_PARAMETER_BOUNDS)
    N_pars=length(lb)
    MAT_TRY=Array{Float64,2}(undef,N,N_pars)

    #Generate Sobol sequence
    ss = skip(SobolSeq(lb, ub),N)
    #Fill out MAT_TRY
    for i in 1:N
        MAT_TRY[i,:]=next!(ss)
    end

    return MAT_TRY, PAR_NAMES
end

function TryDifferentCalibrations(N::Int64,case_column::Int64,SETUP_FILE::String,SETUP_PARAMETER_BOUNDS::String)
    #Unpack parameters for relevant case
    XX=readdlm(SETUP_FILE,',')
    CASE_NAME=XX[1,case_column]
    OUTPUT_FILE_NAME="TriedCalibrations, $CASE_NAME.csv"
    VEC_PAR=1.0*XX[2:end,case_column]

    #Generate Sobol sequence of vectors (α,β,knk,d1,φ)
    MAT_TRY, PAR_NAMES=GenerateMatrixForTries(N,SETUP_PARAMETER_BOUNDS)
    N, N_pars_Sobol=size(MAT_TRY)

    #Loop paralelly over all parameter tries
    MOMENT_NAMES=MomentsNames_RowVector()
    COL_NAMES=hcat(PAR_NAMES,MOMENT_NAMES)
    N_moments=length(MOMENT_NAMES)
    N_pars=length(PAR_NAMES)
    PARAMETER_MOMENTS_MATRIX=SharedArray{Float64,2}(N,N_pars+N_moments)
    @sync @distributed for i in 1:N
        println("Doing i=$i of $N")
        PARAMETER_MOMENTS_MATRIX[i,:]=CheckMomentsForTry(MAT_TRY[i,:],VEC_PAR)
        MAT=vcat(COL_NAMES,PARAMETER_MOMENTS_MATRIX)
        writedlm(OUTPUT_FILE_NAME,MAT,',')
    end
    return nothing
end

################################################################################
### Functions to choose best calibration
################################################################################
function RelevantTargets()
    # ρ_GDP=0.579
    σ_GDP=3.11
    GDP_drop_DefEv=-13.28
    DefaultPr=0.51
    MeanSpreads=2.9
    WK_GDP=8.09
    return vcat(σ_GDP,GDP_drop_DefEv,MeanSpreads,WK_GDP)
end

function RelevantMatricesFromCase(case_column::Int64,SETUP_FILE::String,FOLDER_TRIES::String)
    #Get case name
    XX=readdlm(SETUP_FILE,',')
    CASE_NAME=XX[1,case_column]

    #Get case tried moments
    if FOLDER_TRIES==" "
        MOMENT_MATCHING_FILE_NAME="TriedCalibrations, $CASE_NAME.csv"
    else
        MOMENT_MATCHING_FILE_NAME="$FOLDER_TRIES\\TriedCalibrations, $CASE_NAME.csv"
    end
    XX_MOMENTS=readdlm(MOMENT_MATCHING_FILE_NAME,',')

    #Relevant sub-matrices
    col_σ_GDP=10
    col_GDP_drop_DefEv=18
    col_MeanSpreads=6
    col_DefaultPr=5
    col_WK_GDP=19
    COLUMNS=vcat(col_σ_GDP,col_GDP_drop_DefEv,col_MeanSpreads,col_WK_GDP)

    MAT_PARS=1.0*XX_MOMENTS[2:end,1:4]
    σ_GDP=1.0*XX_MOMENTS[2:end,col_σ_GDP]
    GDP_drop_DefEv=1.0*XX_MOMENTS[2:end,col_GDP_drop_DefEv]
    MeanSpreads=1.0*XX_MOMENTS[2:end,col_MeanSpreads]
    DefaultPr=1.0*XX_MOMENTS[2:end,col_DefaultPr]
    WK_GDP=1.0*XX_MOMENTS[2:end,col_WK_GDP]
    MAT_RELEVANT=hcat(σ_GDP,GDP_drop_DefEv,MeanSpreads,WK_GDP)

    return MAT_PARS, MAT_RELEVANT, COLUMNS
end

function Best_Parameterization(case_column::Int64,SETUP_FILE::String,FOLDER_TRIES::String)
    MAT_PARS, MAT_RELEVANT, COLUMNS=RelevantMatricesFromCase(case_column,SETUP_FILE,FOLDER_TRIES)

    #Targets
    TARGETS=RelevantTargets()

    #Choose best parameters
    dst=Inf
    best_row=0
    Ntries, Nmoms=size(MAT_RELEVANT)
    for i in 1:Ntries
        mom_try=MAT_RELEVANT[i,:]
        dst_Try=norm(mom_try-TARGETS,Inf)
        if dst_Try<dst
            dst=dst_Try
            best_row=i
        end
    end

    return MAT_PARS[best_row,:], COLUMNS
end

function ParameterVector_Calibrated(case_column::Int64,SETUP_FILE::String,FOLDER_TRIES::String)
    XX=readdlm(SETUP_FILE,',')
    VEC_PAR=XX[2:end,case_column]*1.0

    CALIBRATED_PARS, COLUMNS=Best_Parameterization(case_column,SETUP_FILE,FOLDER_TRIES)
    A_cal=CALIBRATED_PARS[1]
    β_cal=CALIBRATED_PARS[2]
    θf_cal=CALIBRATED_PARS[3]
    σ_ϵz_cal=CALIBRATED_PARS[4]
    VEC_PAR[18]=A_cal
    VEC_PAR[19]=β_cal
    VEC_PAR[20]=θf_cal
    VEC_PAR[22]=σ_ϵz_cal

    return VEC_PAR
end

function Setup_Calibrated(case_column::Int64,SETUP_FILE::String,FOLDER_TRIES::String)
    VEC_PAR=ParameterVector_Calibrated(case_column,SETUP_FILE,FOLDER_TRIES)
    return Setup_From_Vector(VEC_PAR)
end

function Model_Calibrated(case_column::Int64,SETUP_FILE::String,FOLDER_TRIES::String)
    XX=readdlm(SETUP_FILE,',')
    NAME=convert(String,XX[1,case_column])
    VEC_PAR=XX[2:end,case_column]*1.0
    par, GRIDS=Setup_Calibrated(case_column,SETUP_FILE,FOLDER_TRIES)

    PrintProg=true; PrintAll=true
    MOD=SolveModel_VFI(PrintProg,GRIDS,par)

    return MOD, NAME
end

function Save_Model_Calibrated(case_column::Int64,SETUP_FILE::String,FOLDER_TRIES::String)
    MOD, NAME=Model_Calibrated(case_column,SETUP_FILE,FOLDER_TRIES)
    SaveModel_Vector("$NAME.csv",MOD)
    return nothing
end

function Save_Setup_Calibrated(SETUP_ORIGINAL_FILE::String,FOLDER_TRIES::String)
    VEC_PAR_4=ParameterVector_Calibrated(4,SETUP_ORIGINAL_FILE,FOLDER_TRIES)
    VEC_PAR_5=ParameterVector_Calibrated(5,SETUP_ORIGINAL_FILE,FOLDER_TRIES)
    MAT=readdlm(SETUP_ORIGINAL_FILE,',')
    MAT[2:end,4] .= VEC_PAR_4
    MAT[2:end,5] .= VEC_PAR_5
    CALIBRATED_SETUP_FILE_NAME="Setup_Calibrated.csv"
    writedlm(CALIBRATED_SETUP_FILE_NAME,MAT,',')
    return nothing
end

function Save_Setup_Calibrated_Append(IsFirst::Bool,case_column::Int64,SETUP_ORIGINAL_FILE::String,FOLDER_TRIES::String)
    VEC_PAR=ParameterVector_Calibrated(case_column,SETUP_ORIGINAL_FILE,FOLDER_TRIES)
    XX=readdlm(SETUP_ORIGINAL_FILE,',')
    NAME_COLUMN=XX[1,case_column]
    VEC_APPEND=vcat(NAME_COLUMN,VEC_PAR)
    CALIBRATED_SETUP_FILE_NAME="Setup_Calibrated.csv"
    if IsFirst
        #This is the first attempt
        MAT=XX[:,1:3]
    else
        MAT=readdlm(CALIBRATED_SETUP_FILE_NAME,',')
    end
    MAT=hcat(MAT,VEC_APPEND)
    writedlm(CALIBRATED_SETUP_FILE_NAME,MAT,',')
    return nothing
end

################################################################################
### Functions to compute welfare gains of discoveries
################################################################################
function Welfare_Gains_t(t::Int64,TS::Paths,MOD::Model)
    @unpack SOLUTION, par = MOD
    @unpack σ = par
    #TS only has periods with no discovery
    #this will give  hypothetical discovery in every period
    #taking each period as a random draw from the ergodic distribution
    z=TS.z[t]; b=TS.B[t]; Def=TS.Def[t]
    if Def==1.0
        @unpack itp_VD = SOLUTION
        v_no_disc=itp_VD(z,1)
        v_disc=itp_VD(z,2)
    else
        @unpack itp_V = SOLUTION
        v_no_disc=itp_V(b,z,1)
        v_disc=itp_V(b,z,2)
    end
    return 100*(((v_disc/v_no_disc)^(1/(1-σ)))-1)
end

function Average_Welfare_Gains(N::Int64,MOD::Model)
    @unpack par = MOD
    @unpack drp = par
    T=drp+N; ForMoments=true
    TS=Simulate_Paths(ForMoments,T,MOD)
    wg=0.0
    for i in 1:N
        println(i)
        t=drp+i
        wg=wg+Welfare_Gains_t(t,TS,MOD)/N
    end
    return wg
end

function Model_Private_Field(setup_coulumn::Int64,SETUP_FILE::String)
    XX=readdlm(SETUP_FILE,',')
    NAME=convert(String,XX[1,setup_coulumn])
    VEC_PAR=XX[2:end,setup_coulumn]*1.0
    par, GRIDS=Setup_From_Vector(VEC_PAR)
    par_Priv=Pars(par,PrivateOilField=true)

    PrintProg=true; PrintAll=true
    MOD=SolveModel_VFI(PrintProg,GRIDS,par_Priv)

    return MOD
end

function Model_Sell_Giant_Field(setup_coulumn::Int64,SETUP_FILE::String)
    XX=readdlm(SETUP_FILE,',')
    NAME=convert(String,XX[1,setup_coulumn])
    VEC_PAR=XX[2:end,setup_coulumn]*1.0
    par, GRIDS=Setup_From_Vector(VEC_PAR)

    OilRents_Discovery=OilRents_FromDiscovery_ss(par)
    NPV_Discovery=NPV_giant_field(par)
    par_Sell=Pars(par,SellOilDiscovery=true,OilRents_Discovery=OilRents_Discovery,NPV_Discovery=NPV_Discovery)

    PrintProg=true; PrintAll=true
    MOD=SolveModel_VFI(PrintProg,GRIDS,par_Sell)

    return MOD
end

function Welfare_Gains_t_Change_Model(AtDiscovery::Bool,t::Int64,TS::Paths,MOD0::Model,MOD1::Model)
    @unpack par = MOD0
    @unpack σ = par
    #TS only has periods with no discovery
    #this will give  hypothetical discovery in every period
    #taking each period as a random draw from the ergodic distribution
    z=TS.z[t]; b=TS.B[t]; Def=TS.Def[t]
    n_ind=TS.n_ind[t]
    if Def==1.0
        if AtDiscovery
            v0=MOD0.SOLUTION.itp_VD(z,2)
            v1=MOD1.SOLUTION.itp_VD(z,2)
        else
            v0=MOD0.SOLUTION.itp_VD(z,n_ind)
            v1=MOD1.SOLUTION.itp_VD(z,n_ind)
        end
    else
        if AtDiscovery
            v0=MOD0.SOLUTION.itp_V(b,z,2)
            v1=MOD1.SOLUTION.itp_V(b,z,2)
        else
            v0=MOD0.SOLUTION.itp_V(b,z,n_ind)
            v1=MOD1.SOLUTION.itp_V(b,z,n_ind)
        end
    end
    return 100*(((v1/v0)^(1/(1-σ)))-1)
end

function Average_Welfare_Gains_Change_Model(AtDiscovery::Bool,N::Int64,MOD0::Model,MOD1::Model)
    @unpack par = MOD
    @unpack drp = par
    T=drp+N
    TS=Simulate_Paths(AtDiscovery,T,MOD0)
    wg=0.0
    for i in 1:N
        println(i)
        t=drp+i
        wg=wg+Welfare_Gains_t_Change_Model(AtDiscovery,t,TS,MOD0,MOD1)/N
    end
    return wg
end
