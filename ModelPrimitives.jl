
using Parameters, Roots, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, Sobol, Plots

################################################################
#### Defining parameters and other structures for the model ####
################################################################
@with_kw struct Pars
    ################################################################
    ######## Preferences and technology ############################
    ################################################################
    #Preferences
    σ::Float64 = 2.0          #CRRA parameter
    β::Float64 = 0.8775        #Discount factor
    r_star::Float64 = 0.04    #Risk-free interest rate
    Wforeign::Float64 = 1000.0       #Oil investors deep pockets
    #Debt parameters
    γ::Float64 = 0.14      #Reciprocal of average maturity
    κ::Float64 = 0.00      #Coupon payments
    #Default cost and debt parameters
    θ::Float64 = 0.40#0.0385       #Probability of re-admission
    knk::Float64 = 0.90
    d1::Float64 = 0.222        #income default cost
    d0::Float64 = -d1*knk       #income default cost
    #Capital accumulation
    δ::Float64 = 0.05       #Depreciation rate
    φ::Float64 = 0.7617     #Capital adjustment cost
    PkO::Float64 = 0.0625      #Normalization of price of oil capital
    #Production functions
    #Final consumption good
    η::Float64 = 0.83
    ωN::Float64 = 0.60
    ωM::Float64 = 0.34
    ωO::Float64 = 0.06
    #Value added of intermediates
    kbar::Float64 = 1.0
    αN::Float64 = 0.66          #Capital share in non-traded sector
    αM::Float64 = 0.57          #Capital share in manufacturing sector
    #Oil production
    ζ::Float64 = 0.38           #Share of oil rents in oil output
    τR::Float64 = 1.0          #Share of oil rent captured by country
    #Stochastic process
    #Oil discoveries
    nL::Float64 = 0.05
    nH::Float64 = 0.06
    PrDisc::Float64 = 0.01           #Probability of discovery
    FieldLife::Float64 = 50.0        #Average duration of giant oil fields
    Twait::Int64 = 5                 #Periods between discovery and production
    IncludeDisc::Int64 = 1           #Indicate whether we want discoveries or no discoveries
    ExactWait::Int64 = 0
    ImmediateField::Int64 = 0
    OilFix::Bool = false
    OilPut::Bool = false
    #Parameters to pin down steady state capital and scaling parameters
    Target_Delay::Float64 = 5.4    #Exact years between discovery and production
    Target_NPV::Float64 = 0.045    #Target value for steaty state NPV/GDP
                                   #used to calibrate nH with steady state
    Target_oilXp::Float64 = 0.025 #Target value for steady state (oil exports)/GDP
                                  #used to calibrate nL
    Target_spreads::Float64 = 0.035  #Target for interest rate spreads
                                     #used to calculate NPV in calibration

    #Targets for moment-matching exercise Mexico
    Target_100spreads::Float64 =  3.5
    Target_std_spreads::Float64 = 1.3
    Target_σinv_σgdp::Float64 = 3.0
    Target_σcon_σgdp::Float64 = 1.0
    Target_debt_gdp::Float64 = 43.0 #percent
    Target_DefPr::Float64 = 2.0 #percent
    Target_RP_share::Float64 = 33.0 #percent

    #Targets for moment-matching exercise Arg
    arTarget_100spreads::Float64 =  8.15
    arTarget_std_spreads::Float64 = 4.43
    arTarget_σinv_σgdp::Float64 = 3.0
    arTarget_σcon_σgdp::Float64 = 1.0
    arTarget_debt_gdp::Float64 = 100.0 #percent
    arTarget_DefPr::Float64 = 2.0 #percent
    arTarget_RP_share::Float64 = 33.0 #percent

    #risk-premium parameters, formulation as in Arellano and Ramanarayanan
    α0::Float64 = 8.25
    α1::Float64 = -0.0
    #parameters for productivity shock
    μ_ϵz::Float64 = 0.0
    σ_ϵz::Float64 = 0.0194
    dist_ϵz::UnivariateDistribution = Normal(μ_ϵz,σ_ϵz)
    ρ_z::Float64 = 0.3737
    μ_z::Float64 = 1.0
    zlow::Float64 = exp(log(μ_z)-3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    zhigh::Float64 = exp(log(μ_z)+3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    #parameters for process of price of oil
    μ_ϵp::Float64 = 0.0
    σL_ϵp::Float64 = 0.230#0.262#0.219
    σH_ϵp::Float64 = 0.314#0.406#0.3784
    dist_ϵpL::UnivariateDistribution = Normal(μ_ϵp,σL_ϵp)
    dist_ϵpH::UnivariateDistribution = Normal(μ_ϵp,σH_ϵp)
    πp_LH::Float64 = 0.211#0.157#0.1437
    πp_HL::Float64 = 0.500#0.889#0.8136
    ρ_p::Float64 = 0.854#0.928#0.8874
    μ_p::Float64 = 1.0
    plow::Float64 = exp(log(μ_p)-2*sqrt((σH_ϵp^2.0)/(1.0-(ρ_p^2.0))))
    phigh::Float64 = exp(log(μ_p)+2*sqrt((σH_ϵp^2.0)/(1.0-(ρ_p^2.0))))
    #Quadrature parameters
    N_GLz::Int64 = 21
    N_GLp::Int64 = 21
    #Grids
    Nz::Int64 = 5
    Np::Int64 = 5
    Nn::Int64 = 2
    Nk_fdi::Int64 = 100
    Nk::Int64 = 21
    Nb::Int64 = 21
    klow::Float64 = 0.02
    khigh::Float64 = 0.075#3.0
    blow::Float64 = 0.0
    bhigh::Float64 = 2.0
    klowOpt::Float64 = 0.99*klow
    khighOpt::Float64 = 1.01*klow
    blowOpt::Float64 = blow-0.1             #Minimum level of debt for optimization
    bhighOpt::Float64 = bhigh+0.1            #Maximum level of debt for optimization
    #Parameters for solution algorithm
    cmin::Float64 = 1e-2
    Tol_V::Float64 = 1e-6             #Tolerance for absolute distance for value functions
    Tol_q::Float64 = 1e-6             #Tolerance for absolute distance for q in VFI
    relTolV::Float64 = 0.5            #Tolerance for relative distance in VFI (0.1%)
    Tol_q_pct::Float64 = 1.0          #Tolerance for % of states for which q has not converged (1%)
    cnt_max_fdi::Int64 = 100
    cnt_max::Int64 = 100              #Maximum number of iterations on VFI
    cnt_maxFDI::Int64 = 1000              #Maximum number of iterations on VFI for FDI
    MaxIter_Opt::Int64 = 1000
    g_tol_Opt::Float64 = 1e-8#1e-4
    #Simulation parameters
    Tsim::Int64 = 10000
    drp::Int64 = 1000
    Tmom::Int64 = 100
    Tpaths::Int64 = 17
    Tpanel::Int64 = 50
    Npanel::Int64 = 100
    TsinceDefault::Int64 = 0
    TsinceExhaustion::Int64 = 50
    NSamplesMoments::Int64 = 300
    NSamplesPaths::Int64 = 1000
    HPFilter_Par::Float64 = 100.0
end

@with_kw struct Grids
    #Grids of states
    GR_z::Array{Float64,1}
    GR_p::Array{Float64,1}
    GR_n::Array{Float64,1}
    GR_k_fdi::Array{Float64,1}
    GR_k::Array{Float64,1}
    GR_b::Array{Float64,1}

    #Quadrature vectors for integrals
    ϵz_weights::Vector{Float64}
    ϵz_nodes::Vector{Float64}

    ϵpL_weights::Vector{Float64}
    ϵpL_nodes::Vector{Float64}

    ϵpH_weights::Vector{Float64}
    ϵpH_nodes::Vector{Float64}

    #Factor to normalize quadrature approximation
    FacQz::Float64
    FacQpL::Float64
    FacQpH::Float64

    #Matrix for discoveries
    PI_n::Array{Float64,2}

    #Matrix for oil volatility
    PI_σp::Array{Float64,2}

    #Matrices for integrals
    ZPRIME::Array{Float64,2}
    PPRIME_L::Array{Float64,2}
    PPRIME_H::Array{Float64,2}
    PDFz::Array{Float64,2}
    PDFpL::Array{Float64,2}
    PDFpH::Array{Float64,2}
end

function CreateOilFieldGrids(par::Pars)
    @unpack Nn, Twait, nL, nH, FieldLife, PrDisc, ExactWait, IncludeDisc, ImmediateField = par
    GR_n=nL*ones(Float64,Nn)
    if IncludeDisc==1
        GR_n[end]=nH
        if ExactWait==0
            #Fill in transition matrix
            PI_n=zeros(Float64,Nn,Nn)
            if ImmediateField==1
                #Probability of discovery and no discovery
                PI_n[1,1]=1.0-PrDisc
                PI_n[1,2]=PrDisc

                PrExhaustion=1.0/FieldLife
                PI_n[end,end]=1.0-PrExhaustion
                PI_n[end,1]=PrExhaustion
                return GR_n, PI_n
            else
                #Probability of discovery and no discovery
                PI_n[1,1]=1.0-PrDisc
                PI_n[1,2]=PrDisc
                PI_n[2,2]=1.0-(1.0/Twait)
                PI_n[2,3]=1.0/Twait
                PrExhaustion=1.0/FieldLife
                PI_n[end,end]=1.0-PrExhaustion
                PI_n[end,1]=PrExhaustion
                return GR_n, PI_n
            end
        else
            #Fill in transition matrix
            PI_n=zeros(Float64,Nn,Nn)
            #Probability of discovery and no discovery
            PI_n[1,1]=1.0-PrDisc
            PI_n[1,2]=PrDisc
            PrExhaustion=1.0/FieldLife
            PI_n[end,end]=1.0-PrExhaustion
            PI_n[end,1]=PrExhaustion
            for s in 2:Nn-1
                PI_n[s,s+1]=1.0
            end
            return GR_n, PI_n
        end
    else
        GR_n=nL*ones(Float64,Nn)
        PI_n=0.5*ones(Float64,Nn,Nn)
        return GR_n, PI_n
    end
end

function CreateQuadratureObjects(dist_ϵ,GR_x::Array{Float64,1},N_GL::Int64,σ_ϵ::Float64,ρ::Float64,μ::Float64)
    GL_nodes, GL_weights = gausslegendre(N_GL)
    ϵlow=-5.0*σ_ϵ
    ϵhigh=5.0*σ_ϵ
    ϵ_nodes=0.5*(ϵhigh-ϵlow).*GL_nodes .+ 0.5*(ϵhigh+ϵlow)
    ϵ_weights=GL_weights .* 0.5*(ϵhigh-ϵlow)
    #Matrices for integration over z
    XPRIME=Array{Float64,2}(undef,length(GR_x),N_GL)
    PDFx=Array{Float64,2}(undef,length(GR_x),N_GL)
    for x_ind in 1:length(GR_x)
        x=GR_x[x_ind]
        XPRIME[x_ind,:]=exp.((1.0-ρ)*log(μ)+ρ*log(x) .+ ϵ_nodes)
        PDFx[x_ind,:]=pdf.(dist_ϵ,ϵ_nodes)
    end
    FacQx=dot(ϵ_weights,pdf.(dist_ϵ,ϵ_nodes))

    return ϵ_weights, ϵ_nodes, XPRIME, PDFx, FacQx
end

function CreateGrids(par::Pars)
    #Grid for z
    @unpack Nz, zlow, zhigh = par
    GR_z=collect(range(zlow,stop=zhigh,length=Nz))

    #Gauss-Legendre vectors for z
    @unpack N_GLz, σ_ϵz, ρ_z, μ_z, dist_ϵz = par
    ϵz_weights, ϵz_nodes, ZPRIME, PDFz, FacQz=CreateQuadratureObjects(dist_ϵz,GR_z,N_GLz,σ_ϵz,ρ_z,μ_z)

    #Grid for p
    @unpack Np, plow, phigh = par
    GR_p=collect(range(plow,stop=phigh,length=Np))

    #Gauss-Legendre vectors for p with low variance
    @unpack N_GLp, σL_ϵp, ρ_p, μ_p, dist_ϵpL = par
    ϵpL_weights, ϵpL_nodes, PPRIME_L, PDFpL, FacQpL=CreateQuadratureObjects(dist_ϵpL,GR_p,N_GLp,σL_ϵp,ρ_p,μ_p)

    #Gauss-Legendre vectors for p with high variance
    @unpack σH_ϵp, dist_ϵpH = par
    ϵpH_weights, ϵpH_nodes, PPRIME_H, PDFpH, FacQpH=CreateQuadratureObjects(dist_ϵpH,GR_p,N_GLp,σH_ϵp,ρ_p,μ_p)

    #Grid for n
    GR_n, PI_n=CreateOilFieldGrids(par)

    #Oil volatility transition
    @unpack πp_LH, πp_HL = par
    PI_σp=Array{Float64,2}(undef,2,2)
    PI_σp[1,1]=1.0-πp_LH
    PI_σp[1,2]=πp_LH
    PI_σp[2,1]=πp_HL
    PI_σp[2,2]=1.0-πp_HL

    #Grid of oil capital for fdi solution
    @unpack Nk_fdi, klow, khigh = par
    GR_k_fdi=collect(range(klow,stop=khigh,length=Nk_fdi))

    #Grid of oil capital
    @unpack Nk, klow, khigh = par
    GR_k=collect(range(klow,stop=khigh,length=Nk))

    #Grid of debt
    @unpack Nb, blow, bhigh = par
    GR_b=collect(range(blow,stop=bhigh,length=Nb))

    return Grids(GR_z,GR_p,GR_n,GR_k_fdi,GR_k,GR_b,ϵz_weights,ϵz_nodes,ϵpL_weights,ϵpL_nodes,ϵpH_weights,ϵpH_nodes,FacQz,FacQpL,FacQpH,PI_n,PI_σp,ZPRIME,PPRIME_L,PPRIME_H,PDFz,PDFpL,PDFpH)
end

@with_kw mutable struct Solution{T1,T2,T3,T4,T5,T6,T7,T8,T9}
    ### Arrays
    #Value Functions
    VD::T1
    VP::T2
    V::T2
    Vfdi::T3
    #Expectations and price
    EVD::T1
    EV::T2
    q1::T2
    q1af::T2
    EVfdi::T3
    EyT::T2
    σyT::T2
    #Policy functions
    kprime::T3
    bprime::T2
    logYT::T2
    logYT2::T2
    ### Interpolation objects
    #Value Functions
    itp_VD::T4
    itp_VP::T5
    itp_V::T5
    itp_Vfdi::T6
    #Expectations and price
    itp_EVD::T4
    itp_EV::T5
    itp_EVfdi::T6
    itp_q1::T7
    itp_q1af::T7
    #Policy functions
    itp_kprime::T8
    itp_bprime::T9
    itp_logYT::T9
    itp_logYT2::T9
end

@with_kw mutable struct AuxExpectations{T1,T2,T3}
    ###Arrays to use for numerical integration
    #Store here expectations over z' and p'
    #while keeping σ' and n' constant
    #Expectations government
    sED::T1
    sEP::T2
    #Expectations oil investors
    sEO::T3
end

@with_kw mutable struct Model
    SOLUTION::Solution
    GRIDS::Grids
    par::Pars
end

@with_kw mutable struct SharedAux{T1,T2,T3}
    ### Arrays
    #Value Functions
    VD::T1
    VP::T2
    V::T2
    Vfdi::T3
    #Expectations and price
    EVD::T1
    EV::T2
    q1::T2
    q1af::T2
    EVfdi::T3
    EyT::T2
    σyT::T2
    #Policy functions
    kprime::T3
    bprime::T2
    logYT::T2
    logYT2::T2
end

################################################################
#################### Auxiliary functions #######################
################################################################
function MyBisection(foo,a,b;xatol::Float64=1e-8)
    s=sign(foo(a))
    x=(a+b)/2.0
    d=(b-a)/2.0
    while d>xatol
        d=d/2.0
        if s==sign(foo(x))
            x=x+d
        else
            x=x-d
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
############# Functions to interpolate matrices ################
################################################################
function CreateInterpolation_ValueFunctions(MAT::Array{Float64},IsDefault::Bool,IsFDI::Bool,GRIDS::Grids)
    @unpack GR_p, GR_n = GRIDS
    SIGMAs=range(1,stop=2,length=2)
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    ORDER_SHOCKS=Linear()
    ORDER_STATES=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))

    if IsFDI
        @unpack GR_k_fdi = GRIDS
        Ks=range(GR_k_fdi[1],stop=GR_k_fdi[end],length=length(GR_k_fdi))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp(),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Ks,Ps,Ns,SIGMAs),Interpolations.Line())
    else
        @unpack GR_z, GR_k = GRIDS
        Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
        Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
        if IsDefault==true
            INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp(),NoInterp())
            return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Ks,Zs,Ps,Ns,SIGMAs),Interpolations.Line())
        else
            @unpack GR_b = GRIDS
            Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
            INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp(),NoInterp())
            return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Ks,Zs,Ps,Ns,SIGMAs),Interpolations.Line())
        end
    end
end

function CreateInterpolation_Price(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_z, GR_p, GR_n, GR_k, GR_b = GRIDS
    SIGMAs=range(1,stop=2,length=2)
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp(),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Ks,Zs,Ps,Ns,SIGMAs),Interpolations.Flat())
end

function CreateInterpolation_PolicyFunctions(MAT::Array{Float64},IsFDI::Bool,GRIDS::Grids)
    @unpack GR_p, GR_n = GRIDS
    SIGMAs=range(1,stop=2,length=2)
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    ORDER_SHOCKS=Linear()
    ORDER_STATES=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))

    if IsFDI
        @unpack GR_k_fdi = GRIDS
        Ks=range(GR_k_fdi[1],stop=GR_k_fdi[end],length=length(GR_k_fdi))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp(),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Ks,Ps,Ns,SIGMAs),Interpolations.Flat())
    else
        @unpack GR_z, GR_k, GR_b = GRIDS
        Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
        Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp(),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Ks,Zs,Ps,Ns,SIGMAs),Interpolations.Flat())
    end
end

################################################################
########## Functions to compute expectations ###################
################################################################
function CreateInterpolation_ForExpectations(MAT::Array{Float64,2},GRIDS::Grids)
    @unpack GR_z, GR_p = GRIDS
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    INT_DIMENSIONS=(BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS))
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Zs,Ps),Interpolations.Line())
end

function Expectation_over_pprime_zprime(foo,p_ind::Int64,z_ind::Int64,σprime_ind::Int64,GRIDS::Grids)
    #foo is a function of floats for z' and p', depends on σprime_ind and nprime_ind
    #k', b', nprime_ind and σprime_ind are given
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    int=0.0
    if σprime_ind==1
        #Prices from low variance regime
        @unpack ϵpL_weights, PPRIME_L, PDFpL, FacQpL = GRIDS
        for i in 1:length(ϵz_weights)
            for j in 1:length(ϵpL_weights)
                int=int+ϵpL_weights[j]*ϵz_weights[i]*PDFpL[p_ind,j]*PDFz[z_ind,i]*foo(ZPRIME[z_ind,i],PPRIME_L[p_ind,j])
            end
        end
        return int/(FacQpL*FacQz)
    else
        #Prices from high variance regime
        @unpack ϵpH_weights, PPRIME_H, PDFpH, FacQpH = GRIDS
        for i in 1:length(ϵz_weights)
            for j in 1:length(ϵpH_weights)
                int=int+ϵpH_weights[j]*ϵz_weights[i]*PDFpH[p_ind,j]*PDFz[z_ind,i]*foo(ZPRIME[z_ind,i],PPRIME_H[p_ind,j])
            end
        end
        return int/(FacQpH*FacQz)
    end
end

function ComputeExpectationOverStates!(IsDefault::Bool,MAT::Array{Float64},E_MAT::SharedArray{Float64},AUX_EX::AuxExpectations,GRIDS::Grids)
    @unpack PI_σp, PI_n = GRIDS
    #It will use MAT to compute expectations
    #It will mutate E_MAT
    if IsDefault
        #Loop over all future states to compute expectations over p and z
        @sync @distributed for I in CartesianIndices(MAT)
            (kprime_ind,z_ind,p_ind,nprime_ind,σprime_ind)=Tuple(I)
            foo_mat=CreateInterpolation_ForExpectations(MAT[kprime_ind,:,:,nprime_ind,σprime_ind],GRIDS)
            AUX_EX.sED[I]=Expectation_over_pprime_zprime(foo_mat,p_ind,z_ind,σprime_ind,GRIDS)
        end
        #Loop over all current states to compute expectations over σp and n
        @sync @distributed for I in CartesianIndices(MAT)
            (kprime_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
            int=0.0
            for i in 1:length(GRIDS.GR_n)
                for j in 1:2
                    int=int+PI_n[n_ind,i]*PI_σp[σ_ind,j]*AUX_EX.sED[kprime_ind,z_ind,p_ind,i,j]
                end
            end
            E_MAT[I]=int
        end
    else
        #Loop over all future states to compute expectations over p and z
        @sync @distributed for I in CartesianIndices(MAT)
            (bprime_ind,kprime_ind,z_ind,p_ind,nprime_ind,σprime_ind)=Tuple(I)
            foo_mat=CreateInterpolation_ForExpectations(MAT[bprime_ind,kprime_ind,:,:,nprime_ind,σprime_ind],GRIDS)
            AUX_EX.sEP[I]=Expectation_over_pprime_zprime(foo_mat,p_ind,z_ind,σprime_ind,GRIDS)
        end
        #Loop over all states to compute expectations over σp and n
        @sync @distributed for I in CartesianIndices(MAT)
            (bprime_ind,kprime_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
            int=0.0
            for i in 1:length(GRIDS.GR_n)
                for j in 1:2
                    int=int+PI_n[n_ind,i]*PI_σp[σ_ind,j]*AUX_EX.sEP[bprime_ind,kprime_ind,z_ind,p_ind,i,j]
                end
            end
            E_MAT[I]=int
        end
    end
    return nothing
end

function ComputeExpectationOverStates_NoParallel!(IsDefault::Bool,MAT::Array{Float64},E_MAT::Array{Float64},AUX_EX::AuxExpectations,GRIDS::Grids)
    @unpack PI_σp, PI_n = GRIDS
    #It will use MAT to compute expectations
    #It will mutate E_MAT
    if IsDefault
        #Loop over all future states to compute expectations over p and z
        for I in CartesianIndices(MAT)
            (kprime_ind,z_ind,p_ind,nprime_ind,σprime_ind)=Tuple(I)
            foo_mat=CreateInterpolation_ForExpectations(MAT[kprime_ind,:,:,nprime_ind,σprime_ind],GRIDS)
            AUX_EX.sED[I]=Expectation_over_pprime_zprime(foo_mat,p_ind,z_ind,σprime_ind,GRIDS)
        end
        #Loop over all current states to compute expectations over σp and n
        for I in CartesianIndices(MAT)
            (kprime_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
            int=0.0
            for i in 1:length(GRIDS.GR_n)
                for j in 1:2
                    int=int+PI_n[n_ind,i]*PI_σp[σ_ind,j]*AUX_EX.sED[kprime_ind,z_ind,p_ind,i,j]
                end
            end
            E_MAT[I]=int
        end
    else
        #Loop over all future states to compute expectations over p and z
        for I in CartesianIndices(MAT)
            (bprime_ind,kprime_ind,z_ind,p_ind,nprime_ind,σprime_ind)=Tuple(I)
            foo_mat=CreateInterpolation_ForExpectations(MAT[bprime_ind,kprime_ind,:,:,nprime_ind,σprime_ind],GRIDS)
            AUX_EX.sEP[I]=Expectation_over_pprime_zprime(foo_mat,p_ind,z_ind,σprime_ind,GRIDS)
        end
        #Loop over all states to compute expectations over σp and n
        for I in CartesianIndices(MAT)
            (bprime_ind,kprime_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
            int=0.0
            for i in 1:length(GRIDS.GR_n)
                for j in 1:2
                    int=int+PI_n[n_ind,i]*PI_σp[σ_ind,j]*AUX_EX.sEP[bprime_ind,kprime_ind,z_ind,p_ind,i,j]
                end
            end
            E_MAT[I]=int
        end
    end
    return nothing
end

################################################################
######## Functions to compute expectations, FDI ################
################################################################
function CreateInterpolation_ForExpectation_FDI(MAT::Array{Float64,1},GRIDS::Grids)
    @unpack GR_p = GRIDS
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    INT_DIMENSIONS=BSpline(ORDER_SHOCKS)
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Ps),Interpolations.Line())
end

function Expectation_over_pprime(foo,p_ind::Int64,σprime_ind::Int64,GRIDS::Grids)
    #foo is a function of floats for z' and p', depends on σprime_ind and nprime_ind
    #k', b', nprime_ind and σprime_ind are given
    int=0.0
    if σprime_ind==1
        #Prices from low variance regime
        @unpack ϵpL_weights, PPRIME_L, PDFpL, FacQpL = GRIDS
        for j in 1:length(ϵpL_weights)
            int=int+ϵpL_weights[j]*PDFpL[p_ind,j]*foo(PPRIME_L[p_ind,j])
        end
        return int/FacQpL
    else
        #Prices from high variance regime
        @unpack ϵpH_weights, PPRIME_H, PDFpH, FacQpH = GRIDS
        for j in 1:length(ϵpH_weights)
            int=int+ϵpH_weights[j]*PDFpH[p_ind,j]*foo(PPRIME_H[p_ind,j])
        end
        return int/FacQpH
    end
end

function ComputeExpectationOverStates_FDI!(MAT::Array{Float64,4},E_MAT::SharedArray{Float64,4},AUX_EX::AuxExpectations,GRIDS::Grids)
    @unpack PI_σp, PI_n = GRIDS
    #It will use MAT to compute expectations
    #It will mutate E_MAT
    #Loop over all future states to compute expectations over p and z
    @sync @distributed for I in CartesianIndices(MAT)
        (kprime_ind,p_ind,nprime_ind,σprime_ind)=Tuple(I)
        foo_mat=CreateInterpolation_ForExpectation_FDI(MAT[kprime_ind,:,nprime_ind,σprime_ind],GRIDS)
        AUX_EX.sEO[I]=Expectation_over_pprime(foo_mat,p_ind,σprime_ind,GRIDS)
    end
    #Loop over all states to compute expectations over σp and n
    @sync @distributed for I in CartesianIndices(MAT)
        (kprime_ind,p_ind,n_ind,σ_ind)=Tuple(I)
        int=0.0
        for i in 1:length(GRIDS.GR_n)
            for j in 1:2
                int=int+PI_n[n_ind,i]*PI_σp[σ_ind,j]*AUX_EX.sEO[kprime_ind,p_ind,i,j]
            end
        end
        E_MAT[I]=int
    end
    return nothing
end

function ComputeExpectationOverStates_FDI_NoParallel!(MAT::Array{Float64,4},E_MAT::Array{Float64,4},AUX_EX::AuxExpectations,GRIDS::Grids)
    @unpack PI_σp, PI_n = GRIDS
    #It will use MAT to compute expectations
    #It will mutate E_MAT
    #Loop over all future states to compute expectations over p and z
    for I in CartesianIndices(MAT)
        (kprime_ind,p_ind,nprime_ind,σprime_ind)=Tuple(I)
        foo_mat=CreateInterpolation_ForExpectation_FDI(MAT[kprime_ind,:,nprime_ind,σprime_ind],GRIDS)
        AUX_EX.sEO[I]=Expectation_over_pprime(foo_mat,p_ind,σprime_ind,GRIDS)
    end
    #Loop over all states to compute expectations over σp and n
    for I in CartesianIndices(MAT)
        (kprime_ind,p_ind,n_ind,σ_ind)=Tuple(I)
        int=0.0
        for i in 1:length(GRIDS.GR_n)
            for j in 1:2
                int=int+PI_n[n_ind,i]*PI_σp[σ_ind,j]*AUX_EX.sEO[kprime_ind,p_ind,i,j]
            end
        end
        E_MAT[I]=int
    end
    return nothing
end

###############################################################################
#Function to Initiate solution and auxiliary objects
###############################################################################
function InitiateAuxExpectations(par::Pars)
    @unpack Nn, Np, Nz, Nk_fdi, Nk, Nb = par
    #Expectations over z and p
    sED=SharedArray{Float64,5}(Nk,Nz,Np,Nn,2)
    sEP=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    sEO=SharedArray{Float64,4}(Nk_fdi,Np,Nn,2)
    return AuxExpectations(sED,sEP,sEO)
end

function InitiateEmptySolution(GRIDS::Grids,par::Pars)
    @unpack Nn, Np, Nz, Nk_fdi, Nk, Nb, N_GLz, N_GLp = par
    ### Allocate all values to object
    VD=zeros(Float64,Nk,Nz,Np,Nn,2)
    VP=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    V=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    Vfdi=zeros(Float64,Nk_fdi,Np,Nn,2)
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,false,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,false,GRIDS)
    itp_Vfdi=CreateInterpolation_ValueFunctions(Vfdi,false,true,GRIDS)
    #Expectations and price
    EVD=zeros(Float64,Nk,Nz,Np,Nn,2)
    EV=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    q1=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    q1af=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    EVfdi=zeros(Float64,Nk_fdi,Np,Nn,2)
    EyT=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    σyT=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,false,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,false,GRIDS)
    itp_EVfdi=CreateInterpolation_ValueFunctions(EVfdi,false,true,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)
    itp_q1af=CreateInterpolation_Price(q1af,GRIDS)
    #Policy functions
    kprime=zeros(Float64,Nk_fdi,Np,Nn,2)
    bprime=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    logYT=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    logYT2=zeros(Float64,Nb,Nk,Nz,Np,Nn,2)
    itp_kprime=CreateInterpolation_PolicyFunctions(kprime,true,GRIDS)
    itp_bprime=CreateInterpolation_PolicyFunctions(bprime,false,GRIDS)
    itp_logYT=CreateInterpolation_PolicyFunctions(logYT,false,GRIDS)
    itp_logYT2=CreateInterpolation_PolicyFunctions(logYT2,false,GRIDS)
    return Solution(VD,VP,V,Vfdi,EVD,EV,q1,q1af,EVfdi,EyT,σyT,kprime,bprime,logYT,logYT2,itp_VD,itp_VP,itp_V,itp_Vfdi,itp_EVD,itp_EV,itp_EVfdi,itp_q1,itp_q1af,itp_kprime,itp_bprime,itp_logYT,itp_logYT2)
end

function InitiateSharedArrays(par::Pars)
    @unpack Nn, Np, Nz, Nk_fdi, Nk, Nb, N_GLz, N_GLp = par
    ### Arrays
    #Value Functions
    VD=SharedArray{Float64,5}(Nk,Nz,Np,Nn,2)
    VP=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    V=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    Vfdi=SharedArray{Float64,4}(Nk_fdi,Np,Nn,2)
    #Expectations and price
    EVD=SharedArray{Float64,5}(Nk,Nz,Np,Nn,2)
    EV=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    q1=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    q1af=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    EVfdi=SharedArray{Float64,4}(Nk_fdi,Np,Nn,2)
    EyT=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    σyT=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    #Policy functions
    kprime=SharedArray{Float64,4}(Nk_fdi,Np,Nn,2)
    bprime=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    logYT=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)
    logYT2=SharedArray{Float64,6}(Nb,Nk,Nz,Np,Nn,2)

    return SharedAux(VD,VP,V,Vfdi,EVD,EV,q1,q1af,EVfdi,EyT,σyT,kprime,bprime,logYT,logYT2)
end

################################################################
############### Preferences and technology #####################
################################################################
function Utility(c::Float64,par::Pars)
    @unpack σ = par
    return (c^(1.0-σ))/(1.0-σ)
end

function UtilityFDI(c::Float64,par::Pars)
    return log(c)
end

function CapitalAdjustment(kprime::Float64,k::Float64,par::Pars)
    @unpack φ = par
    return 0.5*φ*((kprime-k)^2)/k
end

function DefaultCost(y::Float64,par::Pars)
    #Chatterjee and Eyiungongor (2012)
    # @unpack d0, d1 = par
    # return max(0.0,d0*y+d1*y*y)
    #Arellano (2008)
    @unpack knk = par
    return max(0.0,y-knk)
end

function NonTradedProduction(z::Float64,kN::Float64,par::Pars)
    @unpack αN = par
    return z*(kN^αN)
end

function ManufProduction(z::Float64,kM::Float64,par::Pars)
    @unpack αM = par
    return z*(kM^αM)
end

function OilProduction(kO::Float64,n::Float64,par::Pars)
    @unpack ζ = par
    return (kO^(1-ζ))*(n^ζ)
end

function OilRent(pO::Float64,kO::Float64,n::Float64,par::Pars)
    @unpack ζ = par
    yO=OilProduction(kO,n,par)
    return ζ*pO*yO
end

function SDF_Lenders(σprime_ind::Int64,nprime_ind::Int64,zprime::Float64,pprime::Float64,
                     kprime::Float64,bprime::Float64,EyT::Float64,σyT::Float64,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    @unpack r_star, α0 = par
    @unpack itp_logYT = SOLUTION
    nuprime=itp_logYT(bprime,kprime,zprime,pprime,nprime_ind,σprime_ind)-EyT
    return exp(-(r_star+α0*nuprime+0.5*((α0^2.0)*(σyT^2.0))))
end

function OilPriceOptions(p::Float64,par::Pars)
    if par.OilFix
        return 1.0
    else
        if par.OilPut
            return max(p,1.0)
        else
            return p
        end
    end
end

################################################################
################ Production functions ##########################
################################################################
function Final_CES(cN::Float64,cM::Float64,cO::Float64,par::Pars)
    @unpack η, ωN, ωM, ωO = par
    return (ωN*(cN^((η-1.0)/η))+ωM*(cM^((η-1.0)/η))+ωO*(cO^((η-1.0)/η)))^(η/(η-1.0))
end

function PriceNonTraded(λ::Float64,par::Pars)
    @unpack αN, αM, kbar = par
    #Compute share of capital in manufacturing
    return (αM/αN)*(((1.0-λ)^(1.0-αN))/(λ^(1.0-αM)))*(kbar^(αM-αN))
end

function PriceFinalGood(λ::Float64,pO::Float64,par::Pars)
    @unpack ωN, ωM, ωO, η = par
    pN=PriceNonTraded(λ,par)
    return ((ωN^η)*(pN^(1.0-η))+(ωM^η)*(1.0^(1.0-η))+(ωO^η)*(pO^(1.0-η)))^(1.0/(1.0-η))
end

function CapitalAllocation(z::Float64,pO::Float64,n::Float64,kO::Float64,Tr::Float64,par::Pars)
    @unpack η, ωO, ωM, ωN, αM, αN, kbar, τR = par
    #X is net imports: q*(b'-(1-γ)b)-(γ+κ*(1-γ))*b
    LHS(λ::Float64)=(((αM/αN)*(((1.0-λ)^(1.0-αN))/(λ^(1.0-αM)))*(kbar^(αM-αN)))^η)*NonTradedProduction(z,(1.0-λ)*kbar,par)
    RHS(λ::Float64)=((ωN^η)/((ωM^η)+(ωO^η)*(pO^(1.0-η))))*(ManufProduction(z,λ*kbar,par)+τR*OilRent(pO,kO,n,par)+Tr)
    foo(λ::Float64)=LHS(λ)-RHS(λ)
    if foo(1e-4)<0.0
        return 1e-4
    else
        if foo(1.0-1e-4)>=0.0
            return 1.0-1e-4
        else
            return MyBisection(foo,1e-4,1.0-1e-4,xatol=1e-8)
        end
    end
end

function TradableIncome(z::Float64,pO::Float64,n::Float64,kO::Float64,Tr::Float64,par::Pars)
    λ=CapitalAllocation(z,pO,n,kO,Tr,par)
    return ManufProduction(z,λ*par.kbar,par)+par.τR*OilRent(pO,kO,n,par)
end

function Production(Default::Bool,z::Float64,p::Float64,n::Float64,kO::Float64,Tr::Float64,par::Pars)
    @unpack ωM, ωO, η, kbar, τR = par
    #X is net imports: q*(b'-(1-γ)b)-(γ+κ*(1-γ))*b
    pO=OilPriceOptions(p,par)
    #First compute capital allocation
    λ=CapitalAllocation(z,pO,n,kO,Tr,par)

    #Compute consumption of intermediate goods
    cM=(((ωM/1.0)^η)/((ωM^η)+(ωO^η)*(pO^(1.0-η))))*(ManufProduction(z,λ*kbar,par)+τR*OilRent(pO,kO,n,par)+Tr)

    if cM>0.0
        cO=(((ωO/ωM)*(1.0/pO))^η)*cM
        cN=NonTradedProduction(z,(1.0-λ)*kbar,par)
        y=Final_CES(cN,cM,cO,par)
        if Default
            return y-DefaultCost(y,par)
        else
            return y
        end
    else
        return cM
    end
end

################################################################
################### Setup functions ############################
################################################################
function GDP_nominal(Default::Bool,z::Float64,pO::Float64,n::Float64,kO::Float64,Tr::Float64,par::Pars)
    λ=CapitalAllocation(z,pO,n,kO,Tr,par)
    P=PriceFinalGood(λ,pO,par)
    y=Production(Default,z,pO,n,kO,Tr,par)
    yO=OilProduction(kO,n,par)
    return P*y+pO*yO-par.τR*OilRent(pO,kO,n,par)-Tr
end

function OilExports_GDP(z::Float64,pO::Float64,n::Float64,kO::Float64,Tr::Float64,par::Pars)
    @unpack αN, αM, ωM, ωO, η, ζ, kbar, τR = par
    #Compute share of capital in manufacturing
    λ=CapitalAllocation(z,pO,n,kO,Tr,par)
    #Compute consumption of intermediate goods
    yO=OilProduction(kO,n,par)
    cO=(((ωO^η)*(pO^(1-η)))/((ωM^η)+(ωO^η)*(pO^(1.0-η))))*(ManufProduction(z,λ*kbar,par)+τR*OilRent(pO,kO,n,par)+Tr)/pO
    return pO*(yO-cO)/GDP_nominal(false,z,pO,n,kO,Tr,par)
end

function kOss_FDI(pOss::Float64,nss::Float64,par::Pars)
    @unpack r_star, δ, ζ, PkO = par
    βstar=exp(-r_star)
    return (((βstar*(pOss)*(1-ζ))/(1-βstar*(1-δ)))^(1/ζ))*nss
end

function Calibrate_nL(par::Pars)
    @unpack Target_oilXp, μ_z, μ_p = par
    Tr=0.0
    function foo(nnL::Float64)
        kOss=kOss_FDI(μ_p,nnL,par)
        return OilExports_GDP(μ_z,μ_p,nnL,kOss,Tr,par)-Target_oilXp
    end
    #Get bracketing interval
    nLlow=0.10
    while foo(nLlow)>=0.0
        nLlow=0.5*nLlow
    end
    nLhigh=1.50
    while foo(nLhigh)<=0.0
        nLhigh=2.0*nLhigh
    end
    return MyBisection(foo,nLlow,nLhigh,xatol=1e-5)
end

function NPV_disc(nH::Float64,z::Float64,pO::Float64,par::Pars)
    @unpack Twait, FieldLife, r_star, ζ, Target_spreads = par
    rss=r_star+Target_spreads
    a0=1.0/(1.0+rss)
    FACTOR=((a0^(Twait+1.0))-(a0^(Twait+FieldLife+1.0)))/(1.0-a0)
    kOssL=kOss_FDI(pO,par.nL,par)
    kOssH=kOss_FDI(pO,nH,par)
    return pO*(OilProduction(kOssH,nH,par)-OilProduction(kOssL,par.nL,par))*FACTOR
end

function Calibrate_nH(par::Pars)
    @unpack Target_NPV, μ_z, μ_p = par
    Tr=0.0
    kOss=kOss_FDI(μ_p,par.nL,par)
    gdp0=GDP_nominal(false,μ_z,μ_p,par.nL,kOss,Tr,par)
    function foo(nnH::Float64)
        npv=NPV_disc(nnH,μ_z,μ_p,par)
        return npv/gdp0-Target_NPV
    end
    #Get bracketing interval
    nHlow=0.10
    while foo(nHlow)>=0.0
        nHlow=0.5*nHlow
    end
    nHhigh=1.50
    while foo(nHhigh)<=0.0
        nHhigh=2.0*nHhigh
    end
    return MyBisection(foo,nHlow,nHhigh,xatol=1e-5)
end

function InitiateParameters(nHfactor::Float64)
    par=Pars()
    nL=Calibrate_nL(par)
    par=Pars(par,nL=nL)
    if nHfactor>0.0
        nH=nHfactor*nL
    else
        nH=Calibrate_nH(par)
    end
    par=Pars(par,nH=nH)

    kssL=kOss_FDI(par.μ_p,par.nL,par)
    kssH=kOss_FDI(par.μ_p,par.nH,par)
    klow=0.1*kssL
    khigh=10.0*kssH
    par=Pars(par,klow=klow,khigh=khigh)
    par=Pars(par,klowOpt=0.99*par.klow,khighOpt=1.01*par.khigh)

    #Very deep pockets for foreigners
    Wforeign=1000.0*khigh
    par=Pars(par,Wforeign=Wforeign)

    #Check ExactWait
    if par.ExactWait==1
        Nn=1+1+par.Twait+1
        par=Pars(par,Nn=Nn)
    else
        if par.ImmediateField==1
            par=Pars(par,Nn=2)
        else
            par=Pars(par,Nn=3)
        end
    end

    return par
end

################################################################################
### Functions to save Model in CSV
################################################################################
#Save model objects
function StackSolution_Vector(SOLUTION::Solution)
    #Stack vectors of repayment size first
    @unpack VP, V, EV, q1, q1af, EyT, σyT, bprime, logYT, logYT2 = SOLUTION
    VEC=reshape(VP,(:))
    VEC=vcat(VEC,reshape(V,(:)))
    VEC=vcat(VEC,reshape(EV,(:)))
    VEC=vcat(VEC,reshape(q1,(:)))
    VEC=vcat(VEC,reshape(q1af,(:)))
    VEC=vcat(VEC,reshape(EyT,(:)))
    VEC=vcat(VEC,reshape(σyT,(:)))
    VEC=vcat(VEC,reshape(bprime,(:)))
    VEC=vcat(VEC,reshape(logYT,(:)))
    VEC=vcat(VEC,reshape(logYT2,(:)))

    #Satck vectors of FDI next
    @unpack Vfdi, EVfdi, kprime = SOLUTION
    VEC=vcat(VEC,reshape(Vfdi,(:)))
    VEC=vcat(VEC,reshape(EVfdi,(:)))
    VEC=vcat(VEC,reshape(kprime,(:)))

    #Then stack vectors of default
    @unpack VD, EVD = SOLUTION
    VEC=vcat(VEC,reshape(VD,(:)))
    VEC=vcat(VEC,reshape(EVD,(:)))

    return VEC
end

function VectorOfRelevantParameters(par::Pars)
    #Stack important values from parameters
    #Parameters for Grids go first
    #Grids sizes
    VEC=par.N_GLz              #1
    VEC=vcat(VEC,par.N_GLp)    #2
    VEC=vcat(VEC,par.Nz)       #3
    VEC=vcat(VEC,par.Np)       #4
    VEC=vcat(VEC,par.ExactWait)       #5
    VEC=vcat(VEC,par.ImmediateField)       #6
    VEC=vcat(VEC,par.Nk_fdi)   #7
    VEC=vcat(VEC,par.Nk)       #8
    VEC=vcat(VEC,par.Nb)       #9

    #Grids bounds
    VEC=vcat(VEC,par.blow)     #10
    VEC=vcat(VEC,par.bhigh)    #11

    #Parameter values
    VEC=vcat(VEC,par.β)        #12
    VEC=vcat(VEC,par.knk)      #13
    VEC=vcat(VEC,par.d1)       #14
    VEC=vcat(VEC,par.α0)       #15
    VEC=vcat(VEC,par.φ)        #16

    #Extra parameters
    VEC=vcat(VEC,par.cnt_max)  #17

    if par.OilFix
        VEC=vcat(VEC,1.0)  #18
    else
        VEC=vcat(VEC,0.0)  #18
    end

    if par.OilPut
        VEC=vcat(VEC,1.0)  #19
    else
        VEC=vcat(VEC,0.0)  #19
    end

    return VEC
end

function SaveModel_Vector(NAME::String,MODEL::Model)
    @unpack SOLUTION, par = MODEL

    VEC_PAR=VectorOfRelevantParameters(par)
    N_parameters=length(VEC_PAR)
    VEC=vcat(N_parameters,VEC_PAR)

    #Stack SOLUTION in one vector
    VEC_SOL=StackSolution_Vector(SOLUTION)
    VEC=vcat(VEC,VEC_SOL)

    writedlm(NAME,VEC,',')
    return nothing
end

#Extract model objects
function ExtractMatrixFromSolutionVector(start::Int64,size::Int64,IsDefault::Bool,IsCapital::Bool,IsSDF::Bool,VEC::Array{Float64,1},par::Pars)
    @unpack Nn, Np, Nz, Nk_fdi, Nk, Nb, N_GLz, N_GLp = par
    if IsDefault
        I=(Nk,Nz,Np,Nn,2)
    else
        if IsCapital
            I=(Nk_fdi,Np,Nn,2)
        else
            if IsSDF
                I=(Nk,Nz,Np,N_GLz,N_GLp,Nn,2)
            else
                I=(Nb,Nk,Nz,Np,Nn,2)
            end
        end
    end
    finish=start+size-1
    vec=VEC[start:finish]
    return reshape(vec,I)
end

function TransformVectorToSolution(VEC::Array{Float64},GRIDS::Grids,par::Pars)
    #The file SolutionVector.csv must be in FOLDER
    #for this function to work
    @unpack Nn, Np, Nz, Nk_fdi, Nk, Nb = par
    size_repayment=2*Nn*Np*Nz*Nk*Nb
    size_default=2*Nn*Np*Nz*Nk
    size_capital=2*Nn*Np*Nk_fdi

    #Allocate vectors into matrices
    #Repayment
    start=1

    VP=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    V=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    EV=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    q1=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    q1af=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    EyT=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    σyT=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    bprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    logYT=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    logYT2=ExtractMatrixFromSolutionVector(start,size_repayment,false,false,false,VEC,par)
    start=start+size_repayment

    #FDI
    Vfdi=ExtractMatrixFromSolutionVector(start,size_capital,false,true,false,VEC,par)
    start=start+size_capital

    EVfdi=ExtractMatrixFromSolutionVector(start,size_capital,false,true,false,VEC,par)
    start=start+size_capital

    kprime=ExtractMatrixFromSolutionVector(start,size_capital,false,true,false,VEC,par)
    start=start+size_capital

    #Default
    VD=ExtractMatrixFromSolutionVector(start,size_default,true,false,false,VEC,par)
    start=start+size_default

    EVD=ExtractMatrixFromSolutionVector(start,size_default,true,false,false,VEC,par)

    #Create interpolation objects
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,false,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,false,GRIDS)
    itp_Vfdi=CreateInterpolation_ValueFunctions(Vfdi,false,true,GRIDS)
    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,false,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,false,GRIDS)
    itp_EVfdi=CreateInterpolation_ValueFunctions(EVfdi,false,true,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)
    itp_q1af=CreateInterpolation_Price(q1af,GRIDS)
    itp_kprime=CreateInterpolation_PolicyFunctions(kprime,true,GRIDS)
    itp_bprime=CreateInterpolation_PolicyFunctions(bprime,false,GRIDS)
    itp_logYT=CreateInterpolation_PolicyFunctions(logYT,false,GRIDS)
    itp_logYT2=CreateInterpolation_PolicyFunctions(logYT2,false,GRIDS)
    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,Vfdi,EVD,EV,q1,q1af,EVfdi,EyT,σyT,kprime,bprime,logYT,logYT2,itp_VD,itp_VP,itp_V,itp_Vfdi,itp_EVD,itp_EV,itp_EVfdi,itp_q1,itp_q1af,itp_kprime,itp_bprime,itp_logYT,itp_logYT2)
end

function UnpackParameters_Vector(nHfactor::Float64,VEC::Array{Float64})
    par=InitiateParameters(nHfactor)

    #Parameters for Grids go first
    #Grids sizes
    par=Pars(par,N_GLz=convert(Int64,VEC[1]))
    par=Pars(par,N_GLp=convert(Int64,VEC[2]))
    par=Pars(par,Nz=convert(Int64,VEC[3]))
    par=Pars(par,Np=convert(Int64,VEC[4]))
    par=Pars(par,ExactWait=convert(Int64,VEC[5]))
    par=Pars(par,ImmediateField=convert(Int64,VEC[6]))
    #Check ExactWait
    if par.ExactWait==1
        Nn=1+1+par.Twait+1
        par=Pars(par,Nn=Nn)
    else
        if par.ImmediateField==1
            par=Pars(par,Nn=2)
        else
            par=Pars(par,Nn=3)
        end
    end
    par=Pars(par,Nk_fdi=convert(Int64,VEC[7]))
    par=Pars(par,Nk=convert(Int64,VEC[8]))
    par=Pars(par,Nb=convert(Int64,VEC[9]))

    #Grids bounds

    par=Pars(par,blow=VEC[10])
    par=Pars(par,bhigh=VEC[11])
    par=Pars(par,blowOpt=par.blow-0.01,bhighOpt=par.bhigh+0.1)

    #Parameter values
    par=Pars(par,β=VEC[12])
    par=Pars(par,knk=VEC[13])
    par=Pars(par,d1=VEC[14])
    par=Pars(par,α0=VEC[15])
    par=Pars(par,φ=VEC[16])

    #Extra parameters
    par=Pars(par,cnt_max=VEC[17])

    if VEC[18]==1.0
        par=Pars(par,OilFix=true)
    else
        par=Pars(par,OilFix=false)
    end

    if VEC[19]==1.0
        par=Pars(par,OilPut=true)
    else
        par=Pars(par,OilPut=false)
    end

    return par
end

function Setup_From_Vector(nHfactor::Float64,VEC_PAR::Array{Float64})
    #Vector has the correct structure
    par=UnpackParameters_Vector(nHfactor,VEC_PAR)
    GRIDS=CreateGrids(par)
    return par, GRIDS
end

function UnpackModel_Vector(nHfactor::Float64,NAME::String,FOLDER::String)
    #Unpack Vector with data
    if FOLDER==" "
        VEC=readdlm(NAME,',')
    else
        VEC=readdlm("$FOLDER\\$NAME",',')
    end

    #Extract parameters and create grids
    N_parameters=convert(Int64,VEC[1])
    VEC_PAR=VEC[2:N_parameters+1]
    par, GRIDS=Setup_From_Vector(nHfactor,VEC_PAR)

    #Extract solution object
    VEC_SOL=VEC[N_parameters+2:end]
    SOL=TransformVectorToSolution(VEC_SOL,GRIDS,par)

    return Model(SOL,GRIDS,par)
end

function Setup_MomentMatching(β::Float64,knk::Float64,α0::Float64,VEC_PAR::Array{Float64})
    par=UnpackParameters_Vector(0.0,VEC_PAR)
    par=Pars(par,β=β,knk=knk,α0=α0)

    GRIDS=CreateGrids(par)
    return par, GRIDS
end

###############################################################################
#Function to solve FDI problem
###############################################################################
function Consumption_FDI(I::CartesianIndex,kprime::Float64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack δ, τR, Wforeign, PkO = par
    @unpack GR_p, GR_k_fdi, GR_n = GRIDS
    (k_ind,p_ind,n_ind,σ_ind)=Tuple(I)
    #Compute consumption and value
    p=GR_p[p_ind]; k=GR_k_fdi[k_ind]; n=GR_n[n_ind]

    yO=OilProduction(k,n,par)
    iO=kprime-(1-δ)*k
    pO=OilPriceOptions(p,par)
    cons=Wforeign+pO*yO-(1-τR)*OilRent(pO,k,n,par)-iO-CapitalAdjustment(kprime,k,par)
    return cons
end

function ValueFDI(I::CartesianIndex,kprime::Float64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack r_star = par
    @unpack GR_p, GR_k_fdi, GR_n = GRIDS
    @unpack itp_EVfdi = SOLUTION
    (k_ind,p_ind,n_ind,σ_ind)=Tuple(I)
    #Compute consumption and value
    p=GR_p[p_ind]; k=GR_k_fdi[k_ind]; n=GR_n[n_ind]

    βfdi=exp(-r_star)
    cons=Consumption_FDI(I,kprime,MODEL)
    return UtilityFDI(max(cons,1e-3),par)+βfdi*itp_EVfdi(kprime,p,n_ind,σ_ind)
end

function klow_ind_ForSearch(I::CartesianIndex,MODEL::Model)
    #Use this to exploit monotonicity
    (k_ind,p_ind,n_ind,σ_ind)=Tuple(I)
    if k_ind==1
        return k_ind
    else
        @unpack kprime = MODEL.SOLUTION
        kk=kprime[k_ind-1,p_ind,n_ind,σ_ind]
        if kk==0.0
            return 1
        else
            @unpack GR_k_fdi = MODEL.GRIDS
            #find closest grid point
            xx, klow_ind=findmin(abs.(kk .- GR_k_fdi))
            return klow_ind
        end
    end
end

function BoundsSearch_FDI(I::CartesianIndex,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack GR_k_fdi = GRIDS
    val=-Inf
    kpol_ind=0
    klow_ind=klow_ind_ForSearch(I,MODEL)
    for ktry in klow_ind:length(GR_k_fdi)
        kprime=GR_k_fdi[ktry]
        cons=Consumption_FDI(I,kprime,MODEL)
        if cons>0.0
            vv=ValueFDI(I,kprime,MODEL)
            if vv>val
                val=vv
                kpol_ind=ktry
            end
        end
    end

    if kpol_ind<=1
        @unpack klowOpt = par
        return klowOpt, GR_k_fdi[2]
    else
        if kpol_ind==length(GR_k_fdi)
            @unpack khighOpt = par
            return GR_k_fdi[end-1], khighOpt
        else
            return GR_k_fdi[kpol_ind-1], GR_k_fdi[kpol_ind+1]
        end
    end
end

function OptimFDI(I::CartesianIndex,MODEL::Model)

    #Setup function handle for optimization
    f(kprime::Float64)=-ValueFDI(I,kprime,MODEL)
    #Perform optimization with MatLab simplex
    klowOpt, khighOpt=BoundsSearch_FDI(I,MODEL)
    inner_optimizer = GoldenSection()
    res=optimize(f,klowOpt,khighOpt,inner_optimizer)

    return -Optim.minimum(res), Optim.minimizer(res)
end

function Update_FDI!(AUX_EX::AuxExpectations,SHARED_AUX::SharedAux,MODEL::Model)
    @unpack GRIDS = MODEL

    @sync @distributed for I in CartesianIndices(SHARED_AUX.kprime)
        SHARED_AUX.Vfdi[I], SHARED_AUX.kprime[I]=OptimFDI(I,MODEL)
    end
    MODEL.SOLUTION.Vfdi .= SHARED_AUX.Vfdi
    MODEL.SOLUTION.kprime .= SHARED_AUX.kprime
    MODEL.SOLUTION.itp_Vfdi=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.Vfdi,false,true,GRIDS)
    MODEL.SOLUTION.itp_kprime=CreateInterpolation_PolicyFunctions(MODEL.SOLUTION.kprime,true,GRIDS)


    ComputeExpectationOverStates_FDI!(MODEL.SOLUTION.Vfdi,SHARED_AUX.EVfdi,AUX_EX,GRIDS)
    MODEL.SOLUTION.EVfdi .= SHARED_AUX.EVfdi
    MODEL.SOLUTION.itp_EVfdi=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EVfdi,false,true,GRIDS)
    return nothing
end

function Update_FDI_NoParallel!(AUX_EX::AuxExpectations,SHARED_AUX::SharedAux,MODEL::Model)
    @unpack GRIDS = MODEL

    for I in CartesianIndices(SHARED_AUX.kprime)
        MODEL.SOLUTION.Vfdi[I], MODEL.SOLUTION.kprime[I]=OptimFDI(I,MODEL)
    end
    MODEL.SOLUTION.itp_Vfdi=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.Vfdi,false,true,GRIDS)
    MODEL.SOLUTION.itp_kprime=CreateInterpolation_PolicyFunctions(MODEL.SOLUTION.kprime,true,GRIDS)


    ComputeExpectationOverStates_FDI_NoParallel!(MODEL.SOLUTION.Vfdi,MODEL.SOLUTION.EVfdi,AUX_EX,GRIDS)
    MODEL.SOLUTION.itp_EVfdi=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EVfdi,false,true,GRIDS)
    return nothing
end

function ComputeDistance_FDI(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution)
    dst_fdi, Ifdi=findmax(abs.(SOLUTION_CURRENT.Vfdi .- SOLUTION_NEXT.Vfdi))
    dst_k=maximum(abs.(SOLUTION_CURRENT.kprime .- SOLUTION_NEXT.kprime))
    return round(dst_fdi,digits=6), round(dst_k,digits=6), Ifdi
end

function SolveFDIpart!(Parallel::Bool,PrintProgress::Bool,AUX_EX::AuxExpectations,SHARED_AUX::SharedAux,MODEL::Model)
    @unpack par = MODEL
    @unpack Tol_V, cnt_maxFDI = par
    SOLUTION_CURRENT=deepcopy(MODEL.SOLUTION)
    dst=1.0
    cnt=0
    while dst>Tol_V && cnt<=cnt_maxFDI
        if Parallel
            Update_FDI!(AUX_EX,SHARED_AUX,MODEL)
        else
            Update_FDI_NoParallel!(AUX_EX,SHARED_AUX,MODEL)
        end
        dst_fdi, dst_k, Ifdi=ComputeDistance_FDI(SOLUTION_CURRENT,MODEL.SOLUTION)
        dst=dst_fdi
        cnt=cnt+1
        if PrintProgress
            println("FDI, cnt=$cnt, dst_fdi=$dst_fdi at $Ifdi, dst_k=$dst_k")
        end
        SOLUTION_CURRENT=deepcopy(MODEL.SOLUTION)
    end
    return nothing
end

@with_kw mutable struct Paths_FDI
    #Paths of shocks
    p::Array{Float64,1}
    nf::Array{Float64,1}
    n::Array{Int64,1}
    σp::Array{Int64,1}

    #Paths of chosen states
    KO::Array{Float64,1}

    #Path of relevant variables
    Inv::Array{Float64,1}
    yO::Array{Float64,1}
end

function InitiateEmptyPathsFDI(T::Int64)
    #Initiate with zeros to facilitate averages
    #Paths of shocks
    f1=zeros(Float64,T)
    f2=zeros(Float64,T)
    i1=zeros(Int64,T)
    i2=zeros(Int64,T)
    f3=zeros(Float64,T)
    f4=zeros(Float64,T)
    f5=zeros(Float64,T)
    return Paths_FDI(f1,f2,i1,i2,f3,f4,f5)
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

function Simulate_DiscreteShocks(x0::Int64,T::Int64,PI_x::Array{Float64,2})
    X=Array{Int64,1}(undef,T)
    X[1]=x0
    for t in 2:T
        X[t]=Draw_New_discreteShock(X[t-1],PI_x)
    end
    return X
end

function Simulate_p_shocks(T::Int64,σp_TS::Array{Int64,1},GRIDS::Grids,par::Pars)
    @unpack μ_p, ρ_p, dist_ϵpL, dist_ϵpH = par
    ϵpL_TS=rand(dist_ϵpL,T)
    ϵpH_TS=rand(dist_ϵpL,T)
    p_TS=Array{Float64,1}(undef,T)
    for t in 1:T
        if t==1
            p_TS[t]=μ_p
        else
            if σp_TS[t]==1
                #Low variance regime
                p_TS[t]=exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p_TS[t-1])+ϵpL_TS[t])
            else
                #High variance regime
                p_TS[t]=exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p_TS[t-1])+ϵpH_TS[t])
            end
        end
    end
    return p_TS
end

function GenerateNextState_FDI!(t::Int64,PATHS::Paths_FDI,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack itp_kprime = SOLUTION
    @unpack δ = par
    @unpack GR_n = GRIDS

    #Unpack current shocks
    p=PATHS.p[t]; σp=PATHS.σp[t]; n=PATHS.n[t]
    PATHS.nf[t]=GR_n[PATHS.n[t]]

    #Unpack current endogenous state
    kO=PATHS.KO[t]

    #Update next endogenous state
    #Check if it is not final period
    if t<length(PATHS.p)
        PATHS.KO[t+1]=itp_kprime(kO,p,n,σp)
        PATHS.Inv[t]=PATHS.KO[t+1]-(1-δ)*PATHS.KO[t]
    else
        PATHS.Inv[t]=PATHS.Inv[t-1]
    end
    PATHS.yO[t]=OilProduction(PATHS.KO[t],GR_n[n],par)
    return nothing
end

function SimulatePathsOfFDI(T::Int64,MODEL::Model;Only_nL::Bool=false)
    @unpack SOLUTION, GRIDS, par = MODEL
    PATHS=InitiateEmptyPathsFDI(T)
    # Random.seed!(1324)
    #Simulate all shocks
    PATHS.σp .= Simulate_DiscreteShocks(1,T,GRIDS.PI_σp)
    if Only_nL
        PATHS.n .= ones(Int64,length(PATHS.σp))
    else
        PATHS.n .= Simulate_DiscreteShocks(1,T,GRIDS.PI_n)
    end
    p=Simulate_p_shocks(T,PATHS.σp,GRIDS,par)
    PATHS.p .= p

    #Choose initial conditions
    PATHS.KO[1]=kOss_FDI(par.μ_p,par.nL,par)
    for t in 1:T
        GenerateNextState_FDI!(t,PATHS,MODEL)
    end
    return PATHS
end

function Extract_TS_FDI(t1::Int64,tT::Int64,PATHS::Paths_FDI)
    return Paths_FDI(PATHS.p[t1:tT],PATHS.nf[t1:tT],PATHS.n[t1:tT],PATHS.σp[t1:tT],PATHS.KO[t1:tT],PATHS.Inv[t1:tT],PATHS.yO[t1:tT])
end

function SimulatePathsOfDiscovery_FDI(Tbefore::Int64,Tafter::Int64,MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack drp = par
    T=drp+Tbefore+1+Tafter
    PATHS=InitiateEmptyPathsFDI(T)

    #Simulate discovery
    PATHS.n .= ones(Int64,T)
    PATHS.n[drp+Tbefore+1]=2
    # PATHS.n[drp+Tbefore+1:end] .= Simulate_DiscreteShocks(2,Tafter+1,GRIDS.PI_n)
    PATHS.n[drp+Tbefore+1:end] .= 2
    # PATHS.n[drp+Tbefore+1+par.Twait+1:end] .= par.Nn


    #Simulate other shocks
    PATHS.σp .= Simulate_DiscreteShocks(1,T,GRIDS.PI_σp)
    p=Simulate_p_shocks(T,PATHS.σp,GRIDS,par)
    PATHS.p .= p

    #Choose initial conditions
    PP=SimulatePathsOfFDI(1100,MODEL)
    PATHS.KO[1]=mean(PP.KO[101:end])
    for t in 1:T
        GenerateNextState_FDI!(t,PATHS,MODEL)
    end
    return Extract_TS_FDI(drp+1,T,PATHS)
end

function Sum_Paths_FDI!(PATHsum::Paths_FDI,PATH1::Paths_FDI)
    PATHsum.p=PATHsum.p .+ PATH1.p
    PATHsum.nf=PATHsum.nf .+ PATH1.nf
    PATHsum.KO=PATHsum.KO .+ PATH1.KO
    PATHsum.Inv=PATHsum.Inv .+ PATH1.Inv
    PATHsum.yO=PATHsum.yO .+ PATH1.yO
end

function AverageDiscoveryPaths_FDI(N::Int64,Tbefore::Int64,Tafter::Int64,MODEL::Model)
    Random.seed!(1324)
    PATHS=SimulatePathsOfDiscovery_FDI(Tbefore,Tafter,MODEL)
    for i in 2:N
        PATH1=SimulatePathsOfDiscovery_FDI(Tbefore,Tafter,MODEL)
        Sum_Paths_FDI!(PATHS,PATH1)
    end
    PATHS.p=PATHS.p ./ N
    PATHS.nf=PATHS.nf ./ N
    PATHS.KO=PATHS.KO ./ N
    PATHS.Inv=PATHS.Inv ./ N
    PATHS.yO=PATHS.yO ./ N
    return PATHS
end

###############################################################################
#Function to compute consumption
###############################################################################
function ConsNet(y::Float64,par::Pars)
    return y
end

function Calculate_Tr(qq::Float64,b::Float64,bprime::Float64,par::Pars)
    @unpack γ, κ = par
    #Compute net borrowing from the rest of the world
    return qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b
end

###############################################################################
#Functions to compute value given state, policies, and guesses
###############################################################################
function ValueInDefault(I::CartesianIndex,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack β, θ, cmin = par
    @unpack GR_z, GR_p, GR_k, GR_n = GRIDS
    @unpack itp_EV, itp_EVD, itp_kprime = SOLUTION
    (k_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
    #Compute consumption and value
    z=GR_z[z_ind]; p=GR_p[p_ind]; k=GR_k[k_ind]
    n=GR_n[n_ind]
    kkOO=itp_kprime(k,p,n_ind,σ_ind)

    Tr=0.0
    y=Production(true,z,p,n,k,Tr,par)
    cons=ConsNet(y,par)

    #There is no way for consumption to be negative
    return Utility(cons,par)+β*θ*itp_EV(0.0,kkOO,z,p,n_ind,σ_ind)+β*(1.0-θ)*itp_EVD(kkOO,z,p,n_ind,σ_ind)
end

function ValueInRepayment(bprime::Float64,I::CartesianIndex,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack β, cmin = par
    @unpack GR_z, GR_p, GR_k, GR_n, GR_b = GRIDS
    @unpack itp_EV, itp_q1, itp_kprime = SOLUTION
    (b_ind,k_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
    #Compute output
    z=GR_z[z_ind]; p=GR_p[p_ind]; k=GR_k[k_ind]
    n=GR_n[n_ind]; b=GR_b[b_ind]
    kkOO=itp_kprime(k,p,n_ind,σ_ind)

    qq=itp_q1(bprime,kkOO,z,p,n_ind,σ_ind)
    Tr=Calculate_Tr(qq,b,bprime,par)

    y=Production(false,z,p,n,k,Tr,par)
    #Compute consumption
    cons=ConsNet(y,par)
    if cons>0.0
        return Utility(cons,par)+β*itp_EV(bprime,kkOO,z,p,n_ind,σ_ind)
    else
        #Return very low utility penalized by how negative consumption is
        return Utility(cmin,par)+cons
    end
end

###############################################################################
#Functions to optimize given guesses and state
###############################################################################
function BoundsSearch_B(I::CartesianIndex,MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack GR_b = GRIDS

    bpol=0
    val=-Inf
    for btry in 1:length(GR_b)
        bprime=GR_b[btry]
        vv=ValueInRepayment(bprime,I,MODEL)
        if vv>val
            val=vv
            bpol=btry
        end
    end

    if bpol<=1
        @unpack blowOpt = par
        return blowOpt, GR_b[2]
    else
        if bpol==length(GR_b)
            @unpack bhighOpt = par
            return GR_b[end-1], bhighOpt
        else
            return GR_b[bpol-1], GR_b[bpol+1]
        end
    end
end

function OptimInRepayment(I::CartesianIndex,MODEL::Model)
    @unpack SOLUTION = MODEL
    @unpack itp_q1, kprime = SOLUTION
    (b_ind,k_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
    #Get guess and bounds for bprime
    blowOpt, bhighOpt=BoundsSearch_B(I,MODEL)

    #Setup function handle for optimization
    f(bprime::Float64)=-ValueInRepayment(bprime,I,MODEL)
    #Perform optimization with MatLab simplex
    inner_optimizer = GoldenSection()
    res=optimize(f,blowOpt,bhighOpt,inner_optimizer)

    return -Optim.minimum(res), Optim.minimizer(res)
end

###############################################################################
#Update default
###############################################################################
function UpdateDefault!(AUX_EX::AuxExpectations,SHARED_AUX::SharedAux,MODEL::Model)
    @unpack GRIDS = MODEL
    #Loop over all states to fill array of VD
    @sync @distributed for I in CartesianIndices(SHARED_AUX.VD)
        SHARED_AUX.VD[I]=ValueInDefault(I,MODEL)
    end
    MODEL.SOLUTION.VD .= SHARED_AUX.VD
    MODEL.SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.VD,true,false,GRIDS)

    ComputeExpectationOverStates!(true,MODEL.SOLUTION.VD,SHARED_AUX.EVD,AUX_EX,GRIDS)
    MODEL.SOLUTION.EVD .= SHARED_AUX.EVD
    MODEL.SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EVD,true,false,GRIDS)
    return nothing
end

function UpdateDefault_NoParallel!(AUX_EX::AuxExpectations,SHARED_AUX::SharedAux,MODEL::Model)
    @unpack GRIDS = MODEL
    #Loop over all states to fill array of VD
    for I in CartesianIndices(SHARED_AUX.VD)
        MODEL.SOLUTION.VD[I]=ValueInDefault(I,MODEL)
    end
    MODEL.SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.VD,true,false,GRIDS)

    ComputeExpectationOverStates_NoParallel!(true,MODEL.SOLUTION.VD,MODEL.SOLUTION.EVD,AUX_EX,GRIDS)
    MODEL.SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EVD,true,false,GRIDS)
    return nothing
end

###############################################################################
#Update repayment
###############################################################################
function TradableIncome_state(bprime::Float64,I::CartesianIndex,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack GR_z, GR_p, GR_k, GR_n, GR_b = GRIDS
    @unpack itp_q1, itp_kprime = SOLUTION
    (b_ind,k_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
    #Compute output
    z=GR_z[z_ind]; p=GR_p[p_ind]; k=GR_k[k_ind]
    n=GR_n[n_ind]; b=GR_b[b_ind]
    kkOO=itp_kprime(k,p,n_ind,σ_ind)

    qq=itp_q1(bprime,kkOO,z,p,n_ind,σ_ind)
    Tr=Calculate_Tr(qq,b,bprime,par)

    return TradableIncome(z,p,n,k,Tr,par)
end

function RepaymentUpdater!(I::CartesianIndex,SHARED_AUX::SharedAux,MODEL::Model)
    (b_ind,k_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
    SHARED_AUX.VP[I], SHARED_AUX.bprime[I]=OptimInRepayment(I,MODEL)

    if SHARED_AUX.VP[I]<MODEL.SOLUTION.VD[k_ind,z_ind,p_ind,n_ind,σ_ind]
        SHARED_AUX.V[I]=MODEL.SOLUTION.VD[k_ind,z_ind,p_ind,n_ind,σ_ind]
    else
        SHARED_AUX.V[I]=MODEL.SOLUTION.VP[I]
    end

    yT=TradableIncome_state(SHARED_AUX.bprime[I],I,MODEL)
    SHARED_AUX.logYT[I]=log(yT)
    SHARED_AUX.logYT2[I]=log(yT)^2
    return nothing
end

function UpdateRepayment!(AUX_EX::AuxExpectations,SHARED_AUX::SharedAux,MODEL::Model)
    @unpack GRIDS = MODEL
    #Loop over all states to fill value of repayment
    @sync @distributed for I in CartesianIndices(SHARED_AUX.VP)
        RepaymentUpdater!(I,SHARED_AUX,MODEL)
    end
    MODEL.SOLUTION.V .= SHARED_AUX.V
    MODEL.SOLUTION.VP .= SHARED_AUX.VP
    MODEL.SOLUTION.bprime .= SHARED_AUX.bprime
    MODEL.SOLUTION.logYT .= SHARED_AUX.logYT
    MODEL.SOLUTION.logYT2 .= SHARED_AUX.logYT2
    MODEL.SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.VP,false,false,GRIDS)
    MODEL.SOLUTION.itp_V=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.V,false,false,GRIDS)
    MODEL.SOLUTION.itp_bprime=CreateInterpolation_PolicyFunctions(MODEL.SOLUTION.bprime,false,GRIDS)
    MODEL.SOLUTION.itp_logYT=CreateInterpolation_PolicyFunctions(MODEL.SOLUTION.logYT,false,GRIDS)
    MODEL.SOLUTION.itp_logYT2=CreateInterpolation_PolicyFunctions(MODEL.SOLUTION.logYT2,false,GRIDS)

    ComputeExpectationOverStates!(false,MODEL.SOLUTION.V,SHARED_AUX.EV,AUX_EX,GRIDS)
    MODEL.SOLUTION.EV .= SHARED_AUX.EV

    ComputeExpectationOverStates!(false,MODEL.SOLUTION.logYT,SHARED_AUX.EyT,AUX_EX,GRIDS)
    MODEL.SOLUTION.EyT .= SHARED_AUX.EyT

    ComputeExpectationOverStates!(false,MODEL.SOLUTION.logYT2,SHARED_AUX.σyT,AUX_EX,GRIDS)
    MODEL.SOLUTION.σyT .= sqrt.(SHARED_AUX.σyT .- (MODEL.SOLUTION.EyT .^ 2))

    MODEL.SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EV,false,false,GRIDS)
    return nothing
end

function UpdateRepayment_NoParallel!(AUX_EX::AuxExpectations,SHARED_AUX::SharedAux,MODEL::Model)
    @unpack GRIDS = MODEL
    #Loop over all states to fill value of repayment
    for I in CartesianIndices(SHARED_AUX.VP)
        RepaymentUpdater!(I,SHARED_AUX,MODEL)
    end
    MODEL.SOLUTION.V .= SHARED_AUX.V
    MODEL.SOLUTION.VP .= SHARED_AUX.VP
    MODEL.SOLUTION.bprime .= SHARED_AUX.bprime
    MODEL.SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.VP,false,false,GRIDS)
    MODEL.SOLUTION.itp_V=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.V,false,false,GRIDS)
    MODEL.SOLUTION.itp_bprime=CreateInterpolation_PolicyFunctions(MODEL.SOLUTION.bprime,false,GRIDS)

    ComputeExpectationOverStates_NoParallel!(false,MODEL.SOLUTION.V,MODEL.SOLUTION.EV,AUX_EX,GRIDS)

    MODEL.SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EV,false,false,GRIDS)
    return nothing
end

###############################################################################
#Update both prices
###############################################################################
function BothBondsPayoff(σprime_ind::Int64,nprime_ind::Int64,zprime::Float64,pprime::Float64,
                         kprime::Float64,bprime::Float64,EyT::Float64,σyT::Float64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack γ, κ = par
    @unpack itp_VP, itp_VD, itp_q1, itp_bprime, itp_kprime, itp_q1af = SOLUTION
    @unpack GR_z, GR_p, GR_k, GR_b = GRIDS
    if itp_VD(kprime,zprime,pprime,nprime_ind,σprime_ind)>itp_VP(bprime,kprime,zprime,pprime,nprime_ind,σprime_ind)
        return 0.0, 0.0
    else
        SDF=SDF_Lenders(σprime_ind,nprime_ind,zprime,pprime,kprime,bprime,EyT,σyT,MODEL)
        SDFaf=exp(-par.r_star)
        if γ==1.0
            return SDF, SDFaf
        else
            kOkO=itp_kprime(kprime,pprime,nprime_ind,σprime_ind)
            bb=itp_bprime(bprime,kprime,zprime,pprime,nprime_ind,σprime_ind)
            qq=itp_q1(bb,kOkO,zprime,pprime,nprime_ind,σprime_ind)
            qqaf=itp_q1af(bb,kOkO,zprime,pprime,nprime_ind,σprime_ind)
            return SDF*(γ+(1.0-γ)*(κ+qq)), SDFaf*(γ+(1.0-γ)*(κ+qqaf))
        end
    end
end

function Expectation_over_pprime_zprime_BothBonds(p_ind::Int64,z_ind::Int64,nprime_ind::Int64,σprime_ind::Int64,
                                                  kprime::Float64,bprime::Float64,EyT::Float64,σyT::Float64,MODEL::Model)
    #k', b', nprime_ind and σprime_ind are given
    @unpack GRIDS = MODEL
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    int=0.0
    intaf=0.0
    zprime=0.0; pprime=0.0
    if σprime_ind==1
        #Prices from low variance regime
        @unpack ϵpL_weights, PPRIME_L, PDFpL, FacQpL = GRIDS
        for i in 1:length(ϵz_weights)
            zprime=ZPRIME[z_ind,i]
            for j in 1:length(ϵpL_weights)
                pprime=PPRIME_L[p_ind,j]
                ii, iiaf=BothBondsPayoff(σprime_ind,nprime_ind,zprime,pprime,kprime,bprime,EyT,σyT,MODEL)
                int=int+ϵpL_weights[j]*ϵz_weights[i]*PDFpL[p_ind,j]*PDFz[z_ind,i]*ii
                intaf=intaf+ϵpL_weights[j]*ϵz_weights[i]*PDFpL[p_ind,j]*PDFz[z_ind,i]*iiaf
            end
        end
        return int/(FacQpL*FacQz), intaf/(FacQpL*FacQz)
    else
        #Prices from high variance regime
        @unpack ϵpH_weights, PPRIME_H, PDFpH, FacQpH = GRIDS
        for i in 1:length(ϵz_weights)
            zprime=ZPRIME[z_ind,i]
            for j in 1:length(ϵpH_weights)
                pprime=PPRIME_H[p_ind,j]
                ii, iiaf=BothBondsPayoff(σprime_ind,nprime_ind,zprime,pprime,kprime,bprime,EyT,σyT,MODEL)
                int=int+ϵpH_weights[j]*ϵz_weights[i]*PDFpH[p_ind,j]*PDFz[z_ind,i]*ii
                intaf=intaf+ϵpH_weights[j]*ϵz_weights[i]*PDFpH[p_ind,j]*PDFz[z_ind,i]*iiaf
            end
        end
        return int/(FacQpH*FacQz), intaf/(FacQpH*FacQz)
    end
end

function UpdateBothBondsPrices!(SHARED_AUX::SharedAux,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack GR_z, GR_p, GR_k, GR_b, PI_n, PI_σp = GRIDS
    @unpack Nn = par
    #Loop over all states to compute expectation over z' and p'
    @sync @distributed for I in CartesianIndices(SHARED_AUX.q1)
        (bprime_ind,kprime_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
        #Unpack state
        kprime=GR_k[kprime_ind]; bprime=GR_b[bprime_ind]
        EyT=SOLUTION.EyT[I]
        σyT=SOLUTION.σyT[I]
        int=0.0
        intaf=0.0
        for nprime_ind in 1:Nn
            for σprime_ind in 1:2
                ii, iiaf=Expectation_over_pprime_zprime_BothBonds(p_ind,z_ind,nprime_ind,σprime_ind,kprime,bprime,EyT,σyT,MODEL)
                int=int+PI_σp[σ_ind,σprime_ind]*PI_n[n_ind,nprime_ind]*ii
                intaf=intaf+PI_σp[σ_ind,σprime_ind]*PI_n[n_ind,nprime_ind]*iiaf
            end
        end
        SHARED_AUX.q1[I]=int
        SHARED_AUX.q1af[I]=intaf
    end
    MODEL.SOLUTION.q1 .= SHARED_AUX.q1
    MODEL.SOLUTION.q1af .= SHARED_AUX.q1af

    MODEL.SOLUTION.itp_q1=CreateInterpolation_Price(MODEL.SOLUTION.q1,GRIDS)
    MODEL.SOLUTION.itp_q1af=CreateInterpolation_Price(MODEL.SOLUTION.q1af,GRIDS)
    return nothing
end

function UpdateBothBondsPrices_NoParallel!(SHARED_AUX::SharedAux,MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack GR_z, GR_p, GR_k, GR_b, PI_n, PI_σp = GRIDS
    @unpack Nn = par
    #Loop over all states to compute expectation over z' and p'
    for I in CartesianIndices(SHARED_AUX.q1)
        (bprime_ind,kprime_ind,z_ind,p_ind,n_ind,σ_ind)=Tuple(I)
        #Unpack state
        int=0.0
        intaf=0.0
        for nprime_ind in 1:Nn
            for σprime_ind in 1:2
                ii, iiaf=Expectation_over_pprime_zprime_BothBonds(p_ind,z_ind,nprime_ind,σprime_ind,kprime_ind,bprime_ind,MODEL)
                int=int+PI_σp[σ_ind,σprime_ind]*PI_n[n_ind,nprime_ind]*ii
                intaf=intaf+PI_σp[σ_ind,σprime_ind]*PI_n[n_ind,nprime_ind]*iiaf
            end
        end
        MODEL.SOLUTION.q1[I]=int
        MODEL.SOLUTION.q1af[I]=intaf
    end

    MODEL.SOLUTION.itp_q1=CreateInterpolation_Price(MODEL.SOLUTION.q1,GRIDS)
    MODEL.SOLUTION.itp_q1af=CreateInterpolation_Price(MODEL.SOLUTION.q1af,GRIDS)
    return nothing
end

###############################################################################
#Compute risk premium
###############################################################################
#Default probability next period
function Default_Choice(σ_ind::Int64,n_ind::Int64,z::Float64,p::Float64,
                        k::Float64,b::Float64,MODEL::Model)
    @unpack SOLUTION = MODEL
    @unpack itp_VP, itp_VD = SOLUTION
    if itp_VD(k,z,p,n_ind,σ_ind)>itp_VP(b,k,z,p,n_ind,σ_ind)
        return 1.0
    else
        return 0.0
    end
end

function Integrate_DefChoice(z::Float64,p::Float64,n_ind::Int64,σ_ind::Int64,kprime::Float64,
                             bprime::Float64,MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack Nn = par
    @unpack PI_n, PI_σp = GRIDS
    #Integrate over z', p', n'
    Eq1=0.0
    for nprime_ind in 1:Nn
        πn=PI_n[n_ind,nprime_ind]
        πσL=PI_σp[σ_ind,1]
        πσH=PI_σp[σ_ind,2]
        if πn>0.0
            foo_σL(zz::Float64,pp::Float64)=Default_Choice(1,nprime_ind,zz,pp,kOprime,bprime,MODEL)
            epz_L=Expectation_over_pprime_zprime_float(foo_σL,z,p,1,GRIDS,par)
            foo_σH(zz::Float64,pp::Float64)=Default_Choice(2,nprime_ind,zz,pp,kOprime,bprime,MODEL)
            epz_H=Expectation_over_pprime_zprime_float(foo_σH,z,p,2,GRIDS,par)
            Eq1=Eq1+πn*(πσL*epz_L+πσH*epz_H)
        end
    end
    return Eq1
end

#Compute risk premium
function Compute_Risk_Premium(z::Float64,p::Float64,n_ind::Int64,σ_ind::Int64,
                              kprime::Float64,bprime::Float64,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    @unpack Nn, γ, κ = par
    @unpack itp_q1, itp_q1af = SOLUTION
    #Expectation and variance of yT' conditional on current z, p, n
    #and on choices kO' and k'
    qaf=itp_q1af(bprime,kprime,z,p,n_ind,σ_ind)
    qt=itp_q1(bprime,kprime,z,p,n_ind,σ_ind)
    if qt>0.0 && qaf>0.0
        iaf=-log(qaf/(γ+(1.0-γ)*(κ+qaf)))
        ib=-log(qt/(γ+(1.0-γ)*(κ+qt)))
        return 100*(ib-iaf)
    else
        return 0.0
    end
end

###############################################################################
#VFI algorithm
###############################################################################
function UpdateSolution!(Parallel::Bool,AUX_EX::AuxExpectations,SHARED_AUX::SharedAux,MODEL::Model)
    if Parallel
        UpdateDefault!(AUX_EX,SHARED_AUX,MODEL)
        UpdateRepayment!(AUX_EX,SHARED_AUX,MODEL)
        UpdateBothBondsPrices!(SHARED_AUX,MODEL)
    else
        UpdateDefault_NoParallel!(AUX_EX,SHARED_AUX,MODEL)
        UpdateRepayment_NoParallel!(AUX_EX,SHARED_AUX,MODEL)
        UpdateBothBondsPrices_NoParallel!(SHARED_AUX,MODEL)
    end
    return nothing
end

function ComputeDistance_q(MODEL_CURRENT::Model,MODEL_NEXT::Model)
    @unpack par = MODEL_CURRENT
    @unpack Tol_q = par
    dst_q=maximum(abs.(MODEL_CURRENT.SOLUTION.q1 .- MODEL_NEXT.SOLUTION.q1))
    NotConv=sum(abs.(MODEL_CURRENT.SOLUTION.q1 .- MODEL_NEXT.SOLUTION.q1) .> Tol_q)
    NotConvPct=100.0*NotConv/length(MODEL_CURRENT.SOLUTION.q1)
    return round(dst_q,digits=7), round(NotConvPct,digits=2)
end

function ComputeDistance_V(MODEL_CURRENT::Model,MODEL_NEXT::Model)
    dst_D=maximum(abs.(MODEL_CURRENT.SOLUTION.VD .- MODEL_NEXT.SOLUTION.VD))
    dst_V=maximum(abs.(MODEL_CURRENT.SOLUTION.V .- MODEL_NEXT.SOLUTION.V))
    return round(abs(dst_D),digits=7), round(abs(dst_V),digits=7)
end

function SolveModel_VFI(Parallel::Bool,PrintProgress::Bool,SaveProgress::Bool,NAME::String,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, relTolV, Tol_q_pct, cnt_max = par
    if PrintProgress
        println("allocating auxiliary matrices")
    end
    AUX_EX=InitiateAuxExpectations(par)
    SHARED_AUX=InitiateSharedArrays(par)

    if PrintProgress
        println("Preparing solution guess")
    end
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    MODEL_CURRENT=Model(SOLUTION_CURRENT,GRIDS,par)

    if PrintProgress
        println("Solving FDI")
    end
    SolveFDIpart!(Parallel,false,AUX_EX,SHARED_AUX,MODEL_CURRENT)
    if PrintProgress
        println("Pre-compute SDF")
    end

    MODEL_NEXT=deepcopy(MODEL_CURRENT)
    dst_V=1.0
    rdst_V=100.0
    rdst_D=100.0
    rdst_P=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    if PrintProgress
        println("Starting VFI")
    end
    while ((dst_V>Tol_V) || (dst_q>Tol_q)) && cnt<cnt_max
        UpdateSolution!(Parallel,AUX_EX,SHARED_AUX,MODEL_NEXT)
        dst_q, NotConvPct=ComputeDistance_q(MODEL_CURRENT,MODEL_NEXT)
        dst_D, dst_P=ComputeDistance_V(MODEL_CURRENT,MODEL_NEXT)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        MODEL_CURRENT=deepcopy(MODEL_NEXT)
        if PrintProgress
            println("cnt=$cnt,  dst_D=$dst_D, dst_P=$dst_P, dst_q=$dst_q")
        end
        if SaveProgress
            SaveModel_Vector(NAME,MODEL_NEXT)
        end
    end
    return MODEL_NEXT
end

function SolveAndSaveModel_VFI(Parallel::Bool,PrintProgress::Bool,SaveProgress::Bool,NAME::String,GRIDS::Grids,par::Pars)
    MODEL=SolveModel_VFI(Parallel,PrintProgress,SaveProgress,NAME,GRIDS,par)
    SaveModel_Vector(NAME,MODEL)
    return nothing
end

################################################################################
### Functions for simulations
################################################################################
@with_kw mutable struct Paths
    #Paths of shocks
    z::Array{Float64,1}
    p::Array{Float64,1}
    nf::Array{Float64,1}
    n::Array{Int64,1}
    σp::Array{Int64,1}

    #Paths of chosen states
    Def::Array{Float64,1}
    K::Array{Float64,1}
    B::Array{Float64,1}

    #Path of relevant variables
    Spreads::Array{Float64,1}
    GDP::Array{Float64,1}
    nGDP::Array{Float64,1}
    P::Array{Float64,1}
    GDPdef::Array{Float64,1}
    Ypot::Array{Float64,1}
    yT::Array{Float64,1}
    Cons::Array{Float64,1}
    Inv::Array{Float64,1}
    nInv::Array{Float64,1}
    λ::Array{Float64,1}
    yO::Array{Float64,1}
    TB::Array{Float64,1}
    CA::Array{Float64,1}
    RER::Array{Float64,1}
    RiskPremium::Array{Float64,1}
    RP_Spr::Array{Float64,1}
    DefPr::Array{Float64,1}
    NPV::Array{Float64,1}
end

function InitiateEmptyPaths(T::Int64)
    #Initiate with zeros to facilitate averages
    #Paths of shocks
    f1=zeros(Float64,T)
    f2=zeros(Float64,T)
    f3=zeros(Float64,T)
    i1=zeros(Int64,T)
    i2=zeros(Int64,T)
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
    f18=zeros(Float64,T)
    f19=zeros(Float64,T)
    f20=zeros(Float64,T)
    f21=zeros(Float64,T)
    f22=zeros(Float64,T)
    f23=zeros(Float64,T)
    f24=zeros(Float64,T)
    f25=zeros(Float64,T)
    return Paths(f1,f2,i1,i2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25)
end

function Simulate_z_p_shocks(T::Int64,σp_TS::Array{Int64,1},GRIDS::Grids,par::Pars)
    @unpack μ_z, ρ_z, dist_ϵz = par
    @unpack μ_p, ρ_p, dist_ϵpL, dist_ϵpH = par
    ϵz_TS=rand(dist_ϵz,T)
    ϵpL_TS=rand(dist_ϵpL,T)
    ϵpH_TS=rand(dist_ϵpH,T)
    z_TS=Array{Float64,1}(undef,T)
    p_TS=Array{Float64,1}(undef,T)
    for t in 1:T
        if t==1
            z_TS[t]=μ_z
            p_TS[t]=μ_p
        else
            z_TS[t]=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z_TS[t-1])+ϵz_TS[t])
            if σp_TS[t]==1
                #Low variance regime
                p_TS[t]=exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p_TS[t-1])+ϵpL_TS[t])
            else
                #High variance regime
                p_TS[t]=exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p_TS[t-1])+ϵpH_TS[t])
            end
        end
    end
    return z_TS, p_TS
end

function CalculateSpreads(qq::Float64,par::Pars)
    @unpack γ, κ, r_star = par
    ib=-log(qq/(γ+(1.0-γ)*(κ+qq)))
    ia=((1.0+ib)^1.0)-1.0
    rf=((1.0+r_star)^1.0)-1.0
    return 100.0*(ia-rf)
end

function GenerateNextState!(t::Int64,PATHS::Paths,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack itp_VD, itp_VP, itp_kprime, itp_bprime, itp_q1 = SOLUTION
    @unpack θ = par
    #It must be t>=2, initial state defined outside of this function

    #Unpack current shocks
    z=PATHS.z[t]; p=PATHS.p[t]; σp=PATHS.σp[t]; n=PATHS.n[t]
    PATHS.nf[t]=GRIDS.GR_n[n]

    #Unpack current endogenous state and previous default
    if t==1
        d_=PATHS.Def[t]
    else
        d_=PATHS.Def[t-1]
    end
    k=PATHS.K[t]; b=PATHS.B[t]

    #Update next endogenous state
    if d_==1.0
        #Coming from default state yesterday, must draw for readmission
        if rand()<=θ
            #Get readmitted, choose whether to default or not today
            if itp_VD(k,z,p,n,σp)<=itp_VP(b,k,z,p,n,σp)
                #Choose not to default
                PATHS.Def[t]=0.0
                #Check if it is not final period
                if t<length(PATHS.z)
                    #Fill kO' and b'
                    PATHS.K[t+1]=itp_kprime(k,p,n,σp)
                    PATHS.B[t+1]=max(itp_bprime(b,k,z,p,n,σp),0.0)
                    qq=itp_q1(PATHS.B[t+1],PATHS.K[t+1],z,p,n,σp)
                    PATHS.Spreads[t]=CalculateSpreads(qq,par)
                else
                    #Fix spreads
                    if t==1
                        PATHS.Spreads[t]=0.0
                    else
                        PATHS.Spreads[t]=PATHS.Spreads[t-1]
                    end
                end
            else
                #Choose to default
                PATHS.Def[t]=1.0
                #Check if it is not final period
                if t<length(PATHS.z)
                    #Fill kO' and b'
                    PATHS.K[t+1]=itp_kprime(k,p,n,σp)
                    PATHS.B[t+1]=0.0
                    if t==1
                        PATHS.Spreads[t]=0.0
                    else
                        PATHS.Spreads[t]=PATHS.Spreads[t-1]
                    end
                else
                    #Fix spreads
                    if t==1
                        PATHS.Spreads[t]=0.0
                    else
                        PATHS.Spreads[t]=PATHS.Spreads[t-1]
                    end
                end
            end
        else
            #Stay in default
            PATHS.Def[t]=1.0
            #Check if it is not final period
            if t<length(PATHS.z)
                #Fill kO' and b'
                PATHS.K[t+1]=itp_kprime(k,p,n,σp)
                PATHS.B[t+1]=0.0
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            else
                #Fix spreads
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            end
        end
    else
        #Coming from repayment, choose whether to default or not today
        if itp_VD(k,z,p,n,σp)<=itp_VP(b,k,z,p,n,σp)
            #Choose not to default
            PATHS.Def[t]=0.0
            #Check if it is not final period
            if t<length(PATHS.z)
                #Fill kO' and b'
                PATHS.K[t+1]=itp_kprime(k,p,n,σp)
                PATHS.B[t+1]=max(itp_bprime(b,k,z,p,n,σp),0.0)
                qq=itp_q1(PATHS.B[t+1],PATHS.K[t+1],z,p,n,σp)
                PATHS.Spreads[t]=CalculateSpreads(qq,par)
            else
                #Fix spreads
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            end
        else
            #Choose to default
            PATHS.Def[t]=1.0
            #Check if it is not final period
            if t<length(PATHS.z)
                #Fill kO' and b'
                PATHS.K[t+1]=itp_kprime(k,p,n,σp)
                PATHS.B[t+1]=0.0
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            else
                #Fix spreads
                if t==1
                    PATHS.Spreads[t]=0.0
                else
                    PATHS.Spreads[t]=PATHS.Spreads[t-1]
                end
            end
        end
    end
    return nothing
end

function NPV_disc_TS(d::Float64,σ_ind::Int64,z::Float64,p::Float64,
                     k::Float64,b::Float64,Spread::Float64,SOLUTION::Solution,par::Pars)
    @unpack Twait, FieldLife, ζ, nH, nL, Target_spreads = par
    @unpack itp_kprime = SOLUTION
    rss=par.r_star+Spread
    a0=1.0/(1.0+rss)
    FACTOR=((a0^(Twait+1.0))-(a0^(Twait+FieldLife+1.0)))/(1.0-a0)

    #Project path of oil capital as if shocks remained the same
    n_ind=2
    k0=k
    #Compute oil capital installed for period after discovery
    kD=itp_kprime(k0,p,n_ind,σ_ind)

    npv=0.0
    for t=1:Twait+FieldLife
        #Additional oil output in td+t relative to no discovery
        k0=kD
        yO=OilProduction(k0,nH,par)-OilProduction(k,nL,par)
        npv=npv+(a0^t)*p*yO
        #Compute oil capital installed in td+t for next period
        kD=itp_kprime(k0,p,n_ind,σ_ind)
    end
    return npv
end

function CalculateVariables!(t::Int64,PATHS::Paths,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack δ, γ, κ, ωO, ωM, ωN, η, τR, PkO = par
    #It must be t>=2, initial state defined outside of this function

    #Unpack current shocks
    zstate=PATHS.z[t]; p=PATHS.p[t]; σ_ind=PATHS.σp[t]; n_ind=PATHS.n[t]
    n=GRIDS.GR_n[n_ind]

    #Unpack current endogenous state
    d=PATHS.Def[t]; k=PATHS.K[t]; b=PATHS.B[t]
    #z shock to evaluate
    if d==1
        z=zstate#DefaultCost(zstate,par)
    else
        z=zstate
    end

    #Compute policies
    if t==length(PATHS.K)
        #end of time, use policy functions for next state
        if d==1
            #in default
            @unpack itp_kprime = SOLUTION
            kprime=itp_kprime(k,p,n_ind,σ_ind)
            bprime=0.0
            qq=0.0
            Tr=0.0
        else
            #in repayment
            @unpack itp_kprime, itp_bprime, itp_q1 = SOLUTION
            kprime=itp_kprime(k,p,n_ind,σ_ind)
            bprime=itp_bprime(b,k,zstate,p,n_ind,σ_ind)
            qq=itp_q1(bprime,kprime,zstate,p,n_ind,σ_ind)
            Tr=qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b
        end
    else
        #Not end of time, use next state
        kprime=PATHS.K[t+1]
        bprime=PATHS.B[t+1]
        if d==1
            #in default
            qq=0.0
            Tr=0.0
        else
            #in repayment
            @unpack itp_q1 = SOLUTION
            qq=itp_q1(bprime,kprime,zstate,p,n_ind,σ_ind)
            Tr=qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b
        end
    end

    λ=CapitalAllocation(z,p,n,k,Tr,par)
    PATHS.λ[t]=λ

    #Store base prices
    if t==1
        #Use today's prices as P0 and Pk0
        P0=PriceFinalGood(λ,p,par)
        PATHS.P[t]=P0
        pO0=p
    else
        P0=PATHS.P[1]
        pO0=PATHS.p[1]
    end

    #Calculate current nominal and real variables
    if d==1
        Y=Production(true,z,p,n,k,Tr,par)
    else
        Y=Production(false,z,p,n,k,Tr,par)
    end
    PATHS.Ypot[t]=Y
    P=PriceFinalGood(λ,p,par)
    Adj=CapitalAdjustment(kprime,k,par)
    PATHS.P[t]=P
    yM=ManufProduction(z,λ*par.kbar,par)
    yO=OilProduction(k,n,par)
    PATHS.yO[t]=yO
    PATHS.yT[t]=yM+τR*OilRent(p,k,n,par)
    cO=(((ωO/p)^η)/((ωM^η)+(ωO^η)*(p^(1.0-η))))*(PATHS.yT[t]+Tr)
    cM=(((ωM/ωO)*p)^η)*cO

    PATHS.nInv[t]=PkO*(kprime-(1-δ)*k) #Nominal investment, current prices
    PATHS.Inv[t]=PkO*(kprime-(1-δ)*k) #Real investment, fixed prices

    tb=(yM-cM)+p*(yO-cO)-PATHS.nInv[t]    #Nominal net exports of goods and services
    tbr=(yM-cM)+pO0*(yO-cO)-PATHS.Inv[t]  #Real net exports of goods and services

    PATHS.GDP[t]=P0*Y+PATHS.Inv[t]+tbr #Real GDP, constant prices
    PATHS.nGDP[t]=P*Y+PATHS.nInv[t]+tb  #nominal GDP, for ratios as in data
    PATHS.Cons[t]=Y    #Real consumption, constant prices
    PATHS.TB[t]=tb #Nominal trade balance
    PATHS.CA[t]=-(qq*(bprime-(1-γ)*b))-PATHS.nInv[t] #Change in net foreign assets, current prices
    PATHS.GDPdef[t]=PATHS.nGDP[t]/PATHS.GDP[t] #GDP deflator: nominal GDP/real GDP
    PATHS.RER[t]=1/PATHS.P[t]
    if d==1
        PATHS.DefPr[t]=1-par.θ
        if t>1
            PATHS.RiskPremium[t]=PATHS.RiskPremium[t-1]
        else
            PATHS.RiskPremium[t]=0.0
        end
    else
        PATHS.RiskPremium[t]=Compute_Risk_Premium(zstate,p,n_ind,σ_ind,kprime,bprime,MODEL)
        # PATHS.DefPr[t]=Integrate_DefChoice(zstate,p,n_ind,σ_ind,kprime,bprime,SOLUTION,GRIDS,par)
    end
    if PATHS.Spreads[t]==0.0
        PATHS.RP_Spr[t]=0.0
    else
        PATHS.RP_Spr[t]=100*PATHS.RiskPremium[t]/PATHS.Spreads[t]
    end
    if n_ind==2
        if t==1
            PATHS.NPV[t]=0.0
        else
            if PATHS.n[t-1]==1
                npv=NPV_disc_TS(d,σ_ind,zstate,p,k,b,0.01*PATHS.Spreads[t],SOLUTION,par)
                PATHS.NPV[t]=100*npv/PATHS.nGDP[t]
            else
                PATHS.NPV[t]=0.0
            end
        end
    else
        PATHS.NPV[t]=0.0
    end
    return nothing
end

function SimulatePathsOfStates(T::Int64,MODEL::Model;Only_nL::Bool=false,Only_σL::Bool=false)
    @unpack SOLUTION, GRIDS, par = MODEL
    PATHS=InitiateEmptyPaths(T)
    # Random.seed!(132)
    #Simulate all shocks
    if Only_nL
        PATHS.n .= ones(Int64,T)
    else
        PATHS.n .= Simulate_DiscreteShocks(1,T,GRIDS.PI_n)
    end
    if Only_σL
        PATHS.σp .= ones(Int64,T)
    else
        PATHS.σp .= Simulate_DiscreteShocks(1,T,GRIDS.PI_σp)
    end
    z, p=Simulate_z_p_shocks(T,PATHS.σp,GRIDS,par)
    PATHS.z .= z
    PATHS.p .= p

    #Choose initial conditions
    PATHS.K[1]=0.5#kOss_FDI(par.μ_p,par.nL,par)
    PATHS.B[1]=0.0
    PATHS.Def[1]=0.0
    for t in 1:T
        GenerateNextState!(t,PATHS,SOLUTION,GRIDS,par)
    end
    return PATHS
end

function UpdateOtherVariablesInPaths!(PATHS::Paths,MODEL::Model)
    for t=1:length(PATHS.z)
        CalculateVariables!(t,PATHS,MODEL)
    end
end

################################################################################
### Functions to create average paths of discovery
################################################################################
function Extract_TS(t1::Int64,tT::Int64,PATHS::Paths)
    return Paths(PATHS.z[t1:tT],PATHS.p[t1:tT],PATHS.nf[t1:tT],PATHS.n[t1:tT],PATHS.σp[t1:tT],PATHS.Def[t1:tT],PATHS.K[t1:tT],PATHS.B[t1:tT],PATHS.Spreads[t1:tT],PATHS.GDP[t1:tT],PATHS.nGDP[t1:tT],PATHS.P[t1:tT],PATHS.GDPdef[t1:tT],PATHS.Ypot[t1:tT],PATHS.yT[t1:tT],PATHS.Cons[t1:tT],PATHS.Inv[t1:tT],PATHS.nInv[t1:tT],PATHS.λ[t1:tT],PATHS.yO[t1:tT],PATHS.TB[t1:tT],PATHS.CA[t1:tT],PATHS.RER[t1:tT],PATHS.RiskPremium[t1:tT],PATHS.RP_Spr[t1:tT],PATHS.DefPr[t1:tT],PATHS.NPV[t1:tT])
end

function SimulatePathsOfDiscovery(FixedShocks::Bool,Tbefore::Int64,Tafter::Int64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack drp = par
    T=drp+Tbefore+1+Tafter
    PATHS=InitiateEmptyPaths(T)
    PATHS_ND=InitiateEmptyPaths(T)

    #Simulate discovery
    PATHS.n .= ones(Int64,T)
    # PATHS.n .= Simulate_DiscreteShocks(1,T,GRIDS.PI_n)
    PATHS.n[drp+Tbefore+1]=2
    PATHS.σp .= ones(Int64,T)

    #Simulate other shocks
    if FixedShocks
        PATHS.σp .= ones(Int64,length(PATHS.n))
        PATHS.z .= ones(Float64,length(PATHS.n))
        PATHS.p .= ones(Float64,length(PATHS.n))
        PATHS.n[drp+Tbefore+1:end] .= 2
    else
        # PATHS.n[drp+Tbefore+1:end] .= 2
        PATHS.n[drp+Tbefore+1:end] .= Simulate_DiscreteShocks(2,Tafter+1,GRIDS.PI_n)
        PATHS.σp .= Simulate_DiscreteShocks(1,T,GRIDS.PI_σp)
        z, p=Simulate_z_p_shocks(T,PATHS.σp,GRIDS,par)
        PATHS.z .= z
        PATHS.p .= p
    end

    PATHS_ND.σp .= PATHS_ND.σp
    PATHS_ND.z .= PATHS_ND.z
    PATHS_ND.p .= PATHS_ND.p
    PATHS_ND.n .= ones(Int64,length(PATHS_ND.p))

    #Choose initial conditions
    PP=SimulatePathsOfStates(1100,MODEL)
    PATHS.K[1]=mean(PP.K[101:end])
    PATHS.B[1]=mean(PP.B[101:end])

    PATHS_ND.K[1]=mean(PP.K[101:end])
    PATHS_ND.B[1]=mean(PP.B[101:end])
    for t in 1:T
        GenerateNextState!(t,PATHS,SOLUTION,GRIDS,par)
        GenerateNextState!(t,PATHS_ND,SOLUTION,GRIDS,par)
    end
    UpdateOtherVariablesInPaths!(PATHS,MODEL)
    UpdateOtherVariablesInPaths!(PATHS_ND,MODEL)
    return Extract_TS(drp+1,drp+Tbefore+1+Tafter,PATHS), Extract_TS(drp+1,drp+Tbefore+1+Tafter,PATHS_ND)
end

function SumPathForAverage!(N::Int64,PATHS_AV::Paths,PATHS::Paths)
    #Paths of shocks
    PATHS_AV.z=PATHS_AV.z .+ (PATHS.z ./ N)
    PATHS_AV.p=PATHS_AV.p .+ (PATHS.p ./ N)
    PATHS_AV.nf=PATHS_AV.nf .+ PATHS.nf
    PATHS_AV.n=PATHS_AV.n .+ PATHS.n
    PATHS_AV.σp=PATHS_AV.σp .+ PATHS.σp

    #Paths of chosen states
    PATHS_AV.Def=PATHS_AV.Def .+ (PATHS.Def ./ N)
    PATHS_AV.K=PATHS_AV.K .+ (PATHS.K ./ N)
    PATHS_AV.B=PATHS_AV.B .+ (PATHS.B ./ N)

    #Path of relevant variables
    PATHS_AV.Spreads=PATHS_AV.Spreads .+ (PATHS.Spreads ./ N)
    PATHS_AV.GDP=PATHS_AV.GDP .+ (PATHS.GDP ./ N)
    PATHS_AV.nGDP=PATHS_AV.nGDP .+ (PATHS.nGDP ./ N)
    PATHS_AV.P=PATHS_AV.P .+ (PATHS.P ./ N)
    PATHS_AV.GDPdef=PATHS_AV.GDPdef .+ (PATHS.GDPdef ./ N)
    PATHS_AV.Ypot=PATHS_AV.Ypot .+ (PATHS.Ypot ./ N)
    PATHS_AV.Cons=PATHS_AV.Cons .+ (PATHS.Cons ./ N)
    PATHS_AV.Inv=PATHS_AV.Inv .+ (PATHS.Inv ./ N)
    PATHS_AV.λ=PATHS_AV.λ .+ (PATHS.λ ./ N)
    PATHS_AV.yO=PATHS_AV.yO .+ (PATHS.yO ./ N)
    PATHS_AV.yT=PATHS_AV.yT .+ (PATHS.yT ./ N)
    PATHS_AV.TB=PATHS_AV.TB .+ (PATHS.TB ./ N)
    PATHS_AV.CA=PATHS_AV.CA .+ (PATHS.CA ./ N)
    PATHS_AV.RER=PATHS_AV.RER .+ (PATHS.RER ./ N)
    PATHS_AV.RiskPremium=PATHS_AV.RiskPremium .+ (PATHS.RiskPremium ./ N)
    PATHS_AV.RP_Spr=PATHS_AV.RP_Spr .+ (PATHS.RP_Spr ./ N)
    PATHS_AV.DefPr=PATHS_AV.DefPr .+ (PATHS.DefPr ./ N)
    PATHS_AV.NPV=PATHS_AV.NPV .+ (PATHS.NPV ./ N)
    return nothing
end

function AverageDiscoveryPaths(FixedShocks::Bool,N::Int64,Tbefore::Int64,Tafter::Int64,MODEL::Model)
    @unpack par = MODEL
    @unpack drp = par
    Random.seed!(1234)
    PATHS_AV=InitiateEmptyPaths(Tbefore+1+Tafter)
    PATHS_AV_ND=InitiateEmptyPaths(Tbefore+1+Tafter)
    for i in 1:N
        println(i)
        PATHS, PATHS_ND=SimulatePathsOfDiscovery(FixedShocks,Tbefore,Tafter,MODEL)
        SumPathForAverage!(N,PATHS_AV,PATHS)
        SumPathForAverage!(N,PATHS_AV_ND,PATHS_ND)
        #New draw
    end
    return PATHS_AV, PATHS_AV_ND
end

function SubtractPaths(PATHS::Paths,PATHS_ND::Paths)
    PATHS_dif=InitiateEmptyPaths(length(PATHS.z))
    #Paths of shocks
    PATHS_dif.z=PATHS.z .- PATHS_ND.z
    PATHS_dif.p=PATHS.p .- PATHS_ND.p
    PATHS_dif.nf=PATHS.nf .- PATHS_ND.nf
    PATHS_dif.n=PATHS.n .- PATHS_ND.n
    PATHS_dif.σp=PATHS.σp .- PATHS_ND.σp

    #Paths of chosen states
    PATHS_dif.Def=PATHS.Def .- PATHS_ND.Def
    PATHS_dif.K=PATHS.K .- PATHS_ND.K
    PATHS_dif.B=PATHS.B .- PATHS_ND.B

    #Path of relevant variables
    PATHS_dif.Spreads=PATHS.Spreads .- PATHS_ND.Spreads
    PATHS_dif.GDP=PATHS.GDP .- PATHS_ND.GDP
    PATHS_dif.nGDP=PATHS.nGDP .- PATHS_ND.nGDP
    PATHS_dif.P=PATHS.P .+ PATHS_ND.P
    PATHS_dif.GDPdef=PATHS.GDPdef .- PATHS_ND.GDPdef
    PATHS_dif.Ypot=PATHS.Ypot .- PATHS_ND.Ypot
    PATHS_dif.Cons=PATHS.Cons .- PATHS_ND.Cons
    PATHS_dif.Inv=PATHS.Inv .- PATHS_ND.Inv
    PATHS_dif.λ=PATHS.λ .- PATHS_ND.λ
    PATHS_dif.yO=PATHS.yO .- PATHS_ND.yO
    PATHS_dif.yT=PATHS.yT .- PATHS_ND.yT
    PATHS_dif.TB=PATHS.TB .- PATHS_ND.TB
    PATHS_dif.CA=PATHS.CA .- PATHS_ND.CA
    PATHS_dif.RER=PATHS.RER .- PATHS_ND.RER
    PATHS_dif.RiskPremium=PATHS.RiskPremium .- PATHS_ND.RiskPremium
    PATHS_dif.RP_Spr=PATHS.RP_Spr .- PATHS_ND.RP_Spr
    PATHS_dif.DefPr=PATHS.DefPr .- PATHS_ND.DefPr
    PATHS_dif.NPV=PATHS.NPV .- PATHS_ND.NPV

    return PATHS_dif
end

################################################################################
### Functions to compute moments
################################################################################
@with_kw mutable struct Moments
    #Initiate them at 0.0 to facilitate average across samples
    #Default, spreads, and Debt
    DefaultPr::Float64 = 0.0
    Av_Spreads::Float64 = 0.0
    Av_RiskPremium::Float64 = 0.0
    Av_RP_Spr::Float64 = 0.0
    Debt_GDP::Float64 = 0.0
    #Volatilities
    σ_GDP::Float64 = 0.0
    σ_Spreads::Float64 = 0.0
    σ_con::Float64 = 0.0
    σ_inv::Float64 = 0.0
    σ_TB::Float64 = 0.0
    σ_CA::Float64 = 0.0
    σ_RER::Float64 = 0.0
    #Cyclicality
    Corr_Spreads_GDP::Float64 = 0.0
    Corr_con_GDP::Float64 = 0.0
    Corr_inv_GDP::Float64 = 0.0
    Corr_TB_GDP::Float64 = 0.0
    Corr_CA_GDP::Float64 = 0.0
    Corr_RER_GDP::Float64 = 0.0
    Corr_yT_Ypot::Float64 = 0.0
    Corr_TB_yT::Float64 = 0.0
    Corr_CA_yT::Float64 = 0.0
end

function TimeSeriesForMoments(Tmom::Int64,MODEL::Model;Only_nL::Bool=true,Only_σL::Bool=false)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack drp, Tsim = par

    PATHS = SimulatePathsOfStates(drp+Tsim,MODEL;Only_nL=Only_nL,Only_σL=Only_σL)
    t0=drp+1
    cnt=0
    while true
        if maximum(PATHS.Def[t0])==0
            break
        else
            cnt=cnt+1
            PATHS = SimulatePathsOfStates(drp+Tsim,MODEL;Only_nL=Only_nL,Only_σL=Only_σL)
            if cnt>10
                println("Could not find path starting in good standing, $cnt tries")
                break
            end
        end
    end
    tT=t0+Tmom-1
    PATHS_MOM=Extract_TS(t0,tT,PATHS)
    UpdateOtherVariablesInPaths!(PATHS_MOM,MODEL)
    return PATHS_MOM
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

function ComputeMomentsIntoStructure!(Tmom::Int64,MOM::Moments,MODEL::Model;Only_nL::Bool=true,Only_σL::Bool=false)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack HPFilter_Par = par

    #Draw long time series for default frequency with very long series
    PATHS = SimulatePathsOfStates(10000,MODEL)
    DefaultEvents=0.0
    for t in 2:length(PATHS.Def)
        if PATHS.Def[t]==1.0 && PATHS.Def[t-1]==0.0
            DefaultEvents=DefaultEvents+1
        end
    end
    MOM.DefaultPr=100*DefaultEvents/sum(1 .- PATHS.Def)

    #Draw time series with conditions for moments
    PATHS = TimeSeriesForMoments(Tmom,MODEL;Only_nL=Only_nL,Only_σL=Only_σL)
    GoodStanding=1 .- PATHS.Def

    #Compute average spreads, only periods in good standing
    MOM.Av_Spreads=sum(PATHS.Spreads .* GoodStanding)/(sum(GoodStanding)+sqrt(eps(Float64)))

    #Compute the average risk premium, only periods in good standing
    MOM.Av_RiskPremium=sum(PATHS.RiskPremium .* GoodStanding)/(sum(GoodStanding)+sqrt(eps(Float64)))
    MOM.Av_RP_Spr=sum(PATHS.RP_Spr .* GoodStanding)/(sum(GoodStanding)+sqrt(eps(Float64)))

    #Compute the average debt-to-GDP ratio, only periods in good standing
    b_gdp=100*(PATHS.B ./ PATHS.nGDP)
    MOM.Debt_GDP=sum(b_gdp .* GoodStanding)/(sum(GoodStanding)+sqrt(eps(Float64)))

    #Business cycles
    #Compute the natural logarithm of variables
    ln_y=log.(abs.(PATHS.GDP))
    ln_c=log.(abs.(PATHS.Cons))
    ln_i=log.(abs.(PATHS.Inv))
    ln_rer=log.(abs.(PATHS.RER))
    ln_yT=log.(abs.(PATHS.yT))

    #HP-Filtering
    y_trend=hp_filter(ln_y,HPFilter_Par)
    c_trend=hp_filter(ln_c,HPFilter_Par)
    i_trend=hp_filter(ln_i,HPFilter_Par)
    rer_trend=hp_filter(ln_rer,HPFilter_Par)
    yT_trend=hp_filter(ln_yT,HPFilter_Par)

    y_cyc=100.0*(ln_y .- y_trend)
    c_cyc=100.0*(ln_c .- c_trend)
    i_cyc=100.0*(ln_i .- i_trend)
    rer_cyc=100.0*(ln_rer .- rer_trend)
    yT_cyc=100.0*(ln_yT .- yT_trend)

    #Volatilities
    MOM.σ_GDP=std(y_cyc)
    MOM.σ_con=std(c_cyc)
    MOM.σ_inv=std(i_cyc)
    MOM.σ_RER=std(rer_cyc)

    spr2=PATHS.Spreads .^ 2
    Espr2=sum(spr2 .* GoodStanding)/(sum(GoodStanding)+sqrt(eps(Float64)))
    MOM.σ_Spreads=sqrt(Espr2-(MOM.Av_Spreads .^ 2))

    tb_y=100*(PATHS.TB ./ PATHS.nGDP)
    ca_y=100*(PATHS.CA ./ PATHS.nGDP)
    MOM.σ_TB=std(tb_y)
    MOM.σ_CA=std(ca_y)

    #Cyclicality
    MOM.Corr_Spreads_GDP=cor(y_cyc,PATHS.Spreads)
    MOM.Corr_con_GDP=cor(y_cyc,c_cyc)
    MOM.Corr_inv_GDP=cor(y_cyc,i_cyc)
    MOM.Corr_TB_GDP=cor(y_cyc,tb_y)
    MOM.Corr_CA_GDP=cor(y_cyc,ca_y)
    MOM.Corr_RER_GDP=cor(y_cyc,rer_cyc)

    #yT and potential output
    MOM.Corr_yT_Ypot=cor(PATHS.yT,PATHS.Ypot)
    MOM.Corr_TB_yT=cor(yT_cyc,tb_y)
    MOM.Corr_CA_yT=cor(yT_cyc,ca_y)
    return nothing
end

function SumMomentsForAverage!(N::Int64,MOM_AV::Moments,MOM::Moments)
    #Default, spreads, and Debt
    MOM_AV.DefaultPr=MOM_AV.DefaultPr+MOM.DefaultPr/N
    MOM_AV.Av_Spreads=MOM_AV.Av_Spreads+MOM.Av_Spreads/N
    MOM_AV.Av_RiskPremium=MOM_AV.Av_RiskPremium+MOM.Av_RiskPremium/N
    MOM_AV.Av_RP_Spr=MOM_AV.Av_RP_Spr+MOM.Av_RP_Spr/N
    MOM_AV.Debt_GDP=MOM_AV.Debt_GDP+MOM.Debt_GDP/N
    #Volatilities
    MOM_AV.σ_GDP=MOM_AV.σ_GDP+MOM.σ_GDP/N
    MOM_AV.σ_Spreads=MOM_AV.σ_Spreads+MOM.σ_Spreads/N
    MOM_AV.σ_con=MOM_AV.σ_con+MOM.σ_con/N
    MOM_AV.σ_inv=MOM_AV.σ_inv+MOM.σ_inv/N
    MOM_AV.σ_TB=MOM_AV.σ_TB+MOM.σ_TB/N
    MOM_AV.σ_CA=MOM_AV.σ_CA+MOM.σ_CA/N
    MOM_AV.σ_RER=MOM_AV.σ_RER+MOM.σ_RER/N
    #Cyclicality
    MOM_AV.Corr_Spreads_GDP=MOM_AV.Corr_Spreads_GDP+MOM.Corr_Spreads_GDP/N
    MOM_AV.Corr_con_GDP=MOM_AV.Corr_con_GDP+MOM.Corr_con_GDP/N
    MOM_AV.Corr_inv_GDP=MOM_AV.Corr_inv_GDP+MOM.Corr_inv_GDP/N
    MOM_AV.Corr_TB_GDP=MOM_AV.Corr_TB_GDP+MOM.Corr_TB_GDP/N
    MOM_AV.Corr_CA_GDP=MOM_AV.Corr_CA_GDP+MOM.Corr_CA_GDP/N
    MOM_AV.Corr_RER_GDP=MOM_AV.Corr_RER_GDP+MOM.Corr_RER_GDP/N
    MOM_AV.Corr_yT_Ypot=MOM_AV.Corr_yT_Ypot+MOM.Corr_yT_Ypot/N

    MOM_AV.Corr_TB_yT=MOM_AV.Corr_TB_yT+MOM.Corr_TB_yT/N
    MOM_AV.Corr_CA_yT=MOM_AV.Corr_CA_yT+MOM.Corr_CA_yT/N
    return nothing
end

function AverageMomentsManySamples(N::Int64,Tmom::Int64,MODEL::Model;Only_nL::Bool=true,Only_σL::Bool=false)
    @unpack SOLUTION, GRIDS, par = MODEL
    #Initiate average moments at 0
    Random.seed!(1234)
    AV_MOM=Moments()
    MOM=Moments()
    for i in 1:N
        ComputeMomentsIntoStructure!(Tmom,MOM,MODEL;Only_nL=Only_nL,Only_σL=Only_σL)
        SumMomentsForAverage!(N,AV_MOM,MOM)
    end
    return AV_MOM
end

################################################################################
### Results for paper
################################################################################
#Model Fit, responses vs data
include("EmpiricalResults.jl")

function Plot_Figure_6_with3Mods(FOLDER_GRAPHS::String,FILE_REGRESSIONS::String,
                                 PATHS::Paths,PATHS_higher::Paths,PATHS_RN::Paths,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    #Details for graphs
    npv=4.5
    tend=15
    t0=0
    t1=tend

    #Read empirical results
    #Spreads
    COLUMN=14
    reg=Read_Regression_Objects(FILE_REGRESSIONS,COLUMN)
    Spreads_data, cilow, cihigh=ImpulseResponse_TS(npv,t1+1,reg)

    #GDP
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE_REGRESSIONS,COLUMN)
    GDP_data, cilow, cihigh=ImpulseResponse_TS(npv,t1+1,reg)

    #Investment
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE_REGRESSIONS,COLUMN)
    Inv_data, cilow, cihigh=ImpulseResponse_TS(npv,t1+1,reg)

    #Current account
    COLUMN=COLUMN+3
    reg=Read_Regression_Objects(FILE_REGRESSIONS,COLUMN)
    CA_data, cilow, cihigh=ImpulseResponse_TS(npv,t1+1,reg)

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    LABELS=["data" "benchmark" "high α0" "risk neutral"]
    LINESTYLES=[:solid :dash :dot :dashdot]
    COLORS=[:black :blue :green :orange]

    #Plot Spreads
    TITLE="spreads"
    dat=Spreads_data[1:t1+1]
    mod=PATHS.Spreads[2:t1+2] .- PATHS.Spreads[1]
    mod2=PATHS_higher.Spreads[2:t1+2] .- PATHS_higher.Spreads[1]
    mod3=PATHS_RN.Spreads[2:t1+2] .- PATHS_RN.Spreads[1]
    plt_spreads=plot([t0:t1],[dat mod mod2 mod3],label=LABELS,
        linestyle=LINESTYLES,linecolor=COLORS,title=TITLE,
        ylabel="percentage points",xlabel="t",#ylims=[-0.05,1.3],
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot investment
    TITLE="investment"
    dat=Inv_data[1:t1+1]
    mod0=100*PATHS.Inv ./ PATHS.nGDP
    mod=mod0[2:t1+2] .- mod0[1]
    mod02=100*PATHS_higher.Inv ./ PATHS_higher.nGDP
    mod2=mod02[2:t1+2] .- mod02[1]
    mod03=100*PATHS_RN.Inv ./ PATHS_RN.nGDP
    mod3=mod03[2:t1+2] .- mod03[1]
    plt_inv=plot([t0:t1],[dat mod mod2 mod3],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Plot current account
    TITLE="current account"
    dat=CA_data[1:t1+1]
    mod0=100*PATHS.CA ./ PATHS.nGDP
    mod=mod0[2:t1+2] .- mod0[1]
    mod02=100*PATHS_higher.CA ./ PATHS_higher.nGDP
    mod2=mod02[2:t1+2] .- mod02[1]
    mod03=100*PATHS_RN.CA ./ PATHS_RN.nGDP
    mod3=mod03[2:t1+2] .- mod03[1]
    plt_CA=plot([t0:t1],[dat mod mod2 mod3],legend=:bottomright,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage of GDP",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW,label=LABELS)

    #Plot GDP
    TITLE="GDP"
    dat=GDP_data[1:t1+1]
    mod=100*(log.(PATHS.GDP[2:t1+2]) .- log.(PATHS.GDP[1]))
    mod2=100*(log.(PATHS_higher.GDP[2:t1+2]) .- log.(PATHS_higher.GDP[1]))
    mod3=100*(log.(PATHS_RN.GDP[2:t1+2]) .- log.(PATHS_RN.GDP[1]))
    plt_gdp=plot([t0:t1],[dat mod mod2 mod3],legend=false,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        size=SIZE_PLOTS,linewidth=LW)

    #Create plot array
    l = @layout([a b; c d])
    plt=plot(plt_spreads,plt_gdp,
             plt_inv,plt_CA,
             layout=l,size=(size_width*2,size_height*2))
    savefig(plt,"$FOLDER_GRAPHS\\Figure6.pdf")
    return plt
end

function Plot_Figure_7(FOLDER_GRAPHS::String,PATHS::Paths,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    #Details for graphs
    npv=4.5
    tend=15
    t0=-1
    t1=tend

    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0

    #Plot Risk premium
    TITLE="risk premium"
    LINESTYLES=[:solid :dash]
    COLORS=[:blue :red]
    LABELS=["spreads" "risk premium"]
    spr=PATHS.Spreads[1:t1+2]
    rp=PATHS.RiskPremium[1:t1+2]
    plt_rp=plot([t0:t1],rp,label=LABELS,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage points",xlabel="t",
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot tradable income
    TITLE="oil output"
    LINESTYLES=[:solid :dash]
    COLORS=[:blue :red]
    LABELS=["yT" "yOil"]
    yT=PATHS.yT[1:t1+2]
    yO=PATHS.yO[1:t1+2]
    plt_yT=plot([t0:t1],yO,label=LABELS,
        linestyle=LINESTYLES,title=TITLE,
        ylabel="yOil",xlabel="t",linecolor=COLORS,
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot share λ
    TITLE="share of K in manufacturing"
    LINESTYLES=[:solid :dash]
    COLORS=[:blue :red]
    LABELS=["yT" "yOil"]
    λ=PATHS.λ[1:t1+2]
    plt_λ=plot([t0:t1],λ,label=LABELS,
        linestyle=LINESTYLES,title=TITLE,
        ylabel="λ",xlabel="t",linecolor=COLORS,
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Plot real exchange rate
    TITLE="real exchange rate"
    LINESTYLES=[:solid :dash]
    COLORS=[:blue :red]
    LABELS=["yT" "yOil"]
    mod=100*(log.(PATHS.RER[1:t1+2]) .- log.(PATHS.RER[1]))
    plt_rer=plot([t0:t1],mod,label=LABELS,
        linestyle=LINESTYLES,title=TITLE,linecolor=COLORS,
        ylabel="percentage change",xlabel="t",
        legend=false,size=SIZE_PLOTS,linewidth=LW)

    #Create plot array
    l = @layout([a b; c d])
    plt=plot(plt_rp,plt_yT,
             plt_λ,plt_rer,
             layout=l,size=(size_width*2,size_height*2))
    savefig(plt,"$FOLDER_GRAPHS\\Figure7.pdf")
    return plt
end

function WelfareGains(MODEL::Model)
    T=2000
    PATHS=TimeSeriesForMoments(T,MODEL;Only_nL=true,Only_σL=false)
    z=PATHS.z[end]
    p=PATHS.p[end]
    σ_ind=PATHS.σp[end]
    b=PATHS.B[end]
    k=PATHS.K[end]

    wD=MODEL.SOLUTION.itp_V(b,k,z,p,2,σ_ind)
    wND=MODEL.SOLUTION.itp_V(b,k,z,p,1,σ_ind)
    @unpack par = MODEL
    @unpack σ = par

    return 100*(((wD/wND)^(1/(1-σ)))-1)
end

function AverageWelfareGains(N::Int64,MODEL::Model)
    Random.seed!(1234)
    av=0.0
    for i in 1:N
        println(i)
        av=av+WelfareGains(MODEL)/N
    end
    return av
end
