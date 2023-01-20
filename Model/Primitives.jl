
using Parameters, Roots, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, Sobol, Plots

################################################################
#### Defining parameters and other structures for the model ####
################################################################
#Define parameter and grid structure
@with_kw struct Pars
    ################################################################
    ######## Preferences and technology ############################
    ################################################################
    #Preferences
    σ::Float64 = 2.0          #CRRA parameter
    β::Float64 = 0.8775        #Discount factor
    r_star::Float64 = 0.04    #Risk-free interest rate
    #Debt parameters
    γ::Float64 = 0.14      #Reciprocal of average maturity
    κ::Float64 = 0.00      #Coupon payments
    #Default cost and debt parameters
    θ::Float64 = 0.40       #Probability of re-admission
    d0::Float64 = -0.16       #income default cost
    d1::Float64 = 0.222        #income default cost
    #Capital accumulation
    δ::Float64 = 0.05       #Depreciation rate
    φ::Float64 = 2.0        #Capital adjustment cost
    φO::Float64 = 2.0        #Capital adjustment cost
    #Production functions
    #Final consumption good
    η::Float64 = 0.83
    ωN::Float64 = 0.60
    ωM::Float64 = 0.34
    ωO::Float64 = 0.06
    A::Float64 = 0.764404296875            #Scaling factor for final good
    AO::Float64 = 1.0
    #Value added of intermediates
    αN::Float64 = 0.66          #Capital share in non-traded sector
    αM::Float64 = 0.57          #Capital share in manufacturing sector
    αO::Float64 = 0.49/(0.49+0.13)          #Capital share in oil Value Added
    #Oil field and value added
    ϕ::Float64 = 0.4           #Elasticity of substitution between oil field and VA
    ζ::Float64 = 0.38           #Share of oil rents in oil output
    #Stochastic process
    #Oil discoveries
    nL::Float64 = 0.25419921874999996
    nH::Float64 = 0.27437930107116687
    PrDisc::Float64 = 0.01           #Probability of discovery
    FieldLife::Float64 = 50.0        #Average duration of giant oil fields
    Twait::Int64 = 5                 #Periods between discovery and production
    IncludeDisc::Int64 = 1           #Indicate whether we want discoveries or no discoveries
    ExactWait::Int64 = 1
    #Parameters to pin down steady state capital and scaling parameters
    Target_yss::Float64 = 1.0        #Target value for steady state output
    Target_NPV::Float64 = 0.18       #Target value for steaty state NPV/GDP
    Target_oilXp::Float64 = 0.025    #Target value for steady state (oil exports)/GDP
    Target_spreads::Float64 = 0.029  #Target for interest rate spreads
    Target_iss_gdp::Float64 = 0.20   #Target for investment / gdp in steady state
    #Targets for moment-matching exercise
    Target_100spreads::Float64 =  3.5
    Target_std_spreads::Float64 = 2.5
    Target_σinv_σgdp::Float64 = 3.0
    Target_σcon_σgdp::Float64 = 1.0
    Target_debt_gdp::Float64 = 43.0
    Target_DefPr::Float64 = 2.0 #percent
    Target_RP_share::Float64 = 0.33
    #risk-premium parameters as in Arellano and Ramanarayanan
    α0::Float64 = 8.25
    α1::Float64 = -0.0
    #parameters for productivity shock
    μ_ϵz::Float64 = 0.0
    σ_ϵz::Float64 = 0.02
    dist_ϵz::UnivariateDistribution = truncated(Normal(μ_ϵz,σ_ϵz),-2.0*σ_ϵz,2.0*σ_ϵz)
    ρ_z::Float64 = 0.91
    μ_z::Float64 = 1.0
    zlow::Float64 = exp(log(μ_z)-2.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    zhigh::Float64 = exp(log(μ_z)+2.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    #parameters for process of price of oil
    μ_ϵp::Float64 = 0.0
    σ_ϵp::Float64 = 0.28#0.18
    dist_ϵp::UnivariateDistribution = truncated(Normal(μ_ϵp,σ_ϵp),-1.5*σ_ϵp,1.5*σ_ϵp)
    ρ_p::Float64 = 0.94#0.92
    μ_p::Float64 = 1.0
    plow::Float64 = exp(log(μ_p)-1.5*sqrt((σ_ϵp^2.0)/(1.0-(ρ_p^2.0))))
    phigh::Float64 = exp(log(μ_p)+1.5*sqrt((σ_ϵp^2.0)/(1.0-(ρ_p^2.0))))
    #Quadrature parameters
    N_GLz::Int64 = 21
    N_GLp::Int64 = 21
    #Grids
    Nz::Int64 = 11
    Np::Int64 = 11
    Nn::Int64 = Twait+3
    Nk::Int64 = 11
    NkO::Int64 = 11
    Nb::Int64 = 21
    NX::Int64 = 7
    klow::Float64 = 0.25
    khigh::Float64 = 2.0
    kOlow::Float64 = 0.10
    kOhigh::Float64 = 0.55
    blow::Float64 = 0.0
    bhigh::Float64 = 3.5
    #Parameters for solution algorithm
    cmin::Float64 = 1e-2
    Tol_V::Float64 = 1e-3             #Tolerance for absolute distance for value functions
    Tol_q::Float64 = 1e-3             #Tolerance for absolute distance for q in VFI
    relTolV::Float64 = 0.5            #Tolerance for relative distance in VFI (0.1%)
    Tol_q_pct::Float64 = 1.0          #Tolerance for % of states for which q has not converged (1%)
    cnt_max::Int64 = 50#100              #Maximum number of iterations on VFI
    MaxIter_Opt::Int64 = 1000
    g_tol_Opt::Float64 = 1e-8#1e-4
    blowOpt::Float64 = blow-0.1             #Minimum level of debt for optimization
    bhighOpt::Float64 = bhigh+0.1            #Maximum level of debt for optimization
    klowOpt::Float64 = 0.9*klow             #Minimum level of capital for optimization
    khighOpt::Float64 = 1.1*khigh           #Maximum level of capital for optimization
    kOlowOpt::Float64 = 0.9*kOlow            #Minimum level of oil capital for optimization
    kOhighOpt::Float64 = 1.1*kOhigh           #Maximum level of oil capital for optimization
    #Simulation parameters
    Tsim::Int64 = 10000
    drp::Int64 = 1000
    Tmom::Int64 = 50
    Tpaths::Int64 = 17
    Tpanel::Int64 = 50
    Npanel::Int64 = 1000
    TsinceDefault::Int64 = 25
    TsinceExhaustion::Int64 = 50
    NSamplesMoments::Int64 = 300
    NSamplesPaths::Int64 = 1000
    HPFilter_Par::Float64 = 100.0
end

@with_kw struct Grids
    #Grids of states
    GR_z::Array{Float64}
    GR_p::Array{Float64}
    GR_n::Array{Float64}
    GR_k::Array{Float64}
    GR_kO::Array{Float64}
    GR_b::Array{Float64}
    #Quadrature vectors for integrals
    ϵz_weights::Vector{Float64}
    ϵz_nodes::Vector{Float64}
    ϵp_weights::Vector{Float64}
    ϵp_nodes::Vector{Float64}
    #Factor to correct quadrature underestimation
    FacQz::Float64
    FacQp::Float64
    #Vector for discoveries
    PI_n::Array{Float64,2}
    #Bounds for divide and conquer for b
    ind_order_b::Array{Int64,1}
    lower::Array{Int64,1}
    upper::Array{Int64,1}
    #Matrices for integrals
    ZPRIME::Array{Float64,2}
    PPRIME::Array{Float64,2}
    PDFz::Array{Float64,2}
    PDFp::Array{Float64,2}
end

function BoundsDivideAndConquer(N::Int64)
    ind_order=Array{Int64,1}(undef,N)
    lower=Array{Int64,1}(undef,N)
    upper=Array{Int64,1}(undef,N)
    i=1
    #Do for last first
    ind_order[i]=N
    lower[i]=1
    upper[i]=N
    i=i+1
    #Do for first
    ind_order[i]=1
    lower[i]=1
    upper[i]=N
    i=i+1
    #Auxiliary vectors
    d=zeros(Int64,floor(Int64,N/2)+1)
    up=zeros(Int64,floor(Int64,N/2)+1)
    k=1
    d[1]=1
    up[1]=N
    k=1
    while true

        while true

            if up[k]==(d[k]+1)
                break
            end
            k=k+1
            d[k]=d[k-1]
            up[k]=floor(Int64,(d[k-1]+up[k-1])/2)
            # Compute for g of u(k) searching from g(l(k-1)) to g(u(k-1))b
            ind_order[i]=up[k]
            lower[i]=d[k-1]
            upper[i]=up[k-1]
            i=i+1
        end

        while true
            if k==1
                break
            end
            if up[k]!=up[k-1]
                break
            end
            k=k-1
        end
        if k==1
            break
        end
        d[k]=up[k]
        up[k]=up[k-1]
    end
    return ind_order, lower, upper
end

function CreateOilFieldGrids(par::Pars)
    @unpack Nn, Twait, nL, nH, FieldLife, PrDisc, ExactWait, IncludeDisc = par
    if IncludeDisc==1
        if ExactWait==0
            GR_n=nL*ones(Float64,Nn)
            GR_n[end]=nH
            #Fill in transition matrix
            PI_n=zeros(Float64,Nn,Nn)
            #Probability of discovery and no discovery
            PI_n[1,1]=1.0-PrDisc
            PI_n[1,2]=PrDisc
            PI_n[2,2]=1.0-(1.0/Twait)
            PI_n[2,3]=1.0/Twait
            PrExhaustion=1.0/FieldLife
            PI_n[end,end]=1.0-PrExhaustion
            PI_n[end,1]=PrExhaustion
            return GR_n, PI_n
        else
            GR_n=nL*ones(Float64,Nn)
            GR_n[end]=nH
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

function CreateGrids(par::Pars)
    #Grid for z
    @unpack Nz, zlow, zhigh = par
    GR_z=collect(range(zlow,stop=zhigh,length=Nz))
    #Grid for p
    @unpack Np, plow, phigh = par
    GR_p=collect(range(plow,stop=phigh,length=Np))
    #Grid for n
    GR_n, PI_n=CreateOilFieldGrids(par)
    #Gauss-Legendre vectors for z
    @unpack N_GLz, σ_ϵz, ρ_z, μ_z, dist_ϵz = par
    GL_nodes, GL_weights = gausslegendre(N_GLz)
    ϵzlow=-3.0*σ_ϵz
    ϵzhigh=3.0*σ_ϵz
    ϵz_nodes=0.5*(ϵzhigh-ϵzlow).*GL_nodes .+ 0.5*(ϵzhigh+ϵzlow)
    ϵz_weights=GL_weights .* 0.5*(ϵzhigh-ϵzlow)
    #Matrices for integration over z
    ZPRIME=Array{Float64,2}(undef,Nz,N_GLz)
    PDFz=Array{Float64,2}(undef,Nz,N_GLz)
    for z_ind in 1:Nz
        z=GR_z[z_ind]
        ZPRIME[z_ind,:]=exp.((1.0-ρ_z)*log(μ_z)+ρ_z*log(z) .+ ϵz_nodes)
        PDFz[z_ind,:]=pdf.(dist_ϵz,ϵz_nodes)
    end
    FacQz=dot(ϵz_weights,pdf.(dist_ϵz,ϵz_nodes))
    #Gauss-Legendre vectors for p
    @unpack N_GLp, σ_ϵp, ρ_p, μ_p, dist_ϵp = par
    GL_nodes, GL_weights = gausslegendre(N_GLp)
    ϵplow=-3.0*σ_ϵp
    ϵphigh=3.0*σ_ϵp
    ϵp_nodes=0.5*(ϵphigh-ϵplow).*GL_nodes .+ 0.5*(ϵphigh+ϵplow)
    ϵp_weights=GL_weights .* 0.5*(ϵphigh-ϵplow)
    #Matrices for integration over p
    PPRIME=Array{Float64,2}(undef,Np,N_GLp)
    PDFp=Array{Float64,2}(undef,Np,N_GLp)
    for p_ind in 1:Np
        p=GR_p[p_ind]
        PPRIME[p_ind,:]=exp.((1.0-ρ_p)*log(μ_p)+ρ_p*log(p) .+ ϵp_nodes)
        PDFp[p_ind,:]=pdf.(dist_ϵp,ϵp_nodes)
    end
    FacQp=dot(ϵp_weights,pdf.(dist_ϵp,ϵp_nodes))
    #Grid of capital
    @unpack Nk, klow, khigh = par
    GR_k=collect(range(klow,stop=khigh,length=Nk))
    #Grid of oil capital
    @unpack NkO, kOlow, kOhigh = par
    GR_kO=collect(range(kOlow,stop=kOhigh,length=NkO))
    #Grid of debt
    @unpack Nb, blow, bhigh = par
    GR_b=collect(range(blow,stop=bhigh,length=Nb))
    #Bounds for divide and conquer
    ind_order, lower, upper=BoundsDivideAndConquer(Nb)
    return Grids(GR_z,GR_p,GR_n,GR_k,GR_kO,GR_b,ϵz_weights,ϵz_nodes,ϵp_weights,ϵp_nodes,FacQz,FacQp,PI_n,ind_order,lower,upper,ZPRIME,PPRIME,PDFz,PDFp)
end

@with_kw mutable struct Solution{T1,T2,T3,T4,T5,T6,T7,T8}
    ### Arrays
    #Value Functions
    VD::T1
    VP::T2
    V::T2
    #Expectations and price
    EVD::T1
    EV::T2
    q1::T2
    EyT::T1
    σyT::T1
    #Policy functions
    kprime_D::T1
    kOprime_D::T1
    kprime::T2
    kOprime::T2
    bprime::T2
    lnyT::T1
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
    itp_kprime_D::T6
    itp_kOprime_D::T6
    itp_kprime::T7
    itp_kOprime::T7
    itp_bprime::T7
    itp_lnyT::T8
end

@with_kw struct ProductionMatrices{T1,T2}
    MAT_YD::T1
    MAT_lnYT::T1
    MAT_YR::T2
    GR_X::T2
end

#Interpolate equilibrium objects
function CreateInterpolation_ValueFunctions(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_p, GR_n, GR_k, GR_kO = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    KOs=range(GR_kO[1],stop=GR_kO[end],length=length(GR_kO))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KOs,Ks,Zs,Ps,Ns),Interpolations.Line())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KOs,Ks,Zs,Ps,Ns),Interpolations.Line())
    end
end

function CreateInterpolation_Price(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_z, GR_p, GR_n, GR_k, GR_kO, GR_b = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    KOs=range(GR_kO[1],stop=GR_kO[end],length=length(GR_kO))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KOs,Ks,Zs,Ps,Ns),Interpolations.Flat())
end

function CreateInterpolation_PolicyFunctions(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_p, GR_n, GR_k, GR_kO = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    KOs=range(GR_kO[1],stop=GR_kO[end],length=length(GR_kO))
    ORDER_SHOCKS=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))
    ORDER_STATES=Linear()
    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KOs,Ks,Zs,Ps,Ns),Interpolations.Flat())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,KOs,Ks,Zs,Ps,Ns),Interpolations.Flat())
    end
end

function CreateInterpolation_log_yT(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_z, GR_p, GR_n, GR_k, GR_kO = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Ps=range(GR_p[1],stop=GR_p[end],length=length(GR_p))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Ks=range(GR_k[1],stop=GR_k[end],length=length(GR_k))
    KOs=range(GR_kO[1],stop=GR_kO[end],length=length(GR_kO))
    ORDER_SHOCKS=Linear()
    ORDER_STATES=Linear()
    EXT=Interpolations.Line()
    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),KOs,Ks,Zs,Ps,Ns),EXT)
end

################################################################
################ Preference functions ##########################
################################################################
function Utility(c::Float64,par::Pars)
    @unpack σ = par
    return (c^(1.0-σ))/(1.0-σ)
end

function zDefault(z::Float64,par::Pars)
    @unpack d0, d1 = par
    return z-max(0.0,d0*z+d1*z*z)
end

function CapitalAdjustment(kprime::Float64,k::Float64,par::Pars)
    @unpack φ = par
    0.5*φ*((kprime-k)^2.0)/k
end

function OilCapitalAdjustment(kOprime::Float64,kO::Float64,par::Pars)
    @unpack φO = par
    0.5*φO*((kOprime-kO)^2.0)/kO
end

function ExpectedTradableIncome_z_GivenP(z_ind::Int64,pPrime::Float64,n_ind::Int64,
                                         kprime::Float64,kOprime::Float64,
                                         SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, dist_ϵz, cmin = par
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    @unpack itp_lnyT = SOLUTION
    funYT(zprime::Float64)=itp_lnyT(kOprime,kprime,zprime,pPrime,n_ind)
    pdf_VD=PDFz[z_ind,:] .* funYT.(ZPRIME[z_ind,:])
    return dot(ϵz_weights,pdf_VD)/FacQz
end

function ExpectedTradableIncome(z_ind::Int64,p_ind::Int64,n_ind::Int64,
                                kprime::Float64,kOprime::Float64,
                                SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_p, μ_p, dist_ϵp = par
    @unpack ϵp_weights, PPRIME, PDFp, FacQp = GRIDS
    funYT(pPrime::Float64)=ExpectedTradableIncome_z_GivenP(z_ind,pPrime,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    pdf_VD=PDFp[p_ind,:] .* funYT.(PPRIME[p_ind,:])
    return dot(ϵp_weights,pdf_VD)/FacQp
end

function VarTradableIncome_z_GivenP(z_ind::Int64,pPrime::Float64,n_ind::Int64,
                                    kprime::Float64,kOprime::Float64,
                                    SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, dist_ϵz, cmin = par
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    @unpack itp_lnyT = SOLUTION
    funYT(zprime::Float64)=(itp_lnyT(kOprime,kprime,zprime,pPrime,n_ind))^2.0
    pdf_YT=PDFz[z_ind,:] .* funYT.(ZPRIME[z_ind,:])
    return dot(ϵz_weights,pdf_YT)/FacQz
end

function VarTradableIncome(z_ind::Int64,p_ind::Int64,n_ind::Int64,
                           kprime::Float64,kOprime::Float64,
                           SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_p, μ_p, dist_ϵp = par
    @unpack ϵp_weights, PPRIME, PDFp, FacQp = GRIDS
    funYT(pPrime::Float64)=VarTradableIncome_z_GivenP(z_ind,pPrime,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    pdf_YT=PDFp[p_ind,:] .* funYT.(PPRIME[p_ind,:])
    return dot(ϵp_weights,pdf_YT)/FacQp
end

function SDF_Lenders(lnyT::Float64,EyT::Float64,σT::Float64,par::Pars)
    @unpack r_star, α0 = par
    yTtilde=lnyT-EyT
    return exp(-(r_star+α0*yTtilde+0.5*((α0^2.0)*(σT^2.0))))
end

################################################################
################ Production functions ##########################
################################################################
function NonTradedProduction(z::Float64,kN::Float64,par::Pars)
    @unpack αN, A = par
    return z*A*(kN^αN)
end

function ManufProduction(z::Float64,kM::Float64,par::Pars)
    @unpack αM, A = par
    return z*A*(kM^αM)
end

function Oil_CES(VA::Float64,n::Float64,par::Pars)
    @unpack ϕ, ζ = par
    return ((1.0-ζ)*(VA^((ϕ-1.0)/ϕ))+ζ*(n^((ϕ-1.0)/ϕ)))^(ϕ/(ϕ-1.0))
end

function OilProduction(kO::Float64,n::Float64,par::Pars)
    @unpack ϕ, ζ, αO, AO = par
    VA=AO*(kO^αO)
    return Oil_CES(VA,n,par)
end

function Final_CES(cN::Float64,cM::Float64,cO::Float64,par::Pars)
    @unpack A, η, ωN, ωM, ωO = par
    return (ωN*(cN^((η-1.0)/η))+ωM*(cM^((η-1.0)/η))+ωO*(cO^((η-1.0)/η)))^(η/(η-1.0))
end

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

function CapitalAllocation(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,X::Float64,par::Pars)
    @unpack η, ωO, ωM, ωN, αM, αN = par
    LHS(λ::Float64)=(((αM/αN)*(((1.0-λ)^(1.0-αN))/(λ^(1.0-αM)))*(k^(αM-αN)))^η)*NonTradedProduction(z,(1.0-λ)*k,par)
    RHS(λ::Float64)=((ωN^η)/((ωM^η)+(ωO^η)*(pO^(1.0-η))))*(ManufProduction(z,λ*k,par)+pO*OilProduction(kO,n,par)+X)
    foo(λ::Float64)=LHS(λ)-RHS(λ)
    if foo(1e-4)<0.0
        return 1e-4
    else
        if foo(1.0-1e-4)>=0.0
            return 1.0-1e-4
        else
            return MyBisection(foo,1e-4,1.0-1e-4,xatol=1e-3)
        end
    end
end

function Production(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,X::Float64,par::Pars)
    @unpack αN, αM, αO, ωN, ωM, ωO, η, ζ, A = par
    #Compute share of capital in manufacturing
    λ=CapitalAllocation(z,pO,n,k,kO,X,par)
    #Compute consumption of intermediate goods
    cO=(((ωO/pO)^η)/((ωM^η)+(ωO^η)*(pO^(1.0-η))))*(ManufProduction(z,λ*k,par)+pO*OilProduction(kO,n,par)+X)
    cM=(((ωM/ωO)*pO)^η)*cO
    cN=NonTradedProduction(z,(1.0-λ)*k,par)
    if cO>0.0
        return Final_CES(cN,cM,cO,par)
    else
        return cO
    end
end

################################################################
################### Setup functions ############################
################################################################
function MPK(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,X::Float64,par::Pars)
    hh=1e-7
    return (Production(z,pO,n,k+hh,kO,X,par)-Production(z,pO,n,k-hh,kO,X,par))/(2.0*hh)
end

function MPK_O(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,X::Float64,par::Pars)
    hh=1e-7
    return (Production(z,pO,n,k,kO+hh,X,par)-Production(z,pO,n,k,kO-hh,X,par))/(2.0*hh)
end

function kOss_Given_kss(zss::Float64,pOss::Float64,nss::Float64,kss::Float64,par::Pars)
    @unpack β, δ, μ_z, A = par
    foo(kOss::Float64)=β*(MPK_O(μ_z,pOss,nss,kss,kOss,0.0,par)+1.0-δ)-1.0
    #Find a bracketing interval
    kOlow=0.001
    while foo(kOlow)<=0.0
        kOlow=0.5*kOlow
        if kOlow<1e-14
            break
        end
    end
    kOhigh=kss
    while foo(kOhigh)>=0.0
        kOhigh=2.0*kOhigh
    end
    return MyBisection(foo,kOlow,kOhigh,xatol=1e-3)
end

function SteadyStateCapital(par::Pars)
    @unpack β, δ, μ_z, μ_p, nL = par
    foo(kss::Float64)=β*(MPK(μ_z,μ_p,nL,kss,kOss_Given_kss(μ_z,μ_p,nL,kss,par),0.0,par)+1.0-δ)-1.0
    #Find a bracketing interval
    klow=1.0
    while foo(klow)<=0.0
        klow=0.5*klow
    end
    khigh=3.0
    while foo(khigh)>=0.0
        khigh=2.0*khigh
    end
    kss=MyBisection(foo,klow,khigh,xatol=1e-3)
    kOss=kOss_Given_kss(μ_z,μ_p,nL,kss,par)
    return kss, kOss
end

function PriceNonTraded(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,X::Float64,par::Pars)
    @unpack αN, αM = par
    #Compute share of capital in manufacturing
    λ=CapitalAllocation(z,pO,n,k,kO,X,par)
    return (αM/αN)*(((1.0-λ)^(1.0-αN))/(λ^(1.0-αM)))*(k^(αM-αN))
end

function PriceFinalGood(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,X::Float64,par::Pars)
    @unpack ωN, ωM, ωO, η = par
    pN=PriceNonTraded(z,pO,n,k,kO,X,par)
    return (((ωN^η)*(pN^(1.0-η))+(ωM^η)*(1.0^(1.0-η))+(ωO^η)*(pO^(1.0-η)))^(1.0/(1.0-η)))
end

function OilExports_GDP(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,X::Float64,par::Pars)
    @unpack αN, αM, αO, ωM, ωO, η, ζ = par
    P=PriceFinalGood(z,pO,n,k,kO,X,par)
    y=Production(z,pO,n,k,kO,X,par)
    #Compute share of capital in manufacturing
    λ=CapitalAllocation(z,pO,n,k,kO,X,par)
    #Compute consumption of intermediate goods
    yO=OilProduction(kO,n,par)
    cO=(((ωO^η)*(pO^(1.0-η)))/((ωM^η)+(ωO^η)*(pO^(1.0-η))))*(ManufProduction(z,λ*k,par)+pO*yO+X)/pO
    return pO*(yO-cO)/(P*y)
end

function Calibrate_nL_given_A(pOss::Float64,par::Pars)
    @unpack Target_oilXp, μ_z = par
    function foo(nnL::Float64)
        par_nL=Pars(par,nL=nnL)
        kss, kOss=SteadyStateCapital(par_nL)
        return OilExports_GDP(μ_z,pOss,nnL,kss,kOss,0.0,par_nL)-Target_oilXp
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
    return MyBisection(foo,nLlow,nLhigh,xatol=1e-3)
end

function Calibrate_nL_and_A(par::Pars)
    @unpack Target_yss, μ_z, μ_p = par
    function foo(Ass::Float64)
        parA=Pars(par,A=Ass)
        nLss=Calibrate_nL_given_A(μ_p,parA)
        parA=Pars(par,A=Ass,nL=nLss)
        kss, kOss=SteadyStateCapital(parA)
        λss=CapitalAllocation(μ_z,μ_p,nLss,kss,kOss,0.0,parA)
        yT=ManufProduction(μ_z,λss*kss,parA)+μ_p*OilProduction(kOss,nLss,parA)
        return yT-Target_yss
    end
    #Get bracketing interval
    Alow=0.5
    while foo(Alow)>=0.0
        Alow=0.5*Alow
    end
    Ahigh=1.5
    while foo(Ahigh)<=0.0
        Ahigh=2.0*Ahigh
    end
    A_ss=MyBisection(foo,Alow,Ahigh,xatol=1e-4)
    parA=Pars(par,A=A_ss)
    nL=Calibrate_nL_given_A(μ_p,parA)
    return A_ss, nL
end

function SteadyStateCapital_nH(par::Pars)
    @unpack β, δ, μ_z, μ_p, nH = par
    foo(kss::Float64)=β*(MPK(μ_z,μ_p,nH,kss,kOss_Given_kss(μ_z,μ_p,nH,kss,par),0.0,par)+1.0-δ)-1.0
    #Find a bracketing interval
    klow=1.0
    while foo(klow)<=0.0
        klow=0.5*klow
    end
    khigh=3.0
    while foo(khigh)>=0.0
        khigh=2.0*khigh
    end
    kss=MyBisection(foo,klow,khigh,xatol=1e-3)
    kOss=kOss_Given_kss(μ_z,μ_p,nH,kss,par)
    return kss, kOss
end

function NPV_disc(z::Float64,pO::Float64,par::Pars)
    @unpack Twait, FieldLife, r_star, αO, ζ, nH, nL, Target_spreads = par
    rss=r_star+Target_spreads
    a0=1.0/(1.0+rss)
    FACTOR=((a0^(Twait+1.0))-(a0^(Twait+FieldLife+1.0)))/(1.0-a0)
    kssL, kOssL=SteadyStateCapital(par)
    kssH, kOssH=SteadyStateCapital(par)
    return pO*(OilProduction(kOssL,nH,par)-OilProduction(kOssL,nL,par))*FACTOR
end

function NPV_disc_kH(z::Float64,pO::Float64,par::Pars)
    @unpack Twait, FieldLife, r_star, αO, ζ, nH, nL, Target_spreads = par
    rss=r_star+Target_spreads
    a0=1.0/(1.0+rss)
    FACTOR=((a0^(Twait+1.0))-(a0^(Twait+FieldLife+1.0)))/(1.0-a0)
    kssL, kOssL=SteadyStateCapital(par)
    kssH, kOssH=SteadyStateCapital_nH(par)
    return pO*(OilProduction(kOssH,nH,par)-OilProduction(kOssL,nL,par))*FACTOR
end

function Calibrate_nH(par::Pars)
    @unpack r_star, Target_spreads, nL, ζ, αO, μ_p, μ_z, Twait, FieldLife, Target_NPV = par
    rss=r_star+Target_spreads
    a0=1.0/(1.0+rss)
    FACTOR=((a0^Twait)-(a0^(FieldLife+1)))/(1.0-a0)
    kss, kOss=SteadyStateCapital(par)
    P=PriceFinalGood(μ_z,μ_p,nL,kss,kOss,0.0,par)
    y=Production(μ_z,μ_p,nL,kss,kOss,0.0,par)
    function foo(nHss::Float64)
        parNH=Pars(par,nH=nHss)
        return NPV_disc(μ_z,μ_p,parNH)/(P*y)-Target_NPV
    end
    #Get bracketing interval
    nHlow=0.10
    while foo(nHlow)>=0.0
        nHlow=0.5*nHlow
    end
    nHhigh=2.0*nL
    while foo(nHhigh)<=0.0
        nHhigh=2.0*nHhigh
    end
    return MyBisection(foo,nHlow,nHhigh,xatol=1e-3)
end

function Calibrate_nH2(par::Pars)
    @unpack r_star, Target_spreads, nL, ζ, αO, μ_p, μ_z, Twait, FieldLife, Target_NPV = par
    rss=r_star+Target_spreads
    a0=1.0/(1.0+rss)
    FACTOR=((a0^Twait)-(a0^(FieldLife+1)))/(1.0-a0)
    kss, kOss=SteadyStateCapital(par)
    P=PriceFinalGood(μ_z,μ_p,nL,kss,kOss,0.0,par)
    y=Production(μ_z,μ_p,nL,kss,kOss,0.0,par)
    function foo(nHss::Float64)
        parNH=Pars(par,nH=nHss)
        return NPV_disc_kH(μ_z,μ_p,parNH)/(P*y)-Target_NPV
    end
    #Get bracketing interval
    nHlow=0.10
    while foo(nHlow)>=0.0
        nHlow=0.5*nHlow
    end
    nHhigh=2.0*nL
    while foo(nHhigh)<=0.0
        nHhigh=2.0*nHhigh
    end
    return MyBisection(foo,nHlow,nHhigh,xatol=1e-3)
end

function Calibrate_delta_nL_A(par::Pars)
    @unpack Target_iss_gdp, μ_z, μ_p, nL = par
    function foo(δ::Float64)
        par=Pars(par,δ=δ)
        A, nL=Calibrate_nL_and_A(par)
        par=Pars(par,A=A,nL=nL)
        kss, kOss=SteadyStateCapital(par)
        yss=Production(μ_z,μ_p,nL,kss,kOss,0.0,par)
        return δ*(kss+kOss)/yss-Target_iss_gdp
    end
    #Get bracketing interval
    δlow=0.01
    while foo(δlow)>=0.0
        δlow=0.5*δlow
    end
    δhigh=0.15
    while foo(δhigh)<=0.0 && δhigh<0.9
        δhigh=δhigh+0.01
    end
    δ=MyBisection(foo,δlow,δhigh,xatol=1e-4)
    par=Pars(par,δ=δ)
    A, nL=Calibrate_nL_and_A(par)
    return δ, A, nL
end

function OilConsumption(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,X::Float64,par::Pars)
    @unpack αN, αM, αO, ωM, ωO, η, ζ = par
    #Compute share of capital in manufacturing
    λ=CapitalAllocation(z,pO,n,k,kO,X,par)
    #Compute consumption of intermediate goods
    yO=OilProduction(kO,n,par)
    return (((ωO^η)*(pO^(1.0-η)))/((ωM^η)+(ωO^η)*(pO^(1.0-η))))*(ManufProduction(z,λ*k,par)+pO*yO+X)/pO
end

function Setup_MomentMatching(β::Float64,d0::Float64,d1::Float64,
                              α0::Float64,φ::Float64)
    par=Pars(β=β,d0=d0,d1=d1,α0=α0,φ=φ)
    #Setup size of grids for discoveries
    if par.IncludeDisc==1
        if par.ExactWait==0
            par=Pars(par,Nn=3)
        else
            par=Pars(par,Nn=3+par.Twait)
        end
    else
        par=Pars(par,Nn=2)
    end
    #Calibrate A and nL
    A, nL=Calibrate_nL_and_A(par)
    par=Pars(par,A=A,nL=nL)
    nH=Calibrate_nH2(par)
    par=Pars(par,nH=nH)
    #Setup Grids range
    kss, kOss=SteadyStateCapital(par)
    φO=par.φ
    par=Pars(par,φO=φO)
    par=Pars(par,klow=0.25,khigh=2.0*kss)
    par=Pars(par,kOlow=0.05,kOhigh=2.5*kOss)
    GRIDS=CreateGrids(par)
    #Set bounds for optimization algorithm
    par=Pars(par,klowOpt=0.1,khighOpt=1.1*par.khigh)
    par=Pars(par,kOlowOpt=0.01,kOhighOpt=1.1*par.kOhigh)
    return par, GRIDS
end

###############################################################################
#Function to pre-compute output matrices
###############################################################################
function LowerBound_X(z::Float64,pO::Float64,n::Float64,k::Float64,kO::Float64,par::Pars)
    foo(X::Float64)=ManufProduction(z,CapitalAllocation(z,pO,n,k,kO,X,par)*k,par)+pO*OilProduction(kO,n,par)+X
    Xlow=-(ManufProduction(z,k,par)+pO*OilProduction(kO,n,par))
    Xhigh=0.0
    return 0.9*MyBisection(foo,Xlow,Xhigh)
end

function FillOutputMatrices(z_ind::Int64,p_ind::Int64,n_ind::Int64,k_ind::Int64,kO_ind::Int64,par::Pars,GRIDS::Grids)
    @unpack NX, δ = par
    @unpack GR_z, GR_p, GR_n, GR_k, GR_kO, GR_b = GRIDS
    #Unpack States
    z=GR_z[z_ind]
    zD=zDefault(z,par)
    pO=GR_p[p_ind]
    n=GR_n[n_ind]
    k=GR_k[k_ind]
    kO=GR_kO[kO_ind]
    #Fill production in default
    mat_yd=Production(zD,pO,n,k,kO,0.0,par)
    #Fill tradable income with X=0.0
    λ=CapitalAllocation(z,pO,n,k,kO,0.0,par)
    mat_lnyt=log(ManufProduction(z,λ*k,par)+pO*OilProduction(kO,n,par))
    #Fill production in repayment and grid X
    Xlow=LowerBound_X(z,pO,n,k,kO,par)
    Xhigh=GR_b[end]
    gr_x=collect(range(Xlow,stop=Xhigh,length=NX))
    mat_yr=Array{Float64,1}(undef,NX)
    for X_ind in 1:NX
        mat_yr[X_ind]=Production(z,pO,n,k,kO,gr_x[X_ind],par)
    end
    return mat_yd, mat_yr, gr_x, mat_lnyt
end

function OutputMatrices(par::Pars,GRIDS::Grids)
    @unpack Nz, Np, Nn, Nk, NkO, NX, γ = par
    @unpack GR_z, GR_p, GR_n, GR_k, GR_kO, GR_b = GRIDS
    #Preallocate matrices
    MAT_YD=SharedArray{Float64,5}(NkO,Nk,Nz,Np,Nn)
    MAT_lnYT=SharedArray{Float64,5}(NkO,Nk,Nz,Np,Nn)
    MAT_YR=SharedArray{Float64,6}(NX,NkO,Nk,Nz,Np,Nn)
    GR_X=SharedArray{Float64,6}(NX,NkO,Nk,Nz,Np,Nn)
    #Fill matrix in default
    @sync @distributed for n_ind in 1:Nn
        for p_ind in 1:Np
            for z_ind in 1:Nz
                for k_ind in 1:Nk
                    for kO_ind in 1:NkO
                        MAT_YD[kO_ind,k_ind,z_ind,p_ind,n_ind], MAT_YR[:,kO_ind,k_ind,z_ind,p_ind,n_ind], GR_X[:,kO_ind,k_ind,z_ind,p_ind,n_ind], MAT_lnYT[kO_ind,k_ind,z_ind,p_ind,n_ind] = FillOutputMatrices(z_ind,p_ind,n_ind,k_ind,kO_ind,par,GRIDS)
                    end
                end
            end
        end
    end
    return ProductionMatrices(convert(Array{Float64},MAT_YD),convert(Array{Float64},MAT_lnYT),convert(Array{Float64},MAT_YR),convert(Array{Float64},GR_X))
end

function OutputMatrices_NotParallel(par::Pars,GRIDS::Grids)
    @unpack Nz, Np, Nn, Nk, NkO, NX, γ = par
    @unpack GR_z, GR_p, GR_n, GR_k, GR_kO, GR_b = GRIDS
    #Preallocate matrices
    MAT_YD=Array{Float64,5}(undef,NkO,Nk,Nz,Np,Nn)
    MAT_lnYT=Array{Float64,5}(undef,NkO,Nk,Nz,Np,Nn)
    MAT_YR=Array{Float64,6}(undef,NX,NkO,Nk,Nz,Np,Nn)
    GR_X=Array{Float64,6}(undef,NX,NkO,Nk,Nz,Np,Nn)
    #Fill matrix in default
    for n_ind in 1:Nn
        for p_ind in 1:Np
            for z_ind in 1:Nz
                for k_ind in 1:Nk
                    for kO_ind in 1:NkO
                        MAT_YD[kO_ind,k_ind,z_ind,p_ind,n_ind], MAT_YR[:,kO_ind,k_ind,z_ind,p_ind,n_ind], GR_X[:,kO_ind,k_ind,z_ind,p_ind,n_ind], MAT_lnYT[kO_ind,k_ind,z_ind,p_ind,n_ind] = FillOutputMatrices(z_ind,p_ind,n_ind,k_ind,kO_ind,par,GRIDS)
                    end
                end
            end
        end
    end
    return ProductionMatrices(MAT_YD,MAT_lnYT,MAT_YR,GR_X)
end


###############################################################################
#Function to compute consumption net of investment and adjustment cost
###############################################################################
function ConsNet(y::Float64,k::Float64,kO::Float64,kprime::Float64,kOprime::Float64,par::Pars)
    y+(1.0-par.δ)*(k+kO)-kprime-kOprime-CapitalAdjustment(kprime,k,par)-OilCapitalAdjustment(kOprime,kO,par)
end

###############################################################################
#Functions to compute value given state, policies, and guesses
###############################################################################
# transform function
function TransformIntoBounds(x::Float64,min::Float64,max::Float64)
    (max - min) * (1.0/(1.0 + exp(-x))) + min
end

function TransformIntoReals(x::Float64,min::Float64,max::Float64)
    log((x - min)/(max - x))
end

function ValueInDefault(n_ind::Int64,p::Float64,z::Float64,k::Float64,kO::Float64,
                        y::Float64,kprime_REAL::Float64,kOprime_REAL::Float64,
                        VDmin::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, klowOpt, khighOpt, kOlowOpt, kOhighOpt, cmin = par
    @unpack GR_n, GR_k, GR_kO = GRIDS
    @unpack VD, itp_EV, itp_EVD = SOLUTION
    #transform policy tries into interval
    kprime=TransformIntoBounds(kprime_REAL,klowOpt,khighOpt)
    kOprime=TransformIntoBounds(kOprime_REAL,kOlowOpt,kOhighOpt)
    #Compute consumption and value
    cons=ConsNet(y,k,kO,kprime,kOprime,par)
    if cons>0.0
        return Utility(cons,par)+β*θ*min(0.0,itp_EV(0.0,kOprime,kprime,z,p,n_ind))+β*(1.0-θ)*min(0.0,itp_EVD(kOprime,kprime,z,p,n_ind))
    else
        return Utility(cmin,par)-kprime-kOprime
    end
end

function ValueInRepayment(n_ind::Int64,p::Float64,z::Float64,k::Float64,kO::Float64,b::Float64,
                        kprime_REAL::Float64,kOprime_REAL::Float64,bprime_REAL::Float64,
                        itp_YR,VPmin::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, κ, klowOpt, khighOpt, kOlowOpt, kOhighOpt, blowOpt ,bhighOpt, cmin = par
    @unpack GR_k, GR_kO, GR_b, GR_n = GRIDS
    @unpack itp_EV, itp_q1 = SOLUTION
    #transform policy tries into interval
    kprime=TransformIntoBounds(kprime_REAL,klowOpt,khighOpt)
    kOprime=TransformIntoBounds(kOprime_REAL,kOlowOpt,kOhighOpt)
    bprime=TransformIntoBounds(bprime_REAL,blowOpt,bhighOpt)
    #Compute output
    aa=0.0
    qq=itp_q1(bprime,kOprime,kprime,z,p,n_ind)
    if qq==0.0
        aa=-bprime
    end
    X=qq*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
    y=itp_YR(X)
    #Compute consumption
    cons=ConsNet(y,k,kO,kprime,kOprime,par)
    if y>0.0 && cons>0.0
        return Utility(cons,par)+β*min(0.0,itp_EV(bprime,kOprime,kprime,z,p,n_ind))+aa
    else
        return Utility(cmin,par)-kprime-kOprime+min(X,0.0)+aa
    end
end

###############################################################################
#Functions to optimize given guesses and state
###############################################################################

function OptimInDefault(X0_BOUNDS::Array{Float64,1},n_ind::Int64,p_ind::Int64,z_ind::Int64,
                        k_ind::Int64,kO_ind::Int64,PROD_MATS::ProductionMatrices,
                        VDmin::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, θ, δ, MaxIter_Opt, g_tol_Opt, klowOpt, khighOpt, kOlowOpt, kOhighOpt = par
    @unpack GR_z, GR_p, GR_k, GR_kO = GRIDS
    @unpack MAT_YD = PROD_MATS
    #transform policy guess into reals
    X0=Array{Float64,1}(undef,2)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],klowOpt,khighOpt)
    X0[2]=TransformIntoReals(X0_BOUNDS[2],kOlowOpt,kOhighOpt)
    #Setup function handle for optimization
    z=GR_z[z_ind]
    p=GR_p[p_ind]
    k=GR_k[k_ind]
    kO=GR_kO[kO_ind]
    y=MAT_YD[kO_ind,k_ind,z_ind,p_ind,n_ind]
    f(X::Array{Float64,1})=-ValueInDefault(n_ind,p,z,k,kO,y,X[1],X[2],VDmin,SOLUTION,GRIDS,par)
    #Perform optimization
    inner_optimizer = NelderMead()
    res=optimize(f,X0,inner_optimizer)#,
                 #Optim.Options(g_tol = g_tol_Opt,iterations = MaxIter_Opt))
    #Transform optimizer into bounds
    kprime=TransformIntoBounds(Optim.minimizer(res)[1],klowOpt,khighOpt)
    kOprime=TransformIntoBounds(Optim.minimizer(res)[2],kOlowOpt,kOhighOpt)
    return -Optim.minimum(res), kprime, kOprime
end

function GridSearchOverB(n_ind::Int64,p_ind::Int64,z_ind::Int64,
                         k_ind::Int64,kO_ind::Int64,b_ind::Int64,itp_YR,
                         SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, κ, Nb = par
    @unpack GR_k, GR_kO, GR_b, GR_z, GR_p = GRIDS
    @unpack itp_EV, itp_q1 = SOLUTION
    z=GR_z[z_ind]
    p=GR_p[p_ind]
    k=GR_k[k_ind]
    kO=GR_kO[kO_ind]
    b=GR_b[b_ind]
    kprime=k
    kOprime=kO
    bpol=0
    val=-Inf
    for btry in 1:Nb
        bprime=GR_b[btry]
        qq=itp_q1(bprime,kOprime,kprime,z,p,n_ind)
        if qq==0.0 && bpol>0
            break
        end
        X=qq*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
        y=itp_YR(X)
        if y>0.0
            cons=ConsNet(y,k,kO,kprime,kOprime,par)
            vv=Utility(cons,par)+β*itp_EV(bprime,kOprime,kprime,z,p,n_ind)
            if vv>val
                val=vv
                bpol=btry
            end
        end
    end
    return bpol
end

function OptimInRepayment(X0_BOUNDS::Array{Float64,1},n_ind::Int64,p_ind::Int64,z_ind::Int64,
                          k_ind::Int64,kO_ind::Int64,b_ind::Int64,PROD_MATS::ProductionMatrices,
                          VPmin::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack β, γ, δ, MaxIter_Opt, g_tol_Opt, Nb, klowOpt, khighOpt, kOlowOpt, kOhighOpt, blowOpt, bhighOpt = par
    @unpack GR_k, GR_kO, GR_b, GR_z, GR_p = GRIDS
    @unpack MAT_YR, GR_X = PROD_MATS
    #Setup intermpolation object for output
    itp_YR=LinearInterpolation(GR_X[:,kO_ind,k_ind,z_ind,p_ind,n_ind], MAT_YR[:,kO_ind,k_ind,z_ind,p_ind,n_ind], extrapolation_bc = Interpolations.Line())
    #Get guess and bounds for bprime
    bpol=GridSearchOverB(n_ind,p_ind,z_ind,k_ind,kO_ind,b_ind,itp_YR,SOLUTION,GRIDS,par)
    if bpol>0
        b0=GR_b[bpol]
    else
        b0=GR_b[b_ind]
    end
    #transform policy guess into reals
    X0=Array{Float64,1}(undef,3)
    X0[1]=TransformIntoReals(X0_BOUNDS[1],klowOpt,khighOpt)
    X0[2]=TransformIntoReals(X0_BOUNDS[2],kOlowOpt,kOhighOpt)
    X0[3]=TransformIntoReals(b0,blowOpt,bhighOpt)
    #Setup function handle for optimization
    z=GR_z[z_ind]
    p=GR_p[p_ind]
    k=GR_k[k_ind]
    kO=GR_kO[kO_ind]
    b=GR_b[b_ind]
    f(X::Array{Float64,1})=-ValueInRepayment(n_ind,p,z,k,kO,b,X[1],X[2],X[3],itp_YR,VPmin,SOLUTION,GRIDS,par)
    #Perform optimization with MatLab simplex
    inner_optimizer = NelderMead()
    res=optimize(f,X0,inner_optimizer)#,
                 #Optim.Options(g_tol = g_tol_Opt,iterations = MaxIter_Opt))
    #Transform optimizer into bounds
    kprime=TransformIntoBounds(Optim.minimizer(res)[1],klowOpt,khighOpt)
    kOprime=TransformIntoBounds(Optim.minimizer(res)[2],kOlowOpt,kOhighOpt)
    bprime=TransformIntoBounds(Optim.minimizer(res)[3],blowOpt,bhighOpt)
    return -Optim.minimum(res), kprime, kOprime, bprime
end

###############################################################################
#Update default
###############################################################################
function Expectation_Default_z_GivenP(z_ind::Int64,pPrime::Float64,n_ind::Int64,
                                      kprime::Float64,kOprime::Float64,
                                      SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, dist_ϵz = par
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    @unpack itp_VD = SOLUTION
    funVD(zprime::Float64)=min(0.0,itp_VD(kOprime,kprime,zprime,pPrime,n_ind))
    pdf_VD=PDFz[z_ind,:] .* funVD.(ZPRIME[z_ind,:])
    return dot(ϵz_weights,pdf_VD)/FacQz
end

function Expectation_Default(z_ind::Int64,p_ind::Int64,n_ind::Int64,
                                 kprime::Float64,kOprime::Float64,
                                 SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_p, μ_p, dist_ϵp = par
    @unpack ϵp_weights, PPRIME, PDFp, FacQp = GRIDS
    funVD(pPrime::Float64)=Expectation_Default_z_GivenP(z_ind,pPrime,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    pdf_VD=PDFp[p_ind,:] .* funVD.(PPRIME[p_ind,:])
    return dot(ϵp_weights,pdf_VD)/FacQp
end

function UpdateDefault!(PROD_MATS::ProductionMatrices,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Np, Nk, NkO = par
    @unpack GR_k, GR_kO, GR_z, GR_p, PI_n = GRIDS
    #Assign shared arrays
    sVD=SharedArray{Float64,5}(NkO,Nk,Nz,Np,Nn)
    skprime_D=SharedArray{Float64,5}(NkO,Nk,Nz,Np,Nn)
    skOprime_D=SharedArray{Float64,5}(NkO,Nk,Nz,Np,Nn)
    sEVD=SharedArray{Float64,5}(NkO,Nk,Nz,Np,Nn)
    #Loop over all states to fill array of VD
    VDmin=minimum(SOLUTION.VD)
    @sync @distributed for I in CartesianIndices(sVD)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        if (Nn==2 && n_ind==1) || Nn>2
            X0=[GR_k[k_ind]; GR_kO[kO_ind]]
            sVD[I], skprime_D[I], skOprime_D[I]=OptimInDefault(X0,n_ind,p_ind,z_ind,k_ind,kO_ind,PROD_MATS,VDmin,SOLUTION,GRIDS,par)
        end
    end
    if Nn==2
        sVD[:,:,:,:,2].=sVD[:,:,:,:,1]
        skprime_D[:,:,:,:,2].=skprime_D[:,:,:,:,1]
        skOprime_D[:,:,:,:,2].=skOprime_D[:,:,:,:,1]
    end
    SOLUTION.VD .= sVD
    SOLUTION.kprime_D .= skprime_D
    SOLUTION.kOprime_D .= skOprime_D
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_kprime_D=CreateInterpolation_PolicyFunctions(SOLUTION.kprime_D,true,GRIDS)
    SOLUTION.itp_kOprime_D=CreateInterpolation_PolicyFunctions(SOLUTION.kOprime_D,true,GRIDS)
    #Loop over all states to compute expectations over p and z
    @sync @distributed for I in CartesianIndices(sEVD)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        # p=GR_p[p_ind]
        # z=GR_z[z_ind]
        kprime=GR_k[k_ind]
        kOprime=GR_kO[kO_ind]
        # sEVD[I]=Expectation_Default(z,p,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
        sEVD[I]=Expectation_Default(z_ind,p_ind,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    end
    #Loop over all states to compute expectations over n
    for I in CartesianIndices(sEVD)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        SOLUTION.EVD[I]=dot(sEVD[kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
    end
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    return nothing
end

function UpdateDefault_NotParallel!(PROD_MATS::ProductionMatrices,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Np, Nk, NkO = par
    @unpack GR_k, GR_kO, GR_z, GR_p, PI_n = GRIDS
    #Loop over all states to fill array of VD
    VDmin=minimum(SOLUTION.VD)
    for I in CartesianIndices(SOLUTION.VD)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        if (Nn==2 && n_ind==1) || Nn>2
            X0=[GR_k[k_ind]; GR_kO[kO_ind]]
            SOLUTION.VD[I], SOLUTION.kprime_D[I], SOLUTION.kOprime_D[I]=OptimInDefault(X0,n_ind,p_ind,z_ind,k_ind,kO_ind,PROD_MATS,VDmin,SOLUTION,GRIDS,par)
        end
    end
    if Nn==2
        SOLUTION.VD[:,:,:,:,2].=SOLUTION.VD[:,:,:,:,1]
        SOLUTION.kprime_D[:,:,:,:,2].=SOLUTION.kprime_D[:,:,:,:,1]
        SOLUTION.kOprime_D[:,:,:,:,2].=SOLUTION.kOprime_D[:,:,:,:,1]
    end
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_kprime_D=CreateInterpolation_PolicyFunctions(SOLUTION.kprime_D,true,GRIDS)
    SOLUTION.itp_kOprime_D=CreateInterpolation_PolicyFunctions(SOLUTION.kOprime_D,true,GRIDS)
    #Loop over all states to compute expectations over p and z
    sEVD=Array{Float64,5}(undef,NkO,Nk,Nz,Np,Nn)
    for I in CartesianIndices(sEVD)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        # p=GR_p[p_ind]
        # z=GR_z[z_ind]
        kprime=GR_k[k_ind]
        kOprime=GR_kO[kO_ind]
        # sEVD[I]=Expectation_Default(z,p,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
        sEVD[I]=Expectation_Default(z_ind,p_ind,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    end
    #Loop over all states to compute expectations over n
    for I in CartesianIndices(sEVD)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        SOLUTION.EVD[I]=dot(sEVD[kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
    end
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    return nothing
end

###############################################################################
#Update repayment
###############################################################################
function Expectation_Repayment_z_GivenP(z_ind::Int64,pPrime::Float64,n_ind::Int64,
                                        kprime::Float64,kOprime::Float64,bprime::Float64,
                                        SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, dist_ϵz = par
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    @unpack itp_V = SOLUTION
    funV(zprime::Float64)=min(0.0,itp_V(bprime,kOprime,kprime,zprime,pPrime,n_ind))
    pdf_VD=PDFz[z_ind,:] .* funV.(ZPRIME[z_ind,:])
    return dot(ϵz_weights,pdf_VD)/FacQz
end

function Expectation_Repayment(z_ind::Int64,p_ind::Int64,n_ind::Int64,
                               kprime::Float64,kOprime::Float64,bprime::Float64,
                               SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_p, μ_p, dist_ϵp = par
    @unpack ϵp_weights, PPRIME, PDFp, FacQp = GRIDS
    funV(pPrime::Float64)=Expectation_Repayment_z_GivenP(z_ind,pPrime,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
    pdf_V=PDFp[p_ind,:] .* funV.(PPRIME[p_ind,:])
    return dot(ϵp_weights,pdf_V)/FacQp
end

function TradedIncomeGivenPolicy(n_ind::Int64,p_ind::Int64,z_ind::Int64,
                                 k_ind::Int64,kO_ind::Int64,b_ind::Int64,X::Float64,
                                 PROD_MATS::ProductionMatrices,GRIDS::Grids,par::Pars)
    @unpack γ, κ = par
    @unpack GR_k, GR_kO, GR_b, GR_n, GR_z, GR_p = GRIDS
    @unpack MAT_lnYT, GR_X = PROD_MATS
    itp_lnYT=LinearInterpolation(GR_X[:,kO_ind,k_ind,z_ind,p_ind,n_ind], MAT_lnYT[:,kO_ind,k_ind,z_ind,p_ind,n_ind], extrapolation_bc = Interpolations.Line())
    return itp_lnYT(X)
end

function UpdateRepayment!(PROD_MATS::ProductionMatrices,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Np, Nk, NkO, Nb, γ, κ = par
    @unpack GR_k, GR_kO, GR_b, PI_n, GR_z, GR_p = GRIDS
    @unpack kprime_D, kOprime_D, itp_q1 = SOLUTION
    VPmin=minimum(SOLUTION.VP)
    #Allocate shared arrays
    sVP=SharedArray{Float64,6}(Nb,NkO,Nk,Nz,Np,Nn)
    sV=SharedArray{Float64,6}(Nb,NkO,Nk,Nz,Np,Nn)
    sEV=SharedArray{Float64,6}(Nb,NkO,Nk,Nz,Np,Nn)
    skprime=SharedArray{Float64,6}(Nb,NkO,Nk,Nz,Np,Nn)
    skOprime=SharedArray{Float64,6}(Nb,NkO,Nk,Nz,Np,Nn)
    sbprime=SharedArray{Float64,6}(Nb,NkO,Nk,Nz,Np,Nn)
    #Loop over all states to fill value of repayment
    @sync @distributed for I in CartesianIndices(sVP)
        (b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        if (Nn==2 && n_ind==1) || Nn>2
            z=GR_z[z_ind]
            p=GR_p[p_ind]
            X0=[GR_k[k_ind]; GR_kO[kO_ind]; GR_b[b_ind]]
            sVP[I], skprime[I], skOprime[I], sbprime[I]=OptimInRepayment(X0,n_ind,p_ind,z_ind,k_ind,kO_ind,b_ind,PROD_MATS,VPmin,SOLUTION,GRIDS,par)
            if sVP[I]<SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]
                sV[I]=SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]
            else
                sV[I]=sVP[I]
            end
        end
    end
    if Nn==2
        sVP[:,:,:,:,:,2].=sVP[:,:,:,:,:,1]
        skprime[:,:,:,:,:,2].=skprime[:,:,:,:,:,1]
        skOprime[:,:,:,:,:,2].=skOprime[:,:,:,:,:,1]
        sbprime[:,:,:,:,:,2].=sbprime[:,:,:,:,:,1]
        sV[:,:,:,:,:,2].=sV[:,:,:,:,:,1]
    end
    SOLUTION.VP .= sVP
    SOLUTION.V .= sV
    SOLUTION.kprime .= skprime
    SOLUTION.kOprime .= skOprime
    SOLUTION.bprime .= sbprime
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_kprime=CreateInterpolation_PolicyFunctions(SOLUTION.kprime,false,GRIDS)
    SOLUTION.itp_kOprime=CreateInterpolation_PolicyFunctions(SOLUTION.kOprime,false,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_PolicyFunctions(SOLUTION.bprime,false,GRIDS)
    #Loop over all states to compute expectation of EV over p' and z'
    @sync @distributed for I in CartesianIndices(sEV)
        (b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        # p=GR_p[p_ind]
        # z=GR_z[z_ind]
        kprime=GR_k[k_ind]
        kOprime=GR_kO[kO_ind]
        bprime=GR_b[b_ind]
        sEV[I]=Expectation_Repayment(z_ind,p_ind,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
    end
    #Loop over all states to compute expectation of EV over n'
    for I in CartesianIndices(sEV)
        (b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        SOLUTION.EV[I]=dot(sEV[b_ind,kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
    end
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    return nothing
end

function UpdateRepayment_NotParallel!(PROD_MATS::ProductionMatrices,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Np, Nk, NkO, Nb = par
    @unpack GR_k, GR_kO, GR_b, PI_n, GR_z, GR_p = GRIDS
    @unpack kprime_D, kOprime_D = SOLUTION
    VPmin=minimum(SOLUTION.VP)
    #Loop over all states to fill value of repayment
    for I in CartesianIndices(SOLUTION.VP)
        (b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        if (Nn==2 && n_ind==1) || Nn>2
            X0=[GR_k[k_ind]; GR_kO[kO_ind]; GR_b[b_ind]]
            SOLUTION.VP[I], SOLUTION.kprime[I], SOLUTION.kOprime[I], SOLUTION.bprime[I]=OptimInRepayment(X0,n_ind,p_ind,z_ind,k_ind,kO_ind,b_ind,PROD_MATS,VPmin,SOLUTION,GRIDS,par)
            if SOLUTION.VP[I]<SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]
                SOLUTION.V[I]=SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]
            else
                SOLUTION.V[I]=SOLUTION.VP[I]
            end
        end
    end
    if Nn==2
        SOLUTION.VP[:,:,:,:,:,2].=SOLUTION.VP[:,:,:,:,:,1]
        SOLUTION.kprime[:,:,:,:,:,2].=SOLUTION.kprime[:,:,:,:,:,1]
        SOLUTION.kOprime[:,:,:,:,:,2].=SOLUTION.kOprime[:,:,:,:,:,1]
        SOLUTION.bprime[:,:,:,:,:,2].=SOLUTION.bprime[:,:,:,:,:,1]
        SOLUTION.V[:,:,:,:,:,2].=SOLUTION.V[:,:,:,:,:,1]
    end
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_kprime=CreateInterpolation_PolicyFunctions(SOLUTION.kprime,false,GRIDS)
    SOLUTION.itp_kOprime=CreateInterpolation_PolicyFunctions(SOLUTION.kOprime,false,GRIDS)
    SOLUTION.itp_bprime=CreateInterpolation_PolicyFunctions(SOLUTION.bprime,false,GRIDS)
    #Loop over all states to compute expectation of EV over p' and z'
    sEV=Array{Float64,6}(undef,Nb,NkO,Nk,Nz,Np,Nn)
    for I in CartesianIndices(sEV)
        (b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        # p=GR_p[p_ind]
        # z=GR_z[z_ind]
        kprime=GR_k[k_ind]
        kOprime=GR_kO[kO_ind]
        bprime=GR_b[b_ind]
        sEV[I]=Expectation_Repayment(z_ind,p_ind,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
    end
    #Loop over all states to compute expectation of EV over n'
    for I in CartesianIndices(sEV)
        (b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        SOLUTION.EV[I]=dot(sEV[b_ind,kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
    end
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    return nothing
end

###############################################################################
#Update price
###############################################################################
function BondsPayoff(ϵz::Float64,ϵp::Float64,z::Float64,p::Float64,n_ind::Int64,
                     kprime::Float64,kOprime::Float64,bprime::Float64,EyT::Float64,
                     σT::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, ρ_p, μ_p, γ, κ, cmin = par
    @unpack itp_VP, itp_VD, itp_q1, itp_bprime, itp_kprime, itp_kOprime, itp_lnyT = SOLUTION
    @unpack GR_n = GRIDS
    zprime=min(exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z)+ϵz),par.zhigh)
    pprime=min(exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p)+ϵp),par.phigh)
    nprime=GR_n[n_ind]
    if itp_VD(kOprime,kprime,zprime,pprime,n_ind)>itp_VP(bprime,kOprime,kprime,zprime,pprime,n_ind)
        return 0.0
    else
        kk=itp_kprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        kOkO=itp_kOprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        bb=itp_bprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        qq=itp_q1(bb,kOkO,kk,zprime,pprime,n_ind)
        lnyTprime=itp_lnyT(kOprime,kprime,zprime,pprime,n_ind)
        SDF=SDF_Lenders(lnyTprime,EyT,σT,par)
        return SDF*(γ+(1.0-γ)*(κ+qq))
    end
end

function Integrate_BondsPrice_z_GivenP(ϵp::Float64,z::Float64,p::Float64,n_ind::Int64,
                                       kprime::Float64,kOprime::Float64,bprime::Float64,EyT::Float64,
                                       σT::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵz = par
    @unpack ϵz_nodes, ϵz_weights, FacQz = GRIDS
    fun(ϵz::Float64)=pdf(dist_ϵz,ϵz)*BondsPayoff(ϵz,ϵp,z,p,n_ind,kprime,kOprime,bprime,EyT,σT,SOLUTION,GRIDS,par)
    return dot(ϵz_weights,fun.(ϵz_nodes))/FacQz
end

function Integrate_BondsPrice(z_ind::Int64,p_ind::Int64,n_ind::Int64,
                              k_ind::Int64,kO_ind::Int64,b_ind::Int64,
                              SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵp, Nn = par
    @unpack ϵp_nodes, ϵp_weights, FacQp, GR_n, PI_n = GRIDS
    @unpack σyT, EyT = SOLUTION
    #Current z and p for conditional expectation
    z=GRIDS.GR_z[z_ind]
    p=GRIDS.GR_p[p_ind]
    #Policies today that affect default tomorrow
    kprime=GRIDS.GR_k[k_ind]
    kOprime=GRIDS.GR_kO[kO_ind]
    bprime=GRIDS.GR_b[b_ind]
    #Expectation and variance of yT' conditional on current z, p, n
    #and on choices kO' and k'
    σT=σyT[kO_ind,k_ind,z_ind,p_ind,n_ind]
    ET=EyT[kO_ind,k_ind,z_ind,p_ind,n_ind]
    #Integrate over z', p', n'
    Eq1=0.0
    for nprime_ind in 1:Nn
        πn=PI_n[n_ind,nprime_ind]
        if πn>0.0
            fun(ϵp::Float64)=pdf(dist_ϵp,ϵp)*Integrate_BondsPrice_z_GivenP(ϵp,z,p,nprime_ind,kprime,kOprime,bprime,ET,σT,SOLUTION,GRIDS,par)
            Eq1=Eq1+πn*(dot(ϵp_weights,fun.(ϵp_nodes))/FacQp)
        end
    end
    return Eq1
end

function UpdateBondsPrice!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Np, Nk, NkO, Nb = par
    @unpack GR_z, GR_p, GR_k, GR_kO, GR_b, PI_n = GRIDS
    #Allocate shared array
    sq1=SharedArray{Float64,6}(Nb,NkO,Nk,Nz,Np,Nn)
    #Loop over all states to compute expectation over z' and p'
    @sync @distributed for I in CartesianIndices(sq1)
        (b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        if (Nn==2 && n_ind==1) || Nn>2
            p=GR_p[p_ind]
            z=GR_z[z_ind]
            sq1[I]=Integrate_BondsPrice(z_ind,p_ind,n_ind,k_ind,kO_ind,b_ind,SOLUTION,GRIDS,par)
        end
    end
    if Nn==2
        sq1[:,:,:,:,:,2].=sq1[:,:,:,:,:,1]
    end
    SOLUTION.q1 .= sq1
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)
    return nothing
end

function UpdateBondsPrice_NotParallel!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Np, Nk, NkO, Nb = par
    @unpack GR_z, GR_p, GR_k, GR_kO, GR_b, PI_n = GRIDS
    #Loop over all states to compute expectation over z' and p'
    for I in CartesianIndices(SOLUTION.q1)
        (b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        if (Nn==2 && n_ind==1) || Nn>2
            p=GR_p[p_ind]
            z=GR_z[z_ind]
            SOLUTION.q1[I]=Integrate_BondsPrice(z_ind,p_ind,n_ind,k_ind,kO_ind,b_ind,SOLUTION,GRIDS,par)
        end
    end
    if Nn==2
        SOLUTION.q1[:,:,:,:,:,2].=SOLUTION.q1[:,:,:,:,:,1]
    end
    SOLUTION.itp_q1=CreateInterpolation_Price(SOLUTION.q1,GRIDS)
    return nothing
end

###############################################################################
#Compute risk premium
###############################################################################
#Expected AF bond payoff
function Bond_AF_Payoff(ϵz::Float64,ϵp::Float64,z::Float64,p::Float64,n_ind::Int64,
                     kprime::Float64,kOprime::Float64,bprime::Float64,
                     SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, ρ_p, μ_p, γ, κ, cmin, r_star = par
    @unpack itp_VP, itp_VD, itp_q1, itp_bprime, itp_kprime, itp_kOprime = SOLUTION
    @unpack GR_n = GRIDS
    zprime=min(exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z)+ϵz),par.zhigh)
    pprime=min(exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p)+ϵp),par.phigh)
    nprime=GR_n[n_ind]
    if itp_VD(kOprime,kprime,zprime,pprime,n_ind)>itp_VP(bprime,kOprime,kprime,zprime,pprime,n_ind)
        return 0.0
    else
        kk=itp_kprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        kOkO=itp_kOprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        bb=itp_bprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        qq=itp_q1(bb,kOkO,kk,zprime,pprime,n_ind)
        return exp(-r_star)*(γ+(1.0-γ)*(κ+qq))
    end
end

function Integrate_Bond_AF_Payoff_z_GivenP(ϵp::Float64,z::Float64,p::Float64,n_ind::Int64,
                                       kprime::Float64,kOprime::Float64,bprime::Float64,
                                       SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵz = par
    @unpack ϵz_nodes, ϵz_weights, FacQz = GRIDS
    fun(ϵz::Float64)=pdf(dist_ϵz,ϵz)*Bond_AF_Payoff(ϵz,ϵp,z,p,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
    return dot(ϵz_weights,fun.(ϵz_nodes))/FacQz
end

function Integrate_Bond_AF_Payoff(z::Float64,p::Float64,n_ind::Int64,
                              kprime::Float64,kOprime::Float64,bprime::Float64,
                              SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵp, Nn = par
    @unpack ϵp_nodes, ϵp_weights, FacQp, GR_n, PI_n = GRIDS
    #Integrate over z', p', n'
    Eq1=0.0
    for nprime_ind in 1:Nn
        πn=PI_n[n_ind,nprime_ind]
        if πn>0.0
            fun(ϵp::Float64)=pdf(dist_ϵp,ϵp)*Integrate_Bond_AF_Payoff_z_GivenP(ϵp,z,p,nprime_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
            Eq1=Eq1+πn*(dot(ϵp_weights,fun.(ϵp_nodes))/FacQp)
        end
    end
    return Eq1
end

#Expected bond payoff with SDF
function Bond_RP_Payoff(ϵz::Float64,ϵp::Float64,z::Float64,p::Float64,n_ind::Int64,
                     kprime::Float64,kOprime::Float64,bprime::Float64,EyT::Float64,
                     σT::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, ρ_p, μ_p, γ, κ, cmin, r_star = par
    @unpack itp_VP, itp_VD, itp_q1, itp_bprime, itp_kprime, itp_kOprime, itp_lnyT = SOLUTION
    @unpack GR_n = GRIDS
    zprime=min(exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z)+ϵz),par.zhigh)
    pprime=min(exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p)+ϵp),par.phigh)
    nprime=GR_n[n_ind]
    if itp_VD(kOprime,kprime,zprime,pprime,n_ind)>itp_VP(bprime,kOprime,kprime,zprime,pprime,n_ind)
        return 0.0
    else
        kk=itp_kprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        kOkO=itp_kOprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        bb=itp_bprime(bprime,kOprime,kprime,zprime,pprime,n_ind)
        qq=itp_q1(bb,kOkO,kk,zprime,pprime,n_ind)
        lnyTprime=itp_lnyT(kOprime,kprime,zprime,pprime,n_ind)
        SDF=SDF_Lenders(lnyTprime,EyT,σT,par)
        return SDF*(γ+(1.0-γ)*(κ+qq))
    end
end

function Integrate_Bond_RP_Payoff_z_GivenP(ϵp::Float64,z::Float64,p::Float64,n_ind::Int64,
                                       kprime::Float64,kOprime::Float64,bprime::Float64,EyT::Float64,
                                       σT::Float64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵz = par
    @unpack ϵz_nodes, ϵz_weights, FacQz = GRIDS
    fun(ϵz::Float64)=pdf(dist_ϵz,ϵz)*Bond_RP_Payoff(ϵz,ϵp,z,p,n_ind,kprime,kOprime,bprime,EyT,σT,SOLUTION,GRIDS,par)
    return dot(ϵz_weights,fun.(ϵz_nodes))/FacQz
end

function Integrate_Bond_RP_Payoff(z::Float64,p::Float64,n_ind::Int64,
                              kprime::Float64,kOprime::Float64,bprime::Float64,
                              itp_σyT,itp_EyT,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵp, Nn = par
    @unpack ϵp_nodes, ϵp_weights, FacQp, GR_n, PI_n = GRIDS
    #Expectation and variance of yT' conditional on current z, p, n
    #and on choices kO' and k'
    σT=itp_σyT(kOprime,kprime,z,p,n_ind)
    EyT=itp_EyT(kOprime,kprime,z,p,n_ind)
    #Integrate over z', p', n'
    Eq1=0.0
    for nprime_ind in 1:Nn
        πn=PI_n[n_ind,nprime_ind]
        if πn>0.0
            fun(ϵp::Float64)=pdf(dist_ϵp,ϵp)*Integrate_Bond_RP_Payoff_z_GivenP(ϵp,z,p,nprime_ind,kprime,kOprime,bprime,EyT,σT,SOLUTION,GRIDS,par)
            Eq1=Eq1+πn*(dot(ϵp_weights,fun.(ϵp_nodes))/FacQp)
        end
    end
    return Eq1
end

#Default probability next period
function Default_Choice(ϵz::Float64,ϵp::Float64,z::Float64,p::Float64,n_ind::Int64,
                     kprime::Float64,kOprime::Float64,bprime::Float64,
                     SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack ρ_z, μ_z, ρ_p, μ_p = par
    @unpack itp_VP, itp_VD = SOLUTION
    @unpack GR_n = GRIDS
    zprime=min(exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z)+ϵz),par.zhigh)
    pprime=min(exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p)+ϵp),par.phigh)
    nprime=GR_n[n_ind]
    if itp_VD(kOprime,kprime,zprime,pprime,n_ind)>itp_VP(bprime,kOprime,kprime,zprime,pprime,n_ind)
        return 1.0
    else
        return 0.0
    end
end

function Integrate_DefChoice_z_GivenP(ϵp::Float64,z::Float64,p::Float64,n_ind::Int64,
                                       kprime::Float64,kOprime::Float64,bprime::Float64,
                                       SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵz = par
    @unpack ϵz_nodes, ϵz_weights, FacQz = GRIDS
    fun(ϵz::Float64)=pdf(dist_ϵz,ϵz)*Default_Choice(ϵz,ϵp,z,p,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
    return dot(ϵz_weights,fun.(ϵz_nodes))/FacQz
end

function Integrate_DefChoice(z::Float64,p::Float64,n_ind::Int64,
                              kprime::Float64,kOprime::Float64,bprime::Float64,
                              SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵp, Nn = par
    @unpack ϵp_nodes, ϵp_weights, FacQp, GR_n, PI_n = GRIDS
    #Integrate over z', p', n'
    PrDef=0.0
    for nprime_ind in 1:Nn
        πn=PI_n[n_ind,nprime_ind]
        if πn>0.0
            fun(ϵp::Float64)=pdf(dist_ϵp,ϵp)*Integrate_DefChoice_z_GivenP(ϵp,z,p,nprime_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
            PrDef=PrDef+πn*(dot(ϵp_weights,fun.(ϵp_nodes))/FacQp)
        end
    end
    return PrDef
end

#Compute risk premium
function Compute_Risk_Premium(z::Float64,p::Float64,n_ind::Int64,
                             kprime::Float64,kOprime::Float64,bprime::Float64,
                             itp_σyT,itp_EyT,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack dist_ϵp, Nn, γ, κ = par
    @unpack itp_q1 = SOLUTION
    #Expectation and variance of yT' conditional on current z, p, n
    #and on choices kO' and k'
    qaf=Integrate_Bond_AF_Payoff(z,p,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
    # qt=itp_q1(bprime,kOprime,kprime,z,p,n_ind)
    qt=Integrate_Bond_RP_Payoff(z,p,n_ind,kprime,kOprime,bprime,itp_σyT,itp_EyT,SOLUTION,GRIDS,par)
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
function UpdateMoments_yT!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Np, Nk, NkO = par
    @unpack GR_z, GR_p, GR_k, GR_kO, PI_n = GRIDS
    #Allocate shared arrays
    sEyT=SharedArray{Float64,5}(NkO,Nk,Nz,Np,Nn)
    #Loop over all states to compute expectation over z' and p'
    #Compute the expectation first
    @sync @distributed for I in CartesianIndices(sEyT)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        kprime=GR_k[k_ind]
        kOprime=GR_kO[kO_ind]
        sEyT[I]=ExpectedTradableIncome(z_ind,p_ind,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    end
    for I in CartesianIndices(sEyT)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        SOLUTION.EyT[I]=dot(sEyT[kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
    end
    #Compute the variance after
    sσyT=SharedArray{Float64,5}(NkO,Nk,Nz,Np,Nn)
    @sync @distributed for I in CartesianIndices(sEyT)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        kprime=GR_k[k_ind]
        kOprime=GR_kO[kO_ind]
        sσyT[I]=VarTradableIncome(z_ind,p_ind,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    end
    for I in CartesianIndices(sEyT)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        eyt=SOLUTION.EyT[I]
        SOLUTION.σyT[I]=sqrt(dot(sσyT[kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])-(eyt^2.0))
    end
    return nothing
end

function UpdateMoments_yT_NotParallel!(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Np, Nk, NkO = par
    @unpack GR_z, GR_p, GR_k, GR_kO, PI_n = GRIDS
    #Allocate shared arrays
    sEyT=Array{Float64,5}(undef,NkO,Nk,Nz,Np,Nn)
    #Loop over all states to compute expectation over z' and p'
    #Compute the expectation first
    for I in CartesianIndices(sEyT)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        kprime=GR_k[k_ind]
        kOprime=GR_kO[kO_ind]
        sEyT[I]=ExpectedTradableIncome(z_ind,p_ind,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    end
    for I in CartesianIndices(sEyT)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        SOLUTION.EyT[I]=dot(sEyT[kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
    end
    #Compute the variance after
    sσyT=Array{Float64,5}(undef,NkO,Nk,Nz,Np,Nn)
    for I in CartesianIndices(sEyT)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        kprime=GR_k[k_ind]
        kOprime=GR_kO[kO_ind]
        sσyT[I]=VarTradableIncome(z_ind,p_ind,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
    end
    for I in CartesianIndices(sEyT)
        (kO_ind,k_ind,z_ind,p_ind,n_ind)=Tuple(I)
        eyt=SOLUTION.EyT[I]
        SOLUTION.σyT[I]=sqrt(dot(sσyT[kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])-(eyt^2.0))
    end
    return nothing
end

function UpdateSolution!(PROD_MATS::ProductionMatrices,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault!(PROD_MATS,SOLUTION,GRIDS,par)
    UpdateRepayment!(PROD_MATS,SOLUTION,GRIDS,par)
    UpdateBondsPrice!(SOLUTION,GRIDS,par)
    return nothing
end

function UpdateSolution_NotParallel!(PROD_MATS::ProductionMatrices,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    UpdateDefault_NotParallel!(PROD_MATS,SOLUTION,GRIDS,par)
    UpdateRepayment_NotParallel!(PROD_MATS,SOLUTION,GRIDS,par)
    UpdateBondsPrice_NotParallel!(SOLUTION,GRIDS,par)
    return nothing
end

function ComputeDistance_q(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution,par::Pars)
    @unpack Tol_q = par
    dst_q=maximum(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1))
    NotConv=sum(abs.(SOLUTION_CURRENT.q1 .- SOLUTION_NEXT.q1) .> Tol_q)
    NotConvPct=100.0*NotConv/length(SOLUTION_CURRENT.q1)
    return dst_q, round(NotConvPct,digits=2)
end

function ComputeRelativeDistance(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution)
    dst_D=100.0*maximum(abs.(SOLUTION_CURRENT.VD .- SOLUTION_NEXT.VD) ./ abs.(SOLUTION_CURRENT.VD))
    dst_V=100.0*maximum(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V) ./ abs.(SOLUTION_CURRENT.V))
    return round(abs(dst_D),digits=4), round(abs(dst_V),digits=4)
end

function ComputeDistance_V(SOLUTION_CURRENT::Solution,SOLUTION_NEXT::Solution)
    dst_D=maximum(abs.(SOLUTION_CURRENT.VD .- SOLUTION_NEXT.VD))
    dst_V=maximum(abs.(SOLUTION_CURRENT.V .- SOLUTION_NEXT.V))
    return round(abs(dst_D),digits=4), round(abs(dst_V),digits=4)
end

function InitiateEmptySolution(GRIDS::Grids,par::Pars)
    @unpack Nn, Np, Nz, Nk, NkO, Nb, γ, κ, r_star = par
    ### Allocate all values to object
    VD=zeros(Float64,NkO,Nk,Nz,Np,Nn)
    VP=zeros(Float64,Nb,NkO,Nk,Nz,Np,Nn)
    V=zeros(Float64,Nb,NkO,Nk,Nz,Np,Nn)
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)
    #Expectations and price
    EVD=zeros(Float64,NkO,Nk,Nz,Np,Nn)
    EV=zeros(Float64,Nb,NkO,Nk,Nz,Np,Nn)
    er=exp(r_star)
    qbar=(γ+(1.0-γ)*κ)/(er+γ-1.0)
    q1=qbar*ones(Float64,Nb,NkO,Nk,Nz,Np,Nn)
    EyT=zeros(Float64,NkO,Nk,Nz,Np,Nn)
    σyT=zeros(Float64,NkO,Nk,Nz,Np,Nn)
    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)
    #Policy functions
    kprime_D=zeros(Float64,NkO,Nk,Nz,Np,Nn)
    kOprime_D=zeros(Float64,NkO,Nk,Nz,Np,Nn)
    kprime=zeros(Float64,Nb,NkO,Nk,Nz,Np,Nn)
    kOprime=zeros(Float64,Nb,NkO,Nk,Nz,Np,Nn)
    bprime=zeros(Float64,Nb,NkO,Nk,Nz,Np,Nn)
    lnyT=zeros(Float64,NkO,Nk,Nz,Np,Nn)
    itp_kprime_D=CreateInterpolation_PolicyFunctions(kprime_D,true,GRIDS)
    itp_kOprime_D=CreateInterpolation_PolicyFunctions(kOprime_D,true,GRIDS)
    itp_kprime=CreateInterpolation_PolicyFunctions(kprime,false,GRIDS)
    itp_kOprime=CreateInterpolation_PolicyFunctions(kOprime,false,GRIDS)
    itp_bprime=CreateInterpolation_PolicyFunctions(bprime,false,GRIDS)
    itp_lnyT=CreateInterpolation_log_yT(lnyT,GRIDS)
    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,EyT,σyT,kprime_D,kOprime_D,kprime,kOprime,bprime,lnyT,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kprime_D,itp_kOprime_D,itp_kprime,itp_kOprime,itp_bprime,itp_lnyT)
end

function SolutionEndOfTime(PROD_MATS::ProductionMatrices,GRIDS::Grids,par::Pars)
    @unpack Nn, Np, Nz, Nk, NkO, Nb, δ, φ, φO, γ, κ, cmin, r_star = par
    @unpack GR_n, GR_z, GR_p, GR_k, GR_kO, GR_b, PI_n = GRIDS
    @unpack MAT_YD, MAT_YR, GR_X, MAT_lnYT = PROD_MATS
    SOLUTION=InitiateEmptySolution(GRIDS,par)
    er=exp(r_star)
    qbar=(γ+(1.0-γ)*κ)/(er+γ-1.0)
    #Value Functions in very last period
    for n_ind in 1:Nn
        n=GR_n[n_ind]
        for p_ind in 1:Np
            p=GR_p[p_ind]
            for z_ind in 1:Nz
                z=GR_z[z_ind]
                zD=zDefault(z,par)
                for k_ind in 1:Nk
                    k=GR_k[k_ind]
                    if φ>1.0
                        kprime=((φ-1.0)/φ)*k
                    else
                        kprime=0.0
                    end
                    for kO_ind in 1:NkO
                        kO=GR_kO[kO_ind]
                        if φO>1.0
                            kOprime=((φO-1.0)/φO)*kO
                        else
                            kOprime=0.0
                        end
                        yD=MAT_YD[kO_ind,k_ind,z_ind,p_ind,n_ind]
                        cD=ConsNet(yD,k,kO,kprime,kOprime,par)
                        if cD>cmin && yD>0.0
                            SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]=Utility(cD,par)
                        else
                            SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]=Utility(cmin,par)
                        end
                        SOLUTION.lnyT[kO_ind,k_ind,z_ind,p_ind,n_ind]=MAT_lnYT[kO_ind,k_ind,z_ind,p_ind,n_ind]
                        for b_ind in 1:Nb
                            b=GR_b[b_ind]
                            X=-(γ+(1.0-γ)*(κ+qbar))*b
                            yP=Production(z,p,n,k,kO,X,par)
                            cP=ConsNet(yP,k,kO,kprime,kOprime,par)
                            if cP>0.0 && yP>0.0
                                SOLUTION.VP[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=Utility(cP,par)
                            else
                                SOLUTION.VP[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=Utility(cmin,par)-sqrt(eps(Float64))
                            end
                            if SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]>SOLUTION.VP[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]
                                SOLUTION.V[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]
                            else
                                SOLUTION.V[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=SOLUTION.VP[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]
                            end
                        end
                    end
                end
            end
        end
    end
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_lnyT=CreateInterpolation_log_yT(SOLUTION.lnyT,GRIDS)
    #Loop over all states to compute expectations over p and z
    for n_ind in 1:Nn
        for p_ind in 1:Np
            p=GR_p[p_ind]
            for z_ind in 1:Nz
                z=GR_z[z_ind]
                for k_ind in 1:Nk
                    kprime=GR_k[k_ind]
                    for kO_ind in 1:NkO
                        kOprime=GR_kO[kO_ind]
                        SOLUTION.EVD[kO_ind,k_ind,z_ind,p_ind,n_ind]=Expectation_Default(z_ind,p_ind,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
                    end
                end
            end
        end
    end
    #Loop over all states to compute expectations over n
    for n_ind in 1:Nn
        for p_ind in 1:Np
            for z_ind in 1:Nz
                for k_ind in 1:Nk
                    for kO_ind in 1:NkO
                        SOLUTION.EVD[kO_ind,k_ind,z_ind,p_ind,n_ind]=dot(SOLUTION.EVD[kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
                    end
                end
            end
        end
    end
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    #Loop over all states to compute expectation of EV over p' and z'
    for n_ind in 1:Nn
        for p_ind in 1:Np
            p=GR_p[p_ind]
            for z_ind in 1:Nz
                z=GR_z[z_ind]
                for k_ind in 1:Nk
                    kprime=GR_k[k_ind]
                    for kO_ind in 1:NkO
                        kOprime=GR_kO[kO_ind]
                        for b_ind in 1:Nb
                            bprime=GR_b[b_ind]
                            SOLUTION.EV[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=Expectation_Repayment(z_ind,p_ind,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
                        end
                    end
                end
            end
        end
    end
    #Loop over all states to compute expectation of EV over n'
    for n_ind in 1:Nn
        for p_ind in 1:Np
            for z_ind in 1:Nz
                for k_ind in 1:Nk
                    for kO_ind in 1:NkO
                        for b_ind in 1:Nb
                            SOLUTION.EV[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=dot(SOLUTION.EV[b_ind,kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
                        end
                    end
                end
            end
        end
    end
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    UpdateMoments_yT!(SOLUTION,GRIDS,par)
    UpdateBondsPrice!(SOLUTION,GRIDS,par)
    return SOLUTION
end

function SolutionEndOfTime_NotParallel(PROD_MATS::ProductionMatrices,GRIDS::Grids,par::Pars)
    @unpack Nn, Np, Nz, Nk, NkO, Nb, δ, φ, φO, γ, κ, cmin, r_star = par
    @unpack GR_n, GR_z, GR_p, GR_k, GR_kO, GR_b, PI_n = GRIDS
    @unpack MAT_YD, MAT_YR, GR_X, MAT_lnYT = PROD_MATS
    SOLUTION=InitiateEmptySolution(GRIDS,par)
    er=exp(r_star)
    qbar=(γ+(1.0-γ)*κ)/(er+γ-1.0)
    #Value Functions in very last period
    for n_ind in 1:Nn
        n=GR_n[n_ind]
        for p_ind in 1:Np
            p=GR_p[p_ind]
            for z_ind in 1:Nz
                z=GR_z[z_ind]
                zD=zDefault(z,par)
                for k_ind in 1:Nk
                    k=GR_k[k_ind]
                    if φ>1.0
                        kprime=((φ-1.0)/φ)*k
                    else
                        kprime=0.0
                    end
                    for kO_ind in 1:NkO
                        kO=GR_kO[kO_ind]
                        if φO>1.0
                            kOprime=((φO-1.0)/φO)*kO
                        else
                            kOprime=0.0
                        end
                        yD=MAT_YD[kO_ind,k_ind,z_ind,p_ind,n_ind]
                        cD=ConsNet(yD,k,kO,kprime,kOprime,par)
                        if cD>cmin && yD>0.0
                            SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]=Utility(cD,par)
                        else
                            SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]=Utility(cmin,par)
                        end
                        SOLUTION.lnyT[kO_ind,k_ind,z_ind,p_ind,n_ind]=MAT_lnYT[kO_ind,k_ind,z_ind,p_ind,n_ind]
                        for b_ind in 1:Nb
                            b=GR_b[b_ind]
                            X=-(γ+(1.0-γ)*(κ+qbar))*b
                            yP=Production(z,p,n,k,kO,X,par)
                            cP=ConsNet(yP,k,kO,kprime,kOprime,par)
                            if cP>0.0 && yP>0.0
                                SOLUTION.VP[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=Utility(cP,par)
                            else
                                SOLUTION.VP[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=Utility(cmin,par)-sqrt(eps(Float64))
                            end
                            if SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]>SOLUTION.VP[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]
                                SOLUTION.V[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=SOLUTION.VD[kO_ind,k_ind,z_ind,p_ind,n_ind]
                            else
                                SOLUTION.V[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=SOLUTION.VP[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]
                            end
                        end
                    end
                end
            end
        end
    end
    SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(SOLUTION.VD,true,GRIDS)
    SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(SOLUTION.VP,false,GRIDS)
    SOLUTION.itp_V=CreateInterpolation_ValueFunctions(SOLUTION.V,false,GRIDS)
    SOLUTION.itp_lnyT=CreateInterpolation_log_yT(SOLUTION.lnyT,GRIDS)
    #Loop over all states to compute expectations over p and z
    for n_ind in 1:Nn
        for p_ind in 1:Np
            p=GR_p[p_ind]
            for z_ind in 1:Nz
                z=GR_z[z_ind]
                for k_ind in 1:Nk
                    kprime=GR_k[k_ind]
                    for kO_ind in 1:NkO
                        kOprime=GR_kO[kO_ind]
                        SOLUTION.EVD[kO_ind,k_ind,z_ind,p_ind,n_ind]=Expectation_Default(z_ind,p_ind,n_ind,kprime,kOprime,SOLUTION,GRIDS,par)
                    end
                end
            end
        end
    end
    #Loop over all states to compute expectations over n
    for n_ind in 1:Nn
        for p_ind in 1:Np
            for z_ind in 1:Nz
                for k_ind in 1:Nk
                    for kO_ind in 1:NkO
                        SOLUTION.EVD[kO_ind,k_ind,z_ind,p_ind,n_ind]=dot(SOLUTION.EVD[kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
                    end
                end
            end
        end
    end
    SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(SOLUTION.EVD,true,GRIDS)
    #Loop over all states to compute expectation of EV over p' and z'
    for n_ind in 1:Nn
        for p_ind in 1:Np
            p=GR_p[p_ind]
            for z_ind in 1:Nz
                z=GR_z[z_ind]
                for k_ind in 1:Nk
                    kprime=GR_k[k_ind]
                    for kO_ind in 1:NkO
                        kOprime=GR_kO[kO_ind]
                        for b_ind in 1:Nb
                            bprime=GR_b[b_ind]
                            SOLUTION.EV[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=Expectation_Repayment(z_ind,p_ind,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
                        end
                    end
                end
            end
        end
    end
    #Loop over all states to compute expectation of EV over n'
    for n_ind in 1:Nn
        for p_ind in 1:Np
            for z_ind in 1:Nz
                for k_ind in 1:Nk
                    for kO_ind in 1:NkO
                        for b_ind in 1:Nb
                            SOLUTION.EV[b_ind,kO_ind,k_ind,z_ind,p_ind,n_ind]=dot(SOLUTION.EV[b_ind,kO_ind,k_ind,z_ind,p_ind,:],PI_n[n_ind,:])
                        end
                    end
                end
            end
        end
    end
    SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(SOLUTION.EV,false,GRIDS)
    UpdateMoments_yT_NotParallel!(SOLUTION,GRIDS,par)
    UpdateBondsPrice_NotParallel!(SOLUTION,GRIDS,par)
    return SOLUTION
end

################################################################################
### Functions for simulations
################################################################################
@with_kw mutable struct States_TS
    z::Array{Float64,1}
    p::Array{Float64,1}
    n::Array{Int64,1}
    Def::Array{Float64,1}
    K::Array{Float64,1}
    KO::Array{Float64,1}
    B::Array{Float64,1}
    Spreads::Array{Float64,1}
end

@with_kw mutable struct Paths
    #Paths of shocks
    z::Array{Float64,1}
    p::Array{Float64,1}
    n::Array{Int64,1}
    #Paths of chosen states
    Def::Array{Float64,1}
    K::Array{Float64,1}
    KO::Array{Float64,1}
    B::Array{Float64,1}
    #Path of relevant variables
    Spreads::Array{Float64,1}
    GDP::Array{Float64,1}
    nGDP::Array{Float64,1}
    Cons::Array{Float64,1}
    Inv::Array{Float64,1}
    TB::Array{Float64,1}
    CA::Array{Float64,1}
    RER::Array{Float64,1}
    σyT::Array{Float64,1}
    RiskPremium::Array{Float64,1}
    DefPr::Array{Float64,1}
end

function Simulate_z_p_shocks(T::Int64,GRIDS::Grids,par::Pars)
    @unpack μ_z, ρ_z, dist_ϵz = par
    @unpack μ_p, ρ_p, dist_ϵp = par
    ϵz_TS=rand(dist_ϵz,T)
    ϵp_TS=rand(dist_ϵp,T)
    z_TS=Array{Float64,1}(undef,T)
    p_TS=Array{Float64,1}(undef,T)
    for t in 1:T
        if t==1
            z_TS[t]=μ_z
            p_TS[t]=μ_p
        else
            z_TS[t]=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(z_TS[t-1])+ϵz_TS[t])
            p_TS[t]=exp((1.0-ρ_p)*log(μ_p)+ρ_p*log(p_TS[t-1])+ϵp_TS[t])
        end
    end
    return z_TS, p_TS, ϵz_TS, ϵp_TS
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

function Draw_New_n(n_ind::Int64,PI_n::Array{Float64,2})
    PDF=PI_n[n_ind,:]
    CDF=Get_CDF(PDF)
    x=rand()
    n_prime=0
    for i in 1:length(CDF)
        if x<=CDF[i]
            n_prime=i
            break
        else
        end
    end
    return n_prime
end

function Simulate_Discoveries(T::Int64,PI_n::Array{Float64,2})
    X=Array{Int64,1}(undef,T)
    X[1]=1
    for t in 2:T
        X[t]=Draw_New_n(X[t-1],PI_n)
    end
    return X
end

function CalculateSpreads(itp_q1,n::Int64,p::Float64,z::Float64,kprime::Float64,kOprime::Float64,bprime::Float64,par::Pars)
    @unpack r_star, γ, κ = par
    q=max(itp_q1(bprime,kOprime,kprime,z,p,n),1e-1)
    ib=-log(q/(γ+(1.0-γ)*(κ+q)))
    ia=((1.0+ib)^1.0)-1.0
    rf=((1.0+r_star)^1.0)-1.0
    return 100.0*(ia-rf)
end

function SimulateStates_Long(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, Tsim, γ, κ, θ = par
    @unpack GR_n, PI_n = GRIDS
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    #Simulate z, p, and oil discoveries
    z, p, ϵz, ϵp=Simulate_z_p_shocks(Tsim,GRIDS,par)
    n=Simulate_Discoveries(Tsim,PI_n)
    #Initiate vectors
    Def=Array{Float64,1}(undef,Tsim)
    K=Array{Float64,1}(undef,Tsim)
    KO=Array{Float64,1}(undef,Tsim)
    B=Array{Float64,1}(undef,Tsim)
    Spreads=Array{Float64,1}(undef,Tsim)
    for t in 1:Tsim
        if t==1
            K[t], KO[t]=SteadyStateCapital(par)
            B[t]=0.0
            if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                Def[t]=0.0
                K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
            else
                Def[t]=1.0
                K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=0.0
                Spreads[t]=0.0
            end
        else
            if t==Tsim
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            Spreads[t]=Spreads[t-1]
                        else
                            Def[t]=1.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        Spreads[t]=Spreads[t-1]
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            else
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                            Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                        else
                            Def[t]=1.0
                            K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=0.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                        Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            end
        end
    end
    return States_TS(z,p,n,Def,K,KO,B,Spreads)
end

function DrawErgodicState(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, Tsim, γ, κ, θ, μ_z, μ_p = par
    @unpack GR_n, PI_n = GRIDS
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    #Simulate z, p, and oil discoveries
    z, p, ϵz, ϵp=Simulate_z_p_shocks(Tsim,GRIDS,par)
    n=ones(Int64,Tsim)
    #Initiate vectors
    Def=Array{Float64,1}(undef,Tsim)
    K=Array{Float64,1}(undef,Tsim)
    KO=Array{Float64,1}(undef,Tsim)
    B=Array{Float64,1}(undef,Tsim)
    Spreads=Array{Float64,1}(undef,Tsim)
    for t in 1:Tsim
        if t==1
            K[t], KO[t]=SteadyStateCapital(par)
            B[t]=0.0
            if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                Def[t]=0.0
                K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
            else
                Def[t]=1.0
                K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=0.0
                Spreads[t]=0.0
            end
        else
            if t==Tsim
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            Spreads[t]=Spreads[t-1]
                        else
                            Def[t]=1.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        Spreads[t]=Spreads[t-1]
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            else
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                            Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                        else
                            Def[t]=1.0
                            K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=0.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                        Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            end
        end
    end
    return z[end], p[end], n[end], Def[end], K[end], KO[end], B[end], Spreads[end]
end

function DiscoveryTimeSeries(Taft::Int64,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, Tsim, γ, κ, θ, μ_z, μ_p, Nn = par
    @unpack GR_n, PI_n = GRIDS
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    z0, p0, n0, Def0, K0, KO0, B0, Spreads0=DrawErgodicState(SOLUTION_CURRENT,GRIDS,par)
    #Simulate z, p, and oil discoveries
    z=μ_z*ones(Float64,Taft+2)
    p=μ_p*ones(Float64,Taft+2)
    n=Array{Int64,1}(undef,Taft+2)
    n[1]=1
    for t in 2:Taft+2
        n[t]=min(t,Nn)
    end
    #Initiate vectors
    Def=Array{Float64,1}(undef,Taft+2)
    K=Array{Float64,1}(undef,Taft+2)
    KO=Array{Float64,1}(undef,Taft+2)
    B=Array{Float64,1}(undef,Taft+2)
    Spreads=Array{Float64,1}(undef,Taft+2)
    for t in 1:Taft+2
        if t==1
            K[t]=K0
            KO[t]=KO0
            B[t]=B0
            if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                Def[t]=0.0
                K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
            else
                Def[t]=1.0
                K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=0.0
                Spreads[t]=0.0
            end
        else
            if t==Taft+2
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            Spreads[t]=Spreads[t-1]
                        else
                            Def[t]=1.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        Spreads[t]=Spreads[t-1]
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            else
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                            Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                        else
                            Def[t]=1.0
                            K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=0.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                        Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            end
        end
    end
    return States_TS(z,p,n,Def,K,KO,B,Spreads)
end

function DisentangleEffect(PATHS_D::Paths,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, Tsim, γ, κ, θ, μ_z, μ_p, Nn = par
    @unpack itp_q1 = SOLUTION
    @unpack z, p, n, K, KO, B, Spreads = PATHS_D
    Spreads_News=Array{Float64,1}(undef,length(n)-1)
    Spreads_K=Array{Float64,1}(undef,length(n)-1)
    Spreads_KO=Array{Float64,1}(undef,length(n)-1)
    Spreads_B=Array{Float64,1}(undef,length(n)-1)
    for t in 1:length(n)-1
        if t==1
            Spreads_News[t]=Spreads[t]
            Spreads_K[t]=Spreads[t]
            Spreads_KO[t]=Spreads[t]
            Spreads_B[t]=Spreads[t]
        else
            #Just information
            Spreads_News[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[1],KO[1],B[1],par)
            #Information and K
            Spreads_K[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[1],B[1],par)
            #Information and KO
            Spreads_KO[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[1],par)
            #Information and B
            Spreads_B[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
        end
    end
    return Spreads_News, Spreads_K, Spreads_KO, Spreads_B
end

########## Simulate average paths after oil discoveries
function GetOnePathOfStatesAfterDiscovery(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, Twait, TsinceDefault, TsinceExhaustion, Nn, Tpaths, Tsim = par
    STATES_TS=SimulateStates_Long(SOLUTION,GRIDS,par)
    t0=drp+1
    #Look for discovery date in good standing
    #and no exhaustion in first Twait years of life
    #and no default in path
    tsince_Def=0
    tsince_Ex=0
    while true
        if t0+1+Tpaths+100==Tsim
            #Get another long sample
            STATES_TS=SimulateStates_Long(SOLUTION,GRIDS,par)
            t0=drp+1
            tsince_Def=0
            tsince_Ex=0
        end
        if STATES_TS.Def[t0]==1.0
            tsince_Def=0
        else
            tsince_Def=tsince_Def+1
        end
        if STATES_TS.n[t0]==1
            tsince_Ex=tsince_Ex+1
        else
            tsince_Ex=0
        end
        t0=t0+1
        t1=t0+Tpaths
        if STATES_TS.n[t0]==2 # && tsince_Def>=TsinceDefault && tsince_Ex>=TsinceExhaustion
            break
        end
    end
    t00=t0-1
    t1=t0+Tpaths
    return States_TS(STATES_TS.z[t00:t1],STATES_TS.p[t00:t1],STATES_TS.n[t00:t1],STATES_TS.Def[t00:t1],STATES_TS.K[t00:t1],STATES_TS.KO[t00:t1],STATES_TS.B[t00:t1],STATES_TS.Spreads[t00:t1])
end

function Get2ndPathWithoutDiscovery(STATES_D::States_TS,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Simulate 2 paths starting at the same state, with and without discovery
    @unpack r_star, Tsim, γ, κ, θ = par
    @unpack GR_n, PI_n = GRIDS
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    #Initiate copy of states
    z=deepcopy(STATES_D.z)
    p=deepcopy(STATES_D.p)
    #Series of n without discovery
    n=ones(Int64,length(STATES_D.n))
    #Initiate vectors
    Def=Array{Float64,1}(undef,length(STATES_D.n))
    K=Array{Float64,1}(undef,length(STATES_D.n))
    KO=Array{Float64,1}(undef,length(STATES_D.n))
    B=Array{Float64,1}(undef,length(STATES_D.n))
    Spreads=Array{Float64,1}(undef,length(STATES_D.n))
    for t in 1:length(STATES_D.n)
        if t==1
            K[t]=STATES_D.K[1]
            KO[t]=STATES_D.KO[1]
            B[t]=STATES_D.B[1]
            if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                Def[t]=0.0
                K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
            else
                Def[t]=1.0
                K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=0.0
                Spreads[t]=0.0
            end
        else
            if t==length(STATES_D.n)
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            Spreads[t]=Spreads[t-1]
                        else
                            Def[t]=1.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        Spreads[t]=Spreads[t-1]
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            else
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                            Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                        else
                            Def[t]=1.0
                            K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=0.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                        Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            end
        end
    end
    return States_TS(z,p,n,Def,K,KO,B,Spreads)
end

function GetAllPathsAfterDiscovery(STATES_TS::States_TS,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack γ, κ, HPFilter_Par, δ = par
    @unpack GR_n = GRIDS
    @unpack itp_q1, itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    itp_σyT=CreateInterpolation_PolicyFunctions(SOLUTION.σyT,true,GRIDS)
    itp_EyT=CreateInterpolation_PolicyFunctions(SOLUTION.EyT,true,GRIDS)
    #Compute GDP and investment
    GDP_TS=Array{Float64,1}(undef,length(STATES_TS.n))
    nGDP_TS=Array{Float64,1}(undef,length(STATES_TS.n))
    con_TS=Array{Float64,1}(undef,length(STATES_TS.n))
    inv_TS=Array{Float64,1}(undef,length(STATES_TS.n))
    TB_TS=Array{Float64,1}(undef,length(STATES_TS.n))
    CA_TS=Array{Float64,1}(undef,length(STATES_TS.n))
    RER_TS=Array{Float64,1}(undef,length(STATES_TS.n))
    σyT=Array{Float64,1}(undef,length(STATES_TS.n))
    RiskPremium=Array{Float64,1}(undef,length(STATES_TS.n))
    DefPr=Array{Float64,1}(undef,length(STATES_TS.n))
    P0=1.0
    for t in 1:length(STATES_TS.n)
        z=STATES_TS.z[t]
        pO=STATES_TS.p[t]
        n=GR_n[STATES_TS.n[t]]
        k=STATES_TS.K[t]
        kO=STATES_TS.KO[t]
        b=STATES_TS.B[t]
        σyT[t]=itp_σyT(kO,k,z,pO,STATES_TS.n[t])
        if t<length(STATES_TS.n)
            kprime=STATES_TS.K[t+1]
            kOprime=STATES_TS.KO[t+1]
            bprime=STATES_TS.B[t+1]
        else
            if STATES_TS.Def[t]==0.0
                kprime=itp_kprime(b,kO,k,z,pO,STATES_TS.n[t])
                kOprime=itp_kOprime(b,kO,k,z,pO,STATES_TS.n[t])
                bprime=itp_bprime(b,kO,k,z,pO,STATES_TS.n[t])
            else
                kprime=itp_kprime_D(kO,k,z,pO,STATES_TS.n[t])
                kOprime=itp_kOprime_D(kO,k,z,pO,STATES_TS.n[t])
                bprime=0.0
            end
        end
        n_ind=STATES_TS.n[t]
        RiskPremium[t]=Compute_Risk_Premium(z,pO,n_ind,kprime,kOprime,bprime,itp_σyT,itp_EyT,SOLUTION,GRIDS,par)
        DefPr[t]=Integrate_DefChoice(z,pO,n_ind,kprime,kOprime,bprime,SOLUTION,GRIDS,par)
        if STATES_TS.Def[t]==0.0
            q=itp_q1(bprime,kOprime,kprime,z,pO,n_ind)
            X=q*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
            y=Production(z,pO,n,k,kO,X,par)
            P=PriceFinalGood(z,pO,n,k,kO,X,par)
        else
            zD=zDefault(z,par)
            X=0.0
            y=Production(zD,pO,n,k,kO,X,par)
            P=PriceFinalGood(zD,pO,n,k,kO,X,par)
        end
        if t==1
            P0=P
        end
        RER_TS[t]=1.0/P
        GDP_TS[t]=P0*y-X
        nom_GDP=P*y-X
        nGDP_TS[t]=nom_GDP
        nom_inv=P*((kprime+kOprime)-(1.0-δ)*(k+kO))
        inv_TS[t]=100*nom_inv/nom_GDP
        AdjCost=CapitalAdjustment(kprime,k,par)+OilCapitalAdjustment(kOprime,kO,par)
        con_TS[t]=y+(1.0-δ)*(k+kO)-(kprime+kOprime)-AdjCost
        TB_TS[t]=-100*X/nom_GDP
        CA_TS[t]=-100*(bprime-b)/nom_GDP
    end
    return Paths(STATES_TS.z,STATES_TS.p,STATES_TS.n,STATES_TS.Def,STATES_TS.K,STATES_TS.KO,STATES_TS.B,STATES_TS.Spreads,GDP_TS,nGDP_TS,con_TS,inv_TS,TB_TS,CA_TS,RER_TS,σyT,RiskPremium,DefPr)
end

function GetAveragePathsAfterDiscovery(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesPaths, Twait, TsinceDefault, Tpaths = par
    #Initiate vectors to fill with averages
    L=1+1+Tpaths
    z=zeros(Float64,L)
    p=zeros(Float64,L)
    n=zeros(Int64,L)
    Def=zeros(Float64,L)
    K=zeros(Float64,L)
    KO=zeros(Float64,L)
    B=zeros(Float64,L)
    SPREADS=zeros(Float64,L)
    GDP=zeros(Float64,L)
    nGDP=zeros(Float64,L)
    Cons=zeros(Float64,L)
    Inv=zeros(Float64,L)
    TB=zeros(Float64,L)
    CA=zeros(Float64,L)
    RER=zeros(Float64,L)
    σyT=zeros(Float64,L)
    RiskPremium=zeros(Float64,L)
    DefPr=zeros(Float64,L)
    #Average over NSamples samples
    i=1
    while i<=NSamplesPaths
        STATES_TS=GetOnePathOfStatesAfterDiscovery(SOLUTION,GRIDS,par)
        PATHS=GetAllPathsAfterDiscovery(STATES_TS,SOLUTION,GRIDS,par)
        while true
            #reject samples with negative consumption
            println("Doing sample $i for average paths")
            if minimum(PATHS.Cons)>0.0
                break
            else
                STATES_TS=GetOnePathOfStatesAfterDiscovery(SOLUTION,GRIDS,par)
                PATHS=GetAllPathsAfterDiscovery(STATES_TS,SOLUTION,GRIDS,par)
            end
        end
        z=z .+ (PATHS.z ./ NSamplesPaths)
        p=p .+ (PATHS.p ./ NSamplesPaths)
        n=PATHS.n
        Def=Def .+ (PATHS.Def ./ NSamplesPaths)
        K=K .+ (PATHS.K ./ NSamplesPaths)
        KO=KO .+ (PATHS.KO ./ NSamplesPaths)
        B=B .+ (PATHS.B ./ NSamplesPaths)
        SPREADS=SPREADS .+ (PATHS.Spreads ./ NSamplesPaths)
        GDP=GDP .+ (PATHS.GDP ./ NSamplesPaths)
        nGDP=nGDP .+ (PATHS.nGDP ./ NSamplesPaths)
        Cons=Cons .+ (PATHS.Cons ./ NSamplesPaths)
        Inv=Inv .+ (PATHS.Inv ./ NSamplesPaths)
        TB=TB .+ (PATHS.TB ./ NSamplesPaths)
        CA=CA .+ (PATHS.CA ./ NSamplesPaths)
        RER=RER .+ (PATHS.RER ./ NSamplesPaths)
        σyT=σyT .+ (PATHS.σyT ./ NSamplesPaths)
        RiskPremium=RiskPremium .+ (PATHS.RiskPremium ./ NSamplesPaths)
        DefPr=DefPr .+ (PATHS.DefPr ./ NSamplesPaths)
        i=i+1
    end
    return Paths(z,p,n,Def,K,KO,B,SPREADS,GDP,nGDP,Cons,Inv,TB,CA,RER,σyT,RiskPremium,DefPr)
end

function GetAverageDifferenceInPaths(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesPaths, Twait, TsinceDefault, Tpaths = par
    Random.seed!(123)
    #Initiate vectors to fill with averages
    L=1+1+Tpaths
    z=zeros(Float64,L)
    p=zeros(Float64,L)
    n=zeros(Int64,L)
    Def=zeros(Float64,L)
    K=zeros(Float64,L)
    KO=zeros(Float64,L)
    B=zeros(Float64,L)
    SPREADS=zeros(Float64,L)
    GDP=zeros(Float64,L)
    nGDP=zeros(Float64,L)
    Cons=zeros(Float64,L)
    Inv=zeros(Float64,L)
    TB=zeros(Float64,L)
    CA=zeros(Float64,L)
    RER=zeros(Float64,L)
    σyT=zeros(Float64,L)
    RiskPremium=zeros(Float64,L)
    DefPr=zeros(Float64,L)
    #Average over NSamples samples
    i=1
    while i<=NSamplesPaths
        STATES_D=GetOnePathOfStatesAfterDiscovery(SOLUTION,GRIDS,par)
        STATES_ND=Get2ndPathWithoutDiscovery(STATES_D,SOLUTION,GRIDS,par)
        PATHS_D=GetAllPathsAfterDiscovery(STATES_D,SOLUTION,GRIDS,par)
        PATHS_ND=GetAllPathsAfterDiscovery(STATES_ND,SOLUTION,GRIDS,par)
        while true
            #reject samples with negative consumption
            println("Doing sample $i for average difference in paths")
            if minimum(PATHS_D.Cons)>0.0 && minimum(PATHS_ND.Cons)>0.0
                break
            else
                STATES_D=GetOnePathOfStatesAfterDiscovery(SOLUTION,GRIDS,par)
                STATES_ND=Get2ndPathWithoutDiscovery(STATES_D,SOLUTION,GRIDS,par)
                PATHS_D=GetAllPathsAfterDiscovery(STATES_D,SOLUTION,GRIDS,par)
                PATHS_ND=GetAllPathsAfterDiscovery(STATES_ND,SOLUTION,GRIDS,par)
            end
        end
        #Calculate Differences
        Def_dif=PATHS_D.Def .- PATHS_ND.Def
        K_dif=log.(PATHS_D.K) .- log.(PATHS_ND.K)
        KO_dif=log.(PATHS_D.KO) .- log.(PATHS_ND.KO)
        B_dif=(PATHS_D.B .- PATHS_ND.B) ./ PATHS_ND.GDP[1]
        SPREADS_dif=PATHS_D.Spreads .- PATHS_ND.Spreads
        GDP_dif=100.0*((PATHS_D.GDP .- PATHS_ND.GDP) ./ PATHS_ND.GDP)
        nGDP_dif=100.0*((PATHS_D.nGDP .- PATHS_ND.nGDP) ./ PATHS_ND.nGDP)
        Cons_dif=100.0*((PATHS_D.Cons .- PATHS_ND.Cons) ./ PATHS_ND.Cons)
        Inv_dif=PATHS_D.Inv .- PATHS_ND.Inv
        TB_dif=PATHS_D.TB .- PATHS_ND.TB
        CA_dif=PATHS_D.CA .- PATHS_ND.CA
        RER_dif=100.0*((PATHS_D.RER .- PATHS_ND.RER) ./ PATHS_ND.RER)
        σyT_dif=PATHS_D.σyT .- PATHS_ND.σyT
        RiskPremium_dif=PATHS_D.RiskPremium .- PATHS_ND.RiskPremium
        DefPr_dif=PATHS_D.DefPr .- PATHS_ND.DefPr
        #Save average path of differences
        z=z .+ (PATHS_D.z ./ NSamplesPaths)
        p=p .+ (PATHS_D.p ./ NSamplesPaths)
        n=PATHS_D.n
        Def=Def .+ (Def_dif ./ NSamplesPaths)
        K=K .+ (K_dif ./ NSamplesPaths)
        KO=KO .+ (KO_dif ./ NSamplesPaths)
        B=B .+ (B_dif ./ NSamplesPaths)
        SPREADS=SPREADS .+ (SPREADS_dif ./ NSamplesPaths)
        GDP=GDP .+ (GDP_dif ./ NSamplesPaths)
        nGDP=nGDP .+ (nGDP_dif ./ NSamplesPaths)
        Cons=Cons .+ (Cons_dif ./ NSamplesPaths)
        Inv=Inv .+ (Inv_dif ./ NSamplesPaths)
        TB=TB .+ (TB_dif ./ NSamplesPaths)
        CA=CA .+ (CA_dif ./ NSamplesPaths)
        RER=RER .+ (RER_dif ./ NSamplesPaths)
        σyT=σyT .+ (σyT_dif ./ NSamplesPaths)
        RiskPremium=RiskPremium .+ (RiskPremium_dif ./ NSamplesPaths)
        DefPr=DefPr .+ (DefPr_dif ./ NSamplesPaths)
        i=i+1
    end
    return Paths(z,p,n,Def,K,KO,B,SPREADS,GDP,nGDP,Cons,Inv,TB,CA,RER,σyT,RiskPremium,DefPr)
end

########## Generate panel
function SimulateStates_Long_Given_pO(p::Array{Float64,1},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, Tsim, γ, κ, θ = par
    @unpack GR_n, PI_n = GRIDS
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    #Simulate z, p, and oil discoveries
    z, pNOT_USE, ϵz, ϵp=Simulate_z_p_shocks(Tsim,GRIDS,par)
    n=Simulate_Discoveries(Tsim,PI_n)
    #Initiate vectors
    Def=Array{Float64,1}(undef,Tsim)
    K=Array{Float64,1}(undef,Tsim)
    KO=Array{Float64,1}(undef,Tsim)
    B=Array{Float64,1}(undef,Tsim)
    Spreads=Array{Float64,1}(undef,Tsim)
    for t in 1:Tsim
        if t==1
            K[t], KO[t]=SteadyStateCapital(par)
            B[t]=0.0
            if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                Def[t]=0.0
                K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
            else
                Def[t]=1.0
                K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=0.0
                Spreads[t]=0.0
            end
        else
            if t==Tsim
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            Spreads[t]=Spreads[t-1]
                        else
                            Def[t]=1.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        Spreads[t]=Spreads[t-1]
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            else
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                            Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                        else
                            Def[t]=1.0
                            K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=0.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                        Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            end
        end
    end
    return States_TS(z,p,n,Def,K,KO,B,Spreads)
end

function NPV_disc_TS(pO::Float64,kO::Float64,Spread::Float64,par::Pars)
    @unpack Twait, FieldLife, r_star, αO, ζ, nH, nL, Target_spreads = par
    rss=r_star+Spread
    a0=1.0/(1.0+rss)
    FACTOR=((a0^(Twait+1.0))-(a0^(Twait+FieldLife+1.0)))/(1.0-a0)
    kssL, kOssL=SteadyStateCapital(par)
    kssH, kOssH=SteadyStateCapital_nH(par)
    a=kOssH/kOssL
    return pO*(OilProduction(a*kO,nH,par)-OilProduction(kO,nL,par))*FACTOR
end

function TimeSeriesOfNPVdisc(PATHS::Paths,par::Pars)
    @unpack Tpanel = par
    NPV=zeros(Float64,Tpanel)
    for t in 1:Tpanel
        if (PATHS.n[t]==2 && t==1) || (PATHS.n[t]==2 && PATHS.n[t-1]==1)
            NPV[t]=100*NPV_disc_TS(PATHS.p[t],PATHS.KO[t],0.01*PATHS.Spreads[t],par)/PATHS.nGDP[t]
        end
    end
    return NPV
end

function GetOnePathOfStatesForPanel(p::Array{Float64,1},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, TsinceDefault, Nn, Tpanel, Tsim = par
    STATES_TS=SimulateStates_Long_Given_pO(p,SOLUTION,GRIDS,par)
    t0=drp+1
    #Start after drp, start at n=nL with no news
    if STATES_TS.n[t0]>1
        #Keep going until the current large field is exhausted
        #and economy has been in good standing for TsinceDefault periods
        while true
            t0=t0+1
            if STATES_TS.n[t0]==1 && STATES_TS.Def[t0]==0.0
                break
            end
        end
    end
    #Look for discovery date in good standing
    #and small field for Tsince years
    tsince=0
    tsince_f=0
    while true
        if t0+1+Tpanel+100==Tsim
            #Get another long sample
            STATES_TS=SimulateStates_Long_Given_pO(p,SOLUTION,GRIDS,par)
            t0=drp+1
            tsince=0
            tsince_f=0
        end
        if STATES_TS.Def[t0]==1.0
            tsince=0
        else
            tsince=tsince+1
        end
        if STATES_TS.n[t0]==1
            tsince_f=tsince_f+1
        else
            tsince_f=0
        end
        t0=t0+1
        t1=t0+Tpanel-1
        if tsince>=TsinceDefault && tsince_f>=TsinceDefault
            break
        end
    end
    t00=t0
    t1=t0+Tpanel-1
    return States_TS(STATES_TS.z[t00:t1],STATES_TS.p[t00:t1],STATES_TS.n[t00:t1],STATES_TS.Def[t00:t1],STATES_TS.K[t00:t1],STATES_TS.KO[t00:t1],STATES_TS.B[t00:t1],STATES_TS.Spreads[t00:t1])
end

function SimulateSingleEconomy(economy::Int64,p::Array{Float64,1},SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Tpanel = par
    STATES_P=GetOnePathOfStatesForPanel(p,SOLUTION,GRIDS,par)
    PATHS=GetAllPathsAfterDiscovery(STATES_P,SOLUTION,GRIDS,par)
    EC=economy*ones(Float64,Tpanel)
    time=collect(range(1,stop=Tpanel,length=Tpanel))
    NPV_TS=TimeSeriesOfNPVdisc(PATHS,par)
    MAT=[EC time PATHS.z PATHS.p PATHS.n NPV_TS PATHS.Def PATHS.K PATHS.KO PATHS.B PATHS.Spreads PATHS.GDP PATHS.Cons PATHS.Inv PATHS.TB PATHS.CA PATHS.RER]
end

function SavePanelFromModel(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Tsim, Npanel, Tpanel = par
    Random.seed!(123)
    z, p, ϵz, ϵp=Simulate_z_p_shocks(Tsim,GRIDS,par)
    MAT=Array{Float64,2}(undef,Tpanel*Npanel,17)
    for i in 1:Npanel
        first=(i-1)*Tpanel+1
        last=i*Tpanel
        MAT[first:last,:]=SimulateSingleEconomy(i,p,SOLUTION,GRIDS,par)
    end
    NAMES=["economy" "year" "z" "p" "n" "npv" "def" "k" "kO" "b" "spreads" "gdp" "cons" "inv" "tb" "ca" "rer"]
    writedlm("ModelPanel.csv",[NAMES; MAT],',')
    return nothing
end

########## Compute moments
@with_kw mutable struct Moments
    #Initiate them at 0.0 to facilitate average across samples
    #Default, spreads, and Debt
    DefaultPr::Float64 = 0.0
    MeanSpreads::Float64 = 0.0
    StdSpreads::Float64 = 0.0
    Debt_GDP::Float64 = 0.0
    #Volatilities
    σ_GDP::Float64 = 0.0
    σ_con::Float64 = 0.0
    σ_inv::Float64 = 0.0
    σ_yT::Float64 = 0.0
    #Cyclicality
    Corr_con_GDP::Float64 = 0.0
    Corr_inv_GDP::Float64 = 0.0
    Corr_Spreads_GDP::Float64 = 0.0
    Corr_CA_GDP::Float64 = 0.0
    Corr_TB_GDP::Float64 = 0.0
    Av_RP::Float64 = 0.0
end
###### Select moments of length Tmom that start in good standing and have been
###### in good standing after, at least, TsinceDefault=25 periods, consider no
###### discoveries in the sample for the moments
function SimulateStates_Long_NoDiscovery(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, Tsim, γ, κ, θ = par
    @unpack GR_n, PI_n = GRIDS
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    #Simulate z, p, and oil discoveries
    z, p, ϵz, ϵp=Simulate_z_p_shocks(Tsim,GRIDS,par)
    n=ones(Int64,Tsim)
    #Initiate vectors
    Def=Array{Float64,1}(undef,Tsim)
    K=Array{Float64,1}(undef,Tsim)
    KO=Array{Float64,1}(undef,Tsim)
    B=Array{Float64,1}(undef,Tsim)
    Spreads=Array{Float64,1}(undef,Tsim)
    for t in 1:Tsim
        if t==1
            K[t], KO[t]=SteadyStateCapital(par)
            B[t]=0.0
            if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                Def[t]=0.0
                K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
            else
                Def[t]=1.0
                K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=0.0
                Spreads[t]=0.0
            end
        else
            if t==Tsim
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            Spreads[t]=Spreads[t-1]
                        else
                            Def[t]=1.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        Spreads[t]=Spreads[t-1]
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            else
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                            Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                        else
                            Def[t]=1.0
                            K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=0.0
                            Spreads[t]=0.0
                        end
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=0.0
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                        Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=0.0
                    end
                end
            end
        end
    end
    return States_TS(z,p,n,Def,K,KO,B,Spreads)
end

function GetOnePathForMoments(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, Tmom, TsinceDefault, Tsim = par
    STATES_TS=SimulateStates_Long_NoDiscovery(SOLUTION,GRIDS,par)
    #Start after drp, start at good standing Def[t0]=0
    t0=drp+1
    if STATES_TS.Def[t0]==1.0
        #Keep going until good standing
        while true
            t0=t0+1
            if STATES_TS.Def[t0]==0.0
                break
            end
        end
    end
    #Count TsinceDefault periods without default
    #Try at most 5 long samples
    NLongSamples=0
    tsince=0
    while true
        if t0+Tmom+10==Tsim
            NLongSamples=NLongSamples+1
            if NLongSamples<=5
                #Get another long sample
                STATES_TS=SimulateStates_Long_NoDiscovery(SOLUTION,GRIDS,par)
                t0=drp+1
                tsince=0
            else
                # println("failed to get appropriate moment sample")
                break
            end
        end
        if STATES_TS.Def[t0]==1.0
            tsince=0
        else
            tsince=tsince+1
        end
        t0=t0+1
        if STATES_TS.Def[t0]==0.0 && tsince>=TsinceDefault
            break
        end
    end
    t1=t0+Tmom-1
    return States_TS(STATES_TS.z[t0:t1],STATES_TS.p[t0:t1],STATES_TS.n[t0:t1],STATES_TS.Def[t0:t1],STATES_TS.K[t0:t1],STATES_TS.KO[t0:t1],STATES_TS.B[t0:t1],STATES_TS.Spreads[t0:t1])
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

function ComputeMomentsOnce(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Tmom, γ, κ, HPFilter_Par, δ = par
    @unpack GR_n = GRIDS
    @unpack itp_q1, itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    itp_σyT=CreateInterpolation_PolicyFunctions(SOLUTION.σyT,true,GRIDS)
    itp_EyT=CreateInterpolation_PolicyFunctions(SOLUTION.EyT,true,GRIDS)
    STATES_TS=GetOnePathForMoments(SOLUTION,GRIDS,par)
    #Compute easy moments
    #Compute default probability
    sDef_TS=SimulateStates_Long_NoDiscovery(SOLUTION,GRIDS,par)
    DefEpisodes=zeros(Float64,length(sDef_TS.Def))
    for t in 2:length(sDef_TS.Def)
        if sDef_TS.Def[t]==1 && sDef_TS.Def[t-1]==0
            DefEpisodes[t]=1
        end
    end
    DefaultPr=mean(DefEpisodes)
    #Compute spreads moments
    MeanSpreads=mean(STATES_TS.Spreads)
    StdSpreads=std(STATES_TS.Spreads)
    #Compute GDP and investment
    GDP_TS=Array{Float64,1}(undef,Tmom)
    con_TS=Array{Float64,1}(undef,Tmom)
    inv_TS=Array{Float64,1}(undef,Tmom)
    TB_TS=Array{Float64,1}(undef,Tmom)
    CA_TS=Array{Float64,1}(undef,Tmom)
    RiskPremium=Array{Float64,1}(undef,Tmom)
    yT_TS=Array{Float64,1}(undef,Tmom)
    while true
        # STATES_TS=GetOnePathForMoments(SOLUTION,GRIDS,par)
        #Compute easy moments
        DefaultPr=mean(STATES_TS.Def)
        MeanSpreads=mean(STATES_TS.Spreads)
        StdSpreads=std(STATES_TS.Spreads)
        for t in 1:Tmom
            z=STATES_TS.z[t]
            pO=STATES_TS.p[t]
            n=GR_n[STATES_TS.n[t]]
            k=STATES_TS.K[t]
            kO=STATES_TS.KO[t]
            b=STATES_TS.B[t]
            if t<Tmom
                kprime=STATES_TS.K[t+1]
                kOprime=STATES_TS.KO[t+1]
                bprime=STATES_TS.B[t+1]
            else
                if STATES_TS.Def[t]==0.0
                    kprime=itp_kprime(b,kO,k,z,pO,STATES_TS.n[t])
                    kOprime=itp_kOprime(b,kO,k,z,pO,STATES_TS.n[t])
                    bprime=itp_bprime(b,kO,k,z,pO,STATES_TS.n[t])
                else
                    kprime=itp_kprime_D(kO,k,z,pO,STATES_TS.n[t])
                    kOprime=itp_kOprime_D(kO,k,z,pO,STATES_TS.n[t])
                    bprime=0.0
                end
            end
            n_ind=STATES_TS.n[t]
            RiskPremium[t]=Compute_Risk_Premium(z,pO,n_ind,kprime,kOprime,bprime,itp_σyT,itp_EyT,SOLUTION,GRIDS,par)
            if STATES_TS.Def[t]==0.0
                q=itp_q1(bprime,kOprime,kprime,z,pO,n_ind)
                X=q*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
                y=Production(z,pO,n,k,kO,X,par)
                P=PriceFinalGood(z,pO,n,k,kO,X,par)
                λ=CapitalAllocation(z,pO,n,k,kO,X,par)
                yT=ManufProduction(z,λ*k,par)+pO*OilProduction(kO,n,par)
            else
                zD=zDefault(z,par)
                X=0.0
                y=Production(zD,pO,n,k,kO,X,par)
                P=PriceFinalGood(zD,pO,n,k,kO,X,par)
                λ=CapitalAllocation(zD,pO,n,k,kO,X,par)
                yT=ManufProduction(zD,λ*k,par)+pO*OilProduction(kO,n,par)
            end
            yT_TS[t]=yT
            GDP_TS[t]=P*y-X
            AdjCost=CapitalAdjustment(kprime,k,par)+OilCapitalAdjustment(kOprime,kO,par)
            inv_TS[t]=P*((kprime+kOprime)-(1.0-δ)*(k+kO))
            con_TS[t]=GDP_TS[t]-inv_TS[t]+X-P*AdjCost
            TB_TS[t]=-X
            CA_TS[t]=-(bprime-b)
        end
        #reject samples with negative consumption
        if minimum(abs.(con_TS))>=0.0
            break
        else
            STATES_TS=GetOnePathForMoments(SOLUTION,GRIDS,par)
        end
    end
    Debt_GDP=mean(100 .* (STATES_TS.B ./ GDP_TS))
    ###Hpfiltering
    #GDP
    log_GDP=log.(abs.(GDP_TS))
    GDP_trend=hp_filter(log_GDP,HPFilter_Par)
    GDP_cyc=100.0*(log_GDP .- GDP_trend)
    #Investment
    log_inv=log.(abs.(inv_TS))
    inv_trend=hp_filter(log_inv,HPFilter_Par)
    inv_cyc=100.0*(log_inv .- inv_trend)
    #Consumption
    log_con=log.(abs.(con_TS))
    con_trend=hp_filter(log_con,HPFilter_Par)
    con_cyc=100.0*(log_con .- con_trend)
    #Tradable income
    log_yT=log.(abs.(yT_TS))
    yT_trend=hp_filter(log_yT,HPFilter_Par)
    yT_cyc=100.0*(log_yT .- yT_trend)
    #Volatilities
    σ_GDP=std(GDP_cyc)
    σ_con=std(con_cyc)
    σ_inv=std(inv_cyc)
    σ_yT=std(yT_cyc)
    #Correlations with GDP
    Corr_con_GDP=cor(GDP_cyc,con_cyc)
    Corr_inv_GDP=cor(GDP_cyc,inv_cyc)
    Corr_Spreads_GDP=cor(GDP_cyc,STATES_TS.Spreads)
    Corr_CA_GDP=cor(GDP_cyc,100.0 .* (CA_TS ./ GDP_TS))
    Corr_TB_GDP=cor(GDP_cyc,100.0 .* (TB_TS ./ GDP_TS))
    Av_RP=mean(RiskPremium)
    return Moments(DefaultPr,MeanSpreads,StdSpreads,Debt_GDP,σ_GDP,σ_con,σ_inv,σ_yT,Corr_con_GDP,Corr_inv_GDP,Corr_Spreads_GDP,Corr_CA_GDP,Corr_TB_GDP,Av_RP)
end

function AverageMomentsManySamples(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesMoments = par
    Random.seed!(1234)
    #Initiate them at 0.0 to facilitate average across samples
    MOMENTS=Moments()
    for i in 1:NSamplesMoments
        # println("Sample $i for moments")
        MOMS=ComputeMomentsOnce(SOLUTION,GRIDS,par)
        #Default, spreads, and Debt
        MOMENTS.DefaultPr=MOMENTS.DefaultPr+MOMS.DefaultPr/NSamplesMoments
        MOMENTS.MeanSpreads=MOMENTS.MeanSpreads+MOMS.MeanSpreads/NSamplesMoments
        MOMENTS.StdSpreads=MOMENTS.StdSpreads+MOMS.StdSpreads/NSamplesMoments
        MOMENTS.Debt_GDP=MOMENTS.Debt_GDP+MOMS.Debt_GDP/NSamplesMoments
        #Volatilities
        MOMENTS.σ_GDP=MOMENTS.σ_GDP+MOMS.σ_GDP/NSamplesMoments
        MOMENTS.σ_con=MOMENTS.σ_con+MOMS.σ_con/NSamplesMoments
        MOMENTS.σ_inv=MOMENTS.σ_inv+MOMS.σ_inv/NSamplesMoments
        MOMENTS.σ_yT=MOMENTS.σ_yT+MOMS.σ_yT/NSamplesMoments
        #Cyclicality
        MOMENTS.Corr_con_GDP=MOMENTS.Corr_con_GDP+MOMS.Corr_con_GDP/NSamplesMoments
        MOMENTS.Corr_inv_GDP=MOMENTS.Corr_inv_GDP+MOMS.Corr_inv_GDP/NSamplesMoments
        MOMENTS.Corr_Spreads_GDP=MOMENTS.Corr_Spreads_GDP+MOMS.Corr_Spreads_GDP/NSamplesMoments
        MOMENTS.Corr_CA_GDP=MOMENTS.Corr_CA_GDP+MOMS.Corr_CA_GDP/NSamplesMoments
        MOMENTS.Corr_TB_GDP=MOMENTS.Corr_TB_GDP+MOMS.Corr_TB_GDP/NSamplesMoments
        MOMENTS.Av_RP=MOMENTS.Av_RP+MOMS.Av_RP/NSamplesMoments
    end
    return MOMENTS
end

function ComputeAndSaveMoments(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    MOMENTS=AverageMomentsManySamples(SOLUTION,GRIDS,par)
    RowNames=["DefaultPr";
              "MeanSpreads";
              "StdSpreads";
              "Debt_GDP";
              "σ_GDP";
              "σ_con";
              "σ_inv";
              "σ_yT";
              "Corr_con_GDP";
              "Corr_inv_GDP";
              "Corr_Spreads_GDP";
              "Corr_CA_GDP";
              "Corr_TB_GDP";
              "Av_RP"]
    Values=[MOMENTS.DefaultPr;
            MOMENTS.MeanSpreads;
            MOMENTS.StdSpreads;
            MOMENTS.Debt_GDP;
            MOMENTS.σ_GDP;
            MOMENTS.σ_con;
            MOMENTS.σ_inv;
            MOMENTS.σ_yT;
            MOMENTS.Corr_con_GDP;
            MOMENTS.Corr_inv_GDP;
            MOMENTS.Corr_Spreads_GDP;
            MOMENTS.Corr_CA_GDP;
            MOMENTS.Corr_TB_GDP;
            MOMENTS.Av_RP]
    MAT=[RowNames Values]
    writedlm("Moments.csv",MAT,',')
end

################################################################################
### Functions to save solution in CSV
################################################################################

function SaveSolution(SOLUTION::Solution)
    #Save vectors of repayment
    @unpack VP, V, EV, kprime, kOprime, bprime, lnyT, q1, EyT, σyT = SOLUTION
    MAT=reshape(VP,(:))
    MAT=hcat(MAT,reshape(V,(:)))
    MAT=hcat(MAT,reshape(EV,(:)))
    MAT=hcat(MAT,reshape(kprime,(:)))
    MAT=hcat(MAT,reshape(kOprime,(:)))
    MAT=hcat(MAT,reshape(bprime,(:)))
    MAT=hcat(MAT,reshape(q1,(:)))
    writedlm("Repayment.csv",MAT,',')
    #Save vectors of default
    @unpack VD, EVD, kprime_D, kOprime_D = SOLUTION
    MAT=reshape(VD,(:))
    MAT=hcat(MAT,reshape(EVD,(:)))
    MAT=hcat(MAT,reshape(kprime_D,(:)))
    MAT=hcat(MAT,reshape(kOprime_D,(:)))
    MAT=hcat(MAT,reshape(lnyT,(:)))
    MAT=hcat(MAT,reshape(EyT,(:)))
    MAT=hcat(MAT,reshape(σyT,(:)))
    writedlm("Default.csv",MAT,',')
    return nothing
end

function Unpack_Solution(FOLDER::String,GRIDS::Grids,par::Pars)
    #The files Repayment.csv and Default.csv must be in FOLDER
    #for this function to work
    @unpack Nn, Np, Nz, Nk, NkO, Nb = par
    #Unpack Matrices with data
    if FOLDER==" "
        MAT_R=readdlm("Repayment.csv",',')
        MAT_D=readdlm("Default.csv",',')
    else
        MAT_R=readdlm("$FOLDER\\Repayment.csv",',')
        MAT_D=readdlm("$FOLDER\\Default.csv",',')
    end
    #Allocate vectors into matrices
    #Repayment
    I=(Nb,NkO,Nk,Nz,Np,Nn)
    VP=reshape(MAT_R[:,1],I)
    V=reshape(MAT_R[:,2],I)
    EV=reshape(MAT_R[:,3],I)
    kprime=reshape(MAT_R[:,4],I)
    kOprime=reshape(MAT_R[:,5],I)
    bprime=reshape(MAT_R[:,6],I)
    q1=reshape(MAT_R[:,7],I)
    #Default
    I=(NkO,Nk,Nz,Np,Nn)
    VD=reshape(MAT_D[:,1],I)
    EVD=reshape(MAT_D[:,2],I)
    kprime_D=reshape(MAT_D[:,3],I)
    kOprime_D=reshape(MAT_D[:,4],I)
    lnyT=reshape(MAT_D[:,5],I)
    EyT=reshape(MAT_D[:,6],I)
    σyT=reshape(MAT_D[:,7],I)
    #Create interpolation objects
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)
    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)
    itp_kprime_D=CreateInterpolation_PolicyFunctions(kprime_D,true,GRIDS)
    itp_kOprime_D=CreateInterpolation_PolicyFunctions(kOprime_D,true,GRIDS)
    itp_kprime=CreateInterpolation_PolicyFunctions(kprime,false,GRIDS)
    itp_kOprime=CreateInterpolation_PolicyFunctions(kOprime,false,GRIDS)
    itp_bprime=CreateInterpolation_PolicyFunctions(bprime,false,GRIDS)
    itp_lnyT=CreateInterpolation_log_yT(lnyT,GRIDS)
    return Solution(VD,VP,V,EVD,EV,q1,EyT,σyT,kprime_D,kOprime_D,kprime,kOprime,bprime,lnyT,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_kprime_D,itp_kOprime_D,itp_kprime,itp_kOprime,itp_bprime,itp_lnyT)
end

function SolveAndSaveModel_VFI(GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, relTolV, Tol_q_pct, cnt_max = par
    println("pre-computing production")
    PROD_MATS=OutputMatrices(par,GRIDS)
    println("Preparing solution guess")
    SOLUTION_CURRENT=SolutionEndOfTime(PROD_MATS,GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    rdst_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    println("Starting VFI")
    while ((dst_V>Tol_V && rdst_V>relTolV) || (dst_q>Tol_q && NotConvPct>Tol_q_pct)) && cnt<cnt_max
        UpdateSolution!(PROD_MATS,SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        rdst_D, rdst_P=ComputeRelativeDistance(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_D, dst_P=ComputeDistance_V(SOLUTION_CURRENT,SOLUTION_NEXT)
        rdst_V=max(rdst_D,rdst_P)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
        println("cnt=$cnt, rdst_D=$rdst_D%, rdst_P=$rdst_P%, dst_q=$dst_q")
        println("    $cnt,  dst_D=$dst_D , dst_P=$dst_P ,       $NotConvPct% of q not converged")
    end
    SaveSolution(SOLUTION_NEXT)
    ComputeAndSaveMoments(SOLUTION_NEXT,GRIDS,par)
    return nothing
end

################################################################################
### Functions to calibrate β, φ, d0, d1 by matching moments
################################################################################

function SolveModel_VFI_ForMomentMatching(GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, relTolV, Tol_q_pct, cnt_max = par
    PROD_MATS=OutputMatrices_NotParallel(par,GRIDS)
    SOLUTION_CURRENT=SolutionEndOfTime_NotParallel(PROD_MATS,GRIDS,par)
    SOLUTION_NEXT=deepcopy(SOLUTION_CURRENT)
    dst_V=1.0
    rdst_V=100.0
    dst_q=1.0
    NotConvPct=100.0
    cnt=0
    while ((dst_V>Tol_V && rdst_V>relTolV) || (dst_q>Tol_q && NotConvPct>Tol_q_pct)) && cnt<cnt_max
        UpdateSolution_NotParallel!(PROD_MATS,SOLUTION_NEXT,GRIDS,par)
        dst_q, NotConvPct=ComputeDistance_q(SOLUTION_CURRENT,SOLUTION_NEXT,par)
        rdst_D, rdst_P=ComputeRelativeDistance(SOLUTION_CURRENT,SOLUTION_NEXT)
        dst_D, dst_P=ComputeDistance_V(SOLUTION_CURRENT,SOLUTION_NEXT)
        rdst_V=max(rdst_D,rdst_P)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        SOLUTION_CURRENT=deepcopy(SOLUTION_NEXT)
    end
    println("cnt=$cnt,  dst_V=$dst_V , dst_q=$dst_q")
    println("    $cnt, rdst_V=$rdst_V%,       $NotConvPct% of q not converged")
    #Return moments
    return AverageMomentsManySamples(SOLUTION_NEXT,GRIDS,par)
end

function ContrastMomentsWithTargets(MOMENTS::Moments,par::Pars)
    @unpack Target_RP_share, Target_DefPr, Target_100spreads, Target_std_spreads, Target_σinv_σgdp, Target_σcon_σgdp, Target_debt_gdp = par
    @unpack MeanSpreads, StdSpreads, σ_GDP, σ_con, σ_inv, Debt_GDP, DefaultPr, Av_RP = MOMENTS
    diff_1=abs((MeanSpreads/Target_100spreads)-1.0)
    diff_2=abs((StdSpreads/Target_std_spreads)-1.0)
    diff_3=abs(((σ_inv/σ_GDP)/Target_σinv_σgdp)-1.0)
    diff_4=abs((Debt_GDP/Target_debt_gdp)-1.0)
    diff_5=abs((Target_DefPr/(100.0*DefaultPr))-1.0)
    diff_6=abs(((Av_RP/MeanSpreads)/Target_RP_share)-1.0)
    dst=maximum([diff_1 diff_3 diff_4 diff_5 diff_6])
    rnd_MeanSpreads=round(MeanSpreads,digits=1)
    rnd_StdSpreads=round(StdSpreads,digits=1)
    rnd_invGDPvol=round((σ_inv/σ_GDP),digits=1)
    rnd_debtGDP=round(Debt_GDP,digits=1)
    rnd_DefPr=round(100.0*DefaultPr,digits=1)
    rnd_RP=round(Av_RP/MeanSpreads,digits=2)
    @unpack β, d0, d1, α0, φ = par
    println("  ")
    println("β=$β; d0=$d0; d1=$d1; α0=$α0; φ=$φ")
    println("Targets: E[r-r*]=$Target_100spreads, Var(r-r*)=$Target_std_spreads, σi/σy=$Target_σinv_σgdp, b/y=$Target_debt_gdp, DefPr=$Target_DefPr, RP/Spr=$Target_RP_share")
    println("Moments: E[r-r*]=$rnd_MeanSpreads, Var(r-r*)=$rnd_StdSpreads, σi/σy=$rnd_invGDPvol, b/y=$rnd_debtGDP, DefPr=$rnd_DefPr, RP/Spr=$rnd_RP")
    println("supnorm of relative error for targets = $dst")
    return dst
end

function CheckMomentsForTry(PARS_TRY::Array{Float64,1})
    β=PARS_TRY[1]
    d0=PARS_TRY[2]
    knk=PARS_TRY[3]
    d1=-d0/knk
    α0=PARS_TRY[4]
    φ=PARS_TRY[5]
    par, GRIDS=Setup_MomentMatching(β,d0,d1,α0,φ)
    f(z::Float64)=zDefault(z,par)
    if minimum(f.(GRIDS.GR_z))>0.0
        MOMENTS=SolveModel_VFI_ForMomentMatching(GRIDS,par)
        MOM_VEC=Array{Float64,1}(undef,10)
        MOM_VEC[1]=MOMENTS.MeanSpreads
        MOM_VEC[2]=MOMENTS.StdSpreads
        MOM_VEC[3]=MOMENTS.σ_inv/MOMENTS.σ_GDP
        MOM_VEC[4]=MOMENTS.Debt_GDP
        MOM_VEC[5]=MOMENTS.DefaultPr
        MOM_VEC[6]=MOMENTS.σ_con/MOMENTS.σ_GDP
        MOM_VEC[7]=MOMENTS.Corr_Spreads_GDP
        MOM_VEC[8]=MOMENTS.Corr_CA_GDP
        MOM_VEC[9]=MOMENTS.Corr_TB_GDP
        MOM_VEC[10]=MOMENTS.Av_RP
        dst=ContrastMomentsWithTargets(MOMENTS,par)
    else
        mz=minimum(f.(GRIDS.GR_z))
        d0=par.d0
        d1=par.d1
        knk=-d0/d1
        println("skipped d0=$d0, d1=$d1, knk=$knk, mz=$mz")
        MOM_VEC=zeros(Float64,10)
        dst=100.0
    end
    return dst, MOM_VEC
end

function CalibrateMatchingMoments(N::Int64,lb::Vector{Float64},ub::Vector{Float64})
    #Generate Sobol sequence
    ss = skip(SobolSeq(lb, ub),N)
    MAT_TRY=Array{Float64,2}(undef,N,5)
    for i in 1:N
        MAT_TRY[i,:]=next!(ss)
    end
    #Loop paralelly over all parameter tries
    DistVector=SharedArray{Float64,1}(N)
    #There are 9 moments, columns should be 10+number of parameters
    PARAMETER_MOMENTS_MATRIX=SharedArray{Float64,2}(N,15)
    @sync @distributed for i in 1:N
        PARAMETER_MOMENTS_MATRIX[i,1:5]=MAT_TRY[i,:]
        DistVector[i], PARAMETER_MOMENTS_MATRIX[i,6:15]=CheckMomentsForTry(MAT_TRY[i,:])
    end
    COL_NAMES=["beta" "d0" "d1" "alpha0" "phi" "mean spreads" "sdt spreads" "sigma_inv/sigma_y" "debt_GDP" "Default Pr." "sigma_c/sigma_y" "cor(r-r*,y)" "cor(CA/y,y)" "cor(TB/y,y)" "Av RP"]
    MAT=[COL_NAMES; PARAMETER_MOMENTS_MATRIX]
    writedlm("TriedCalibrations.csv",MAT,',')
    #Get best calibration
    dst, i_best=findmin(DistVector)
    #Save solution with best calibration
    β=MAT_TRY[i_best,1]
    d0=MAT_TRY[i_best,2]
    knk=MAT_TRY[i_best,3]
    d1=-d0/knk
    α0=MAT_TRY[i_best,4]
    φ=MAT_TRY[i_best,5]
    par0, GRIDS0=Setup_MomentMatching(β,d0,d1,α0,φ)
    par=Pars(par0,IncludeDisc=1,Nn=3+par0.Twait)
    GRIDS=CreateGrids(par)
    SolveAndSaveModel_VFI(GRIDS,par)
    println("Best calibration: β=$β; d0=$d0; d1=$d1; α0=$α0; φ=$φ")
    return nothing
end

################################################################################
### Functions for specific results for paper
################################################################################
#Default probabilities unconditional and after discovery
function DefaultProbabilitiesAfterDisc_Once(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Compute default probability with no discovery
    sDef_ALL=SimulateStates_Long(SOLUTION,GRIDS,par)
    DefEpisodes=zeros(Float64,length(sDef_ALL.Def))
    for t in 2:length(sDef_ALL.Def)
        if sDef_ALL.Def[t]==1 && sDef_ALL.Def[t-1]==0
            DefEpisodes[t]=1
        end
    end
    DefaultPr_Unc=mean(DefEpisodes)
    #Compute default probability conditional 10years after discovery
    DefEpisodes=0
    disc_eps=0
    current_discovery=0
    for t in 1:length(sDef_ALL.Def)
        #Check if t is a discovery episode
        if sDef_ALL.n[t]==2
            disc_eps=disc_eps+1
            current_discovery=1
        end
        #Reset current episode
        if current_discovery>10
            current_discovery=0
        end
        #Count default episodes
        if current_discovery>0 && sDef_ALL.Def[t]==1 && sDef_ALL.Def[t-1]==0
            DefEpisodes=DefEpisodes+1
        end
        #Count periods after discovery
        if current_discovery>0
            current_discovery=current_discovery+1
        end
    end
    DefaultPr_Con=DefEpisodes/(10*disc_eps)
    #Convert to 10 years
    U_PrDef10=1.0-((1.0-DefaultPr_Unc)^10.0)
    D_PrDef10=1.0-((1.0-DefaultPr_Con)^10.0)
    return 100.0*U_PrDef10, 100.0*D_PrDef10
end

function DefaultProbabilitiesAfterDisc(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesMoments = par
    Random.seed!(1234)
    #Compute default probability with no discovery
    U_PrDef10=0.0
    D_PrDef10=0.0
    for i in 1:NSamplesMoments
        u, d=DefaultProbabilitiesAfterDisc_Once(SOLUTION,GRIDS,par)
        U_PrDef10=U_PrDef10+u/NSamplesMoments
        D_PrDef10=D_PrDef10+d/NSamplesMoments
    end
    return U_PrDef10, D_PrDef10
end

#Model Fit, responses vs data
function Read_IR_FromRegressions(FOLDER::String)
    #Unpack Matrix with data
    if FOLDER==" "
        MAT=readdlm("DataResponses.csv",',')
    else
        MAT=readdlm("$FOLDER\\DataResponses.csv",',')
    end
    return MAT
end

function PlotResponsesVsData(FOLDER_GRAPHS::String,FOLDER::String,PATHS::Paths,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Tpaths, NSamplesPaths = par
    #Read data from paper
    DATA=Read_IR_FromRegressions(FOLDER)
    Spreads_data=DATA[:,1]
    Inv_data=DATA[:,2]
    CA_data=DATA[:,3]
    GDP_data=DATA[:,4]
    Cons_data=DATA[:,5]
    RER_data=DATA[:,6]
    #Generate paths of IR from model
    tend=15
    t0=0
    t1=tend
    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    #Plot Spreads
    dat=Spreads_data[1:t1+1]
    mod=PATHS.Spreads[2:t1+2] .- PATHS.Spreads[1]
    plt_spreads=plot([t0:t1],[dat, mod],label=["data" "model"],
        linestyle=[:solid :dash],title="spreads",
        ylabel="percentage points",xlabel="t",ylims=[-0.05,1.3],
        legend=:topleft,size=SIZE_PLOTS,linewidth=LW)
    #Plot real exchange rate
    dat=RER_data[1:t1+1]
    mod=PATHS.RER[2:t1+2]
    plt_rer=plot([t0:t1],[dat, mod],label=["data" "model"],
        linestyle=[:solid :dash],title="real exchange rate",
        ylabel="percentage change",xlabel="t",
        legend=false,size=SIZE_PLOTS,linewidth=LW)
    #Plot investment
    dat=Inv_data[1:t1+1]
    mod=PATHS.Inv[2:t1+2]
    plt_inv=plot([t0:t1],[dat, mod],label=["data" "model"],
        linestyle=[:solid :dash],title="investment",
        ylabel="percentage of GDP",xlabel="t",
        legend=false,size=SIZE_PLOTS,linewidth=LW)
    #Plot current account
    dat=CA_data[1:t1+1]
    mod=PATHS.CA[2:t1+2]
    plt_CA=plot([t0:t1],[dat, mod],label=["data" "model"],
        linestyle=[:solid :dash],title="current account",
        ylabel="percentage of GDP",xlabel="t",
        legend=false,size=SIZE_PLOTS,linewidth=LW)
    #Plot GDP
    dat=GDP_data[1:t1+1]
    mod=PATHS.GDP[2:t1+2]
    plt_gdp=plot([t0:t1],[dat, mod],label=["data" "model"],
        linestyle=[:solid :dash],title="GDP",
        ylabel="percentage change",xlabel="t",
        legend=false,size=SIZE_PLOTS,linewidth=LW)
    #Plot consumption
    dat=Cons_data[1:t1+1]
    mod=PATHS.Cons[2:t1+2]
    plt_cons=plot([t0:t1],[dat, mod],label=["data" "model"],
        linestyle=[:solid :dash],title="consumption",
        ylabel="percentage change",xlabel="t",
        legend=false,size=SIZE_PLOTS,linewidth=LW)
    #Create plot array
    l = @layout([a b; c d; e f])
    plt=plot(plt_spreads,plt_rer,
             plt_inv,plt_CA,
             plt_gdp,plt_cons,
             layout=l,size=(size_width*2,size_height*3))
    savefig(plt,"$FOLDER_GRAPHS\\Figure6.pdf")
    return plt
end

#Discussion figures
function PlotFigure8_Transition_kb(FOLDER_GRAPHS::String,PATHS_LEV::Paths,GRIDS::Grids,par::Pars)
    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    t0=-1
    t1=par.Tpaths-1
    #Left panel: capital
    kk=log.(PATHS_LEV.K[1:end-1]) .- log(PATHS_LEV.K[1])
    kkoo=log.(PATHS_LEV.KO[1:end-1]) .- log(PATHS_LEV.KO[1])
    plt_k=plot([t0:t1],[kk, kkoo],legend=:topleft,
         label=["capital" "oil capital"],linestyle=[:solid :dash],
         title="change in capital stocks",xlabel="t",
         ylabel="log difference",size=SIZE_PLOTS,linewidth=LW)
    #Right panel: debt/gdp
    bb=(PATHS_LEV.B[1:end-1] .- PATHS_LEV.B[1]) ./ PATHS_LEV.GDP[1]
    plt_b=plot([t0:t1],bb,legend=false,
         title="change in stock of debt",xlabel="t",
         ylabel="Δb/Av(GDP)",size=SIZE_PLOTS,linewidth=LW)
    #Create plot array
    l = @layout([a b])
    plt=plot(plt_k,plt_b,
             layout=l,size=(size_width*2,size_height))
    savefig(plt,"$FOLDER_GRAPHS\\Figure8.pdf")
    return plt
end

function PlotFigure9_PriceSchedule(FOLDER_GRAPHS::String,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_b, GR_k, GR_kO = GRIDS
    @unpack q1, itp_q1 = SOLUTION
    #Details for plots
    size_width=500
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    b_ind=1; kO_ind=4; k_ind=6; p_ind=6; z_ind=6; n_ind=1
    k, kO=SteadyStateCapital(par)
    b=0.0; p=1.0; z=1.0
    klow=0.75*k
    khigh=1.25*k
    kOlow=0.5*kO
    kOhigh=2.5*kO
    blow=0.95
    bhigh=1.15
    #As a function of b, different kO
    foobL(bb::Float64)=itp_q1(bb,kO,klow,z,p,n_ind)
    foobH(bb::Float64)=itp_q1(bb,kO,khigh,z,p,n_ind)
    # qqL=q1[:,kOlow,k_ind,z_ind,p_ind,n_ind]
    # qqH=q1[:,kOhigh,k_ind,z_ind,p_ind,n_ind]
    qqbL=foobL.(GR_b)
    qqbH=foobH.(GR_b)
    plt_b=plot(GR_b,[qqbL qqbH],label=["low k'" "high k'"],
         ylims=[0,0.85],legend=:topright,linestyle=[:solid :dash],
         size=SIZE_PLOTS,linewidth=LW,xlabel="b'")
    #As a function of k, different b
    fookL(kk::Float64)=itp_q1(blow,kO,kk,z,p,n_ind)
    fookH(kk::Float64)=itp_q1(bhigh,kO,kk,z,p,n_ind)
    # qqL=q1[blow,kOlow,:,z_ind,p_ind,n_ind]
    # qqH=q1[bhigh,kOhigh,:,z_ind,p_ind,n_ind]
    qqkL=fookL.(GR_k)
    qqkH=fookH.(GR_k)
    plt_k=plot(GR_k,[qqkL qqkH],label=["low b'" "high b'"],
         ylims=[0,0.85],legend=:bottomright,linestyle=[:solid :dash],
         size=SIZE_PLOTS,linewidth=LW,xlabel="k'")
    #As a function of kO, different b
    fookOL(kkO::Float64)=itp_q1(blow,kkO,k,z,p,n_ind)
    fookOH(kkO::Float64)=itp_q1(bhigh,kkO,k,z,p,n_ind)
    # qqL=q1[blow,kOlow,:,z_ind,p_ind,n_ind]
    # qqH=q1[bhigh,kOhigh,:,z_ind,p_ind,n_ind]
    qqkOL=fookOL.(GR_kO)
    qqkOH=fookOH.(GR_kO)
    plt_kO=plot(GR_kO,[qqkOL qqkOH],label=["low b'" "high b'"],
         ylims=[0,0.85],legend=:bottomright,linestyle=[:solid :dash],
         size=SIZE_PLOTS,linewidth=LW,xlabel="kO'")
    #Create plot array
    l = @layout([a b c])
    plt=plot(plt_k,plt_kO,plt_b,
             layout=l,size=(size_width*3,size_height))
    savefig(plt,"$FOLDER_GRAPHS\\Figure9.pdf")
    return plt
end

function PlotFigure10_GapVP_VD(FOLDER_GRAPHS::String,SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack GR_b, GR_k, GR_kO = GRIDS
    @unpack itp_VP, itp_VD = SOLUTION
    #Details for plots
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    b_ind=1; kO_ind=4; k_ind=6; p_ind=6; z_ind=6
    k, kO=SteadyStateCapital(par)
    k=4.26; kO=0.24
    b=0.0; p=1.0; z=1.0
    nlow=1
    nhigh=par.Nn
    blow=0.00
    bhigh=1.5
    #As a function of k, different n, low b
    fookbLnL(kk::Float64)=itp_VP(blow,kO,kk,z,p,nlow)-itp_VD(kO,kk,z,p,nlow)
    fookbLnH(kk::Float64)=itp_VP(blow,kO,kk,z,p,nhigh)-itp_VD(kO,kk,z,p,nhigh)
    vvkbLnL=fookbLnL.(GR_k[2:end])
    vvkbLnH=fookbLnH.(GR_k[2:end])
    plt_kbL=plot(GR_k[2:end],[vvkbLnL vvkbLnH],label=["nL" "nH"],
         legend=:topright,linestyle=[:solid :dash],
         size=SIZE_PLOTS,linewidth=LW,xlabel="k",ylabel="VP-VD",
         title="as a function of k, b=$blow")
    #As a function of kO, different n, low b
    fookObLnL(kkO::Float64)=itp_VP(blow,kkO,k,z,p,nlow)-itp_VD(kkO,k,z,p,nlow)
    fookObLnH(kkO::Float64)=itp_VP(blow,kkO,k,z,p,nhigh)-itp_VD(kkO,k,z,p,nhigh)
    vvkObLnL=fookObLnL.(GR_kO[2:end])
    vvkObLnH=fookObLnH.(GR_kO[2:end])
    plt_kObL=plot(GR_kO[2:end],[vvkObLnL vvkObLnH],label=["nL" "nH"],
         legend=false,linestyle=[:solid :dash],
         size=SIZE_PLOTS,linewidth=LW,xlabel="kO",ylabel="VP-VD",
         title="as a function of kO, b=$blow")
    #As a function of k, different n, high b
    fookbHnL(kk::Float64)=itp_VP(bhigh,kO,kk,z,p,nlow)-itp_VD(kO,kk,z,p,nlow)
    fookbHnH(kk::Float64)=itp_VP(bhigh,kO,kk,z,p,nhigh)-itp_VD(kO,kk,z,p,nhigh)
    vvkbHnL=fookbHnL.(GR_k[2:end])
    vvkbHnH=fookbHnH.(GR_k[2:end])
    plt_kbH=plot(GR_k[2:end],[vvkbHnL vvkbHnH],label=["nL" "nH"],
         legend=false,linestyle=[:solid :dash],
         size=SIZE_PLOTS,linewidth=LW,xlabel="k",ylabel="VP-VD",
         title="as a function of k, b=$bhigh")
    #As a function of kO, different n, high b
    fookObHnL(kkO::Float64)=itp_VP(bhigh,kkO,k,z,p,nlow)-itp_VD(kkO,k,z,p,nlow)
    fookObHnH(kkO::Float64)=itp_VP(bhigh,kkO,k,z,p,nhigh)-itp_VD(kkO,k,z,p,nhigh)
    vvkObHnL=fookObHnL.(GR_kO[2:end])
    vvkObHnH=fookObHnH.(GR_kO[2:end])
    plt_kObH=plot(GR_kO[2:end],[vvkObHnL vvkObHnH],label=["nL" "nH"],
         legend=false,linestyle=[:solid :dash],
         size=SIZE_PLOTS,linewidth=LW,xlabel="kO",ylabel="VP-VD",
         title="as a function of kO, b=$bhigh")
    #Create plot array
    l = @layout([a b; c D])
    plt=plot(plt_kbL,plt_kObL,plt_kbH,plt_kObH,
             layout=l,size=(size_width*2,size_height*2))
    savefig(plt,"$FOLDER_GRAPHS\\Figure10.pdf")
    return plt
end

function Plot_SigmaT_and_RP(FOLDER_GRAPHS::String,PATHS::Paths,PATHS_LEV::Paths,GRIDS::Grids,par::Pars)
    #Details for plots
    size_width=650
    size_height=450
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    t0=-1
    t1=par.Tpaths-1
    #Left panel: volatility of yT
    plt_σT=plot([t0:t1],PATHS.σyT[1:end-1],legend=false,marker="-o",
         title="change in standard deviation of log(yT)",ylabel="Δ σ(log(yT))",xlabel="t")
    #Right panel: Spreads and risk-premium
    ss=PATHS.Spreads[1:end-1]
    rp=PATHS.RiskPremium[1:end-1]
    plt_rp=plot([t0:t1],[ss rp],legend=:bottomright,linestyle=[:solid :dash],
         title="change in spreads and risk premium",
         ylabel="percentage points",xlabel="t",label=["spreads" "risk premium"])
    #Create plot array
    l = @layout([a b])
    plt=plot(plt_σT,plt_rp,
             layout=l,size=(size_width*2,size_height))
    savefig(plt,"$FOLDER_GRAPHS\\Figure11.pdf")
    return plt
end

#Welfare calculations
function WG_one_Discovery(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack σ = par
    @unpack itp_V = SOLUTION
    #Get a random draw from ergodic distribution
    z, p, n_ind, Def, K, KO, B, Spreads=DrawErgodicState(SOLUTION,GRIDS,par)
    while Def>0
        #Make sure it is a draw in good standing
        z, p, n_ind, Def, K, KO, B, Spreads=DrawErgodicState(SOLUTION,GRIDS,par)
    end
    #Calculate welfare gains of discovery
    vd=itp_V(B,KO,K,z,p,2)
    vnd0=itp_V(B,KO,K,z,p,1)
    return ((vd/vnd0)^(1.0/(1.0-σ)))-1.0
end

function AvWG_Discovery(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesMoments = par
    WG=0.0
    for i in 1:NSamplesMoments
        WG=WG+WG_one_Discovery(SOLUTION,GRIDS,par)/NSamplesMoments
    end
    return 100.0*WG
end

#Table 4
function SimulateStates_Long_nH(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack r_star, Tsim, γ, κ, θ, Nn = par
    @unpack GR_n, PI_n = GRIDS
    @unpack itp_VD, itp_VP, itp_q1 = SOLUTION
    @unpack itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    #Simulate z, p, and oil discoveries
    z, p, ϵz, ϵp=Simulate_z_p_shocks(Tsim,GRIDS,par)
    n=Nn*ones(Int64,Tsim)
    #Initiate vectors
    Def=Array{Float64,1}(undef,Tsim)
    K=Array{Float64,1}(undef,Tsim)
    KO=Array{Float64,1}(undef,Tsim)
    B=Array{Float64,1}(undef,Tsim)
    Spreads=Array{Float64,1}(undef,Tsim)
    for t in 1:Tsim
        if t==1
            K[t], KO[t]=SteadyStateCapital(par)
            B[t]=0.0
            if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                Def[t]=0.0
                K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
            else
                Def[t]=1.0
                K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                B[t+1]=0.0
                Spreads[t]=0.0
            end
        else
            if t==Tsim
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            Spreads[t]=Spreads[t-1]
                        else
                            Def[t]=1.0
                            Spreads[t]=Spreads[t-1]
                        end
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        Spreads[t]=Spreads[t-1]
                    else
                        Def[t]=1.0
                        Spreads[t]=Spreads[t-1]
                    end
                end
            else
                if Def[t-1]==1.0
                    if rand()<=θ
                        if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                            Def[t]=0.0
                            K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                            Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                        else
                            Def[t]=1.0
                            K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                            KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                            B[t+1]=0.0
                            Spreads[t]=0.0
                        end
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=0.0
                    end
                else
                    if itp_VD(KO[t],K[t],z[t],p[t],n[t])<=itp_VP(B[t],KO[t],K[t],z[t],p[t],n[t])
                        Def[t]=0.0
                        K[t+1]=itp_kprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime(B[t],KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=max(itp_bprime(B[t],KO[t],K[t],z[t],p[t],n[t]),0.0)
                        Spreads[t]=CalculateSpreads(itp_q1,n[t],p[t],z[t],K[t+1],KO[t+1],B[t+1],par)
                    else
                        Def[t]=1.0
                        K[t+1]=itp_kprime_D(KO[t],K[t],z[t],p[t],n[t])
                        KO[t+1]=itp_kOprime_D(KO[t],K[t],z[t],p[t],n[t])
                        B[t+1]=0.0
                        Spreads[t]=0.0
                    end
                end
            end
        end
    end
    return States_TS(z,p,n,Def,K,KO,B,Spreads)
end

function GetOnePathForMoments_nH(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack drp, Tmom, TsinceDefault, Tsim = par
    STATES_TS=SimulateStates_Long_nH(SOLUTION,GRIDS,par)
    #Start after drp, start at good standing Def[t0]=0
    t0=drp+1
    if STATES_TS.Def[t0]==1.0
        #Keep going until good standing
        while true
            t0=t0+1
            if STATES_TS.Def[t0]==0.0
                break
            end
        end
    end
    #Count TsinceDefault periods without default
    #Try at most 5 long samples
    NLongSamples=0
    tsince=0
    while true
        if t0+Tmom+10==Tsim
            NLongSamples=NLongSamples+1
            if NLongSamples<=5
                #Get another long sample
                STATES_TS=SimulateStates_Long_nH(SOLUTION,GRIDS,par)
                t0=drp+1
                tsince=0
            else
                # println("failed to get appropriate moment sample")
                break
            end
        end
        if STATES_TS.Def[t0]==1.0
            tsince=0
        else
            tsince=tsince+1
        end
        t0=t0+1
        if STATES_TS.Def[t0]==0.0 && tsince>=TsinceDefault
            break
        end
    end
    t1=t0+Tmom-1
    return States_TS(STATES_TS.z[t0:t1],STATES_TS.p[t0:t1],STATES_TS.n[t0:t1],STATES_TS.Def[t0:t1],STATES_TS.K[t0:t1],STATES_TS.KO[t0:t1],STATES_TS.B[t0:t1],STATES_TS.Spreads[t0:t1])
end

function ComputeMomentsOnce_nH(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack Tmom, γ, κ, HPFilter_Par, δ = par
    @unpack GR_n = GRIDS
    @unpack itp_q1, itp_kprime, itp_kOprime, itp_bprime = SOLUTION
    @unpack itp_kprime_D, itp_kOprime_D = SOLUTION
    itp_σyT=CreateInterpolation_PolicyFunctions(SOLUTION.σyT,true,GRIDS)
    itp_EyT=CreateInterpolation_PolicyFunctions(SOLUTION.EyT,true,GRIDS)
    STATES_TS=GetOnePathForMoments_nH(SOLUTION,GRIDS,par)
    #Compute easy moments
    #Compute default probability
    sDef_TS=SimulateStates_Long_nH(SOLUTION,GRIDS,par)
    DefEpisodes=zeros(Float64,length(sDef_TS.Def))
    for t in 2:length(sDef_TS.Def)
        if sDef_TS.Def[t]==1 && sDef_TS.Def[t-1]==0
            DefEpisodes[t]=1
        end
    end
    DefaultPr=mean(DefEpisodes)
    #Compute spreads moments
    MeanSpreads=mean(STATES_TS.Spreads)
    StdSpreads=std(STATES_TS.Spreads)
    #Compute GDP and investment
    GDP_TS=Array{Float64,1}(undef,Tmom)
    con_TS=Array{Float64,1}(undef,Tmom)
    inv_TS=Array{Float64,1}(undef,Tmom)
    TB_TS=Array{Float64,1}(undef,Tmom)
    CA_TS=Array{Float64,1}(undef,Tmom)
    RiskPremium=Array{Float64,1}(undef,Tmom)
    yT_TS=Array{Float64,1}(undef,Tmom)
    while true
        # STATES_TS=GetOnePathForMoments(SOLUTION,GRIDS,par)
        #Compute easy moments
        DefaultPr=mean(STATES_TS.Def)
        MeanSpreads=mean(STATES_TS.Spreads)
        StdSpreads=std(STATES_TS.Spreads)
        for t in 1:Tmom
            z=STATES_TS.z[t]
            pO=STATES_TS.p[t]
            n=GR_n[STATES_TS.n[t]]
            k=STATES_TS.K[t]
            kO=STATES_TS.KO[t]
            b=STATES_TS.B[t]
            if t<Tmom
                kprime=STATES_TS.K[t+1]
                kOprime=STATES_TS.KO[t+1]
                bprime=STATES_TS.B[t+1]
            else
                if STATES_TS.Def[t]==0.0
                    kprime=itp_kprime(b,kO,k,z,pO,STATES_TS.n[t])
                    kOprime=itp_kOprime(b,kO,k,z,pO,STATES_TS.n[t])
                    bprime=itp_bprime(b,kO,k,z,pO,STATES_TS.n[t])
                else
                    kprime=itp_kprime_D(kO,k,z,pO,STATES_TS.n[t])
                    kOprime=itp_kOprime_D(kO,k,z,pO,STATES_TS.n[t])
                    bprime=0.0
                end
            end
            n_ind=STATES_TS.n[t]
            RiskPremium[t]=Compute_Risk_Premium(z,pO,n_ind,kprime,kOprime,bprime,itp_σyT,itp_EyT,SOLUTION,GRIDS,par)
            if STATES_TS.Def[t]==0.0
                q=itp_q1(bprime,kOprime,kprime,z,pO,n_ind)
                X=q*(bprime-(1.0-γ)*b)-(γ+κ*(1.0-γ))*b
                y=Production(z,pO,n,k,kO,X,par)
                P=PriceFinalGood(z,pO,n,k,kO,X,par)
                λ=CapitalAllocation(z,pO,n,k,kO,X,par)
                yT=ManufProduction(z,λ*k,par)+pO*OilProduction(kO,n,par)
            else
                zD=zDefault(z,par)
                X=0.0
                y=Production(zD,pO,n,k,kO,X,par)
                P=PriceFinalGood(zD,pO,n,k,kO,X,par)
                λ=CapitalAllocation(zD,pO,n,k,kO,X,par)
                yT=ManufProduction(zD,λ*k,par)+pO*OilProduction(kO,n,par)
            end
            yT_TS[t]=yT
            GDP_TS[t]=P*y-X
            AdjCost=CapitalAdjustment(kprime,k,par)+OilCapitalAdjustment(kOprime,kO,par)
            inv_TS[t]=P*((kprime+kOprime)-(1.0-δ)*(k+kO))
            con_TS[t]=GDP_TS[t]-inv_TS[t]+X-P*AdjCost
            TB_TS[t]=-X
            CA_TS[t]=-(bprime-b)
        end
        #reject samples with negative consumption
        if minimum(abs.(con_TS))>=0.0
            break
        else
            STATES_TS=GetOnePathForMoments_nH(SOLUTION,GRIDS,par)
        end
    end
    Debt_GDP=mean(100 .* (STATES_TS.B ./ GDP_TS))
    ###Hpfiltering
    #GDP
    log_GDP=log.(abs.(GDP_TS))
    GDP_trend=hp_filter(log_GDP,HPFilter_Par)
    GDP_cyc=100.0*(log_GDP .- GDP_trend)
    #Investment
    log_inv=log.(abs.(inv_TS))
    inv_trend=hp_filter(log_inv,HPFilter_Par)
    inv_cyc=100.0*(log_inv .- inv_trend)
    #Consumption
    log_con=log.(abs.(con_TS))
    con_trend=hp_filter(log_con,HPFilter_Par)
    con_cyc=100.0*(log_con .- con_trend)
    #Tradable income
    log_yT=log.(abs.(yT_TS))
    yT_trend=hp_filter(log_yT,HPFilter_Par)
    yT_cyc=100.0*(log_yT .- yT_trend)
    #Volatilities
    σ_GDP=std(GDP_cyc)
    σ_con=std(con_cyc)
    σ_inv=std(inv_cyc)
    σ_yT=std(yT_cyc)
    #Correlations with GDP
    Corr_con_GDP=cor(GDP_cyc,con_cyc)
    Corr_inv_GDP=cor(GDP_cyc,inv_cyc)
    Corr_Spreads_GDP=cor(GDP_cyc,STATES_TS.Spreads)
    Corr_CA_GDP=cor(GDP_cyc,100.0 .* (CA_TS ./ GDP_TS))
    Corr_TB_GDP=cor(GDP_cyc,100.0 .* (TB_TS ./ GDP_TS))
    Av_RP=mean(RiskPremium)
    return Moments(DefaultPr,MeanSpreads,StdSpreads,Debt_GDP,σ_GDP,σ_con,σ_inv,σ_yT,Corr_con_GDP,Corr_inv_GDP,Corr_Spreads_GDP,Corr_CA_GDP,Corr_TB_GDP,Av_RP)
end

function AverageMomentsManySamples_nH(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    @unpack NSamplesMoments = par
    Random.seed!(1234)
    #Initiate them at 0.0 to facilitate average across samples
    MOMENTS=Moments()
    for i in 1:NSamplesMoments
        # println("Sample $i for moments")
        MOMS=ComputeMomentsOnce_nH(SOLUTION,GRIDS,par)
        #Default, spreads, and Debt
        MOMENTS.DefaultPr=MOMENTS.DefaultPr+MOMS.DefaultPr/NSamplesMoments
        MOMENTS.MeanSpreads=MOMENTS.MeanSpreads+MOMS.MeanSpreads/NSamplesMoments
        MOMENTS.StdSpreads=MOMENTS.StdSpreads+MOMS.StdSpreads/NSamplesMoments
        MOMENTS.Debt_GDP=MOMENTS.Debt_GDP+MOMS.Debt_GDP/NSamplesMoments
        #Volatilities
        MOMENTS.σ_GDP=MOMENTS.σ_GDP+MOMS.σ_GDP/NSamplesMoments
        MOMENTS.σ_con=MOMENTS.σ_con+MOMS.σ_con/NSamplesMoments
        MOMENTS.σ_inv=MOMENTS.σ_inv+MOMS.σ_inv/NSamplesMoments
        MOMENTS.σ_yT=MOMENTS.σ_yT+MOMS.σ_yT/NSamplesMoments
        #Cyclicality
        MOMENTS.Corr_con_GDP=MOMENTS.Corr_con_GDP+MOMS.Corr_con_GDP/NSamplesMoments
        MOMENTS.Corr_inv_GDP=MOMENTS.Corr_inv_GDP+MOMS.Corr_inv_GDP/NSamplesMoments
        MOMENTS.Corr_Spreads_GDP=MOMENTS.Corr_Spreads_GDP+MOMS.Corr_Spreads_GDP/NSamplesMoments
        MOMENTS.Corr_CA_GDP=MOMENTS.Corr_CA_GDP+MOMS.Corr_CA_GDP/NSamplesMoments
        MOMENTS.Corr_TB_GDP=MOMENTS.Corr_TB_GDP+MOMS.Corr_TB_GDP/NSamplesMoments
        MOMENTS.Av_RP=MOMENTS.Av_RP+MOMS.Av_RP/NSamplesMoments
    end
    return MOMENTS
end

function Table4Calculations(SOLUTION::Solution,GRIDS::Grids,par::Pars)
    #Default probabilities 10 years after discovery vs 10 years
    U_PrDef10, D_PrDef10=DefaultProbabilitiesAfterDisc(SOL,GRIDS,par)
    println("Table 4")
    println("DefPr(10 years)=$U_PrDef10, DefPr(10 years|Disc)=$D_PrDef10")
    println("Table 4")
    #Volatilities of consumption and tradable income
    MOM_nL=AverageMomentsManySamples(SOLUTION,GRIDS,par)
    σc_nL=MOM_nL.σ_con
    σyT_nL=MOM_nL.σ_yT
    println("σc_nL=$σc_nL, σyT_nL=$σyT_nL")
    MOM_nH=AverageMomentsManySamples_nH(SOLUTION,GRIDS,par)
    σc_nH=MOM_nH.σ_con
    σyT_nH=MOM_nH.σ_yT
    println("σc_nH=$σc_nH, σyT_nH=$σyT_nH")
end
