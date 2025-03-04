
using Parameters, Roots, Interpolations, Optim, SharedArrays, DelimitedFiles,
      Distributions, FastGaussQuadrature, LinearAlgebra, Random, Statistics,
      SparseArrays, Sobol, Plots, Distributed

################################################################
#### Defining parameters and other structures for the model ####
################################################################
@with_kw struct Pars
    ################################################################
    ######## Preferences and technology ############################
    ################################################################
    #Preferences
    σ::Float64 = 2.0          #CRRA parameter
    β::Float64 = 0.8775       #Discount factor
    r_star::Float64 = 0.04    #Risk-free interest rate
    φ::Float64 = 10.0           #Robustness parameter
    #Debt parameters
    γ::Float64 = 0.14      #Reciprocal of average maturity
    κ::Float64 = 0.00      #Coupon payments
    #Government consumption parameters
    σG::Float64 = 2.0          #CRRA parameter
    ψ::Float64 = 0.19
    τ::Float64 = 0.15
    DomesticField::Bool = true  #If field is operated by domestic private sector
    #Default cost
    θ::Float64 = 0.40#0.0385       #Probability of re-admission
    knk::Float64 = 0.90
    d1::Float64 = 0.222        #income default cost
    d0::Float64 = -d1*knk       #income default cost
    OilEmbargo::Bool = false
    #"Oil discoveries"
    nL::Float64 = 0.05
    nH::Float64 = 0.10
    πLH::Float64 = 0.01#0.01           #Probability of discovery
    πHL::Float64 = 0.02#1/50           #Probability of exhaustion
    Twait::Int64 = 0                   #Periods between discovery and production
    SellField::Bool = false
    FieldToPayDebt::Bool = false
    SpreadToDiscount::Float64 = 0.0
    #Income shock
    μ_ϵz::Float64 = 0.0
    σ_ϵz::Float64 = 0.025
    dist_ϵz::UnivariateDistribution = Normal(μ_ϵz,σ_ϵz)
    ρ_z::Float64 = 0.95
    μ_z::Float64 = 1.00
    zlow::Float64 = exp(log(μ_z)-3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    zhigh::Float64 = exp(log(μ_z)+3.0*sqrt((σ_ϵz^2.0)/(1.0-(ρ_z^2.0))))
    #Quadrature parameters
    N_GLz::Int64 = 21
    #Grids
    Nn::Int64 = 2+Twait
    Nz::Int64 = 5
    Nb::Int64 = 21
    blow::Float64 = 0.0
    bhigh::Float64 = 2.0
    blowOpt::Float64 = blow-0.1             #Minimum level of debt for optimization
    bhighOpt::Float64 = bhigh+0.1            #Maximum level of debt for optimization
    #Parameters for solution algorithm
    cmin::Float64 = 1e-2
    Tol_V::Float64 = 1e-6             #Tolerance for absolute distance for value functions
    Tol_q::Float64 = 1e-6             #Tolerance for absolute distance for q in VFI
    cnt_max::Int64 = 100              #Maximum number of iterations on VFI
    #Parameters for simulation
    drp::Int64 = 1000
    HPFilter_Par::Float64 = 100.0
end

@with_kw struct Grids
    #Grids of states
    GR_n::Array{Float64,1}
    GR_z::Array{Float64,1}
    GR_b::Array{Float64,1}

    #Quadrature vectors for integrals
    ϵz_weights::Vector{Float64}
    ϵz_nodes::Vector{Float64}

    #Factor to normalize quadrature approximation
    FacQz::Float64

    #Matrix for discoveries
    PI_n::Array{Float64,2}

    #Matrices for integrals
    ZPRIME::Array{Float64,2}
    PDFz::Array{Float64,2}
end

function CreateOilFieldGrids(par::Pars)
    @unpack πLH, πHL, Twait, Nn, nL, nH = par
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

    #Gauss-Legendre vectors for z with high mean
    @unpack N_GLz, σ_ϵz, ρ_z, μ_z, dist_ϵz = par
    ϵz_weights, ϵz_nodes, ZPRIME, PDFz, FacQz=CreateQuadratureObjects(dist_ϵz,GR_z,N_GLz,σ_ϵz,ρ_z,μ_z)

    #Grid for n
    GR_n, PI_n=CreateOilFieldGrids(par)

    #Grid of debt
    @unpack Nb, blow, bhigh = par
    GR_b=collect(range(blow,stop=bhigh,length=Nb))

    return Grids(GR_n,GR_z,GR_b,ϵz_weights,ϵz_nodes,FacQz,PI_n,ZPRIME,PDFz)
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
    #Policy functions
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

@with_kw mutable struct Model
    SOLUTION::Solution
    GRIDS::Grids
    par::Pars
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
function CreateInterpolation_ValueFunctions(MAT::Array{Float64},IsDefault::Bool,GRIDS::Grids)
    @unpack GR_z, GR_n = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    ORDER_SHOCKS=Linear()
    ORDER_STATES=Linear()

    if IsDefault==true
        INT_DIMENSIONS=(BSpline(ORDER_STATES),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Zs,Ns),Interpolations.Line())
    else
        @unpack GR_b = GRIDS
        Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
        INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
        return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Zs,Ns),Interpolations.Line())
    end
end

function CreateInterpolation_Price(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_z, GR_n, GR_b = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))
    ORDER_SHOCKS=Linear()
    ORDER_STATES=Linear()

    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Zs,Ns),Interpolations.Flat())
end

function CreateInterpolation_PolicyFunctions(MAT::Array{Float64},GRIDS::Grids)
    @unpack GR_z, GR_n, GR_b = GRIDS
    Ns=range(1,stop=length(GR_n),length=length(GR_n))
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    Bs=range(GR_b[1],stop=GR_b[end],length=length(GR_b))

    ORDER_SHOCKS=Linear()
    ORDER_STATES=Linear()
    # ORDER_STATES=Cubic(Line(OnGrid()))

    INT_DIMENSIONS=(BSpline(ORDER_STATES),BSpline(ORDER_SHOCKS),NoInterp())
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Bs,Zs,Ns),Interpolations.Flat())
end

################################################################
########## Functions to compute expectations ###################
################################################################
function CreateInterpolation_ForExpectations(MAT::Array{Float64,1},GRIDS::Grids,par::Pars)
    @unpack GR_z = GRIDS
    Zs=range(GR_z[1],stop=GR_z[end],length=length(GR_z))
    ORDER_SHOCKS=Linear()
    INT_DIMENSIONS=BSpline(ORDER_SHOCKS)
    return extrapolate(Interpolations.scale(interpolate(MAT,INT_DIMENSIONS),Zs),Interpolations.Line())
end

function Expectation_over_zprime(foo,z_ind::Int64,GRIDS::Grids)
    #foo is a function of floats for z'
    #μ, μ', and b'
    int=0.0
    #Expectation over low mean tomorrow
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    for i in 1:length(ϵz_weights)
        int=int+ϵz_weights[i]*PDFz[z_ind,i]*foo(ZPRIME[z_ind,i])
    end
    return int/FacQz
end

function ComputeExpectationOverStates!(IsDefault::Bool,MAT::Array{Float64},E_MAT::Array{Float64},GRIDS::Grids,par::Pars)
    @unpack PI_n, GR_n = GRIDS
    Nn=length(GR_n)
    #It will use MAT to compute expectations
    #It will mutate E_MAT
    if IsDefault
        for I in CartesianIndices(MAT)
            (z_ind,n_ind)=Tuple(I)
            int=0.0
            for nprime_ind in 1:Nn
                if PI_n[n_ind,nprime_ind]>0.0
                    foo_mat=CreateInterpolation_ForExpectations(MAT[:,nprime_ind],GRIDS,par)
                    EVD=Expectation_over_zprime(foo_mat,z_ind,GRIDS)
                    int=int+PI_n[n_ind,nprime_ind]*EVD
                end
            end
            E_MAT[I]=int
        end
    else
        for I in CartesianIndices(MAT)
            (b_ind,z_ind,n_ind)=Tuple(I)
            int=0.0
            for nprime_ind in 1:Nn
                if PI_n[n_ind,nprime_ind]>0.0
                    foo_mat=CreateInterpolation_ForExpectations(MAT[b_ind,:,nprime_ind],GRIDS,par)
                    EVP=Expectation_over_zprime(foo_mat,z_ind,GRIDS)
                    int=int+PI_n[n_ind,nprime_ind]*EVP
                end
            end
            E_MAT[I]=int
        end
    end
    return nothing
end

###############################################################################
#Function to Initiate solution and auxiliary objects
###############################################################################
function InitiateEmptySolution(GRIDS::Grids,par::Pars)
    @unpack Nn, Nz, Nb, N_GLz = par
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
    #Policy functions
    bprime=zeros(Float64,Nb,Nz,Nn)
    itp_bprime=CreateInterpolation_PolicyFunctions(bprime,GRIDS)
    return Solution(VD,VP,V,EVD,EV,q1,bprime,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_bprime)
end

################################################################
############### Preferences and technology #####################
################################################################
function Small_u(x::Float64,σ::Float64)
    if σ==1
        return log(x)
    else
        return (x^(1-σ))/(1-σ)
    end
end

function Utility(c::Float64,g::Float64,par::Pars)
    @unpack σ, σG, ψ = par
    return (1-ψ)*Small_u(c,σ)+ψ*Small_u(g,σG)
end

function DefaultCost(y::Float64,Ey::Float64,par::Pars)
    #Chatterjee and Eyiungongor (2012)
    # @unpack d0, d1 = par
    # return max(0.0,d0*y+d1*y*y)
    #Arellano (2008)
    @unpack knk = par
    return max(0.0,y-knk*Ey)
end

function SDF_Lenders(par)
    @unpack r_star = par
    return 1/(1+r_star)
end

function ValueOfField(par::Pars)
    @unpack r_star, Twait, πHL, nL, nH, SpreadToDiscount = par
    r=r_star+SpreadToDiscount
    yField=nH-nL
    return ((1/(1+r))^Twait)*yField*(1+r)/(r+πHL)
end

################################################################################
### Functions to save Model in CSV
################################################################################
#Save model objects
function StackSolution_Vector(SOLUTION::Solution)
    #Stack vectors of repayment size first
    @unpack VP, V, EV, q1, bprime = SOLUTION
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
    #Parameters for Grids go first
    #Grids sizes
    VEC=par.N_GLz              #1
    VEC=vcat(VEC,par.Twait)    #2
    VEC=vcat(VEC,par.Nz)       #3
    VEC=vcat(VEC,par.Nb)       #4

    #Grids bounds
    VEC=vcat(VEC,par.blow)     #5
    VEC=vcat(VEC,par.bhigh)    #6

    #Parameter values
    VEC=vcat(VEC,par.β)        #7
    VEC=vcat(VEC,par.knk)      #8
    VEC=vcat(VEC,par.d0)       #9
    VEC=vcat(VEC,par.d1)       #10

    #Extra parameters
    VEC=vcat(VEC,par.cnt_max)  #11
    VEC=vcat(VEC,par.nL)       #12
    VEC=vcat(VEC,par.nH)       #13
    VEC=vcat(VEC,par.ρ_z)      #14
    VEC=vcat(VEC,par.σ_ϵz)     #15
    VEC=vcat(VEC,par.γ)        #16

    VEC=vcat(VEC,par.τ)        #17
    VEC=vcat(VEC,par.ψ)        #18

    if par.DomesticField              #19
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end

    if par.OilEmbargo           #20
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end

    if par.SellField           #21
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end

    if par.FieldToPayDebt           #22
        VEC=vcat(VEC,1.0)
    else
        VEC=vcat(VEC,0.0)
    end

    VEC=vcat(VEC,par.SpreadToDiscount)        #23

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
function ExtractMatrixFromSolutionVector(start::Int64,size::Int64,IsDefault::Bool,VEC::Array{Float64,1},par::Pars)
    @unpack Nz, Nb, N_GLz = par
    if IsDefault
        I=(Nz,2)
    else
        I=(Nb,Nz,2)
    end
    finish=start+size-1
    vec=VEC[start:finish]
    return reshape(vec,I)
end

function TransformVectorToSolution(VEC::Array{Float64},GRIDS::Grids,par::Pars)
    #The file SolutionVector.csv must be in FOLDER
    #for this function to work
    @unpack Nz, Nb = par
    size_repayment=2*Nz*Nb
    size_default=2*Nz

    #Allocate vectors into matrices
    #Repayment
    start=1

    VP=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment

    V=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment

    EV=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment

    q1=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment

    bprime=ExtractMatrixFromSolutionVector(start,size_repayment,false,VEC,par)
    start=start+size_repayment

    #Default
    VD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)
    start=start+size_default

    EVD=ExtractMatrixFromSolutionVector(start,size_default,true,VEC,par)

    #Create interpolation objects
    itp_VD=CreateInterpolation_ValueFunctions(VD,true,GRIDS)
    itp_VP=CreateInterpolation_ValueFunctions(VP,false,GRIDS)
    itp_V=CreateInterpolation_ValueFunctions(V,false,GRIDS)
    itp_EVD=CreateInterpolation_ValueFunctions(EVD,true,GRIDS)
    itp_EV=CreateInterpolation_ValueFunctions(EV,false,GRIDS)
    itp_q1=CreateInterpolation_Price(q1,GRIDS)
    itp_bprime=CreateInterpolation_PolicyFunctions(bprime,GRIDS)
    ### Interpolation objects
    #Input values for end of time
    return Solution(VD,VP,V,EVD,EV,q1,bprime,itp_VD,itp_VP,itp_V,itp_EVD,itp_EV,itp_q1,itp_bprime)
end

function UnpackParameters_Vector(VEC::Array{Float64})
    par=Pars()

    #Parameters for Grids go first
    #Grids sizes
    par=Pars(par,N_GLz=convert(Int64,VEC[1]))
    par=Pars(par,Twait=convert(Int64,VEC[2]))
    par=Pars(par,Nn=2+par.Twait)
    par=Pars(par,Nz=convert(Int64,VEC[3]))
    par=Pars(par,Nb=convert(Int64,VEC[4]))

    #Grids bounds
    par=Pars(par,blow=VEC[5])
    par=Pars(par,blowOpt=VEC[5]-0.05)
    par=Pars(par,bhigh=VEC[6])
    par=Pars(par,bhighOpt=VEC[6]+0.05)

    #Parameter values
    par=Pars(par,β=VEC[7])
    par=Pars(par,knk=VEC[8])
    par=Pars(par,d0=VEC[9])
    par=Pars(par,d1=VEC[10])

    #Extra parameters
    par=Pars(par,cnt_max=VEC[11])
    par=Pars(par,nL=VEC[12])
    par=Pars(par,nH=VEC[13])
    par=Pars(par,ρ_z=VEC[14])
    par=Pars(par,σ_ϵz=VEC[15])

    zlow=exp(log(par.μ_z)-3.0*sqrt((par.σ_ϵz^2.0)/(1.0-(par.ρ_z^2.0))))
    zhigh=exp(log(par.μ_z)+3.0*sqrt((par.σ_ϵz^2.0)/(1.0-(par.ρ_z^2.0))))
    par=Pars(par,zlow=zlow,zhigh=zhigh)

    par=Pars(par,γ=VEC[16])
    par=Pars(par,τ=VEC[17])
    par=Pars(par,ψ=VEC[18])

    if VEC[19]==1.0
        par=Pars(par,DomesticField=true)
    else
        par=Pars(par,DomesticField=false)
    end

    if VEC[20]==1.0
        par=Pars(par,OilEmbargo=true)
    else
        par=Pars(par,OilEmbargo=false)
    end

    if VEC[21]==1.0
        par=Pars(par,SellField=true)
    else
        par=Pars(par,SellField=false)
    end

    if VEC[22]==1.0
        par=Pars(par,FieldToPayDebt=true)
    else
        par=Pars(par,FieldToPayDebt=false)
    end

    par=Pars(par,SpreadToDiscount=VEC[23])

    return par
end

function Setup_From_Vector(VEC_PAR::Array{Float64})
    #Vector has the correct structure
    par=UnpackParameters_Vector(VEC_PAR)
    GRIDS=CreateGrids(par)
    return par, GRIDS
end

function UnpackModel_Vector(NAME::String,FOLDER::String)
    #Unpack Vector with data
    if FOLDER==" "
        VEC=readdlm(NAME,',')
    else
        VEC=readdlm("$FOLDER\\$NAME",',')
    end

    #Extract parameters and create grids
    N_parameters=convert(Int64,VEC[1])
    VEC_PAR=VEC[2:N_parameters+1]
    par, GRIDS=Setup_From_Vector(VEC_PAR)

    #Extract solution object
    VEC_SOL=VEC[N_parameters+2:end]
    SOL=TransformVectorToSolution(VEC_SOL,GRIDS,par)

    return Model(SOL,GRIDS,par)
end

###############################################################################
#Function to compute consumption given state
###############################################################################
function Calculate_Tr(n_ind::Int64,qq::Float64,b::Float64,bprime::Float64,par::Pars)
    @unpack γ, κ, SellField, FieldToPayDebt = par
    #Compute net borrowing from the rest of the world
    if SellField
        if n_ind==2
            npv=ValueOfField(par)
        else
            npv=0.0
        end
        if FieldToPayDebt
            bb=b-npv
            return qq*(bprime-(1-γ)*bb)-(γ+κ*(1-γ))*bb
        else
            return qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b+npv
        end
    else
        return qq*(bprime-(1-γ)*b)-(γ+κ*(1-γ))*b
    end
end

function Output_Consumption_and_g(IsDefault::Bool,z::Float64,n_ind::Int64,Tr::Float64,MODEL::Model)
    @unpack par, GRIDS = MODEL
    @unpack SellField, FieldToPayDebt, OilEmbargo, DomesticField, μ_z, nL, τ = par
    @unpack GR_n = GRIDS
    n=GR_n[n_ind]

    if IsDefault
        if DomesticField
            y=z+n
            yD=y-DefaultCost(y,μ_z+nL,par)
            c=(1-τ)*yD
            g=τ*yD
            gdp=yD
            return c, g, gdp
        else
            y=z
            yD=y-DefaultCost(y,μ_z,par)
            if OilEmbargo
                nD=n-DefaultCost(n,nL,par)
                c=(1-τ)*yD
                g=τ*yD+nD
                gdp=yD+nD
                return c, g, gdp
            else
                if SellField
                    if n_ind==2
                        npv=ValueOfField(par)
                    else
                        npv=0.0
                    end
                    nD=nL
                    c=(1-τ)*yD
                    g=τ*yD+nD+npv
                    gdp=yD+n
                    return c, g, gdp
                else
                    nD=n
                    c=(1-τ)*yD
                    g=τ*yD+nD
                    gdp=yD+nD
                    return c, g, gdp
                end
            end
        end
    else
        if DomesticField
            y=z+n
            c=(1-τ)*y
            g=τ*y+Tr
            gdp=y
            return c, g, gdp
        else
            if SellField
                #Sell proceeds come in Tr
                y=z
                c=(1-τ)*y
                g=τ*y+nL+Tr
                gdp=y+n
                return c, g, gdp
            else
                y=z
                c=(1-τ)*y
                g=τ*y+n+Tr
                gdp=y+n
                return c, g, gdp
            end
        end
    end
end

###############################################################################
#Functions to compute value given state, policies, and guesses
###############################################################################
function Period_Utility(IsDefault::Bool,I::CartesianIndex,bprime::Float64,MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack cmin = par
    @unpack GR_z, GR_n, GR_b = GRIDS
    if IsDefault
        (z_ind,n_ind)=Tuple(I)
        z=GR_z[z_ind]; n=GR_n[n_ind]
        Tr=0.0
        cons, g, y=Output_Consumption_and_g(true,z,n_ind,Tr,MODEL)
        return Utility(cons,g,par)
    else
        @unpack SOLUTION = MODEL
        @unpack itp_q1 = SOLUTION
        (b_ind,z_ind,n_ind)=Tuple(I)
        z=GR_z[z_ind]; n=GR_n[n_ind]
        b=GR_b[b_ind]
        qq=itp_q1(bprime,z,n_ind)
        Tr=Calculate_Tr(n_ind,qq,b,bprime,par)
        cons, g, y=Output_Consumption_and_g(false,z,n_ind,Tr,MODEL)
        if g>0.0 && cons>0.0
            return Utility(cons,g,par)
        else
            if g>0.0 && cons<=0.0
                return Utility(cmin,g,par)+cons
            else
                if g<=0.0 && cons>0.0
                    return Utility(cons,cmin,par)+g
                else
                    return Utility(cmin,cmin,par)+g+cons
                end
            end
        end
    end
end

function ValueInDefault(I::CartesianIndex,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack β, θ, cmin = par
    @unpack GR_z = GRIDS
    @unpack itp_EV, itp_EVD = SOLUTION
    (z_ind,n_ind)=Tuple(I)
    z=GR_z[z_ind]
    U=Period_Utility(true,I,0.0,MODEL)

    CON_REP=itp_EV(0.0,z,n_ind)
    CON_DEF=itp_EVD(z,n_ind)

    return U+β*θ*CON_REP+β*(1.0-θ)*CON_DEF
end

function ValueInRepayment(bprime::Float64,I::CartesianIndex,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack β, cmin = par
    @unpack GR_z, GR_b = GRIDS
    @unpack itp_EV, itp_q1 = SOLUTION
    (b_ind,z_ind,n_ind)=Tuple(I)
    z=GR_z[z_ind]
    U=Period_Utility(false,I,bprime,MODEL)
    CON_REP=itp_EV(bprime,z,n_ind)

    return U+β*CON_REP
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
    @unpack itp_q1 = SOLUTION
    (b_ind,z_ind,μ_ind)=Tuple(I)
    #Get guess and bounds for bprime
    blowOpt, bhighOpt=BoundsSearch_B(I,MODEL)

    #Setup function handle for optimization
    f(bprime::Float64)=-ValueInRepayment(bprime,I,MODEL)

    #Perform optimization
    inner_optimizer = GoldenSection()
    res=optimize(f,blowOpt,bhighOpt,inner_optimizer)

    return -Optim.minimum(res), Optim.minimizer(res)
end

###############################################################################
#Update default
###############################################################################
function UpdateDefault!(MODEL::Model)
    @unpack GRIDS, par = MODEL
    #Loop over all states to fill array of VD
    for I in CartesianIndices(MODEL.SOLUTION.VD)
        MODEL.SOLUTION.VD[I]=ValueInDefault(I,MODEL)
    end
    MODEL.SOLUTION.itp_VD=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.VD,true,GRIDS)

    ComputeExpectationOverStates!(true,MODEL.SOLUTION.VD,MODEL.SOLUTION.EVD,GRIDS,par)
    MODEL.SOLUTION.itp_EVD=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EVD,true,GRIDS)
    return nothing
end

###############################################################################
#Update repayment
###############################################################################
function RepaymentUpdater!(I::CartesianIndex,MODEL::Model)
    (b_ind,z_ind,μ_ind)=Tuple(I)
    MODEL.SOLUTION.VP[I], MODEL.SOLUTION.bprime[I]=OptimInRepayment(I,MODEL)

    if MODEL.SOLUTION.VP[I]<MODEL.SOLUTION.VD[z_ind,μ_ind]
        MODEL.SOLUTION.V[I]=MODEL.SOLUTION.VD[z_ind,μ_ind]
    else
        MODEL.SOLUTION.V[I]=MODEL.SOLUTION.VP[I]
    end

    return nothing
end

function UpdateRepayment!(MODEL::Model)
    @unpack GRIDS, par = MODEL
    #Loop over all states to fill value of repayment
    for I in CartesianIndices(MODEL.SOLUTION.VP)
        RepaymentUpdater!(I,MODEL)
    end
    MODEL.SOLUTION.itp_VP=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.VP,false,GRIDS)
    MODEL.SOLUTION.itp_V=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.V,false,GRIDS)
    MODEL.SOLUTION.itp_bprime=CreateInterpolation_PolicyFunctions(MODEL.SOLUTION.bprime,GRIDS)

    ComputeExpectationOverStates!(false,MODEL.SOLUTION.V,MODEL.SOLUTION.EV,GRIDS,par)

    MODEL.SOLUTION.itp_EV=CreateInterpolation_ValueFunctions(MODEL.SOLUTION.EV,false,GRIDS)
    return nothing
end

###############################################################################
#Update bond prices
###############################################################################
function BondsPayoff(nprime_ind::Int64,zprime::Float64,bprime::Float64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack γ, κ = par
    @unpack itp_VP, itp_VD, itp_q1, itp_bprime = SOLUTION
    @unpack GR_z, GR_b = GRIDS
    if itp_VD(zprime,nprime_ind)>itp_VP(bprime,zprime,nprime_ind)
        return 0.0
    else
        SDF=SDF_Lenders(par)
        if γ==1.0
            return SDF
        else
            bb=itp_bprime(bprime,zprime,nprime_ind)
            qq=itp_q1(bb,zprime,nprime_ind)
            return SDF*(γ+(1.0-γ)*(κ+qq))
        end
    end
end

function Bonds_Expectation_over_zprime(z_ind::Int64,n_ind::Int64,nprime_ind::Int64,bprime::Float64,MODEL::Model)
    @unpack GRIDS = MODEL
    #foo is a function of floats for z'
    #μ, μ', and b'
    int=0.0
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    for i in 1:length(ϵz_weights)
        int=int+ϵz_weights[i]*PDFz[z_ind,i]*BondsPayoff(nprime_ind,ZPRIME[z_ind,i],bprime,MODEL)
    end
    return int/FacQz
end

function UpdateBondsPrices!(MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack PI_n, GR_b = GRIDS
    #It will use MAT to compute expectations
    #It will mutate E_MAT
    for I in CartesianIndices(MODEL.SOLUTION.q1)
        (bprime_ind,z_ind,n_ind)=Tuple(I)
        bprime=GR_b[bprime_ind]
        int=0.0
        for nprime_ind in 1:par.Nn
            if PI_n[n_ind,nprime_ind]>0.0
                Eqz=Bonds_Expectation_over_zprime(z_ind,n_ind,nprime_ind,bprime,MODEL)
                int=int+PI_n[n_ind,nprime_ind]*Eqz
            end
        end
        MODEL.SOLUTION.q1[I]=int
    end

    MODEL.SOLUTION.itp_q1=CreateInterpolation_Price(MODEL.SOLUTION.q1,GRIDS)
    return nothing
end

###############################################################################
#Update bond prices with robustness
###############################################################################
function Numerator_for_bonds(nprime_ind::Int64,zprime::Float64,bprime::Float64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack γ, κ, φ = par
    @unpack itp_VP, itp_VD, itp_q1, itp_bprime = SOLUTION
    @unpack GR_z, GR_b = GRIDS

    if itp_VD(zprime,nprime_ind)>itp_VP(bprime,zprime,nprime_ind)
        return 0.0
    else
        SDF=SDF_Lenders(par)
        if γ==1.0
            V1=bprime
            return SDF*exp(-V1/φ)
        else
            bb=itp_bprime(bprime,zprime,nprime_ind)
            qq=itp_q1(bb,zprime,nprime_ind)
            V1=(γ+(1.0-γ)*(κ+qq))*bprime
            return SDF*exp(-V1/φ)*(γ+(1.0-γ)*(κ+qq))
        end
    end
end

function Denominator_for_bonds(nprime_ind::Int64,zprime::Float64,bprime::Float64,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack γ, κ, φ = par
    @unpack itp_VP, itp_VD, itp_q1, itp_bprime = SOLUTION
    @unpack GR_z, GR_b = GRIDS

    if γ==1.0
        V1rep=bprime
    else
        bb=itp_bprime(bprime,zprime,nprime_ind)
        qq=itp_q1(bb,zprime,nprime_ind)
        V1rep=(γ+(1.0-γ)*(κ+qq))*bprime
    end

    if itp_VD(zprime,nprime_ind)>itp_VP(bprime,zprime,nprime_ind)
        return 1.0
    else
        return exp(-V1rep/φ)
    end
end

function RobustBonds_Expectations_over_zprime(z_ind::Int64,n_ind::Int64,nprime_ind::Int64,bprime::Float64,MODEL::Model)
    @unpack GRIDS = MODEL
    #foo is a function of floats for z'
    #μ, μ', and b'
    int_num=0.0
    int_den=0.0
    @unpack ϵz_weights, ZPRIME, PDFz, FacQz = GRIDS
    for i in 1:length(ϵz_weights)
        int_num=int_num+ϵz_weights[i]*PDFz[z_ind,i]*Numerator_for_bonds(nprime_ind,ZPRIME[z_ind,i],bprime,MODEL)
        int_den=int_den+ϵz_weights[i]*PDFz[z_ind,i]*Denominator_for_bonds(nprime_ind,ZPRIME[z_ind,i],bprime,MODEL)
    end
    return int_num/FacQz, int_den/FacQz
end

###############################################################################
#VFI algorithm
###############################################################################
function UpdateSolution!(MODEL::Model)
    UpdateDefault!(MODEL)
    UpdateRepayment!(MODEL)
    UpdateBondsPrices!(MODEL)
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

function SolveModel_VFI(PrintProgress::Bool,GRIDS::Grids,par::Pars)
    @unpack Tol_q, Tol_V, cnt_max = par
    if PrintProgress
        println("Preparing solution guess")
    end
    SOLUTION_CURRENT=InitiateEmptySolution(GRIDS,par)
    MODEL_CURRENT=Model(SOLUTION_CURRENT,GRIDS,par)
    MODEL_NEXT=deepcopy(MODEL_CURRENT)

    if PrintProgress
        println("Starting VFI")
    end
    dst_V=1.0; dst_q=1.0; cnt=0
    while ((dst_V>Tol_V) || (dst_q>Tol_q)) && cnt<cnt_max
        UpdateSolution!(MODEL_NEXT)
        dst_q, NotConvPct=ComputeDistance_q(MODEL_CURRENT,MODEL_NEXT)
        dst_D, dst_P=ComputeDistance_V(MODEL_CURRENT,MODEL_NEXT)
        dst_V=max(dst_D,dst_P)
        cnt=cnt+1
        MODEL_CURRENT=deepcopy(MODEL_NEXT)
        if PrintProgress
            println("cnt=$cnt,  dst_D=$dst_D, dst_P=$dst_P, dst_q=$dst_q")
        end
    end
    return MODEL_NEXT
end

function SolveAndSaveModel_VFI(PrintProgress::Bool,NAME::String,GRIDS::Grids,par::Pars)
    MODEL=SolveModel_VFI(PrintProgress,GRIDS,par)
    SaveModel_Vector(NAME,MODEL)
    return nothing
end

################################################################################
### Functions for simulations
################################################################################
@with_kw mutable struct Paths
    #Paths of shocks
    z::Array{Float64,1}
    n_ind::Array{Int64,1}
    n::Array{Float64,1}

    #Paths of chosen states
    Def::Array{Float64,1}
    B::Array{Float64,1}

    #Path of relevant variables
    Spreads::Array{Float64,1}
    GDP::Array{Float64,1}
    L::Array{Float64,1}
    Cons::Array{Float64,1}
    G::Array{Float64,1}
    TB::Array{Float64,1}
    CA::Array{Float64,1}

    #Other variables
    τ::Array{Float64,1}
end

function InitiateEmptyPaths(T::Int64)
    #Initiate with zeros to facilitate averages
    #Paths of shocks
    f1=zeros(Float64,T)
    i2=zeros(Int64,T)
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
    return Paths(f1,i2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13)
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

function Simulate_Discrete_nShocks!(PATHS::Paths,MODEL::Model)
    @unpack GRIDS, par = MODEL
    @unpack PI_n, GR_n = GRIDS
    T=length(PATHS.n_ind)
    #initial μ state is defined outside this function
    for t in 2:T
        PATHS.n_ind[t]=Draw_New_discreteShock(PATHS.n_ind[t-1],PI_n)
        PATHS.n[t]=GR_n[PATHS.n_ind[t]]
    end
    return nothing
end

function Simulate_z_shocks!(PATHS::Paths,MODEL::Model)
    #PATHS must already have a simulated vector of μ
    @unpack par = MODEL
    @unpack ρ_z, μ_z, dist_ϵz = par
    T=length(PATHS.z)
    ϵz_TS=rand(dist_ϵz,T)

    PATHS.z[1]=1.0

    for t in 2:T
        PATHS.z[t]=exp((1.0-ρ_z)*log(μ_z)+ρ_z*log(PATHS.z[t-1])+ϵz_TS[t])
    end
    return nothing
end

function CalculateSpreads(qq::Float64,par::Pars)
    @unpack γ, κ, r_star = par
    ib=-log(qq/(γ+(1.0-γ)*(κ+qq)))
    ia=((1.0+ib)^1.0)-1.0
    rf=((1.0+r_star)^1.0)-1.0
    return 100.0*(ia-rf)
end

function GenerateNextState!(t::Int64,PATHS::Paths,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack itp_VD, itp_VP, itp_bprime, itp_q1 = SOLUTION
    @unpack θ, SellField, FieldToPayDebt = par
    #It must be t>=2, initial state defined outside of this function

    #Unpack current exogenous state
    z=PATHS.z[t]; n_ind=PATHS.n_ind[t]

    #Unpack current endogenous state and previous default
    if t==1
        d_=PATHS.Def[t]
    else
        #Def is the default choice in the previous period
        d_=PATHS.Def[t-1]
    end
    b=PATHS.B[t]

    #Update next endogenous state
    if d_==1.0
        #Coming from default state yesterday, must draw for readmission
        if rand()<=θ
            #Get readmitted, choose whether to default or not today
            if itp_VD(z,n_ind)<=itp_VP(b,z,n_ind)
                #Choose not to default
                PATHS.Def[t]=0.0
                #Check if it is not final period
                if t<length(PATHS.z)
                    #Fill b'
                    PATHS.B[t+1]=max(itp_bprime(b,z,n_ind),0.0)
                    qq=itp_q1(PATHS.B[t+1],z,n_ind)
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
                    #Fill b'
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
                #Fill b'
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
        if itp_VD(z,n_ind)<=itp_VP(b,z,n_ind)
            #Choose not to default
            PATHS.Def[t]=0.0
            #Check if it is not final period
            if t<length(PATHS.z)
                #Fill kO' and b'
                PATHS.B[t+1]=max(itp_bprime(b,z,n_ind),0.0)
                qq=itp_q1(PATHS.B[t+1],z,n_ind)
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

function SimulateEndogenousStates!(PATHS::Paths,MODEL::Model)
    #PATHS has initial state defined outside of this function
    T=length(PATHS.z)
    for t in 1:T
        GenerateNextState!(t,PATHS,MODEL)
    end
end

function GenerateOtherPathVariables!(t::Int64,PATHS::Paths,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack γ, κ, SellField, FieldToPayDebt = par
    @unpack itp_bprime, itp_q1 = SOLUTION
    #It must be t>=2, initial state defined outside of this function

    #Unpack current exogenous state
    z=PATHS.z[t]; n_ind=PATHS.n_ind[t]; n=PATHS.n[t]

    #Unpack current endogenous state and previous default
    if t==1
        d_=PATHS.Def[t]
    else
        #Def is the default choice in the previous period
        d_=PATHS.Def[t-1]
    end
    b=PATHS.B[t]

    #Compute policies
    if d_==1
        #in default
        bprime=0.0
        qq=0.0
        Tr=0.0
        c, g, gdp=Output_Consumption_and_g(true,z,n_ind,Tr,MODEL)
        PATHS.L[t]=0.0
        PATHS.GDP[t]=gdp
        PATHS.Cons[t]=c
        PATHS.G[t]=g
        PATHS.TB[t]=0.0
        PATHS.CA[t]=0.0
    else
        #in repayment
        if t==length(PATHS.B)
            #end of time, use policy functions for next state
            @unpack itp_bprime, itp_q1 = SOLUTION
            bprime=itp_bprime(PATHS.B[t],z,n_ind)
        else
            #Not end of time, use next state
            bprime=PATHS.B[t+1]
        end
        qq=itp_q1(bprime,z,n_ind)
        Tr=Calculate_Tr(n_ind,qq,b,bprime,par)
        c, g, gdp=Output_Consumption_and_g(false,z,n_ind,Tr,MODEL)
        PATHS.L[t]=0.0
        PATHS.GDP[t]=gdp
        PATHS.Cons[t]=c
        PATHS.G[t]=g
        PATHS.TB[t]=PATHS.GDP[t]-PATHS.Cons[t]-PATHS.G[t]
        PATHS.CA[t]=-(bprime-PATHS.B[t]) #Change in net foreign assets, current prices
    end
    if SellField
        if FieldToPayDebt
            if n_ind==2
                npv=ValueOfField(par)
            else
                npv=0.0
            end
            PATHS.B[t]=PATHS.B[t]-npv
        end
    end

    return nothing
end

function SimulateOtherPathVariables!(PATHS::Paths,MODEL::Model)
    #PATHS has time series of states already simulated
    T=length(PATHS.z)
    for t in 1:T
        GenerateOtherPathVariables!(t,PATHS,MODEL)
    end
end

function SimulateRandomPath(T::Int64,MODEL::Model)
    PATHS=InitiateEmptyPaths(T)
    PATHS.n_ind[1]=1
    if PATHS.n_ind[1]==1
        PATHS.n[1]=MODEL.par.nL
    else
        PATHS.n[1]=MODEL.par.nH
    end

    Simulate_Discrete_nShocks!(PATHS,MODEL)
    Simulate_z_shocks!(PATHS,MODEL)
    SimulateEndogenousStates!(PATHS,MODEL)
    SimulateOtherPathVariables!(PATHS,MODEL)

    return PATHS
end

################################################################################
### Functions to create average paths of discovery
################################################################################
function Extract_TS(t1::Int64,tT::Int64,PATHS::Paths)
    return Paths(PATHS.z[t1:tT],PATHS.n_ind[t1:tT],PATHS.n[t1:tT],PATHS.Def[t1:tT],PATHS.B[t1:tT],PATHS.Spreads[t1:tT],PATHS.GDP[t1:tT],PATHS.L[t1:tT],PATHS.Cons[t1:tT],PATHS.G[t1:tT],PATHS.TB[t1:tT],PATHS.CA[t1:tT],PATHS.τ[t1:tT])
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
    SimulateEndogenousStates!(PATHS,MODEL)
    SimulateOtherPathVariables!(PATHS,MODEL)

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
    PATHS_AV.z=PATHS_AV.z .+ (PATHS.z ./ N)
    PATHS_AV.n=PATHS_AV.n .+ (PATHS.n ./ N)

    #Paths of chosen states
    PATHS_AV.Def=PATHS_AV.Def .+ (PATHS.Def ./ N)
    PATHS_AV.B=PATHS_AV.B .+ (PATHS.B ./ N)

    #Path of relevant variables
    PATHS_AV.Spreads=PATHS_AV.Spreads .+ (PATHS.Spreads ./ N)
    PATHS_AV.GDP=PATHS_AV.GDP .+ (PATHS.GDP ./ N)
    PATHS_AV.L=PATHS_AV.L .+ (PATHS.L ./ N)
    PATHS_AV.Cons=PATHS_AV.Cons .+ (PATHS.Cons ./ N)
    PATHS_AV.G=PATHS_AV.G .+ (PATHS.G ./ N)
    PATHS_AV.TB=PATHS_AV.TB .+ (PATHS.TB ./ N)
    PATHS_AV.CA=PATHS_AV.CA .+ (PATHS.CA ./ N)
    PATHS_AV.τ=PATHS_AV.τ .+ (PATHS.τ ./ N)
    return nothing
end

function AverageDiscoveryPaths(DropDefaults::Bool,N::Int64,Tbefore::Int64,Tafter::Int64,MODEL::Model)
    @unpack par = MODEL
    @unpack drp = par
    Random.seed!(1234)
    PATHS_AV=InitiateEmptyPaths(Tbefore+1+Tafter)
    for i in 1:N
        PATHS=SimulatePathsOfDiscovery(DropDefaults,Tbefore,Tafter,MODEL)
        SumPathForAverage!(N,PATHS_AV,PATHS)
    end
    return PATHS_AV
end

################################################################################
### Functions to compute moments
################################################################################
@with_kw mutable struct Moments
    #Initiate them at 0.0 to facilitate average across samples
    #Default, spreads, and Debt
    DefaultPr::Float64 = 0.0
    Av_Spreads::Float64 = 0.0
    Debt_GDP::Float64 = 0.0
    #Volatilities
    σ_Spreads::Float64 = 0.0
    σ_GDP::Float64 = 0.0
    σ_con::Float64 = 0.0
    σ_TB::Float64 = 0.0
    σ_CA::Float64 = 0.0
    #Cyclicality
    Corr_Spreads_GDP::Float64 = 0.0
    Corr_con_GDP::Float64 = 0.0
    Corr_TB_GDP::Float64 = 0.0
    Corr_CA_GDP::Float64 = 0.0
    #Other moments
    PeakSpreadResponse::Float64 = 0.0
end

function ColumnVectorMomentNames()
    NAMES=["DefaultPr";
                "Av_Spreads";
                "Debt_GDP";
                "std_Spreads";
                "std_GDP";
                "std_con";
                "std_TB";
                "std_CA";
                "Corr_Spreads_GDP";
                "Corr_con_GDP";
                "Corr_TB_GDP";
                "Corr_CA_GDP";
                "PeakSpreadResponse"]

    return NAMES
end

function RowVectorMomentNames()
    NAMES=["DefaultPr" "Av_Spreads" "Debt_GDP" "std_Spreads" "std_GDP" "std_con" "std_TB" "std_CA" "Corr_Spreads_GDP" "Corr_con_GDP" "Corr_TB_GDP" "Corr_CA_GDP" "PeakSpreadResponse"]

    return NAMES
end

function TimeSeriesForMoments_LongRun(T_LongRun::Int64,Conditional::Bool,Only_nL::Bool,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack drp = par

    PATHS = InitiateEmptyPaths(drp+T_LongRun)
    if Conditional
        if Only_nL
            @unpack nL = par
            PATHS.n_ind .= ones(Int64,length(PATHS.n_ind))
            PATHS.n .= nL*ones(Float64,length(PATHS.n_ind))
        else
            @unpack nH, Nn = par
            PATHS.n_ind .= Nn*ones(Int64,length(PATHS.n_ind))
            PATHS.n .= nH*ones(Float64,length(PATHS.n_ind))
        end
        Simulate_z_shocks!(PATHS,MODEL)
        SimulateEndogenousStates!(PATHS,MODEL)
        SimulateOtherPathVariables!(PATHS,MODEL)
        return Extract_TS(drp+1,drp+T_LongRun,PATHS)
    else
        PATHS = SimulateRandomPath(drp+T_LongRun,MODEL)
        return Extract_TS(drp+1,drp+T_LongRun,PATHS)
    end
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

function ComputeMomentsIntoStructure_LongRun!(MOM::Moments,T_LongRun::Int64,Conditional::Bool,Only_nL::Bool,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    @unpack HPFilter_Par = par

    #Draw long time series for default frequency with very long series
    PATHS=TimeSeriesForMoments_LongRun(T_LongRun,Conditional,Only_nL,MODEL)
    DefaultEvents=0.0
    for t in 2:length(PATHS.Def)
        if PATHS.Def[t]==1.0 && PATHS.Def[t-1]==0.0
            DefaultEvents=DefaultEvents+1
        end
    end
    MOM.DefaultPr=100*DefaultEvents/sum(1 .- PATHS.Def)
    GoodStanding = 1 .- PATHS.Def

    #Compute average spreads, only periods in good standing
    MOM.Av_Spreads=sum(PATHS.Spreads .* GoodStanding)/(sum(GoodStanding)+sqrt(eps(Float64)))

    #Compute the average debt-to-GDP ratio, only periods in good standing
    b_gdp=100*(PATHS.B ./ PATHS.GDP)
    MOM.Debt_GDP=sum(b_gdp .* GoodStanding)/(sum(GoodStanding)+sqrt(eps(Float64)))

    #Business cycles
    #Compute the natural logarithm of variables
    ln_y=log.(abs.(PATHS.GDP))
    ln_c=log.(abs.(PATHS.Cons))

    #HP-Filtering
    y_trend=hp_filter(ln_y,HPFilter_Par)
    c_trend=hp_filter(ln_c,HPFilter_Par)

    y_cyc=100.0*(ln_y .- y_trend)
    c_cyc=100.0*(ln_c .- c_trend)

    #Volatilities
    MOM.σ_GDP=std(y_cyc)
    MOM.σ_con=std(c_cyc)

    spr2=PATHS.Spreads .^ 2
    Espr2=sum(spr2 .* GoodStanding)/(sum(GoodStanding)+sqrt(eps(Float64)))
    MOM.σ_Spreads=sqrt(Espr2-(MOM.Av_Spreads .^ 2))

    tb_y=100*(PATHS.TB ./ PATHS.GDP)
    ca_y=100*(PATHS.CA ./ PATHS.GDP)
    MOM.σ_TB=std(tb_y)
    MOM.σ_CA=std(ca_y)

    #Cyclicality
    MOM.Corr_Spreads_GDP=cor(y_cyc,PATHS.Spreads)
    MOM.Corr_con_GDP=cor(y_cyc,c_cyc)
    MOM.Corr_TB_GDP=cor(y_cyc,tb_y)
    MOM.Corr_CA_GDP=cor(y_cyc,ca_y)

    #Other moments
    MOM.PeakSpreadResponse=0.0#maximum(ts_sp)

    return nothing
end

function SumMomentsForAverage!(N::Int64,MOM_AV::Moments,MOM::Moments)
    #Default, spreads, and Debt
    MOM_AV.DefaultPr=MOM_AV.DefaultPr+MOM.DefaultPr/N
    MOM_AV.Av_Spreads=MOM_AV.Av_Spreads+MOM.Av_Spreads/N
    MOM_AV.Debt_GDP=MOM_AV.Debt_GDP+MOM.Debt_GDP/N
    #Volatilities
    MOM_AV.σ_GDP=MOM_AV.σ_GDP+MOM.σ_GDP/N
    MOM_AV.σ_Spreads=MOM_AV.σ_Spreads+MOM.σ_Spreads/N
    MOM_AV.σ_con=MOM_AV.σ_con+MOM.σ_con/N
    MOM_AV.σ_TB=MOM_AV.σ_TB+MOM.σ_TB/N
    MOM_AV.σ_CA=MOM_AV.σ_CA+MOM.σ_CA/N
    #Cyclicality
    MOM_AV.Corr_Spreads_GDP=MOM_AV.Corr_Spreads_GDP+MOM.Corr_Spreads_GDP/N
    MOM_AV.Corr_con_GDP=MOM_AV.Corr_con_GDP+MOM.Corr_con_GDP/N
    MOM_AV.Corr_TB_GDP=MOM_AV.Corr_TB_GDP+MOM.Corr_TB_GDP/N
    MOM_AV.Corr_CA_GDP=MOM_AV.Corr_CA_GDP+MOM.Corr_CA_GDP/N

    #Other moments
    MOM_AV.PeakSpreadResponse=MOM_AV.PeakSpreadResponse+MOM.PeakSpreadResponse/N

    return nothing
end

function AverageMomentsManySamples_LongRun(N::Int64,T_LongRun::Int64,Conditional::Bool,Only_nL::Bool,MODEL::Model)
    @unpack SOLUTION, GRIDS, par = MODEL
    #Initiate average moments at 0
    Random.seed!(1234)
    AV_MOM=Moments()
    MOM=Moments()
    for i in 1:N
        ComputeMomentsIntoStructure_LongRun!(MOM,T_LongRun,Conditional,Only_nL,MODEL)
        SumMomentsForAverage!(N,AV_MOM,MOM)
    end
    return AV_MOM
end

function ColumnVectorFromMoments(MOM::Moments)
    VEC=vcat(MOM.DefaultPr,MOM.Av_Spreads)
    VEC=vcat(VEC,MOM.Debt_GDP)
    VEC=vcat(VEC,MOM.σ_Spreads)
    VEC=vcat(VEC,MOM.σ_GDP)
    VEC=vcat(VEC,MOM.σ_con)
    VEC=vcat(VEC,MOM.σ_TB)
    VEC=vcat(VEC,MOM.σ_CA)
    VEC=vcat(VEC,MOM.Corr_Spreads_GDP)
    VEC=vcat(VEC,MOM.Corr_con_GDP)
    VEC=vcat(VEC,MOM.Corr_TB_GDP)
    VEC=vcat(VEC,MOM.Corr_CA_GDP)
    return vcat(VEC,MOM.PeakSpreadResponse)
end

################################################################################
### Functions to calibrate parameters by matching moments
################################################################################
function Setup_MomentMatching(β::Float64,knk::Float64,VEC_PAR::Array{Float64,1})
    par=UnpackParameters_Vector(VEC_PAR)
    par=Pars(par,β=β,knk=knk)
    GRIDS=CreateGrids(par)

    return par, GRIDS
end

function SolveModel_VFI_ForMomentMatching(GRIDS::Grids,par::Pars)
    #Solve model
    PrintProgress=false
    MOD=SolveModel_VFI(PrintProgress,GRIDS,par)
    N=100; T_LongRun=10000

    Conditional=false; Only_nL=false
    MOM_u=AverageMomentsManySamples_LongRun(N,T_LongRun,Conditional,Only_nL,MOD)

    Conditional=true; Only_nL=true
    MOM_nL=AverageMomentsManySamples_LongRun(N,T_LongRun,Conditional,Only_nL,MOD)

    Conditional=true; Only_nL=false
    MOM_nH=AverageMomentsManySamples_LongRun(N,T_LongRun,Conditional,Only_nL,MOD)

    return MOM_u, MOM_nL, MOM_nH
end

function CheckMomentsForTry(PARS_TRY::Array{Float64,1},VEC_PAR::Array{Float64,1})
    β=PARS_TRY[1]
    knk=PARS_TRY[2]
    par, GRIDS=Setup_MomentMatching(β,knk,VEC_PAR)
    return SolveModel_VFI_ForMomentMatching(GRIDS,par)
end

function CalibrateMatchingMoments(N::Int64,lb::Vector{Float64},ub::Vector{Float64},country_column::Int64)
    XX=readdlm("Setup.csv",',')
    VEC_PAR=XX[2:end,country_column]*1.0

    #Name file, case
    FILE_NAME_u="TriedCalibrations, Unconditional moments.csv"
    FILE_NAME_nL="TriedCalibrations, Moments conditional on n=nL.csv"
    FILE_NAME_nH="TriedCalibrations, Moments conditional on n=nH.csv"

    #Generate Sobol sequence
    MAT_TRY=Array{Float64,2}(undef,N,length(lb))
    i=0; skp=0
    s=SobolSeq(lb, ub)
    for x in skip(s,skp,exact=true)
        i=i+1
        MAT_TRY[i,:]=x
        if i==N
            break
        end
    end

    #Loop paralelly over all parameter tries
    #There are Nmom moments, columns should be Nmom+number of parameters
    MOMENT_NAMES=RowVectorMomentNames()
    N_moments=length(MOMENT_NAMES)
    N_parameters=length(lb)
    N_columns=N_moments+N_parameters
    MATRIX_u=SharedArray{Float64,2}(N,N_columns)
    MATRIX_nL=SharedArray{Float64,2}(N,N_columns)
    MATRIX_nH=SharedArray{Float64,2}(N,N_columns)
    COL_NAMES=hcat(["beta" "knk"],MOMENT_NAMES)
    @sync @distributed for i in 1:N
        MATRIX_u[i,1:N_parameters]=MAT_TRY[i,:]
        MATRIX_nL[i,1:N_parameters]=MAT_TRY[i,:]
        MATRIX_nH[i,1:N_parameters]=MAT_TRY[i,:]
        MOM_u, MOM_nL, MOM_nH=CheckMomentsForTry(MAT_TRY[i,:],VEC_PAR)

        MATRIX_u[i,N_parameters+1:end]=ColumnVectorFromMoments(MOM_u)'
        MATRIX_nL[i,N_parameters+1:end]=ColumnVectorFromMoments(MOM_nL)'
        MATRIX_nH[i,N_parameters+1:end]=ColumnVectorFromMoments(MOM_nH)'

        MAT_u=[COL_NAMES; MATRIX_u]
        MAT_nL=[COL_NAMES; MATRIX_nL]
        MAT_nH=[COL_NAMES; MATRIX_nH]

        writedlm(FILE_NAME_u,MAT_u,',')
        writedlm(FILE_NAME_nL,MAT_nL,',')
        writedlm(FILE_NAME_nH,MAT_nH,',')
        println("Done with $i of $N")
    end

    return nothing
end

################################################################################
### Welfare analysis
################################################################################
function WelfareGains_one_Discovery(d::Float64,z::Float64,b::Float64,MODEL::Model)
    @unpack SOLUTION, par = MODEL
    @unpack σ = par
    if d==1.0
        @unpack itp_VD = SOLUTION
        v0=itp_VD(z,1)
        v1=itp_VD(z,2)
    else
        @unpack itp_V = SOLUTION
        v0=itp_V(b,z,1)
        v1=itp_V(b,z,2)
    end
    return 100*(((v1/v0)^(1/(1-σ)))-1)
end

function AverageWelfareGains(N::Int64,MODEL::Model)
    Conditional=true; Only_nL=true
    TS=TimeSeriesForMoments_LongRun(N,Conditional,Only_nL,MODEL)
    wg=0.0
    for i in 1:N
        z=TS.z[i]; b=TS.B[i]; d=TS.Def[i]
        wg=wg+WelfareGains_one_Discovery(d,z,b,MODEL)/N
    end
    return wg
end
