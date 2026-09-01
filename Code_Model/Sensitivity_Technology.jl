################################################################################
### Sensitivity of the mechanism to the technology gaps between the two sectors
###
### Figure 8 (Figure_Technology_Exposure) and Figure 9 (Figure_Technology_Responses),
### plus Figures 20 to 23 from Figures_Armington_Sensitivity and
### Figures_DixitStiglitz_Sensitivity. The same variants are the rows of Table 3.
###
### Needs Primitives.jl and ModelResults.jl included first. Figure 9 needs the four
### files in VARIANT_FILES solved and saved in Code_Model, which PaperResults.jl does
### with Solve_Technology_Variants. Parameter changes are internal Pars overrides;
### Setup_Calibrated.csv is read but never written
################################################################################

################################################################################
### Relative exposure of oil to the cost of default, over the three technology
### pairs (αMf,αMo), (λf,λo), (θf,θo)
###
### The statistic is R = (% drop of oil output in default)/(% drop of final
### output in default): R<1 is oil insulated, R=1 is the sign flip
################################################################################

function Pars_NewTechnology(par::Pars;αMf::Float64=par.αMf,λf::Float64=par.λf,
                            θf::Float64=par.θf,αMo::Float64=par.αMo,
                            λo::Float64=par.λo,θo::Float64=par.θo,
                            PickBoth::Bool=true)
    #Reconstructing Pars does not re-evaluate derived defaults, so αN is set by hand.
    #nL and nH are re-picked so oil/GDP stays at the calibration targets; PickBoth=false
    #skips nL for speed when only nH will be used
    par_new=Pars(par,αMf=αMf,λf=λf,θf=θf,αMo=αMo,λo=λo,θo=θo,αN=1.0-αMo-par.αLo)
    nH=Pick_n_Given_Target(par_new.yoil_GDP_target_H,par_new)
    if PickBoth
        nL=Pick_n_Given_Target(par_new.yoil_GDP_target_L,par_new)
        return Pars(par_new,nL=nL,nH=nH)
    else
        return Pars(par_new,nH=nH)
    end
end

function Sector_Drops_InDefault(z::Float64,n::Float64,par::Pars)
    xD=State(true,1,n,z,0.0)
    xR=State(false,1,n,z,0.0)
    gdpfD, gdpoD, LD, yD, yoD, wkD=Compute_Quantities_GivenState(xD,par)
    gdpfR, gdpoR, LR, yR, yoR, wkR=Compute_Quantities_GivenState(xR,par)
    drop_f=100*((yD/yR)-1)
    drop_o=100*((yoD/yoR)-1)
    return drop_f, drop_o
end

function Relative_Exposure_Oil(par::Pars;Nz_band::Int64=11)
    #R at mean z and its extremes over the z grid, at the large field
    @unpack μ_z, zlow, zhigh, nH = par
    drop_f, drop_o=Sector_Drops_InDefault(μ_z,nH,par)
    R_mid=drop_o/drop_f
    R_lo=R_mid; R_hi=R_mid
    for z in range(zlow,stop=zhigh,length=Nz_band)
        drop_fz, drop_oz=Sector_Drops_InDefault(z,nH,par)
        Rz=drop_oz/drop_fz
        R_lo=min(R_lo,Rz)
        R_hi=max(R_hi,Rz)
    end
    return R_mid, R_lo, R_hi
end

function Sweep_Relative_Exposure(PARS_VEC::Array{Pars,1};Nz_band::Int64=11)
    N=length(PARS_VEC)
    R=zeros(Float64,N); R_lo=zeros(Float64,N); R_hi=zeros(Float64,N)
    for i in 1:N
        R[i], R_lo[i], R_hi[i]=Relative_Exposure_Oil(PARS_VEC[i];Nz_band=Nz_band)
    end
    return R, R_lo, R_hi
end

function Lambda_o_From_Shares(sd_f::Float64,sd_o::Float64,par::Pars)
    #A country's measured domestic shares of the intermediate bill -> the model's CES
    #weight for oil, anchored at λf. This is the odds ratio of Subsection 4.1, which is
    #how λo is calibrated
    @unpack μo, λf = par
    Oo=(λf/(1-λf))*(((sd_o/(1-sd_o))/(sd_f/(1-sd_f)))^(1/μo))
    return Oo/(1+Oo)
end

function Gamma_SufficientStatistic(par::Pars;z::Float64=par.μ_z)
    #Γ of equation (33), the state-invariant approximation to Δlog(yo)/Δlog(yf). The two
    #imported-expenditure shares are taken from the model's own prices. Panel (d) and
    #Table 4 read s* off the input-output data instead -- see Gamma_Blocks
    @unpack αMf, αMo, αK, αN, λf, λo, θf, θo, μ, μo, ν, νo, nH = par
    x=State(false,1,nH,z,0.0)
    pd=pd_Given_State(x,par)
    s_f=1.0-(λf^μ)*((pd/Price_M_h(true,pd,x,par))^(1-μ))
    s_o=1.0-(λo^μo)*((pd/Price_M_h(false,pd,x,par))^(1-μo))
    return ((αMo/αN)/(αMf/αK))*(s_o/s_f)*((log(1-θo)/(1-νo))/(log(1-θf)/(1-ν)))
end

function ShareGap_From_Lambda_o(λo::Float64,par::Pars;sd_f::Float64=0.62)
    #Inverse of Lambda_o_From_Shares: the difference in domestic shares of the
    #intermediate bill that a CES weight λo implies, anchored at the final sector's
    #measured share. The map is monotone, so it relabels the λo axis rather than
    #changing it
    @unpack μo, λf = par
    Oo=(sd_f/(1-sd_f))*(((λo/(1-λo))/(λf/(1-λf)))^μo)
    return (Oo/(1+Oo))-sd_f
end

function Lambda_o_From_ShareGap(gap::Float64,par::Pars;sd_f::Float64=0.62)
    return Lambda_o_From_Shares(sd_f,sd_f+gap,par)
end

#Mexico's measured domestic shares of the intermediate bill, from INEGI's 2018 matrix,
#which the last row of Table4.csv prints as the imported shares s*_f=0.38 and s*_o=0.28.
#s^d_f is also the calibrated λf, but s^d_o is NOT λo: λo=0.656 is the CES weight that
#equation (34) backs out of s^d_o=0.72
const SD_F_MEXICO=0.62
const SD_O_MEXICO=0.72

function Theta_Block(par::Pars)
    #C of equation (33), the working-capital block. Mexico's calibrated θ, so it is the
    #same constant for every country and the Γ=1 locus is the hyperbola A*B=1/C
    return (log(1-par.θo)/log(1-par.θf))*((1-par.ν)/(1-par.νo))
end

function Gamma_Blocks(αMf::Float64,αMo::Float64,sd_f::Float64,sd_o::Float64,par::Pars)
    #The two blocks of equation (33) that an input-output table measures, built exactly as
    #Gamma_Rows builds them for Table 4: A is intermediates per unit of the fixed factor
    #and B is the ratio of imported shares of the intermediate bill. Γ=A*B*C.
    #sd_f and sd_o are the measured DOMESTIC shares, as the ICIO panel stores them
    @unpack αLf, αLo = par
    A=(αMo/(1-αMo-αLo))/(αMf/(1-αMf-αLf))
    return A, (1-sd_o)/(1-sd_f)
end

function Gamma_Blocks(par::Pars)
    #The same two blocks for the calibration, on the INEGI shares rather than on λo, so
    #the red diamond reproduces the Mexico row of the ICIO table exactly
    return Gamma_Blocks(par.αMf,par.αMo,SD_F_MEXICO,SD_O_MEXICO,par)
end

function Map_Gamma_And_R(GR_gap_α::Array{Float64,1},GR_gap_s::Array{Float64,1},
                         par::Pars;IdenticalFrictions::Bool=false)
    #Γt and Γ at mean z over the (α gap, domestic share gap) plane, moving the oil-side
    #parameters with the final sector fixed at the benchmark. Rows are share gaps,
    #columns are α gaps, which is the orientation heatmap wants
    θo=IdenticalFrictions ? par.θf : par.θo
    R_MAT=zeros(Float64,length(GR_gap_s),length(GR_gap_α))
    Γ_MAT=zeros(Float64,length(GR_gap_s),length(GR_gap_α))
    for i in 1:length(GR_gap_s)
        for j in 1:length(GR_gap_α)
            par_ij=Pars_NewTechnology(par,αMo=par.αMf-GR_gap_α[j],
                                      λo=Lambda_o_From_ShareGap(GR_gap_s[i],par),
                                      θo=θo,PickBoth=false)
            drop_f, drop_o=Sector_Drops_InDefault(par_ij.μ_z,par_ij.nH,par_ij)
            R_MAT[i,j]=drop_o/drop_f
            Γ_MAT[i,j]=Gamma_SufficientStatistic(par_ij)
        end
    end
    return R_MAT, Γ_MAT
end

function Crossing_R_Equal_1(GR::Array{Float64,1},R::Array{Float64,1})
    #Linear interpolation of the first sign change of R-1; NaN if none
    for i in 1:length(GR)-1
        a=R[i]-1.0; b=R[i+1]-1.0
        if a*b<0.0
            t=a/(a-b)
            return GR[i]+t*(GR[i+1]-GR[i])
        end
    end
    return NaN
end

function Frontier_From_Map(GR_x::Array{Float64,1},GR_y::Array{Float64,1},
                           R_MAT::Array{Float64,2})
    #The R=1 level set, one crossing per row of the map
    xs=Float64[]; ys=Float64[]
    for i in 1:length(GR_y)
        x=Crossing_R_Equal_1(GR_x,R_MAT[i,:])
        if !isnan(x)
            push!(xs,x); push!(ys,GR_y[i])
        end
    end
    return xs, ys
end

function Technology_Points_ICIO(FOLDER_ICIO::String;MinOilShare::Float64=0.01,
                               OnlyDiscovery::Bool=false)
    #Country coordinates for the frontier panel: the ICIO countries with a measured
    #import-use table and oil above MinOilShare of gross output, each on its most recent
    #measured year, Mexico excluded. Returns the four measured levels rather than their
    #two gaps, because the blocks of Γ depend on the levels. OnlyDiscovery restricts to
    #the countries with a giant discovery in the sample. abspath because include inside a
    #function resolves against the calling script's folder, invokelatest because
    #Country_Row may be defined by that include
    isdefined(Main,:Technology_Shares) || include(abspath(joinpath(FOLDER_ICIO,"ICIO_Technology.jl")))
    PANEL,_=readdlm(joinpath(FOLDER_ICIO,"ICIO_Technology_Panel.csv"),',',header=true)
    AV,_=readdlm(joinpath(FOLDER_ICIO,"ICIO_ImportUse_Availability.csv"),',',header=true)
    KEEP=OnlyDiscovery ? DISCOVERY_COUNTRIES : sort(unique(String.(AV[:,1])))
    NAMES=String[]; AMF=Float64[]; AMO=Float64[]; SD_F=Float64[]; SD_O=Float64[]
    YEARS=Int64[]
    for c in sort(KEEP)
        c in TABLE_EXCLUDE && continue
        S, y=Base.invokelatest(Country_Row,FOLDER_ICIO,PANEL,c)
        (S===nothing || S.Oil_Share<MinOilShare) && continue
        push!(NAMES,get(COUNTRY_NAMES,c,c))
        push!(AMF,S.αMf)
        push!(AMO,S.αMo)
        push!(SD_F,S.λf)
        push!(SD_O,S.λo)
        push!(YEARS,y)
    end
    return NAMES, AMF, AMO, SD_F, SD_O, YEARS
end

#Where each country's label sits relative to its dot in the frontier panel; the entries
#are the pairs that would otherwise print on top of each other. Anything the ICIO adds
#later falls back to the rule
const LABEL_NUDGE=Dict("Thailand"=>(0.03,0.09,:left),
                       "Canada"=>(0.03,0.045,:left),
                       "Peru"=>(0.03,-0.045,:left),
                       "United States"=>(-0.03,0.0,:right))

function Label_Offset(NAME::String,A::Float64)
    return get(LABEL_NUDGE,NAME,A<1.4 ? (0.03,0.0,:left) : (-0.03,0.0,:right))
end

function Figure_Technology_Exposure(SETUP_FILE::String,FOLDER_GRAPHS::String;
                                    FOLDER_ICIO::String=joinpath("Code_Data","Evidence_of_Mechanism"),
                                    OnlyDiscovery::Bool=false,
                                    N_sweep::Int64=41,N_map::Int64=25,Nz_band::Int64=11,
                                    col_Benchmark::Int64=4,FILE_NAME::String="Figure8.pdf",
                                    μf::Float64=NaN,μo::Float64=NaN,
                                    νf::Float64=NaN,νo::Float64=NaN,
                                    ThreePanels::Bool=false)
    par, GRIDS=Setup_From_File(col_Benchmark,SETUP_FILE)

    #Elasticity sensitivity. These variants are static, so they are overrides here rather
    #than columns of the setup file: nothing is solved and nothing is serialized
    if !isnan(μf) || !isnan(μo) || !isnan(νf) || !isnan(νo)
        par=Pars(par,μ=isnan(μf) ? par.μ : μf,μo=isnan(μo) ? par.μo : μo,
                     ν=isnan(νf) ? par.ν : νf,νo=isnan(νo) ? par.νo : νo)
        #λo is the CES weight equation (34) reads off s^d_o, and that mapping runs through
        #μo alone, so only a move in μ needs it rebuilt
        if !isnan(μf) || !isnan(μo)
            par=Pars(par,λo=Lambda_o_From_Shares(SD_F_MEXICO,SD_O_MEXICO,par))
        end
        par=Pars(par,nH=Pick_n_Given_Target(par.yoil_GDP_target_H,par),
                     nL=Pick_n_Given_Target(par.yoil_GDP_target_L,par))
    end

    #Benchmark statistics, common to all four panels. Γt is the exact ratio
    #Δlog(yo)/Δlog(yf), Γ the state-invariant approximation to it of equation (33)
    Γt_ben, Γt_ben_lo, Γt_ben_hi=Relative_Exposure_Oil(par;Nz_band=Nz_band)
    Γ_ben=Gamma_SufficientStatistic(par)
    gap_α_ben=par.αMf-par.αMo
    #Domestic gap, which is what the λo mapping is written in. Panel (b) is drawn on the
    #imported gap, its negative
    gap_s_ben=ShareGap_From_Lambda_o(par.λo,par)

    #Country coordinates from the ICIO, the two blocks of Γ each country's own table
    #measures. A dot's product with the working-capital block is that country's Γ, the
    #same number the ICIO table reports, so the two exhibits carry one statistic
    if FOLDER_ICIO==""
        NAMES=String[]; A_C=Float64[]; B_C=Float64[]
    else
        NAMES, AMF_C, AMO_C, SD_F_C, SD_O_C, YEARS=Technology_Points_ICIO(FOLDER_ICIO;
                                                            OnlyDiscovery=OnlyDiscovery)
        BLOCKS=[Gamma_Blocks(AMF_C[i],AMO_C[i],SD_F_C[i],SD_O_C[i],par)
                for i in 1:length(NAMES)]
        A_C=[b[1] for b in BLOCKS]; B_C=[b[2] for b in BLOCKS]
    end

    #Sweeps, one pair at a time, everything else at the benchmark
    GR_gap_α=collect(range(-0.20,stop=0.45,length=N_sweep))
    PARS_A=[Pars_NewTechnology(par,αMo=par.αMf-g) for g in GR_gap_α]
    Γt_A, Γt_A_lo, Γt_A_hi=Sweep_Relative_Exposure(PARS_A;Nz_band=Nz_band)
    Γ_A=[Gamma_SufficientStatistic(p) for p in PARS_A]

    #The imported gap s*_o-s*_f, which is what the ICIO table reports and what panel (d)
    #puts on its vertical axis. The λo mapping is written in domestic gaps, so the sweep
    #hands it the negative. Range is symmetric and unchanged
    GR_gap_s=collect(range(-0.15,stop=0.15,length=N_sweep))
    PARS_L=[Pars_NewTechnology(par,λo=Lambda_o_From_ShareGap(-g,par)) for g in GR_gap_s]
    Γt_L, Γt_L_lo, Γt_L_hi=Sweep_Relative_Exposure(PARS_L;Nz_band=Nz_band)
    Γ_L=[Gamma_SufficientStatistic(p) for p in PARS_L]

    #Runs to 0.95 because θ enters Γ as log(1-θo)/log(1-θf), so everything this margin
    #does happens close to one
    GR_θo=collect(range(0.0,stop=0.95,length=N_sweep))
    PARS_T=[Pars_NewTechnology(par,θo=g) for g in GR_θo]
    Γt_T, Γt_T_lo, Γt_T_hi=Sweep_Relative_Exposure(PARS_T;Nz_band=Nz_band)
    Γ_T=[Gamma_SufficientStatistic(p) for p in PARS_T]

    #Map over both gaps, θo and θf at the benchmark
    #The share gap runs over the range the ICIO cross-section occupies, the same range
    #panel b sweeps
    GR_map_α=collect(range(-0.20,stop=0.45,length=N_map))
    GR_map_s=collect(range(-0.15,stop=0.15,length=N_map))
    R_MAT, Γ_MAT=Map_Gamma_And_R(GR_map_α,GR_map_s,par)

    #Details for plots
    size_width=600
    size_height=400
    LW=3.0
    COLOR_R=:blue
    ymax=1.05*max(maximum(Γt_A_hi),maximum(Γt_L_hi),maximum(Γt_T_hi),
                  maximum(Γ_A),maximum(Γ_L),maximum(Γ_T),1.15)
    YLIMS=(0.0,ymax)
    #Smaller than the paper's default, which runs negative tick labels together here
    TICKFONT=14
    YLABEL_R="\$\\Gamma_{t}\$ = oil drop / final drop"

    #Solid line is Γt at mean z, band its range over the whole z grid, dashed line the
    #approximation Γ along the same sweep, red diamond the Mexican calibration, open
    #circle the crossing of Γt=1 where MarkCrossing asks for it
    function Panel_Sweep(GR,R,R_lo,R_hi,Γ,TITLE,XLABEL,x_cal;YLABEL="",LEGPOS=:topright,
                         MarkCrossing::Bool=true)
        plt=plot(GR,R,ribbon=(R.-R_lo,R_hi.-R),fillalpha=0.25,
                 linecolor=COLOR_R,fillcolor=COLOR_R,linewidth=LW,label="\$\\Gamma_{t}\$",
                 title=TITLE,xlabel=XLABEL,ylabel=YLABEL,ylims=YLIMS,
                 legend=LEGPOS,size=(size_width,size_height),
                 tickfontsize=TICKFONT)
        plot!(plt,GR,Γ,linecolor=:black,linestyle=:dash,linewidth=2.0,label="\$\\Gamma\$")
        hline!(plt,[1.0],linecolor=:black,linestyle=:dot,linewidth=1.5,label="")
        x_cross=Crossing_R_Equal_1(GR,R)
        if MarkCrossing && !isnan(x_cross)
            scatter!(plt,[x_cross],[1.0],markershape=:circle,markersize=9,
                     markercolor=:white,markerstrokecolor=:black,label="")
        end
        scatter!(plt,[x_cal],[Γt_ben],markershape=:diamond,markersize=11,
                 markercolor=:red,markerstrokecolor=:black,label="")
        return plt
    end

    plt_a=Panel_Sweep(GR_gap_α,Γt_A,Γt_A_lo,Γt_A_hi,Γ_A,"intermediates share",
                      "\$\\alpha_{Mf}-\\alpha_{Mo}\$",gap_α_ben;YLABEL=YLABEL_R,
                      MarkCrossing=false)
    plt_b=Panel_Sweep(GR_gap_s,Γt_L,Γt_L_lo,Γt_L_hi,Γ_L,"imported share of intermediates",
                      "\$s^{*}_{o}-s^{*}_{f}\$",-gap_s_ben)
    plt_c=Panel_Sweep(GR_θo,Γt_T,Γt_T_lo,Γt_T_hi,Γ_T,"working capital need of oil",
                      "\$\\theta_{o}\$",par.θo;YLABEL=YLABEL_R,LEGPOS=:topleft)
    vline!(plt_c,[par.θf],linecolor=:gray,linestyle=:dash,linewidth=1.5,label="")
    annotate!(plt_c,par.θf+0.02,0.92*ymax,text("\$\\theta_{o}=\\theta_{f}\$",12,:left))

    #The cross-section in the plane of the two blocks, so nothing here depends on the
    #state. Γ is the product of the two coordinates with the working-capital block, so the
    #frontier is the hyperbola A*B=1/C and everything above it has Γ>1. Red diamond is the
    #calibration, whose λo is a CES weight rather than a measured share
    C_θ=Theta_Block(par)
    A_ben, B_ben=Gamma_Blocks(par)
    XLIMS_D=(0.0,1.05*max(maximum(A_C),A_ben,1.0))
    YLIMS_D=(0.9*min(minimum(B_C),B_ben),1.10*max(maximum(B_C),B_ben,1.0))
    plt_d=plot(xlims=XLIMS_D,ylims=YLIMS_D,
               title="\$\\Gamma=1\$ frontier over the two blocks",
               xlabel="\$\\left(\\alpha_{Mo}/\\alpha_{N}\\right)/\\left(\\alpha_{Mf}/\\alpha_{K}\\right)\$",
               ylabel="\$s^{*}_{o}/s^{*}_{f}\$",legend=false,
               size=(size_width,size_height),tickfontsize=TICKFONT)
    vline!(plt_d,[1.0],linecolor=:black,linestyle=:dot,linewidth=1.5,label="")
    hline!(plt_d,[1.0],linecolor=:black,linestyle=:dot,linewidth=1.5,label="")
    A_FR=collect(range(1.0/(C_θ*YLIMS_D[2]),stop=XLIMS_D[2],length=200))
    plot!(plt_d,A_FR,(1.0/C_θ)./A_FR,linecolor=:black,linewidth=4,linestyle=:dash,
          label="")
    if length(NAMES)>0
        scatter!(plt_d,A_C,B_C,markershape=:circle,markersize=5,
                 markercolor=:black,label="")
        for i in 1:length(NAMES)
            dx, dy, ALIGN=Label_Offset(NAMES[i],A_C[i])
            annotate!(plt_d,A_C[i]+dx,B_C[i]+dy,text(NAMES[i],9,ALIGN))
        end
    end
    scatter!(plt_d,[A_ben],[B_ben],markershape=:diamond,markersize=11,
             markercolor=:red,markerstrokecolor=:black,label="")
    annotate!(plt_d,A_ben+0.045,B_ben,text("Mexico",9,:left))

    #What the panels are worth quoting for, and the one convention the ICIO table and
    #this figure do not share
    @printf("Figure 8: calibration Gamma_t=%.4f  Gamma=%.4f  (model prices, panels a-c)\n",
            Γt_ben,Γ_ben)
    @printf("Figure 8: calibration sits at an imported share gap of %.4f\n",-gap_s_ben)
    @printf("Figure 8: Gamma_t=1 along alpha at %.4f, along the imported share gap at %.4f, along theta_o at %.4f\n",
            Crossing_R_Equal_1(GR_gap_α,Γt_A),Crossing_R_Equal_1(GR_gap_s,Γt_L),
            Crossing_R_Equal_1(GR_θo,Γt_T))
    @printf("Figure 8: max |Gamma_t-Gamma| over the map is %.4f\n",
            maximum(abs.(R_MAT.-Γ_MAT)))
    #Panel (d) is the cross-section, so its Gamma is the table's and not the model's.
    #The diamond has to reproduce the Mexico row of Table4.csv
    @printf("Figure 8: panel (d) diamond A=%.4f B=%.4f Gamma=%.4f (Mexico row of the ICIO table)\n",
            A_ben,B_ben,A_ben*B_ben*C_θ)
    for i in 1:length(NAMES)
        @printf("Figure 8: panel (d) %-16s A=%.4f B=%.4f Gamma=%.4f\n",
                NAMES[i],A_C[i],B_C[i],A_C[i]*B_C[i]*C_θ)
    end

    #Panel (d) is the cross-section and carries no μ at all, so the Armington variants
    #drop it and run the three model sweeps in a row
    if ThreePanels
        plt=plot(plt_a,plt_b,plt_c,
                 layout=(1,3),size=(size_width*3,size_height))
    else
        l = @layout([a b; c d])
        plt=plot(plt_a,plt_b,
                 plt_c,plt_d,
                 layout=l,size=(size_width*2,size_height*2))
    end
    savefig(plt,joinpath(FOLDER_GRAPHS,FILE_NAME))
    return plt
end

function Figures_Armington_Sensitivity(SETUP_FILE::String,FOLDER_GRAPHS::String;
                                       FOLDER_ICIO::String=joinpath("Code_Data","Evidence_of_Mechanism"),
                                       μ_high::Float64=5.1)
    #The benchmark takes μ=2.9 from Mendoza and Yue for both sectors. These two variants
    #raise it to 5.1, the top of the range Bajzik et al (2020) report, once in both
    #sectors and once only in the final one. Static, so nothing is solved
    plt_both=Figure_Technology_Exposure(SETUP_FILE,FOLDER_GRAPHS;FOLDER_ICIO=FOLDER_ICIO,
                                        μf=μ_high,μo=μ_high,ThreePanels=true,
                                        FILE_NAME="Figure20.pdf")
    plt_final=Figure_Technology_Exposure(SETUP_FILE,FOLDER_GRAPHS;FOLDER_ICIO=FOLDER_ICIO,
                                         μf=μ_high,ThreePanels=true,
                                         FILE_NAME="Figure21.pdf")
    return plt_both, plt_final
end

function Figures_DixitStiglitz_Sensitivity(SETUP_FILE::String,FOLDER_GRAPHS::String;
                                           FOLDER_ICIO::String=joinpath("Code_Data","Evidence_of_Mechanism"),
                                           ν_factor::Float64=2.0)
    #The same pair for the Dixit-Stiglitz curvature, which the benchmark takes as ν=2.44
    #from Mendoza and Yue. Doubling it lands at 4.88. ν leaves λo alone, since equation
    #(34) runs through μo
    par_ben, _=Setup_From_File(4,SETUP_FILE)
    ν_high=ν_factor*par_ben.ν
    plt_both=Figure_Technology_Exposure(SETUP_FILE,FOLDER_GRAPHS;FOLDER_ICIO=FOLDER_ICIO,
                                        νf=ν_high,νo=ν_high,ThreePanels=true,
                                        FILE_NAME="Figure22.pdf")
    plt_final=Figure_Technology_Exposure(SETUP_FILE,FOLDER_GRAPHS;FOLDER_ICIO=FOLDER_ICIO,
                                         νf=ν_high,ThreePanels=true,
                                         FILE_NAME="Figure23.pdf")
    return plt_both, plt_final
end

################################################################################
### Responses to a discovery across technology variants
###
### Four solved models: the benchmark, θo=θf, an oil sector identical to the final one
### in every input share and friction, and the calibration procedure applied to Brazil's
### own 2015 ICIO row, which takes αMf, αMo and λo from Brazil and sets θo=θf. One saved
### model file each
################################################################################

const VARIANT_FILES=["Model_Benchmark.csv","Model_SameTheta.csv","Model_Identical.csv",
                     "Model_Brazil.csv"]
const VARIANT_LABELS=["benchmark" "\$\\theta_o=\\theta_f\$" "identical" "Brazil"]
const VARIANT_COLORS=[:green :blue :black :red]
const VARIANT_STYLES=[:solid :dash :solid :dashdot]
#Row labels for Table 3, which spells out what the figure's legend abbreviates
const VARIANT_LABELS_TABLE=["benchmark","\$\\theta_{o}=\\theta_{f}\$","identical",
                            "Brazil"]
#Columns of Setup_Calibrated.csv, in the same order as VARIANT_FILES
const VARIANT_COLUMNS=[4,6,8,7]

function Moments_Technology_Variants(FOLDER::String;FILES::Array{String,1}=VARIANT_FILES[2:end])
    #Untargeted moments of the solved variants, for the rows of Table 3. Same
    #sample sizes as the benchmark, which are the ones the table note reports
    MOM=Array{Moments,1}(undef,length(FILES))
    for i in 1:length(FILES)
        MOD=UnpackModel_File(FILES[i],FOLDER)
        MOM[i]=AverageMomentsManySamples(MOD.par.Tmom,MOD.par.NSamplesMoments,MOD)
    end
    return MOM
end

function Solve_Technology_Variants(SETUP_FILE::String,FOLDER::String;
                                   FILES::Array{String,1}=VARIANT_FILES[2:end],
                                   COLUMNS::Array{Int64,1}=VARIANT_COLUMNS[2:end])
    #Solves and saves the variants other than the benchmark, which
    #Results_Benchmark_Calibration already solves
    for i in 1:length(FILES)
        MOD, NAME=Model_FromSetup(COLUMNS[i],SETUP_FILE)
        SaveModel_Vector(joinpath(FOLDER,FILES[i]),MOD)
    end
    return nothing
end

function Discovery_Paths_And_Counterfactual(MOD::Model;N::Int64=10000,Tbefore::Int64=2,
                                            Tafter::Int64=15,DropDefaults::Bool=false)
    #One pass over the same draws AverageDiscoveryPaths uses (same seed), accumulating the
    #average paths and a counterfactual spread series in which ONLY the field state moves:
    #z and the debt chosen both stay at their t=-1 values. That separates the shift of the
    #spread schedule from the movement along it that comes from borrowing
    T=Tbefore+1+Tafter
    k_m1=Tbefore  #index of t=-1

    Random.seed!(1234)
    PATHS_AV=InitiateEmptyPaths(T)
    SPR_CF=zeros(Float64,T)
    for i in 1:N
        PATHS=SimulatePathsOfDiscovery(DropDefaults,Tbefore,Tafter,MOD)
        SumPathForAverage!(N,PATHS_AV,PATHS)
        z0=PATHS.z[k_m1]
        bprime0=PATHS.B[k_m1+1]
        for k in 1:T
            SPR_CF[k]=SPR_CF[k]+ComputeSpreads(PATHS.n_ind[k],z0,bprime0,MOD)/N
        end
    end
    return PATHS_AV, SPR_CF
end

function Discovery_Responses_Variants(FOLDER::String;N::Int64=10000,Tbefore::Int64=2,
                                      Tafter::Int64=15,DropDefaults::Bool=false,
                                      FILES::Array{String,1}=VARIANT_FILES)
    TS=Array{Paths,1}(undef,length(FILES))
    SPR_CF=Array{Array{Float64,1},1}(undef,length(FILES))
    for i in 1:length(FILES)
        MOD=UnpackModel_File(FILES[i],FOLDER)
        TS[i], SPR_CF[i]=Discovery_Paths_And_Counterfactual(MOD;N=N,Tbefore=Tbefore,
                                            Tafter=Tafter,DropDefaults=DropDefaults)
    end
    return TS, SPR_CF
end

function Plot_Discovery_Responses_Variants(Tbefore::Int64,Tafter::Int64,TS::Array{Paths,1},
                                           SPR_CF::Array{Array{Float64,1},1};
                                           LABELS::Array{String,2}=VARIANT_LABELS,
                                           COLORS=VARIANT_COLORS,STYLES=VARIANT_STYLES)
    t0=-Tbefore
    t1=Tafter
    size_width=600
    size_height=400
    SIZE_PLOTS=(size_width,size_height)
    LW=3.0
    NM=length(TS)
    XLABEL="t"

    function Panel(MAT,TITLE,YLABEL,LEGEND;ZeroLine::Bool=true,YLIMS=:auto)
        plt=plot([t0:t1],MAT,label=LABELS,
                 linecolor=COLORS,linestyle=STYLES,linewidth=LW,
                 title=TITLE,ylabel=YLABEL,xlabel=XLABEL,ylims=YLIMS,
                 legend=LEGEND,legendfontsize=17,size=SIZE_PLOTS)
        ZeroLine && hline!(plt,[0.0],linecolor=:gray,linestyle=:dot,linewidth=1.5,label="")
        return plt
    end

    #Spreads and debt are deviations from the first pre-discovery period, so models
    #with different averages are comparable. The fraction in default is a level
    SPR=hcat([TS[i].Spreads .- TS[i].Spreads[1] for i in 1:NM]...)
    #The same spread response with only the field state moving, so it is the schedule
    #shifting under the discovery with the borrowing response shut down. Flat before
    #the discovery by construction, so its base period is the same as the other panels
    CF=hcat([SPR_CF[i] .- SPR_CF[i][1] for i in 1:NM]...)
    DEBT=hcat([100 .*(TS[i].B .- TS[i].B[1]) ./ TS[i].GDP[1] for i in 1:NM]...)
    DEF=hcat([TS[i].Def for i in 1:NM]...)

    #The two spread panels sit side by side and share a vertical axis, so the distance
    #between the total response and the schedule alone can be read across them
    spr_lo=min(minimum(SPR),minimum(CF))
    spr_hi=max(maximum(SPR),maximum(CF))
    pad=0.08*(spr_hi-spr_lo)
    YLIMS_SPR=(spr_lo-pad,spr_hi+pad)

    #The legend sits top right of the first panel, where the lines have already fallen back
    plt_spr=Panel(SPR,"spreads","percentage points",:topright;YLIMS=YLIMS_SPR)
    plt_q=Panel(CF,"spreads, no borrowing response","percentage points",false;
                YLIMS=YLIMS_SPR)
    plt_def=Panel(DEF,"fraction in default","fraction",false;ZeroLine=false)
    plt_b=Panel(DEBT,"government debt","percentage of Av(GDP)",false)

    l = @layout([a b; c d])
    plt=plot(plt_spr,plt_q,
             plt_def,plt_b,
             layout=l,size=(size_width*2,size_height*2))
    return plt
end

function Figure_Technology_Responses(FOLDER::String,FOLDER_GRAPHS::String;
                                     N::Int64=10000,Tbefore::Int64=2,Tafter::Int64=15)
    TS, SPR_CF=Discovery_Responses_Variants(FOLDER;N=N,Tbefore=Tbefore,Tafter=Tafter)
    plt=Plot_Discovery_Responses_Variants(Tbefore,Tafter,TS,SPR_CF)
    savefig(plt,joinpath(FOLDER_GRAPHS,"Figure9.pdf"))
    return plt, TS, SPR_CF
end
