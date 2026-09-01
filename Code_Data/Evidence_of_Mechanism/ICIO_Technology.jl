################################################################################
### Technology of oil extraction versus the rest of the economy, across countries
###
### Computes alpha_M and the domestic share of intermediates for oil and for the rest
### of the economy, for every country in the OECD Inter-Country Input-Output database,
### 2025 edition, and turns each country's four numbers into the sufficient statistic
### Gamma of equation (33). Oil is ICIO industry B06, ISIC Rev.4 Division 06, the
### counterpart of the NAICS 211 the calibration uses; B09, oilfield services, is
### reported as an alternative definition rather than folded in. Table 4 reports the
### two share columns as the IMPORTED share of the intermediate bill, s*=1-s^d
###
### Two steps, of very different cost:
###   Build_ICIO_Panel  downloads five zip archives (about 600 MB) and reads 28 annual
###                     tables of 4,053 x 4,537. Roughly forty minutes. Writes
###                     ICIO_Technology_Panel.csv, which ships with this package
###   Write_ICIO_Table  reads that panel and writes the table rows. Instant
################################################################################

using Printf, DelimitedFiles, Statistics

#The 12 countries with giant discoveries in the sample that the ICIO covers. Ecuador,
#Gabon, Ghana, Iraq and Venezuela are not in the database.
const DISCOVERY_COUNTRIES=["BRA","CHN","COL","EGY","IDN","KAZ","MYS","MEX","NGA","PER",
                           "RUS","VNM"]

const ICIO_URL="https://webfs-sti.oecd.org/files/STI-PIE/ICIO/2025/"
const ICIO_ARCHIVES=["1995-2000","2001-2005","2006-2010","2011-2015","2016-2022"]

################################################################################
### Reading one annual table
################################################################################
Prefix(s::AbstractString)=String(split(s,'_')[1])
Suffix(s::AbstractString)=occursin('_',s) ? String(split(s,'_';limit=2)[2]) : ""

function Read_ICIO(PATH::String)
    #Row and column labels are country_industry. The last three rows are TLS (taxes less
    #subsidies on products), VA and OUT, and OUT = intermediates + TLS + VA, so the
    #intermediate flows are at basic prices and TLS is the ICIO counterpart of INEGI's
    #D.21-D.31 row
    io=open(PATH,"r")
    COLNAMES=String.(split(readline(io),',')[2:end])
    NCOL=length(COLNAMES)
    ROWNAMES=String[]; ROWS=Vector{Vector{Float64}}()
    for line in eachline(io)
        F=split(line,',')
        push!(ROWNAMES,String(F[1]))
        V=Vector{Float64}(undef,NCOL)
        @inbounds for i in 1:NCOL
            V[i]=parse(Float64,F[i+1])
        end
        push!(ROWS,V)
    end
    close(io)
    A=Matrix{Float64}(undef,length(ROWS),NCOL)
    @inbounds for r in 1:length(ROWS)
        A[r,:].=ROWS[r]
    end
    return A, ROWNAMES, COLNAMES
end

function Append_Year!(PATH::String,YEAR::Int,PATH_OUT::String)
    #One row per country and sector definition, holding the sums every share is built from:
    #total intermediates M at basic prices, the domestically sourced part D, net taxes on
    #products T, gross output Y, and the water transport component of M and D (Mw, Dw)
    A,ROWNAMES,COLNAMES=Read_ICIO(PATH)
    ROW_OUT=findfirst(==("OUT"),ROWNAMES)
    ROW_TLS=findfirst(==("TLS"),ROWNAMES)
    #Industry columns come first and are followed by six final demand columns per country.
    #Rows 1:last(IND) are the matching industry rows, so the same range indexes both
    FINAL_DEMAND=["HFCE","NPISH","GGFC","GFCF","INVNT","DPABR"]
    IND=1:findlast(x->!(Suffix(x) in FINAL_DEMAND) && x!="OUT",COLNAMES)
    NIND=count(x->Prefix(x)==Prefix(COLNAMES[1]),COLNAMES[IND])
    INDUSTRIES=Suffix.(COLNAMES[1:NIND])
    iB06=findfirst(==("B06"),INDUSTRIES)
    iB09=findfirst(==("B09"),INDUSTRIES)
    iH50=findfirst(==("H50"),INDUSTRIES)
    ROWS_H50=iH50:NIND:last(IND)

    io=open(PATH_OUT,"a")
    for c in unique(Prefix.(COLNAMES[IND]))
        COLS_C=findall(x->Prefix(x)==c,COLNAMES[IND])
        ROWS_C=findall(x->Prefix(x)==c,ROWNAMES[IND])
        OIL=[COLS_C[iB06]]
        OIL09=COLS_C[[iB06,iB09]]
        DEFINITIONS=(("B06",OIL),("B0609",OIL09),
                     ("F",setdiff(COLS_C,OIL)),("F09",setdiff(COLS_C,OIL09)))
        for (NAME,S) in DEFINITIONS
            M=sum(A[IND,S]); D=sum(A[ROWS_C,S]); Y=sum(A[ROW_OUT,S]); T=sum(A[ROW_TLS,S])
            Mw=sum(A[ROWS_H50,S]); Dw=sum(A[ROWS_C[iH50],S])
            @printf(io,"%d,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",YEAR,c,NAME,M,D,Y,Mw,Dw,T)
        end
    end
    close(io)
    return nothing
end

function Build_ICIO_Panel(FOLDER::String)
    #Downloads and reads every year, then deletes each 87 MB csv as it goes
    RAW=joinpath(FOLDER,"ICIO_raw")
    isdir(RAW) || mkpath(RAW)
    PATH_OUT=joinpath(FOLDER,"ICIO_Technology_Panel.csv")
    open(PATH_OUT,"w") do io
        println(io,"year,country,sector,M,D,OUT,Mw,Dw,TLS")
    end
    for a in ICIO_ARCHIVES
        ZIP=joinpath(RAW,"$(a)_SML.zip")
        #The OECD file server sits behind a bot filter that rejects the default user agent
        isfile(ZIP) || download_with_browser_agent("$ICIO_URL$(a)_SML.zip",ZIP)
        for YEAR in parse(Int,first(split(a,'-'))):parse(Int,last(split(a,'-')))
            CSV=joinpath(RAW,"$(YEAR)_SML.csv")
            Extract_One(ZIP,"$(YEAR)_SML.csv",CSV)
            Append_Year!(CSV,YEAR,PATH_OUT)
            rm(CSV)
            println("read $YEAR")
        end
    end
    return PATH_OUT
end

function download_with_browser_agent(URL::String,PATH::String)
    UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "*
       "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    run(`curl -sS -o $PATH -A $UA -H "Referer: https://www.oecd.org/" $URL`)
    return nothing
end

function Extract_One(ZIP::String,ENTRY::String,PATH::String)
    CMD="Add-Type -A System.IO.Compression.FileSystem; "*
        "\$z=[IO.Compression.ZipFile]::OpenRead('$ZIP'); "*
        "\$e=\$z.Entries | Where-Object {\$_.Name -eq '$ENTRY'}; "*
        "[IO.Compression.ZipFileExtensions]::ExtractToFile(\$e,'$PATH',\$true); \$z.Dispose()"
    run(pipeline(`powershell.exe -NoProfile -Command $CMD`,devnull))
    return nothing
end

################################################################################
### The four shares, and the table
################################################################################
struct Technology_Shares
    αMo::Float64
    αMf::Float64
    λo::Float64
    λf::Float64
    Exposure_o::Float64 #(1-lambda_o)*alpha_Mo, imported intermediates over gross output
    Exposure_f::Float64
    λo_xw::Float64  #lambda_o excluding water transport inputs
    λf_xw::Float64
    Water_o::Float64 #water transport as a share of the oil sector's intermediates
    Oil_Share::Float64 #oil extraction as a share of the country's gross output
end

function Shares_From_Panel(PANEL::Matrix,YEAR::Int,COUNTRY::String;WithB09::Bool=false)
    #Both shares are built exactly as in Input-Output analysis - INEGI.xlsx, sheet
    #"Domestic", which is where the calibrated values come from:
    #  lambda  = UON/(UON+IET), domestic over total intermediates, both at basic prices
    #            and so net of taxes and subsidies. Here D/M.
    #  alpha_M = UPC/P.1, intermediates at PURCHASER prices over gross output, so the
    #            numerator adds net taxes on products (INEGI's D.21-D.31, the ICIO's TLS
    #            row). Here (M+T)/Y
    OIL=WithB09 ? "B0609" : "B06"
    REST=WithB09 ? "F09" : "F"
    Row(s)=(r=findfirst(i->PANEL[i,1]==YEAR && PANEL[i,2]==COUNTRY && PANEL[i,3]==s,
                        1:size(PANEL,1)); r===nothing ? nothing : Float64.(PANEL[r,4:9]))
    o=Row(OIL); f=Row(REST)
    (o===nothing || f===nothing || o[3]<=0) && return nothing
    αMo=(o[1]+o[6])/o[3]; αMf=(f[1]+f[6])/f[3]; λo=o[2]/o[1]; λf=f[2]/f[1]
    #(1-lambda)*alpha_M is what the mechanism turns on: the share of gross output spent on
    #imported intermediates, which is what a default disrupts through working capital
    return Technology_Shares(αMo,αMf,λo,λf,(1-λo)*αMo,(1-λf)*αMf,
                             (o[2]-o[5])/(o[1]-o[4]), (f[2]-f[5])/(f[1]-f[4]),
                             o[4]/o[1], o[3]/(o[3]+f[3]))
end

function Measured_Years(FOLDER::String,COUNTRY::String)
    #Years for which the OECD holds an actual import use or import input-output table,
    #listed in ICIO_ImportUse_Availability.csv. Everywhere else the imported/domestic split
    #is imputed under import proportionality, under which a difference between lambda_o and
    #lambda_f can only come from the product mix an industry buys, so imputed country-years
    #are not evidence
    T,_=readdlm(joinpath(FOLDER,"ICIO_ImportUse_Availability.csv"),',',header=true)
    return sort([Int(T[i,2]) for i in 1:size(T,1) if T[i,1]==COUNTRY])
end

function Country_Row(FOLDER::String,PANEL::Matrix,COUNTRY::String;WithB09::Bool=false)
    #The country's most recent measured year, and nothing else: no averaging across years
    #and no imputed year ever enters.
    #Watch the water transport line: the ICIO books international shipping as an imported
    #service of the industry whose goods move, so it lands in the oil column as an import
    #even though it is a distribution margin on output. Water_o reports how large it is,
    #and lambda_o_xw the domestic share without it
    YEARS=Measured_Years(FOLDER,COUNTRY)
    for y in reverse(YEARS)
        S=Shares_From_Panel(PANEL,y,COUNTRY;WithB09=WithB09)
        S===nothing || return S, y
    end
    return nothing, 0
end

#The twelve discovery countries, plus every other ICIO country that turns up with a measured
#import use table and an oil sector above one percent of gross output
const COUNTRY_NAMES=Dict("BRA"=>"Brazil","CHN"=>"China","COL"=>"Colombia","EGY"=>"Egypt",
                         "IDN"=>"Indonesia","KAZ"=>"Kazakhstan","MEX"=>"Mexico",
                         "MYS"=>"Malaysia","NGA"=>"Nigeria","PER"=>"Peru",
                         "RUS"=>"Russia","VNM"=>"Viet Nam",
                         "AUS"=>"Australia","CAN"=>"Canada","NLD"=>"Netherlands",
                         "NOR"=>"Norway","THA"=>"Thailand","UKR"=>"Ukraine",
                         "USA"=>"United States")

#The paper's benchmark, from Mexico's 2018 national input-output matrix (NAICS 211
#against the total economy minus 211). Carried as the last row of the table
const CALIBRATION=(αMo=0.30,αMf=0.46,λo=0.72,λf=0.62)

#Mexico is left out of the ICIO rows: the calibration stays on the INEGI matrix, whose
#numbers appear as the last row instead
const TABLE_EXCLUDE=["MEX"]

function Technology_Rows(FOLDER::String;WithB09::Bool=false,AllCountries::Bool=true,
                         MinOilShare::Float64=0.01)
    #The rows of the exhibit, unrounded and in the order they are printed: every ICIO
    #country with a measured import use table and an oil sector of at least MinOilShare,
    #each on its most recent measured year, plus Mexico on the national INEGI matrix --
    #the same four numbers the calibration uses. The ICIO's own Mexico is held out by
    #TABLE_EXCLUDE. Sorted by the alpha margin, strongest first
    PANEL,_=readdlm(joinpath(FOLDER,"ICIO_Technology_Panel.csv"),',',header=true)
    AV,_=readdlm(joinpath(FOLDER,"ICIO_ImportUse_Availability.csv"),',',header=true)
    KEEP=AllCountries ? unique(String.(AV[:,1])) : DISCOVERY_COUNTRIES
    ROWS=NamedTuple{(:name,:year,:αMf,:αMo,:λo,:λf),
                    Tuple{String,Int,Float64,Float64,Float64,Float64}}[]
    for c in KEEP
        c in TABLE_EXCLUDE && continue
        S,YEAR=Country_Row(FOLDER,PANEL,c;WithB09=WithB09)
        (S===nothing || S.Oil_Share<MinOilShare) && continue
        push!(ROWS,(name=get(COUNTRY_NAMES,c,c),year=YEAR,αMf=S.αMf,αMo=S.αMo,
                    λo=S.λo,λf=S.λf))
    end
    push!(ROWS,(name=COUNTRY_NAMES["MEX"],year=2018,αMf=CALIBRATION.αMf,
                αMo=CALIBRATION.αMo,λo=CALIBRATION.λo,λf=CALIBRATION.λf))
    sort!(ROWS,by=r->-(r.αMf-r.αMo))
    return ROWS
end

function Gamma_Parameters(SETUP_FILE::String)
    #The parameters Gamma needs that the ICIO cannot measure. theta_f and theta_o are read
    #from the benchmark column of the setup file rather than hardcoded, because theta_f
    #moves with the SMM; the two labor shares are calibrated directly and are the values
    #in Table 1
    S,_=readdlm(SETUP_FILE,',',header=true)
    Value(name)=(r=findfirst(i->strip(String(S[i,1]))==name,1:size(S,1));
                 r===nothing ? error("$name is not a row of $SETUP_FILE") : Float64(S[r,4]))
    return (αLf=0.14,αLo=0.04,θf=Value("theta_f"),θo=Value("theta_o"))
end

function Gamma_Rows(FOLDER::String;SETUP_FILE::String=joinpath("Code_Model","Setup_Calibrated.csv"),
                    WithB09::Bool=false,AllCountries::Bool=true)
    #Each country's value of the sufficient statistic of equation (33), the product of
    #three blocks:
    #  A = (alpha_Mo/alpha_N)/(alpha_Mf/alpha_K)  intermediates per unit of fixed factor
    #  B = s*_o/s*_f                              imported share of the intermediate bill
    #  C = log(1-theta_o)/log(1-theta_f)          working capital
    #Gamma<1 is the ordering the mechanism requires. Labor shares are held at Mexico's for
    #every country, because the ICIO reports value added as one row and does not split it.
    #C is Mexico's alone, so it enters every row as the same constant; Gamma_SameTheta
    #drops it and is built only from the country's own input-output data
    P=Gamma_Parameters(SETUP_FILE)
    C=log(1-P.θo)/log(1-P.θf)
    ROWS=Technology_Rows(FOLDER;WithB09=WithB09,AllCountries=AllCountries)
    OUT=NamedTuple{(:name,:year,:αMf,:αMo,:λo,:λf,:Γ_SameTheta,:Γ),
                   Tuple{String,Int,Float64,Float64,Float64,Float64,Float64,Float64}}[]
    for r in ROWS
        A=(r.αMo/(1-r.αMo-P.αLo))/(r.αMf/(1-r.αMf-P.αLf))
        #The lambda fields carry the measured DOMESTIC shares s^d, so s*=1-lambda
        B=(1-r.λo)/(1-r.λf)
        push!(OUT,(name=r.name,year=r.year,αMf=r.αMf,αMo=r.αMo,λo=r.λo,λf=r.λf,
                   Γ_SameTheta=A*B,Γ=A*B*C))
    end
    sort!(OUT,by=r->r.Γ)
    return OUT
end

function Write_ICIO_Table(FOLDER::String,FOLDER_TABLES::String;WithB09::Bool=false,
                          AllCountries::Bool=true,
                          SETUP_FILE::String=joinpath("Code_Model","Setup_Calibrated.csv"))
    #Six numbers per country: the two intermediate shares, the two IMPORTED shares of the
    #intermediate bill s*=1-s^d, and the sufficient statistic twice -- once with the
    #working-capital block switched off by theta_o=theta_f, once at the calibrated
    #frictions. Every column reads the same direction, bigger meaning more exposed to
    #default, and Gamma below one is the ordering the mechanism requires. Sorted by the last
    #
    #ONLY THE ROWS ARE WRITTEN; the float that wraps them is in the manuscript. The column
    #numbering (1)-(6) starts at alpha_Mf. WithB09=true writes the B06+B09 variant instead
    ROWS=Gamma_Rows(FOLDER;SETUP_FILE=SETUP_FILE,WithB09=WithB09,AllCountries=AllCountries)
    f2(x)=@sprintf("%.2f",x)

    LABEL=[r.name for r in ROWS]; SPAN=["$(r.year)" for r in ROWS]
    CELLS=[[f2(r.αMf),f2(r.αMo),f2(1-r.λf),f2(1-r.λo),
            f2(r.Γ_SameTheta),f2(r.Γ)] for r in ROWS]

    TEX=Array{String,1}(undef,length(LABEL))
    for i in 1:length(LABEL)
        TEX[i]=join(["{\\footnotesize{}"*c*"}" for c in vcat(LABEL[i],SPAN[i],CELLS[i])],
                    " & ")*"\\tabularnewline"
    end
    BASE=WithB09 ? "Table_ICIO_B0609" : "Table4"
    open(joinpath(FOLDER_TABLES,"$(BASE)_rows.tex"),"w") do io
        for r in TEX
            println(io,r)
        end
    end

    MAT=hcat(LABEL,SPAN,permutedims(hcat(CELLS...)))
    HEAD=["country" "measured_year" "alpha_Mf" "alpha_Mo" "sstar_f" "sstar_o" "Gamma_SameTheta" "Gamma"]
    writedlm(joinpath(FOLDER_TABLES,"$(BASE).csv"),vcat(HEAD,MAT),',')
    return LABEL, CELLS
end

function Summarize_Gamma(FOLDER::String;SETUP_FILE::String=joinpath("Code_Model","Setup_Calibrated.csv"),
                         WithB09::Bool=false,AllCountries::Bool=true)
    #The counts quoted in the text, for the rows of the table as it is printed. Mexico's
    #rank says whether the calibration is a favourable draw from the cross-section
    for B09 in (WithB09 ? (true,) : (false,true))
        ROWS=Gamma_Rows(FOLDER;SETUP_FILE=SETUP_FILE,WithB09=B09,AllCountries=AllCountries)
        G=[r.Γ for r in ROWS]; GT=[r.Γ_SameTheta for r in ROWS]
        i=findfirst(r->r.name=="Mexico",ROWS)
        @printf("%-8s n=%2d  Gamma<1 %2d  Gamma<1 at theta_o=theta_f %2d  range %.2f-%.2f  Mexico %.2f, rank %d\n",
                B09 ? "B06+B09" : "B06",length(G),sum(G.<1),sum(GT.<1),
                minimum(G),maximum(G),G[i],i)
    end
    return nothing
end

function Summarize_ICIO(FOLDER::String;MinOilShare::Float64=0.01,WithB09::Bool=false)
    #How often each half of the technology gap holds, counted two ways: on the most recent
    #measured year of each country, which is what Table 4 reports, and over every measured
    #country-year with oil at least MinOilShare of gross output
    PANEL,_=readdlm(joinpath(FOLDER,"ICIO_Technology_Panel.csv"),',',header=true)
    AV,_=readdlm(joinpath(FOLDER,"ICIO_ImportUse_Availability.csv"),',',header=true)
    ALL=unique(String.(AV[:,1]))
    for (TAG,KEEP) in (("discovery countries",DISCOVERY_COUNTRIES),("all ICIO countries",ALL))
        NY=zeros(Int,4); NC=zeros(Int,4)
        for c in KEEP
            c in ALL || continue
            S,_=Country_Row(FOLDER,PANEL,c;WithB09=WithB09)
            (S===nothing || S.Oil_Share<MinOilShare) && continue
            NC[1]+=1
            S.λo>S.λf && (NC[2]+=1); S.αMo<S.αMf && (NC[3]+=1)
            S.Exposure_o<S.Exposure_f && (NC[4]+=1)
            for y in Measured_Years(FOLDER,c)
                T=Shares_From_Panel(PANEL,y,c;WithB09=WithB09)
                (T===nothing || T.Oil_Share<MinOilShare) && continue
                NY[1]+=1
                T.λo>T.λf && (NY[2]+=1); T.αMo<T.αMf && (NY[3]+=1)
                T.Exposure_o<T.Exposure_f && (NY[4]+=1)
            end
        end
        for (UNIT,N) in (("latest year each",NC),("all measured    ",NY))
            @printf("%-20s %s n=%3d  lambda_o>lambda_f %3d  alpha_Mo<alpha_Mf %3d  oil less exposed %3d\n",
                    TAG,UNIT,N[1],N[2],N[3],N[4])
        end
    end
    return nothing
end
