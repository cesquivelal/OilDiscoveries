** Falsification test for "The Sovereign Default Risk of Giant Oil Discoveries"
**
** Writes Falsification_Draws.csv, read by EmpiricalResults.jl to draw Figure 12.
**
** Equation (2) estimated unchanged, with the discovery histories of the 15 countries that
** had giant discoveries reassigned to 15 countries drawn at random from the 37 in the EMBI.
** Sizes, counts and timing are the actual ones; only which country received them changes.
** The benchmark is the draw in which every history returns to its own country.
**
** The panel is not trimmed, because lags of a reassigned treatment have to be built on the
** full panel. The estimation window is imposed with an if condition instead.

clear all
set seed 20260803
local REPS 1000

** Run Stata from the package root
insheet using "Code_Data/EsquivelOilDiscoveries_data.csv"

sort ifscode year
xtset ifscode year

** Keep the 37 EMBI countries
keep if in_embi_dumm==1

by ifscode: gen npv=sizerealistic
by ifscode: gen ln_oil_price=100*log(oil_price)
by ifscode: gen int_p_dep = oil_rents_93*(log(oil_price)-log(L.oil_price))
gen spreads1=L.spreads
gen ar1=spreads1

gen byte inwin = (year>=1993 & year<=2012)
gen byte disc_in = (npv>0 & npv<. & inwin)
bysort ifscode: egen anydisc = max(disc_in)

** Hold each discovery country's history in a variable defined for every country-year, so a
** replication can hand it to whichever country drew that donor
levelsof ifscode if anydisc==1, local(donors)
local j = 0
foreach d of local donors {
  local ++j
  capture drop dtmp
  gen double dtmp = npv if ifscode==`d'
  bysort year: egen double donor`j'_npv = max(dtmp)
  drop dtmp
}
local NDONORS = `j'
display "donor histories = `NDONORS'"
sort ifscode year

** Country specific quadratic trends, defined only inside the estimation window
egen tt=group(year) if inwin
gen ttsquare=tt^2
egen float dum = group(ifscode)
sum dum
local dummax=r(max)
xi I.dum , prefix(cc_)

local CS_quad_trend
  forvalues i=2/`dummax' {
  g ttsquare`i'=ttsquare*cc_dum_`i'
  local CS_quad_trend `CS_quad_trend' ttsquare`i'
  }

***********************************************************************
********** Falsification, discovery histories reassigned **************
***********************************************************************

tempname pf
postfile `pf' int rep double(rho psi0 psi1 psi2 psi3 psi4 psi5 psi6 psi7 psi8 psi9 psi10) int nobs int ngroups using "Code_Data/Falsification_Draws.dta", replace

forvalues r=1/`REPS' {
  quietly {
    capture drop ucty
    capture drop rk
    capture drop crank
    capture drop npvp
    capture drop npvp_*
    capture drop pb_disc
    capture drop pb_int_*

    ** Rank the 37 countries at random; the first NDONORS receive the histories
    bysort ifscode (year): gen double ucty = runiform() if _n==1
    egen double rk = rank(ucty)
    bysort ifscode: egen double crank = max(rk)

    gen double npvp = 0
    forvalues j=1/`NDONORS' {
      replace npvp = donor`j'_npv if crank==`j'
    }
    gen byte pb_disc = (npvp>0 & npvp<.)

    sort ifscode year
    forvalues i=1/10 {
      by ifscode: gen npvp_`i'=npvp[_n-`i']
      by ifscode: gen pb_int_`i'=pb_disc[_n-`i']*ln_oil_price
    }

    xtscc spreads ar1 npvp npvp_1 npvp_2 npvp_3 npvp_4 npvp_5 npvp_6 npvp_7 npvp_8 npvp_9 npvp_10 int_p_dep pb_int_1 pb_int_2 pb_int_3 pb_int_4 pb_int_5 pb_int_6 pb_int_7 pb_int_8 pb_int_9 pb_int_10 `CS_quad_trend' i.year if inwin , fe

    post `pf' (`r') (_b[ar1]) (_b[npvp]) (_b[npvp_1]) (_b[npvp_2]) (_b[npvp_3]) (_b[npvp_4]) (_b[npvp_5]) (_b[npvp_6]) (_b[npvp_7]) (_b[npvp_8]) (_b[npvp_9]) (_b[npvp_10]) (e(N)) (e(N_g))
  }
}
postclose `pf'

***********************************************************************
********************* Export for Julia ********************************
***********************************************************************
use "Code_Data/Falsification_Draws.dta", clear
export delimited using "Code_Data/Falsification_Draws.csv", replace
erase "Code_Data/Falsification_Draws.dta"

clear all
