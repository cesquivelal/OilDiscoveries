** Import full data base and do exercise with all possible observations
clear all
*set more off
insheet using "C:\Users\Carlos\Documents\Minnesota\6th Year\Sovereign Risk and Dutch Disease\Data\Oil_Spreads_Macro.csv"

** Sort variables, declare panel and destring main variable
sort ifscode year
xtset ifscode year

** Generate lags of discovery size variable and interaction of price of oil with discovery
forval i=1/10{

by ifscode: gen sizerealistic_`i'=sizerealistic[_n-`i']

}



local sizerealistic_lag sizerealistic_1 sizerealistic_2 sizerealistic_3 sizerealistic_4 sizerealistic_5 sizerealistic_6 sizerealistic_7 sizerealistic_8 sizerealistic_9 sizerealistic_10

** Generate group of dependent variables and 4 lags
local depvar embi inv_gdp ca_gdp lngdplcu lnclcu ln_rer_def

foreach var of varlist `depvar' {
gen `var'1=L.`var'
*gen `var'2=L2.`var'
*gen `var'3=L3.`var'
*gen `var'4=L4.`var'
}

*** BENCHMARK SETTING: EMBI countries during 1993-2012
drop if year<1993 | year>2012
drop if in_embi<1

** Generate country specific quadratic trends

egen tt=group(year)
gen ttsquare=tt^2

egen float dum = group(ifscode) 
sum dum
local dummax=r(max)
xi I.dum , prefix(cc_)
*g constant=1
g tt1=tt
g ttsquare1=ttsquare
  forvalues i=2/`dummax' {
  g tt`i'=tt*cc_dum_`i'
  g ttsquare`i'=ttsquare*cc_dum_`i'
  }
  
local CS_quad_trend ttsquare2 ttsquare3 ttsquare4 ttsquare5 ttsquare6 ttsquare7 ttsquare8 ttsquare9 ttsquare10 ttsquare11 ttsquare12 ttsquare13 ttsquare14 ttsquare15 ttsquare16 ttsquare17 ttsquare18 ttsquare19 ttsquare20 ttsquare21 ttsquare22 ttsquare23 ttsquare24 ttsquare25 ttsquare26 ttsquare27 ttsquare28 ttsquare29 ttsquare30 ttsquare31 ttsquare32 ttsquare33 ttsquare34 ttsquare35 ttsquare36 ttsquare37

*Spreads
xtscc embi embi1 sizerealistic `sizerealistic_lag' `CS_quad_trend' i.year , fe

*Investment, current account, GDP, consumption
xtscc inv_gdp inv_gdp1 sizerealistic `sizerealistic_lag' i.year , fe
xtscc ca_gdp ca_gdp1 sizerealistic `sizerealistic_lag' i.year , fe
xtscc lngdplcu lngdplcu1 sizerealistic `sizerealistic_lag' `CS_quad_trend' i.year , fe
xtscc lnclcu lnclcu1 sizerealistic `sizerealistic_lag' `CS_quad_trend' i.year , fe

*Dutch Disease
xtscc ln_rer_def ln_rer_def1 sizerealistic `sizerealistic_lag' `CS_quad_trend' i.year , fe

clear all
