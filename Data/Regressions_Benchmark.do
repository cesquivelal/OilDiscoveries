** Import full data base and do exercise with all possible observations
clear all
*set more off
cd "C:\Users\ce265\Google Drive\Rutgers\Research\Working Papers\Default Risk of Giant Oil Discoveries (new)\Draft\Data"
insheet using "Oil_Spreads_Macro.csv"

** Sort variables, declare panel and destring main variable
sort ifscode year
xtset ifscode year

** Generate lags of discovery size variable and interaction of price of oil with discovery
forval i=1/10{

by ifscode: gen sizerealistic_`i'=sizerealistic[_n-`i']

}


by ifscode: gen ln_oil_price=100*log(oil_price)
forval i=1/10{

by ifscode: gen ln_oil_int_`i'=discovery[_n-`i']*ln_oil_price[_n-0]

}

local sizerealistic_lag sizerealistic_1 sizerealistic_2 sizerealistic_3 sizerealistic_4 sizerealistic_5 sizerealistic_6 sizerealistic_7 sizerealistic_8 sizerealistic_9 sizerealistic_10
local ln_oil_int_lag ln_oil_int_1 ln_oil_int_2 ln_oil_int_3 ln_oil_int_4 ln_oil_int_5 ln_oil_int_6 ln_oil_int_7 ln_oil_int_8 ln_oil_int_9 ln_oil_int_10

** Generate group of dependent variables and 4 lags
gen tot_inv=inv_n+inv_m+inv_c+inv_o
gen shr_n=100*inv_n/tot_inv
gen shr_m=100*inv_m/tot_inv
gen shr_c=100*inv_c/tot_inv
gen shr_o=100*inv_o/tot_inv
gen shr_cc=100*(inv_o+inv_c)/tot_inv
local depvar embi inv_gdp ca_gdp lngdplcu lnclcu ln_rer_def fuel_exp tb_gdp shr_n shr_m shr_c shr_o shr_cc

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
xtscc embi embi1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using TABLE_benchmark.doc, tex replace ctitle(spreads) keep(embi1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag') addtext(Country FE, Yes, Year FE, Yes)

*Investment, current account, GDP, consumption
xtscc inv_gdp inv_gdp1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' i.year , fe
xtscc ca_gdp ca_gdp1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' i.year , fe
xtscc lngdplcu lngdplcu1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' `CS_quad_trend' i.year , fe
xtscc lnclcu lnclcu1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' `CS_quad_trend' i.year , fe

*Dutch Disease
xtscc ln_rer_def ln_rer_def1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' `CS_quad_trend' i.year , fe

clear all
