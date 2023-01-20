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
local depvar shr_n shr_m shr_c shr_o shr_cc

foreach var of varlist `depvar' {
gen `var'1=L.`var'
*gen `var'2=L2.`var'
*gen `var'3=L3.`var'
*gen `var'4=L4.`var'
}

*** BENCHMARK SETTING: all countries with available data during 1993-2012
drop if year<1993 | year>2012

**Benchmark specification
xtscc shr_n shr_n1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' i.year , fe
xtscc shr_m shr_m1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' i.year , fe
xtscc shr_cc shr_cc1 sizerealistic `sizerealistic_lag' `ln_oil_int_lag' i.year , fe

**Specification without control for interaction between price of oil and indicator of recent discoveries	
xtscc shr_n shr_n1 sizerealistic `sizerealistic_lag' i.year , fe
xtscc shr_m shr_m1 sizerealistic `sizerealistic_lag' i.year , fe
xtscc shr_cc shr_cc1 sizerealistic `sizerealistic_lag' i.year , fe

clear all
