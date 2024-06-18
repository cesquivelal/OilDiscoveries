** Import full data base and do exercise with all possible observations
clear all
*set more off
cd "C:\Users\ce265\Box\Research\Submitted Papers\OilDiscoveries\OilDiscoveries_2023_07_JIE_R&R\Draft"
insheet using "EsquivelOilDiscoveries_data.csv"

** Sort variables, declare panel and destring main variable
sort ifscode year
xtset ifscode year

** Generate lags of discovery size variable and interaction of price of oil with discovery
by ifscode: gen npv=sizerealistic
forval i=1/10{

by ifscode: gen sizerealistic_`i'=sizerealistic[_n-`i']
by ifscode: gen npv_`i'=npv[_n-`i']
by ifscode: gen urr_`i'=urr[_n-`i']

}

by ifscode: gen ln_oil_price=100*log(oil_price)
*by ifscode: gen int_p_dep = oil_rents_av*(log(oil_price)-log(L.oil_price))
by ifscode: gen int_p_dep = oil_rents_93*(log(oil_price)-log(L.oil_price))
forval i=1/10{

by ifscode: gen ln_oil_int_`i'=discovery[_n-`i']*ln_oil_price[_n-0]

}

local sizerealistic_lag sizerealistic_1 sizerealistic_2 sizerealistic_3 sizerealistic_4 sizerealistic_5 sizerealistic_6 sizerealistic_7 sizerealistic_8 sizerealistic_9 sizerealistic_10

local ln_oil_int_lag ln_oil_int_1 ln_oil_int_2 ln_oil_int_3 ln_oil_int_4 ln_oil_int_5 ln_oil_int_6 ln_oil_int_7 ln_oil_int_8 ln_oil_int_9 ln_oil_int_10

local npv_lag npv_1 npv_2 npv_3 npv_4 npv_5 npv_6 npv_7 npv_8 npv_9 npv_10

local urr_lag urr_1 urr_2 urr_3 urr_4 urr_5 urr_6 urr_7 urr_8 urr_9 urr_10

** Generate group of dependent variables and 1 lag
*gen spreads=embi/100
gen tot_inv=inv_n+inv_m+inv_c+inv_o
gen shr_n=100*inv_n/tot_inv
gen shr_m=100*inv_m/tot_inv
gen shr_c=100*inv_c/tot_inv
gen shr_o=100*inv_o/tot_inv
gen shr_cc=100*(inv_o+inv_c)/tot_inv
local depvar spreads embi inv_gdp ca_gdp lngdplcu ln_rer_def shr_n shr_m shr_c shr_o shr_cc fdi lncon_priv lncon_gov lncon_tot cent_gov_debt_gdp fdi_net_pos debt_net_pos porteq_net_pos net_iip prim_bal net_debt

foreach var of varlist `depvar' {
gen `var'1=L.`var'
*gen `var'2=L2.`var'
*gen `var'3=L3.`var'
*gen `var'4=L4.`var'
}

***********************************************************************
****************** Figure 4, Dutch disease ****************************
***********************************************************************
*Use all countries due to data availablity for investment
drop if year<1993 | year>2012

*Share in non-traded
gen ar1=shr_n1
xtscc shr_n ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway replace noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(sh_n) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Share in manufacturing
replace ar1=shr_m1
xtscc shr_m ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(sh_m) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Share in commodities
replace ar1=shr_cc1
xtscc shr_cc ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(sh_c) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)
  
*Use only EMBI countries hereafter
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

local CS_quad_trend
forvalues i=2/`dummax' {
  local CS_quad_trend `CS_quad_trend' ttsquare`i'
  }
  
*Real exchange rate
replace ar1=ln_rer_def1
xtscc ln_rer_def ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(rer) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********* Figure 2, spreads, GDP, investment, current account *********
***********************************************************************

*Spreads
replace ar1=spreads1
xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(spreads) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*GDP
replace ar1=lngdplcu1
xtscc lngdplcu ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(gdp) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Investment
replace ar1=inv_gdp1
xtscc inv_gdp ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(investment) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Current account
replace ar1=ca_gdp1
xtscc ca_gdp ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(ca) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
************************ Figure 3, IIP ********************************
***********************************************************************
*FDI net position, EWN data
replace ar1=fdi_net_pos1
xtscc fdi_net_pos ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(FDI) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*IIP net, EWN data
replace ar1=net_iip1
xtscc net_iip ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(net iip) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Primary balance, EWN data
replace ar1=prim_bal1
xtscc prim_bal ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(primary balance) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Net debt, IMF data
replace ar1=net_debt1
xtscc net_debt ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(net debt) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********************* Figure 4, Consumption ***************************
***********************************************************************
*consumption
replace ar1=lncon_tot1
xtscc lncon_tot ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(cons) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Private consumption
replace ar1=lncon_priv1
xtscc lncon_priv ar1 npv `npv_lag' int_p_dep `ln_oil_dep_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(private) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Government consumption
replace ar1=lncon_gov1
xtscc lncon_gov ar1 npv `npv_lag' int_p_dep `ln_oil_dep_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(government) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

clear all
