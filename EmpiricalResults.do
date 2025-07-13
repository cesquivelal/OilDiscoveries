** Import full data base and do exercise with all possible observations
clear all
*set more off
cd "C:\Users\ce265\Box\Research\Papers, submitted\OilDiscoveries\OilDiscoveries_2025_06_\DraftAndCode\Code_Data"
insheet using "EsquivelOilDiscoveries_data.csv"

** Sort variables, declare panel and destring main variable
sort ifscode year
xtset ifscode year

** Generate lags of discovery size variable and interaction of price of oil with discovery
by ifscode: gen npv=sizerealistic
by ifscode: gen ln_reserves=100*log(reserves)
forval i=1/10{

by ifscode: gen sizerealistic_`i'=sizerealistic[_n-`i']
by ifscode: gen npv_`i'=npv[_n-`i']
by ifscode: gen urr_`i'=urr[_n-`i']
by ifscode: gen ln_reserves_`i'=ln_reserves[_n-`i']

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

local ln_reserves_lag ln_reserves_1 ln_reserves_2 ln_reserves_3 ln_reserves_4 ln_reserves_5 ln_reserves_6 ln_reserves_7 ln_reserves_8 ln_reserves_9 ln_reserves_10

** Generate group of dependent variables and 1 lag
*gen spreads=embi/100
local depvar spreads embi inv_gdp ca_gdp lngdplcu fdi lncon_priv lncon_gov lncon_tot cent_gov_debt_gdp fdi_net_pos debt_net_pos porteq_net_pos net_iip prim_bal net_debt

foreach var of varlist `depvar' {
gen `var'1=L.`var'
*gen `var'2=L2.`var'
*gen `var'3=L3.`var'
*gen `var'4=L4.`var'
}

drop if year<1993 | year>2012

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

***********************************************************************
********************** Figure 2, spreads ******************************
***********************************************************************
*Spreads
gen ar1=spreads1
xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway replace noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(spreads) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
**************** Figure 3, GDP, current account ***********************
***********************************************************************

*GDP
replace ar1=lngdplcu1
xtscc lngdplcu ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(gdp) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Current account
replace ar1=ca_gdp1
xtscc ca_gdp ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(ca) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Net debt, IMF data
replace ar1=net_debt1
xtscc net_debt ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(net debt) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********************* Figure 5, Consumption ***************************
***********************************************************************
*Private consumption
replace ar1=lncon_priv1
xtscc lncon_priv ar1 npv `npv_lag' int_p_dep `ln_oil_dep_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(private) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Government consumption
replace ar1=lncon_gov1
xtscc lncon_gov ar1 npv `npv_lag' int_p_dep `ln_oil_dep_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(government) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********************* Figure Appendix Reserves ************************
***********************************************************************

replace ar1=spreads1
xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' ln_reserves i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(spreads_res) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' ln_reserves `ln_reserves_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(spreads_res_lags) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)


***********************************************************************
********************* Figures Appendix only EMBI **********************
***********************************************************************
drop if in_embi<1
*GDP
replace ar1=lngdplcu1
xtscc lngdplcu ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(gdp_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Current account
replace ar1=ca_gdp1
xtscc ca_gdp ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(ca_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Net debt, IMF data
replace ar1=net_debt1
xtscc net_debt ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(net debt_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Private consumption
replace ar1=lncon_priv1
xtscc lncon_priv ar1 npv `npv_lag' int_p_dep `ln_oil_dep_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(private_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Government consumption
replace ar1=lncon_gov1
xtscc lncon_gov ar1 npv `npv_lag' int_p_dep `ln_oil_dep_lag' `CS_quad_trend' i.year , fe
outreg2 using Regressions_Benchmark.xls, excel sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(government_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

clear all
