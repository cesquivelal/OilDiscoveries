** Empirical results for "The Sovereign Default Risk of Giant Oil Discoveries"
**
** Writes Regressions_Benchmark.txt, read directly by EmpiricalResults.jl to draw
** Figures 2, 3 and 4 in the main text and Figures 11, 15 and 16 in the appendix.
**
** The Julia code finds each regression by the name given in ctitle() below, so the
** regressions can be reordered or added to freely, but a ctitle must not be renamed
** without renaming it in EmpiricalResults.jl as well.

** Run Stata from the package root
clear all
insheet using "Code_Data/EsquivelOilDiscoveries_data.csv"

** Sort variables, declare panel and destring main variable
sort ifscode year
xtset ifscode year

** Generate lags of discovery size variable and interaction of price of oil with discovery
by ifscode: gen npv=sizerealistic
by ifscode: gen ln_reserves=100*log(reserves)
forval i=1/10{

by ifscode: gen npv_`i'=npv[_n-`i']
by ifscode: gen ln_reserves_`i'=ln_reserves[_n-`i']

}

** Alternative measures of discovery size
by ifscode: gen npv_risk=sizerisk
by ifscode: gen npv_com5=sizecom5
by ifscode: gen npv_urr=urr
forval i=1/10{

by ifscode: gen npv_risk_`i'=npv_risk[_n-`i']
by ifscode: gen npv_com5_`i'=npv_com5[_n-`i']
by ifscode: gen npv_urr_`i'=npv_urr[_n-`i']

}

** Discovery size with the discoveries of 2009 and later removed as treatments
by ifscode: gen npv_pre=npv
replace npv_pre=0 if year>=2009 & npv_pre<.
forval i=1/10{

by ifscode: gen npv_pre_`i'=npv_pre[_n-`i']

}

** Leads of discovery size, for the pre-trend test
forval i=1/3{

by ifscode: gen npv_f`i'=npv[_n+`i']

}

by ifscode: gen ln_oil_price=100*log(oil_price)
by ifscode: gen dln_oil_price=100*(log(oil_price)-log(L.oil_price))
by ifscode: gen int_p_dep = oil_rents_93*(log(oil_price)-log(L.oil_price))
forval i=1/10{

by ifscode: gen ln_oil_int_`i'=discovery[_n-`i']*ln_oil_price[_n-0]

}

** Alternatives to the interactions: the price change in place of the price level,
** and the lagged discovery dummies on their own. disc is also the treatment variable in
** the exercise that replaces the net present value with the plain discovery indicator
by ifscode: gen disc=discovery
forval i=1/10{

by ifscode: gen d_oil_int_`i'=discovery[_n-`i']*dln_oil_price[_n-0]
by ifscode: gen disc_`i'=discovery[_n-`i']

}

local npv_lag npv_1 npv_2 npv_3 npv_4 npv_5 npv_6 npv_7 npv_8 npv_9 npv_10

local npv_lead1 npv_f1

local npv_lead3 npv_f3 npv_f2 npv_f1

local npv_risk_lag npv_risk_1 npv_risk_2 npv_risk_3 npv_risk_4 npv_risk_5 npv_risk_6 npv_risk_7 npv_risk_8 npv_risk_9 npv_risk_10

local npv_com5_lag npv_com5_1 npv_com5_2 npv_com5_3 npv_com5_4 npv_com5_5 npv_com5_6 npv_com5_7 npv_com5_8 npv_com5_9 npv_com5_10

local npv_urr_lag npv_urr_1 npv_urr_2 npv_urr_3 npv_urr_4 npv_urr_5 npv_urr_6 npv_urr_7 npv_urr_8 npv_urr_9 npv_urr_10

local npv_pre_lag npv_pre_1 npv_pre_2 npv_pre_3 npv_pre_4 npv_pre_5 npv_pre_6 npv_pre_7 npv_pre_8 npv_pre_9 npv_pre_10

local ln_oil_int_lag ln_oil_int_1 ln_oil_int_2 ln_oil_int_3 ln_oil_int_4 ln_oil_int_5 ln_oil_int_6 ln_oil_int_7 ln_oil_int_8 ln_oil_int_9 ln_oil_int_10

local d_oil_int_lag d_oil_int_1 d_oil_int_2 d_oil_int_3 d_oil_int_4 d_oil_int_5 d_oil_int_6 d_oil_int_7 d_oil_int_8 d_oil_int_9 d_oil_int_10

local disc_lag disc_1 disc_2 disc_3 disc_4 disc_5 disc_6 disc_7 disc_8 disc_9 disc_10

local ln_reserves_lag ln_reserves_1 ln_reserves_2 ln_reserves_3 ln_reserves_4 ln_reserves_5 ln_reserves_6 ln_reserves_7 ln_reserves_8 ln_reserves_9 ln_reserves_10

** Generate group of dependent variables and 1 lag
local depvar spreads lngdplcu ca_gdp net_debt lncon_priv lncon_gov

foreach var of varlist `depvar' {
gen `var'1=L.`var'
}

drop if year<1993 | year>2012

** Generate country specific quadratic trends
egen tt=group(year)
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
********************** Figure 2, spreads ******************************
***********************************************************************
*Spreads
gen ar1=spreads1
xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway replace noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(spreads) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********* Figure 3, GDP, current account, net government debt *********
***********************************************************************

*GDP
replace ar1=lngdplcu1
xtscc lngdplcu ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(gdp) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Current account
replace ar1=ca_gdp1
xtscc ca_gdp ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(ca) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Net debt, IMF data
replace ar1=net_debt1
xtscc net_debt ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(net debt) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********************* Figure 4, consumption ***************************
***********************************************************************

*Private consumption
replace ar1=lncon_priv1
xtscc lncon_priv ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(private) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Government consumption
replace ar1=lncon_gov1
xtscc lncon_gov ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(government) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
************ Figure 11, spreads controlling for reserves **************
***********************************************************************

replace ar1=spreads1
xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' ln_reserves i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(spreads_res) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' ln_reserves `ln_reserves_lag' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(spreads_res_lags) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********** Figure 14, alternative measures of discovery size **********
***********************************************************************
*Country-specific discount rate, constant profile
replace ar1=spreads1
xtscc spreads ar1 npv_risk `npv_risk_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(e1b_risk) keep(ar1 npv_risk `npv_risk_lag') addtext(Country FE, Yes, Year FE, Yes)

*Common discount rate of 5 percent, constant profile
xtscc spreads ar1 npv_com5 `npv_com5_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(e1b_com5) keep(ar1 npv_com5 `npv_com5_lag') addtext(Country FE, Yes, Year FE, Yes)

*Ultimately recoverable reserves, no discounting
xtscc spreads ar1 npv_urr `npv_urr_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(e1b_urr) keep(ar1 npv_urr `npv_urr_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
***************** Figure 13, pre-trends with leads ********************
***********************************************************************
*One and three leads of discovery size added to the Figure 2 specification
replace ar1=spreads1
xtscc spreads ar1 `npv_lead1' npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(e1a_lead1) keep(ar1 `npv_lead1' npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

xtscc spreads ar1 `npv_lead3' npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(e1a_lead3) keep(ar1 `npv_lead3' npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********** Figure 17, dropping the years of Chinese lending ***********
***********************************************************************
*Panel truncated at 2008, so no outcome falls in the lending wave. Not plotted
replace ar1=spreads1
xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year if year<=2008 , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(e2c_obs) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Figure 17, full panel with the discoveries of 2009 and later removed as treatments
xtscc spreads ar1 npv_pre `npv_pre_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(e2c_treat) keep(ar1 npv_pre `npv_pre_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********** Spreads without the oil-price interactions ******************
***********************************************************************
*The Figure 2 specification with the ten discovery-by-oil-price interactions removed.
*int_p_dep, the predetermined 1993 oil-rents share times the price change, is kept
replace ar1=spreads1
xtscc spreads ar1 npv `npv_lag' int_p_dep `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_noint) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*The interactions rebuilt on the change in the log oil price instead of its level
xtscc spreads ar1 npv `npv_lag' int_p_dep `d_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_dint) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Diagnostic only, never plotted in the paper: the ten lagged discovery dummies replace the
*interactions, which holds the extensive margin of the treatment fixed and identifies the
*coefficients on npv off differences in size among discoveries alone
xtscc spreads ar1 npv `npv_lag' int_p_dep `disc_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_dum) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Figure 2 with the plain discovery indicator in place of the net present value, so that the
*treatment is the fact of a discovery rather than its size. The sample is held at the
*benchmark's, since dropping npv from the specification would otherwise widen it
replace ar1=spreads1
xtscc spreads ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
capture drop insamp_r24
gen byte insamp_r24=e(sample)
count if insamp_r24

*Diagnostic only, never plotted: with disc in the specification the interactions are the
*same indicator scaled by the log price, so the coefficients on disc are evaluated at a log
*price of zero, which is an oil price of one dollar against a sample range of 277 to 468 on
*that scale. The estimates are an extrapolation and are not interpretable
xtscc spreads ar1 disc `disc_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year if insamp_r24 , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_disc) keep(ar1 disc `disc_lag') addtext(Country FE, Yes, Year FE, Yes)

*Without them
xtscc spreads ar1 disc `disc_lag' int_p_dep `CS_quad_trend' i.year if insamp_r24 , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_disc_noint) keep(ar1 disc `disc_lag') addtext(Country FE, Yes, Year FE, Yes)

*The URR specification with and without the interactions. The sample is held at the URR
*regression's own, which is 437 rather than 427 because reserves are recorded for ten
*country-years in which no net present value is
replace ar1=spreads1
xtscc spreads ar1 npv_urr `npv_urr_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
capture drop insamp_urr
gen byte insamp_urr=e(sample)
count if insamp_urr

xtscc spreads ar1 npv_urr `npv_urr_lag' int_p_dep `CS_quad_trend' i.year if insamp_urr , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_urr_noint) keep(ar1 npv_urr `npv_urr_lag') addtext(Country FE, Yes, Year FE, Yes)

***********************************************************************
********* The other five outcomes with and without *********************
***********************************************************************
*Every outcome of Figures 3 and 4 re-estimated without the discovery-by-oil-price
*interactions, so that all six responses can be shown both ways. Everything else follows
*each outcome's own benchmark: the current account and net debt carry no country-specific
*trends, because those series have no trend

*GDP without the interactions
replace ar1=lngdplcu1
xtscc lngdplcu ar1 npv `npv_lag' int_p_dep `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_gdp_noint) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Current account without the interactions
replace ar1=ca_gdp1
xtscc ca_gdp ar1 npv `npv_lag' int_p_dep i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_ca_noint) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Net government debt without the interactions
replace ar1=net_debt1
xtscc net_debt ar1 npv `npv_lag' int_p_dep i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_nd_noint) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Private consumption without the interactions
replace ar1=lncon_priv1
xtscc lncon_priv ar1 npv `npv_lag' int_p_dep `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_priv_noint) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Government consumption without the interactions
replace ar1=lncon_gov1
xtscc lncon_gov ar1 npv `npv_lag' int_p_dep `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(r24_gov_noint) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)
***********************************************************************
********* Figures 15 and 16, EMBI countries only **********************
***********************************************************************
drop if in_embi_dumm<1

*GDP
replace ar1=lngdplcu1
xtscc lngdplcu ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(gdp_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Current account
replace ar1=ca_gdp1
xtscc ca_gdp ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(ca_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Net debt, IMF data
replace ar1=net_debt1
xtscc net_debt ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(net debt_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Private consumption
replace ar1=lncon_priv1
xtscc lncon_priv ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(private_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

*Government consumption
replace ar1=lncon_gov1
xtscc lncon_gov ar1 npv `npv_lag' int_p_dep `ln_oil_int_lag' `CS_quad_trend' i.year , fe
outreg2 using "Code_Data/Regressions_Benchmark.txt", sideway append noaster noparen auto(6) st(coef ci_low ci_high) level(90) ctitle(government_embi) keep(ar1 npv `npv_lag') addtext(Country FE, Yes, Year FE, Yes)

clear all
