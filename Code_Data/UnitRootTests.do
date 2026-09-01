** Unit root tests. Run Stata from the package root
clear all
*set more off
insheet using "Code_Data/EsquivelOilDiscoveries_data.csv"

** Sort variables, declare panel and destring main variable
sort ifscode year
xtset ifscode year

local varlist lncon_priv lncon_gov lngdplcu spreads
local min_obs 10
foreach var of local varlist {

    * Drop any temporary variables from previous loop
    capture drop valid_panel n_obs

    * Generate number of non-missing obs per panel for this variable
    * Apply in_embi < 1 filter only if variable is spreads
    if "`var'" == "spreads" {
        gen byte hasval = !missing(`var') & in_embi >= 1
    }
    else {
        gen byte hasval = !missing(`var')
    }

    bysort ifscode (year): gen byte first = _n == 1
    by ifscode (year): gen n_obs = sum(hasval)
    by ifscode (year): replace n_obs = n_obs[_N]
    drop hasval first

    * Create valid_panel dummy for ifscode groups with enough obs
    gen byte valid_panel = n_obs >= `min_obs'

    * Run the IPS test only on valid panels
    if "`var'" == "spreads" {
        xtunitroot ips `var' if valid_panel & in_embi >= 1
    }
    else {
        xtunitroot ips `var' if valid_panel
    }

    * Optional: Clean up to avoid conflicts in next loop
    drop valid_panel n_obs
}

* Choose number of lags for dfuller test
drop if in_embi<1

local lags 1

* Create a temp file to store results
tempname results
postfile `results' int(ifscode) double(pval) using "Code_Data/adf_summary.dta", replace

* Loop over countries
levelsof ifscode, local(countries)
foreach c of local countries {
    
    * Count non-missing observations
    quietly count if ifscode == `c' & !missing(spreads)
    local n = r(N)

    if `n' >= `min_obs' {
        * Run ADF test
        quietly dfuller spreads if ifscode == `c', lags(`lags')

        * Extract and store p-value
        local p = r(p)
        post `results' (`c') (`p')
    }
}

* Close the postfile and load results
postclose `results'
use "Code_Data/adf_summary.dta", clear

* Rename the p-value column for clarity
rename pval Dickey_Fuller_test_p_value

* Format and display the table
format Dickey_Fuller_test_p_value %9.4f
list ifscode Dickey_Fuller_test_p_value, sep(0) noobs