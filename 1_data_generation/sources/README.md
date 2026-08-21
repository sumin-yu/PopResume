# `sources/` — inputs for the data-generation pipeline

The notebooks and scripts in `1_data_generation/` read everything from this directory.
Files marked **shipped** are already here; the rest you must obtain yourself, either because
the provider's terms forbid redistribution or because the file is too large to vendor.

| File | Size | Status | Where it comes from |
|---|---|---|---|
| `job_spec_deterministic.json` | 2 KB | **shipped** | Written by us. Per-occupation O\*NET-SOC code, job zone, and skill list. |
| `pool_company.json` | 29 KB | **shipped** | Written by us. Synthetic employer names per occupation. |
| `pool_major.json` | 7 KB | **shipped** | Written by us. Fields of study per occupation. |
| `pool_education_school.json` | 40 KB | **shipped** | Written by us. Synthetic institution names. |
| `pool_education_bullet.json` | 38 KB | **shipped** | Written by us. Coursework bullet templates. |
| `pool_work_bullet.json` | 30 KB | **shipped** | Written by us. Work-history bullet templates. |
| `census2018_occ_to_soc.xlsx` | 54 KB | **shipped** | U.S. Census 2018 OCC → SOC crosswalk (public domain). |
| `Names_2010Census_Top1000.xlsx` | 94 KB | **shipped** | U.S. Census 2010 surname statistics (public domain). |
| `names.zip` | 7.7 MB | **download** | SSA national baby-name data. See below. |
| `usa_00005.dat` | ~200 MB | **you create** | IPUMS ACS PUMS extract. See below. |
| `PSID/psid_famind_final_filtered_0121.csv` | — | **you create** | PSID family–individual file. See below. |

## `names.zip` — SSA baby names

Public domain, downloaded straight from the SSA:

```bash
curl -L -o names.zip https://www.ssa.gov/oact/babynames/names.zip
```

`01-3_name_sampler.ipynb` reads `yob<year>.txt` members directly out of the archive — leave it zipped.

## `usa_00005.dat` — IPUMS ACS PUMS

IPUMS does not permit redistribution of extracts, so you need to create your own at
<https://usa.ipums.org/usa/>. Register, then build an extract with:

- **Sample:** 2023 ACS 1-year
- **Format:** fixed-width `.dat` (with the accompanying `.xml` DDI codebook)
- **Variables:** `YEAR SAMPLE SERIAL PERNUM PERWT STATEFIP SEX AGE RACE RACED EDUC EDUCD
  DEGFIELD DEGFIELDD DEGFIELD2 DEGFIELD2D EMPSTAT EMPSTATD LABFORCE OCC WORKEDYR`

Place the `.dat` (and its `.xml`) here. `01-1_structured_profile_dataset_generation.ipynb` restricts
the sample to people who are currently employed with a valid occupation code, then samples per
occupation — see §4 of the paper.

## `psid_famind_final_filtered_0121.csv` — PSID work experience

PSID requires registration and forbids redistribution: <https://psidonline.isr.umich.edu/>.
Build a family–individual file covering employment history, then filter to the analysis population
(ages 18–44, valid work-history records) and save it as `sources/PSID/psid_famind_final_filtered_0121.csv`.
`01-2_work_experience.ipynb` uses it to fit the experience-year model that is then transferred to the ACS
sample by density-ratio reweighting.

## Pipeline order

```
01-1  usa_00005.dat + census2018_occ_to_soc.xlsx + job_spec_deterministic.json
        -> processed_job_data_0102.csv
01-2  + PSID/psid_famind_final_filtered_0121.csv
        -> processed_job_data_0102_with_exp_pred.csv
01-3  + names.zip + Names_2010Census_Top1000.xlsx
        -> processed_job_data_0102_with_exp_pred_with_names.csv
           name_grouping_first_final.csv, name_grouping_surname_final.csv
01-4  (no new inputs; collapses edu_level / state and attaches the name proxies)
        -> processed_job_data_0102_with_exp_pred_with_names_and_grouping.csv
02-2  + pool_*.json      -> resume content
03    -> rendered resumes
```
