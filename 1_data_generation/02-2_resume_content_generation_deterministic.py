import pandas as pd
import json
import requests
from tqdm import tqdm
import argparse
import os
import random
import re
import math

STATE_TO_USPS = {
    "Alabama": "AL",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "Alaska": "AK",
    "Hawaii": "HI",
}
REQUIRED_COLUMNS = ["id", "full_name", "job", "age", "edu_level", "pred_exp"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="processed_job_data_0102_with_exp_pred_with_names.csv")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--data_num", type=int, default=-1, help="Number of data points to process. -1 means all.")
    parser.add_argument("--job", type=str, default=None, help="Specific job to process. If None, raise an error.")
    return parser.parse_args()

with open("./sources/job_spec_deterministic.json", "r") as f:
    job_spec = json.load(f)
with open("./sources/pool_education_school.json", "r") as f:
    edu_pool = json.load(f)
with open("./sources/pool_education_bullet.json", "r") as f:
    edu_bullet_pool = json.load(f)
with open("./sources/pool_major.json", "r") as f:
    major_pool = json.load(f)
with open("./sources/pool_company.json", "r") as f:
    company_pool = json.load(f)
with open("./sources/pool_work_bullet.json", "r") as f:
    work_bullet_pool = json.load(f)

EDU_STRUCTURE = {
    "High school graduate/GED": {
        "n_entries": 1,
        "n_bullets": 1
    },
    "Some college (no degree)": {
        "n_entries": 1,
        "n_bullets": 2
    },
    "Associate's degree": {
        "n_entries": 1,
        "n_bullets": 2
    },
    "Bachelor's": {
        "n_entries": 1,
        "n_bullets": 2
    },
    "Master's": {
        "n_entries": 2,   # Bachelor's + Master's
        "n_bullets": 2
    },
    "Professional degree": {
        "n_entries": 2,   # Bachelor's + Professional
        "n_bullets": 2
    },
    "Doctorate": {
        "n_entries": 2,   # Bachelor's + Doctorate
        "n_bullets": 3
    }
}
WORK_ROLE_RULES = [
    ((0, 1), 1),   
    ((1, 3), 1),  
    ((3, 6), 2),   
    ((6, 10), 3), 
    ((10, 15), 4),  
    ((15, float("inf")), 4), 
]

def sample_grad_year(age, degree, current_year=2023):
    # Validate age
    if age is None or (isinstance(age, float) and math.isnan(age)):
        print(f"[ERROR] sample_grad_year: age is None or NaN (age={age}, degree={degree})")
        raise ValueError(f"age is None or NaN: age={age}, degree={degree}")
    if not isinstance(age, (int, float)) or age <= 0:
        print(f"[ERROR] sample_grad_year: age must be a positive number (age={age}, degree={degree})")
        raise ValueError(f"age must be a positive number: age={age}, degree={degree}")

    base_age = {
        "High school graduate/GED": 18,
        "Some college (no degree)": 22,
        "Associate's degree": 22,
        "Bachelor's": 23,
        "Master's": 25,
        "Professional degree": 27,
        "Doctorate": 28
    }.get(degree, None)

    # Validate degree
    if base_age is None:
        print(f"[ERROR] sample_grad_year: Unknown degree type (age={age}, degree={degree})")
        raise ValueError(f"Unknown degree type: degree={degree}")

    if age < base_age:
        base_age = age

    return current_year - (age - base_age)

def role_seniority(role_index, n_roles):
    # oldest role = junior, latest role = senior
    ratio = role_index / max(n_roles - 1, 1)

    if ratio < 0.33:
        return "junior"
    elif ratio < 0.66:
        return "mid"
    else:
        return "senior"

def build_year_ranges_forward(start_year, total_years, n_roles, current_year=2023):
    """
    start_year: int (education_end_year)
    total_years: float
    returns: list of (start_yyyy_mm, end_yyyy_mm) forward in time
    """
    # Validate total_years
    if total_years is None or (isinstance(total_years, float) and math.isnan(total_years)):
        print(f"[ERROR] build_year_ranges_forward: total_years is None or NaN (total_years={total_years})")
        raise ValueError(f"total_years is None or NaN: total_years={total_years}")
    if total_years < 0:
        print(f"[ERROR] build_year_ranges_forward: total_years is negative (total_years={total_years})")
        raise ValueError(f"total_years is negative: total_years={total_years}")

    # Validate start_year
    if start_year is None or (isinstance(start_year, float) and math.isnan(start_year)):
        print(f"[ERROR] build_year_ranges_forward: start_year is None or NaN (start_year={start_year})")
        raise ValueError(f"start_year is None or NaN: start_year={start_year}")

    # Validate n_roles
    if n_roles is None or n_roles <= 0:
        print(f"[ERROR] build_year_ranges_forward: n_roles must be positive (n_roles={n_roles})")
        raise ValueError(f"n_roles must be positive: n_roles={n_roles}")

    total_months = int(round(total_years * 12))
    if total_months < 1:
        total_months = 1  # guarantee at least one month

    # total length of the timeline
    timeline_months = (current_year - start_year) * 12

    # is there enough slack to insert an employment gap?
    slack = timeline_months - total_months

    gap_months = []
    if slack >= 36:  # only insert a gap when at least 3 years of slack remain
        n_gaps = random.choice([1, 2])
        remaining = min(slack, random.randint(12, 36))  # cap the gap at 3 years

        for i in range(n_gaps):
            if i == n_gaps - 1:
                gap_months.append(remaining)
            else:
                # Validate randint range
                lower_bound = 6
                upper_bound = min(18, remaining - 6 * (n_gaps - i - 1))
                if upper_bound < lower_bound:
                    print(f"[ERROR] build_year_ranges_forward: Invalid gap range (lower={lower_bound}, upper={upper_bound}, remaining={remaining})")
                    raise ValueError(f"Invalid gap range: lower={lower_bound}, upper={upper_bound}")
                g = random.randint(lower_bound, upper_bound)
                gap_months.append(g)
                remaining -= g

    # role durations
    base = total_months // n_roles
    remainder = total_months % n_roles
    role_durations = [base + (1 if i < remainder else 0) for i in range(n_roles)]

    # Check for zero-duration roles
    for i, duration in enumerate(role_durations):
        if duration <= 0:
            print(f"[ERROR] build_year_ranges_forward: Role {i} has zero or negative duration "
                  f"(duration={duration}, total_months={total_months} (total_years={total_years}), n_roles={n_roles})")
            raise ValueError(f"Role has zero or negative duration: role_index={i}, duration={duration}")

    # build the timeline: role + gap + role + ...
    segments = []
    for i, d in enumerate(role_durations):
        segments.append(("work", d))
        if i < len(gap_months):
            segments.append(("gap", gap_months[i]))

    # lay roles out backwards so the most recent one lands near 2023
    end_idx = current_year * 12
    ranges = []

    cur = end_idx
    for seg_type, months in reversed(segments):
        start = cur - months
        if seg_type == "work":
            ranges.append((start, cur))
        cur = start

    ranges.reverse()

    def idx_to_ym(idx):
        return f"{idx // 12}.{(idx % 12) + 1:02d}"

    return [(idx_to_ym(s), idx_to_ym(e)) for s, e in ranges]

def generate_education_section(row, edu_pool, major_pool, edu_bullet_pool):
    row_id = row.get("id", "unknown")
    educ = row["edu_level"]
    job = row["job"]
    age = row["age"]

    # Validate edu_level
    if educ not in EDU_STRUCTURE:
        print(f"[ERROR] generate_education_section: Unknown edu_level (row_id={row_id}, edu_level={educ})")
        raise ValueError(f"Unknown edu_level: edu_level={educ}, row_id={row_id}")

    structure = EDU_STRUCTURE[educ]
    n_entries = structure["n_entries"]
    n_bullets = structure["n_bullets"]

    entries = []
    grad_years = []
    if n_entries == 1:
        degrees = [educ]
    else:
        degrees = [educ, "Bachelor's"]

    for deg in degrees:
        # Validate edu_pool has the degree
        if deg not in edu_pool:
            print(f"[ERROR] generate_education_section: Degree not found in edu_pool (row_id={row_id}, degree={deg})")
            raise KeyError(f"Degree not found in edu_pool: degree={deg}, row_id={row_id}")
        if not edu_pool[deg]:
            print(f"[ERROR] generate_education_section: edu_pool[{deg}] is empty (row_id={row_id})")
            raise ValueError(f"edu_pool[{deg}] is empty, row_id={row_id}")

        school = random.choice(edu_pool[deg])
        # major_list = major_pool.get(job, {}).get(deg, [])
        major_list = major_pool.get(job,[])
        major = random.choice(major_list) if major_list else None

        grad_year = sample_grad_year(age, deg)
        if grad_year is None:
            print(f"[ERROR] generate_education_section: Cannot sample grad year (row_id={row_id}, age={age}, degree={deg})")
            raise ValueError(f"Cannot sample grad year: age={age}, degree={deg}, row_id={row_id}")
        grad_years.append(grad_year)

        line = f"{deg}, {major}, {school}, {grad_year}" if major else f"{deg}, {school}, {grad_year}"
        entries.append(line)

    # Validate edu_bullet_pool
    # if educ not in edu_bullet_pool:
    #     print(f"[ERROR] generate_education_section: edu_level not found in edu_bullet_pool (row_id={row_id}, edu_level={educ})")
    #     raise KeyError(f"edu_level not found in edu_bullet_pool: edu_level={educ}, row_id={row_id}")
    # if len(edu_bullet_pool[educ]) < n_bullets:
    #     print(f"[ERROR] generate_education_section: Not enough bullets in edu_bullet_pool "
    #           f"(row_id={row_id}, edu_level={educ}, available={len(edu_bullet_pool[educ])}, required={n_bullets})")
    #     raise ValueError(f"Not enough bullets in edu_bullet_pool: edu_level={educ}, "
    #                     f"available={len(edu_bullet_pool[educ])}, required={n_bullets}, row_id={row_id}")

    bullets = random.sample(edu_bullet_pool[job], k=n_bullets)
    # bullets = random.sample(edu_bullet_pool.get(job, {}).get(educ, []), k=n_bullets)
    bullets = [f"- {b}" for b in bullets]

    education_text = "\n".join(entries) + ("\n" + "\n".join(bullets) if bullets else "")
    education_end_year = max(grad_years)

    return education_text, education_end_year

def generate_work_history_section(row, education_end_year, company_pool, work_bullet_pool):
    row_id = row.get("id", "unknown")
    job = row["job"]
    years = row["pred_exp"]

    # Validate pred_exp
    if years is None or (isinstance(years, float) and math.isnan(years)):
        print(f"[ERROR] generate_work_history_section: pred_exp is None or NaN (row_id={row_id}, pred_exp={years})")
        raise ValueError(f"pred_exp is None or NaN: pred_exp={years}, row_id={row_id}")
    if years < 0:
        print(f"[ERROR] generate_work_history_section: pred_exp is negative (row_id={row_id}, pred_exp={years})")
        raise ValueError(f"pred_exp is negative: pred_exp={years}, row_id={row_id}")

    # Validate job in company_pool
    if job not in company_pool:
        print(f"[ERROR] generate_work_history_section: job not found in company_pool (row_id={row_id}, job={job})")
        raise KeyError(f"job not found in company_pool: job={job}, row_id={row_id}")
    if not company_pool[job]:
        print(f"[ERROR] generate_work_history_section: company_pool[{job}] is empty (row_id={row_id})")
        raise ValueError(f"company_pool[{job}] is empty, row_id={row_id}")

    # Validate job in work_bullet_pool
    if job not in work_bullet_pool:
        print(f"[ERROR] generate_work_history_section: job not found in work_bullet_pool (row_id={row_id}, job={job})")
        raise KeyError(f"job not found in work_bullet_pool: job={job}, row_id={row_id}")

    # 1) decide the structure
    n_roles = None
    for (year_range, roles) in WORK_ROLE_RULES:
        if year_range[0] <= years < year_range[1]:
            n_roles = roles
            break

    if n_roles is None:
        print(f"[ERROR] generate_work_history_section: Could not determine n_roles for years={years} (row_id={row_id})")
        raise ValueError(f"Could not determine n_roles: years={years}, row_id={row_id}")

    year_ranges = build_year_ranges_forward(start_year=education_end_year, total_years=years, n_roles=n_roles)

    # Validate year_ranges count matches expected n_roles
    if len(year_ranges) != n_roles:
        print(f"[ERROR] generate_work_history_section: year_ranges count mismatch "
              f"(row_id={row_id}, expected={n_roles}, got={len(year_ranges)})")
        raise ValueError(f"year_ranges count mismatch: expected={n_roles}, got={len(year_ranges)}, row_id={row_id}")

    roles_text = []

    for i, (start, end) in enumerate(year_ranges):
        company = random.choice(company_pool[job])
        title = job.replace("_", " ").title()

        seniority = role_seniority(i, n_roles)

        # Validate seniority in work_bullet_pool
        if seniority not in work_bullet_pool[job]:
            print(f"[ERROR] generate_work_history_section: seniority not found in work_bullet_pool "
                  f"(row_id={row_id}, job={job}, seniority={seniority})")
            raise KeyError(f"seniority not found in work_bullet_pool: job={job}, seniority={seniority}, row_id={row_id}")

        available_bullets = len(work_bullet_pool[job][seniority])
        required_bullets = 2
        if available_bullets < required_bullets:
            print(f"[ERROR] generate_work_history_section: Not enough bullets in work_bullet_pool "
                  f"(row_id={row_id}, job={job}, seniority={seniority}, available={available_bullets}, required={required_bullets})")
            raise ValueError(f"Not enough bullets in work_bullet_pool: job={job}, seniority={seniority}, "
                           f"available={available_bullets}, required={required_bullets}, row_id={row_id}")

        bullets = random.sample(
            work_bullet_pool[job][seniority],
            k=required_bullets
        )

        role_block = f"{title}, {company}, {start}–{end}\n"
        for b in bullets:
            role_block += f"  - {b}\n"

        roles_text.append(role_block.strip())

    # reverse to have latest role first
    roles_text.reverse()

    return "\n\n".join(roles_text)

def generate_resume(row, pools):
    row_id = row.get("id", "unknown")
    job = row["job"]

    # Validate required fields in row
    if "full_name" not in row or pd.isna(row["full_name"]):
        print(f"[ERROR] generate_resume: full_name is missing or NaN (row_id={row_id})")
        raise ValueError(f"full_name is missing or NaN: row_id={row_id}")

    # Validate job in job_spec
    if job not in job_spec:
        print(f"[ERROR] generate_resume: job not found in job_spec (row_id={row_id}, job={job})")
        raise KeyError(f"job not found in job_spec: job={job}, row_id={row_id}")
    if "skills" not in job_spec[job]:
        print(f"[ERROR] generate_resume: 'skills' not found in job_spec[{job}] (row_id={row_id})")
        raise KeyError(f"'skills' not found in job_spec[{job}]: row_id={row_id}")
    if not job_spec[job]["skills"]:
        print(f"[ERROR] generate_resume: job_spec[{job}]['skills'] is empty (row_id={row_id})")
        raise ValueError(f"job_spec[{job}]['skills'] is empty: row_id={row_id}")

    address = f"{STATE_TO_USPS[row['state_name']]}, USA"
    phone = f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"
    email = f"{row['full_name'].lower().replace(' ', '.')}@example.com"

    education, education_end_year = generate_education_section(
        row,
        pools["edu_pool"],
        pools["major_pool"],
        pools["edu_bullet_pool"]
    )

    work_history = generate_work_history_section(
        row,
        education_end_year,
        pools["company_pool"],
        pools["work_bullet_pool"]
    )

    skills = "\n".join([f"- {s}" for s in job_spec[job]["skills"]])

    return {
        "id": row["id"],
        "name": row["full_name"],
        "address": address,
        "contact": {"phone": phone, "email": email},
        "education": education,
        "work_history": work_history,
        "skills": skills
    }

def append_jsonl(path, obj):
    """Append a single JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")

def generate_resume_and_save(row, output_path, args=None):
    resume = generate_resume(row, pools={
        "edu_pool": edu_pool,
        "major_pool": major_pool,
        "company_pool": company_pool,
        "work_bullet_pool": work_bullet_pool,
        "edu_bullet_pool": edu_bullet_pool
    })
    append_jsonl(output_path, resume)
  
def validate_row(row, row_idx):
    """Validate a single row has all required fields with valid values."""
    row_id = row.get("id", f"index_{row_idx}")
    errors = []

    for col in REQUIRED_COLUMNS:
        if col not in row:
            errors.append(f"Missing column '{col}'")
        elif pd.isna(row[col]):
            errors.append(f"Column '{col}' is NaN")

    if errors:
        error_msg = f"Row validation failed (row_id={row_id}): " + "; ".join(errors)
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    # Additional type/value checks
    if not isinstance(row["age"], (int, float)) or row["age"] <= 0:
        error_msg = f"Invalid age value (row_id={row_id}, age={row['age']})"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    if not isinstance(row["pred_exp"], (int, float)) or row["pred_exp"] < 0:
        error_msg = f"Invalid pred_exp value (row_id={row_id}, pred_exp={row['pred_exp']})"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

def main():
    args = parse_args()
    if args.data_num == -1:
        if args.job is not None:
            print(f"[INFO] Processing all data points for job: {args.job}.")
        else:
            print("[ERROR] When --job is not specified, all jobs must be processed.")
            raise ValueError("When --job is not specified, all jobs must be processed.")
    else:
        if args.job is not None:
            print(f"[INFO] Processing {args.data_num} data points for job: {args.job}.")
        else:
            print("[ERROR] When --job is not specified, all jobs must be processed.")
            raise ValueError("When --job is not specified, all jobs must be processed.")

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    # Validate required columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns in input file: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    if args.job is not None:
        df = df[df["job"] == args.job].reset_index(drop=True)
        if len(df) == 0:
            print(f"[ERROR] No data found for job: {args.job}")
            raise ValueError(f"No data found for job: {args.job}")

    if args.data_num > 0:
        df = df.head(args.data_num)

    output_jsonl = args.input.replace(".csv", f"_with_resumes.jsonl")
    if args.data_num > 0:
        output_jsonl = output_jsonl.replace(".jsonl", f"_data{args.data_num}.jsonl")
    else:
        output_jsonl = output_jsonl.replace(".jsonl", f"_datafull.jsonl")
    if args.job is not None:
        output_jsonl = output_jsonl.replace(".jsonl", f"_{args.job}.jsonl")
    if args.test_mode:
        output_jsonl = output_jsonl.replace(".jsonl", f"_testmode.jsonl")

    with open(output_jsonl, "w") as f:
        pass

    success_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            validate_row(row, idx)
            generate_resume_and_save(row, output_jsonl, args=args)
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"[ERROR] Failed to generate resume for row index {idx}: {e}")
            raise  # Re-raise to stop execution; remove this line if you want to continue on errors

    print(f"[SAVED] JSONL saved: {output_jsonl}")
    print(f"[SUMMARY] Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    main()