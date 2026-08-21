#!/bin/bash
# Render resume PDFs to JPG for VLM scoring.
# Produces both variants: with the profile photo, and with images stripped.
#   export DATA_ROOT=/your/CausalFair && ./pdf_to_jpg.sh
set -euo pipefail
: "${DATA_ROOT:?set DATA_ROOT to your CausalFair checkout}"

BASE_DIR="${DATA_ROOT}/Resume/job-distribution"
PREFIX="processed_job_data_0102_with_exp_pred_with_names_with_resumes_datafull"
JOBS=(accountants_auditors construction_laborers elementary_middle_school_teachers
      registered_nurses software_developers)

# pdf_to_jpg.py derives its output directory from the input: it replaces "pdf" with
# "jpg" and appends "_no_images" when --remove_images is given. So the input must be
# *_resumes_pdf for the outputs to land in the *_resumes_jpg[_no_images] folders that
# run_scoring.sh reads.
for job in "${JOBS[@]}"; do
    folder="${BASE_DIR}/${PREFIX}_${job}_resumes_pdf"
    python pdf_to_jpg.py --folder_dir "${folder}" --dpi 150                    # -> *_resumes_jpg
    python pdf_to_jpg.py --folder_dir "${folder}" --dpi 150 --remove_images    # -> *_resumes_jpg_no_images
done
