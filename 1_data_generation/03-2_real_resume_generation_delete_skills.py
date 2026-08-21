import json
import re
import os
import pandas as pd
import ast
import argparse
from tqdm import tqdm
# ----------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--processing_folder",
    type=str,
    required=True,
    help="Folder containing resume text files",
)
args = parser.parse_args()

processing_folder = args.processing_folder
resumes = os.listdir(processing_folder)
resumes = [f for f in resumes if "no_skills" not in f and f.endswith(".txt")]

# ----------------------------
for resume_file in tqdm(resumes):
    resume_id = int(resume_file.split("_")[-1].replace(".txt", ""))
    with open(os.path.join(processing_folder, resume_file), "r", encoding="utf-8") as f:
        resume_text = f.read() 
        
    pattern = r"(\nSKILLS\s*\n)(.*?)(\n[A-Z ]{2,}|\Z)"
    new_resume_text = re.sub(pattern, r"\3", resume_text, flags=re.DOTALL)

    new_resume_file = resume_file.replace(".txt", "_no_skills.txt")
    with open(os.path.join(processing_folder, new_resume_file), "w", encoding="utf-8") as f:
        f.write(new_resume_text)
# ----------------------------