import json
import re
import os
import pandas as pd
import ast
import argparse
from tqdm import tqdm


# ----------------------------
# Utilities
# ----------------------------

def parse_contact_string(contact_str):
    if not isinstance(contact_str, str):
        return contact_str
    match = re.search(r"\{.*\}", contact_str, flags=re.DOTALL)
    if not match:
        return contact_str
    try:
        return json.loads(match.group(0))
    except Exception:
        return contact_str 


def _format_contact(contact):
    """
    If contact is a string containing JSON → parse to dict.
    If dict → format nicely.
    If plain string → return stripped string.
    """
    if isinstance(contact, str):
        parsed = parse_contact_string(contact)
        if isinstance(parsed, dict):
            contact = parsed
        else:
            return contact.strip()

    if isinstance(contact, dict):
        parts = []
        for key in ["location", "phone", "email", "linkedin"]:
            val = contact.get(key)
            if val:
                parts.append(str(val).strip())
        return " | ".join(parts)

    return ""


def _strip_section_headers(text):
    """
    Remove duplicated section headers like 'Education:', 'Work History:' but keep formatting.
    """
    if not text:
        return ""
    text = str(text).strip()

    # remove leading headers only
    text = re.sub(
        r"^\s*(\*\*)?\s*(Education|Work History)\s*:?\s*(\*\*)?\s*\n*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


# ----------------------------
# Main Formatter (PASSTHROUGH)
# ----------------------------

def format_resume_manager(resume_json: dict) -> str:
    # only convert when a dict arrived as a string
    if isinstance(resume_json, str):
        resume_json = ast.literal_eval(resume_json)

    # coerce a pandas Series into a dict
    if isinstance(resume_json, pd.Series):
        resume_json = resume_json.to_dict()

    name = str(resume_json.get("name", "")).strip() or "NAME SURNAME"
    address = str(resume_json.get("address", "")).strip()
    contact = _format_contact(resume_json.get("contact", ""))

    # use as-is, without touching the formatting
    skills_block = str(resume_json.get("skills", "")).strip()
    work_block = _strip_section_headers(resume_json.get("work_history", ""))
    edu_block = _strip_section_headers(resume_json.get("education", ""))

    lines = []
    lines.append(name.upper())
    if address and contact:
        lines.append(f"{address} | {contact}")
    # if contact:
    #     lines.append(contact)
    lines.append("")

    if skills_block:
        lines.append("SKILLS")
        lines.append(skills_block)
        lines.append("")

    if work_block:
        lines.append("WORK HISTORY")
        lines.append(work_block)
        lines.append("")

    if edu_block:
        lines.append("EDUCATION")
        lines.append(edu_block)
        lines.append("")

    return "\n".join(lines).rstrip()


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input jsonl file",
    )
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)

    BASE_DIR = args.input.replace(".jsonl", "_resumes_txt")
    os.makedirs(BASE_DIR, exist_ok=True)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = format_resume_manager(row)

        filename = os.path.join(BASE_DIR, f"resume_{int(row['id']):05d}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

    print(f"Resumes saved to {BASE_DIR}")