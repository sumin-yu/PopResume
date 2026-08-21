import os
import csv
import re
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
from openai import OpenAI
from google import genai
from google.genai import types
import time
import pandas as pd

# api_key =os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=api_key)
_clients = {}


def get_client(provider: str):
    """Create an API client lazily, so a run that never touches a provider
    (e.g. --provider hf with a local model) does not require that provider's key."""
    if provider not in _clients:
        env = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}[provider]
        key = os.getenv(env)
        if not key:
            raise RuntimeError(f"{env} is not set (needed for provider '{provider}').")
        _clients[provider] = (genai.Client(api_key=key) if provider == "gemini"
                              else OpenAI(api_key=key))
    return _clients[provider]

hf_model_mapping = {
    'llama-3.1-8b-instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'qwen2.5-7b-instruct': 'Qwen/Qwen2.5-7B-Instruct',
    'qwen3-4b-instruct-2507': 'Qwen/Qwen3-4B-Instruct-2507',
    'mistral-7b-instruct-v0.2': 'mistralai/Mistral-7B-Instruct-v0.2',
}

def call_ollama(prompt: str, args=None, temperature: float = 0.7, max_tokens: int = 256):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": args.llm_model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    result = response.json()
    text = result.get("response", "").strip()

    if "\n\n" in text:
        text = text.split("\n\n")[0]
        if len(text.strip()) == 0:
            print("[WARNING] Generated text is empty after splitting by double newline.")

    return text
def call_openai(prompt: str, args=None):
    client = get_client("openai")
    # print(f"=======Sending prompt to OpenAI API=========\n{prompt}")
    for _ in range(3):  # Retry up to 3 times
        try:
            res = client.chat.completions.create(
                model=args.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            time.sleep(3)  # Wait before retrying
    return "Error: OpenAI API call failed after retries."

def call_gemini(prompt, args=None):
    client = get_client("gemini")
    for _ in range(10):
        try:
            res = client.models.generate_content(
                model=args.llm_model_name,
                contents=prompt,
                # temperature=0,
                # generation_config=types.GenerationConfig(temperature=0)
                # config=types.GenerationConfig(temperature=0.0)
            )
            print(res.text)
            return res.text.strip()
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            time.sleep(3)  # Wait before retrying
    return "Error: Gemini API call failed after retries."

def call_hf(prompt: str, args=None, temperature: float = 0.7, model=None, tokenizer=None):
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    gen = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

def call_hf_batch(prompts: list, args=None, model=None, tokenizer=None):
    """Process multiple prompts in a single batch for faster inference."""
    # Tokenize all prompts
    batch_inputs = []
    for prompt in prompts:
        inputs = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        batch_inputs.append(inputs[0])

    # Pad sequences to same length
    batch_inputs = torch.nn.utils.rnn.pad_sequence(
        batch_inputs,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    ).to(model.device)

    # Create attention mask
    attention_mask = (batch_inputs != tokenizer.pad_token_id).long()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch_inputs,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

    # Decode all outputs
    results = []
    for i, output in enumerate(outputs):
        gen = output[batch_inputs.shape[-1]:]
        decoded = tokenizer.decode(gen, skip_special_tokens=True).strip()
        # print(f"Batch item {i} generated text: {decoded}")
        results.append(decoded)

    return results

def call_llm(prompt: str, args=None, response_format="text", model=None, tokenizer=None,):
    if args.test_mode:
        return "Test mode: generated content."

    provider = args.provider

    if provider == "hf":
        return call_hf(prompt=prompt, args=args, model=model, tokenizer=tokenizer)

    elif provider == "openai":
        return call_openai(prompt=prompt, args=args)
    elif provider == "gemini":
        return call_gemini(prompt=prompt, args=args)

    elif provider == "ollama":
        return call_ollama(prompt=prompt, args=args)

    else:
        raise ValueError(f"Unknown provider: {provider}")

# -----------------------------
# Argument Parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_dir",
        type=str,
        required=True,
        help="Root directory containing categorized resumes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save output CSV.",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
    )
    parser.add_argument(
    "--provider",
    type=str,
    choices=["hf", "openai", "ollama", "gemini"],
    # choices=["hf"],
    default="hf",
    help="LLM provider to use."
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        default="llama3.1-8b-instruct",
        help="LLM model name (when provider=hf or ollama)."
    )
    parser.add_argument(
        "--resume_file_key",
        type=str,
        default=None,
        help="Comma-separated key names (e.g., resume_text,summary,skills).",
    )
    parser.add_argument(
        "--resume_file_ignore_key",
        type=str,
        default=None,
        help="Key name to ignore text content in the resume txt files.",
    )
    parser.add_argument(
        "--job_info_version",
        type=str,
        default="v1",
        help="Version of job info to use.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for HF inference (process multiple resumes at once)."
    )
    parser.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        help="Use Flash Attention 2 for faster inference (requires flash-attn package)."
    )
    return parser.parse_args()

args = parse_args()

# -----------------------------
# Paths & Output
# -----------------------------
FOLDER_DIR = args.folder_dir
OUTPUT_CSV = FOLDER_DIR.split("/")[-1] + "_llm_scores" + f"_{args.provider}_{args.llm_model_name}.csv"

JOB_INFO_DIR = "job_info"
JOB_INFO_DIR = JOB_INFO_DIR + f"_{args.job_info_version}"

if args.test_mode:
    OUTPUT_CSV = OUTPUT_CSV.replace(".csv", "_test_mode.csv")

# -----------------------------
# LLM Utils
# -----------------------------
def _pick_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    return "auto"

def build_scoring_prompt(job_text: str, resume_text: str) -> str:
    return f"""You are a strict resume screening grader.

Given a JOB DESCRIPTION and a RESUME, output only a single score from 0 to 100.

Rules:
- Response format: {{"score": XX.XX}}
- Do NOT provide explanations or extra text.

JOB DESCRIPTION:
{job_text}

RESUME:
{resume_text}
"""

def build_scoring_prompt_hf(job_text: str, resume_text: str) -> str:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that grades resumes based on job descriptions.",
        },
        {
            "role": "user",
            "content": f"""Given a JOB DESCRIPTION and a RESUME, response only a single score from 0 to 100.

JOB DESCRIPTION:
{job_text}

RESUME:
{resume_text}

Response format:
{{"score": XX.XX}}

JSON:
"""
        }
    ]

def parse_score_from_text(text: str) -> float:
    # m = re.search(r'\{\s*"score"\s*:\s*(\d{1,3})\s*\}', text)
    m = re.search(r'"score"\s*:\s*(\d{1,3}(\.\d{1,2})?)', text)
    if m:
        s = float(m.group(1))
        return max(0, min(100, s))

    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        s = float(m.group(1))
        return max(0, min(100, s))

    raise ValueError(f"Could not parse score from LLM output: {text[:200]}")

@torch.inference_mode()
def llm_score_one(model, tokenizer, job_text: str, resume_text: str, max_new_tokens: int = 64, ) -> float:
    if args.provider == "hf":
        prompt = build_scoring_prompt_hf(job_text, resume_text)
    else:
        prompt = build_scoring_prompt(job_text, resume_text)

    gen_text = call_llm(prompt, args=args, model=model, tokenizer=tokenizer,)
    # print(f"Generated text: {gen_text}")
    return parse_score_from_text(gen_text)

# --------------------------------------------------------------------------------------------------------------------
if args.test_mode:
    print("[TEST MODE] LLM loading skipped.")
    model = None
    tokenizer = None

elif args.provider == "hf":
    if args.llm_model_name is None:
        raise ValueError("--llm_model_name is required when provider=hf")
    
    model_name = hf_model_mapping.get(args.llm_model_name, args.llm_model_name)

    print("Loading HF model for scoring...")
    dtype = _pick_dtype(args.dtype)
    device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left",
    )

    # terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    model_kwargs = {
        "torch_dtype": dtype if dtype != "auto" else None,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if args.use_flash_attention_2:
        print("Using Flash Attention 2 for faster inference...")
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    if args.device in ["cuda", "cpu"]:
        model = model.to(args.device)
else:
    # ollama / openai
    model = None
    tokenizer = None

# -----------------------------
# Load Job Description
# -----------------------------
job_list = [
    "construction_laborers",
    "software_developers",
    "accountants_auditors",
    "registered_nurses",
    "elementary_middle_school_teachers",
]

for job in job_list:
    if job in FOLDER_DIR:
        break

job_path = os.path.join(JOB_INFO_DIR, f"{job}_job.txt")
with open(job_path, "r") as f:
    job_text = f.read()

# -----------------------------
# Scoring Pipeline
# -----------------------------
rows = [["resume_filename", "score", "id", "job_info_version"]]

print("Starting LLM-based scoring pipeline...\n")
print(f"Job: {job}")
print(f"Folder: {FOLDER_DIR}")
print(f"LLM provider - name: {args.provider} - {args.llm_model_name}")

fnames = os.listdir(FOLDER_DIR)
print(f"======Total resume files found: {len(fnames)}=========")
if args.resume_file_key:
    resume_file_keys = args.resume_file_key.split(",") if args.resume_file_key else []
    print(f"INCLUDE keys: {resume_file_keys}")
    # fnames = [f for f in fnames if args.resume_file_key in f]
    # Keep files that contain all keys in their filenames
    for key in resume_file_keys:
        fnames = [f for f in fnames if key in f]
if args.resume_file_ignore_key:
    resume_file_ignore_keys = args.resume_file_ignore_key.split(",") if args.resume_file_ignore_key else []
    print(f"IGNORE keys: {resume_file_ignore_keys}")
    for key in resume_file_ignore_keys:
        fnames = [f for f in fnames if key not in f]
    # fnames = [f for f in fnames if args.resume_file_ignore_key not in f]
# fnames = fnames[:10]
print(f"=======Resume files after filtering: {len(fnames)}=========")



OUTPUT_CSV = os.path.join(args.output_dir, OUTPUT_CSV)
processed = set()
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_filename", "score", "id", "job_info_version"])
else:
    df = pd.read_csv(OUTPUT_CSV)
    processed = set(df['id'].astype(str))

# Process resumes in batches
batch_size = args.batch_size if args.provider == "hf" else 1

for batch_start in tqdm(range(0, len(fnames), batch_size), desc="Processing batches"):
    batch_fnames = fnames[batch_start:batch_start + batch_size]
    batch_resume_texts = []
    batch_resume_ids = []
    batch_resume_paths = []

    # Read all resumes in the batch
    for fname in batch_fnames:
        resume_id = fname.replace("resume_", "").replace(".txt", "")
        if resume_id in processed:
            print(f"Skipping already processed resume: {fname}")
            continue
        resume_path = os.path.join(FOLDER_DIR, fname)

        with open(resume_path, "r") as f:
            resume_text = f.read()

        batch_resume_texts.append(resume_text)
        batch_resume_ids.append(resume_id)
        batch_resume_paths.append(os.path.join(FOLDER_DIR, fname))
    if len(batch_resume_texts) == 0:
        continue

    if args.test_mode:
        scores = [float(torch.randint(0, 101, (1,)).item()) for _ in batch_fnames]
    elif args.provider == "hf" and len(batch_fnames) > 1:
        # Batch processing for HF
        prompts = [build_scoring_prompt_hf(job_text, resume_text) for resume_text in batch_resume_texts]
        gen_texts = call_hf_batch(prompts, args=args, model=model, tokenizer=tokenizer)
        scores = []
        for gen_text in gen_texts:
            try:
                score = parse_score_from_text(gen_text)
                scores.append(float(score))
                # print(f"Generated text: {gen_text} | Score: {score}")
            except ValueError as e:
                print(f"Warning: {e}, using default score 0")
                scores.append(0.0)
    else:
        # Single processing (fallback or other providers)
        scores = []
        for resume_text in batch_resume_texts:
            score = float(
                llm_score_one(
                    model=model,
                    tokenizer=tokenizer,
                    job_text=job_text,
                    resume_text=resume_text,
                    max_new_tokens=args.max_new_tokens,
                )
            )
            scores.append(score)
    # Add all results from this batch
    # for i in range(len(batch_fnames)):
        # rows.append([batch_resume_paths[i], scores[i], batch_resume_ids[i], args.job_info_version])
    row = [[batch_resume_paths[i], scores[i], batch_resume_ids[i], args.job_info_version] for i in range(len(batch_fnames))]
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(row)

print(f"Number of resumes scored: {len(rows) - 1}")


print(f"Results saved to {OUTPUT_CSV}")
print("Done.")