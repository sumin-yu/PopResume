# PopResume

**Causal Fairness Evaluation of LLM/VLM Resume Screeners with a Population-Representative Dataset**
(EMNLP 2026)

Reproduction code. The pipeline (1) generates a population-representative resume dataset from U.S.
population statistics, (2) scores resumes with LLM/VLM screeners, and (3) estimates path-specific
causal effects (TE / NDE / NIE / BIE / RIE).

- 🌐 Project page: https://sumin-yu.github.io/PopResume
- 📄 Paper: [arXiv:2603.22714](https://arxiv.org/abs/2603.22714) · EMNLP 2026
- 🤗 Dataset: [sumin-yu/PopResume](https://huggingface.co/datasets/sumin-yu/PopResume)

## Pipeline

```
1_data_generation/     (1)(2)(3)  population statistics -> structured profiles -> resumes
  sources/                                             pipeline inputs — see sources/README.md
  01-1_structured_profile_dataset_generation.ipynb     ACS PUMS sampling (X, Z, B, R)
  01-2_work_experience.ipynb                           PSID work-experience imputation
  01-3_name_sampler.ipynb                              SSA / Census name sampling + grouping labels
  01-4_var_grouping.ipynb                              collapse edu_level / state, attach name proxies
  02-2_resume_content_generation_deterministic.py      rule-based resume content
  03_real_resume_generation.py                         render resumes (text)
  03-2_real_resume_generation_delete_skills.py         drop the skills section (paper setup)
  resume_pdf_generation.py                             text -> PDF
  profile_image.ipynb                                  inspect the generated profile photos

2_scoring/             (4)        LLM/VLM resume scoring
  run_scoring.sh                                       batch runner (LLM + VLM, all providers)
  score_resumes_llm_score.py                           text-resume scoring (LLM)
  score_resumes_vlm_score.py                           image-resume scoring (VLM)
  pdf_to_jpg.py / pdf_to_jpg.sh                        PDF -> JPG for VLM input
  job_info_v2/                                         job descriptions per occupation

3_causal_estimation/   (5)        path-specific effect estimation (DML-UCA)
  causal_effect.py                                     TE / NDE / NIE / BIE / RIE
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env         # fill in OPENAI_API_KEY / GEMINI_API_KEY / DATA_ROOT
set -a; source .env; set +a  # nothing here auto-loads .env
```

A provider's key is only needed when you use that provider; local HF models need none.

## Running it

**Score resumes.** `2_scoring/run_scoring.sh` is the only entry point and is fixed to the paper's
setup (five occupations, resumes without a skills section, job descriptions `v2`):

```bash
./run_scoring.sh                                          # Llama + Mistral (local HF)
MODE=llm PROVIDER=openai MODELS="gpt-4o-mini"   ./run_scoring.sh
MODE=llm PROVIDER=gemini MODELS="gemini-2.5-flash-lite" ./run_scoring.sh
MODE=vlm MODELS="qwen2.5-vl-7b-instruct internvl2-8b"   ./run_scoring.sh
```

**Estimate effects.** The two grouping flags are required — they select the grouped `W` variables the
paper uses; without them the script asks for columns the dataset does not carry.

```bash
python causal_effect.py --score_path <scores>.csv \
    --X sex --x0 Female --resume_format no_demographics --with_skill 0 \
    --bootstraps 500 --clip 0.01 --with_state_region --with_new_name_group
```

`--resume_format with_demographics_img` for resume images with a profile photo;
`--X race --x0 White` for the race analysis.

**Read the output.** The result CSV keeps the upstream `VDE` / `WDE` names — `VDE` is the paper's
**BIE** (business-necessity path) and `WDE` is **RIE** (redlining path). Each is computed once per
decomposition order; the paper reports the self-normalized, order-symmetric average:

```python
BIE = (df["Estimated_VDE_sn1"] + df["Estimated_VDE_sn2"]) / 2
RIE = (df["Estimated_WDE_sn1"] + df["Estimated_WDE_sn2"]) / 2
```

`Estimated_TE_sn`, `Estimated_NDE_sn`, `Estimated_NIE_sn` are the other three reported effects.

## Notes

- **Paths.** Shell scripts read `${DATA_ROOT}`; Python files and notebooks use the placeholder
  `/path/to/CausalFair` — replace it with your local path.
- **Data.** Code only. The resumes and attributes are the
  [PopResume dataset](https://huggingface.co/datasets/sumin-yu/PopResume), rendered without a skills
  section to match the paper. Scores and effect estimates are experiment outputs and are not included.
- **Inputs.** The content pools and Census tables ship in `1_data_generation/sources/`; the IPUMS,
  PSID, and SSA inputs do not. `1_data_generation/sources/README.md` says where to get each one.
- **Profile photos** are generated outside this repo with
  [Face-MoGLE](https://arxiv.org/abs/2508.16094), prompted *"a professional headshot portrait photo of
  a {age} year old {race} {gender}, white background, formal attire, front-facing"* (paper §4).
  `profile_image.ipynb` only inspects them.
- **Scope.** The figure-rendering notebooks and exploratory work that did not reach the paper are not
  included — ask the authors if you need them.
- **Models.** LLMs: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, GPT-4o-mini, Gemini-2.5-Flash-Lite.
  VLMs: Qwen2.5-VL-7B-Instruct, InternVL2-8B, GPT-4o, Gemini-2.5-Flash-Lite.

## Citation

```bibtex
@inproceedings{yu2026popresume,
  title                                                = {PopResume: Causal Fairness Evaluation of LLM/VLM Resume Screeners
               with Population-Representative Dataset},
  author                                               = {Yu, Sumin and Park, Juhyeon and Moon, Taesup},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in
               Natural Language Processing (EMNLP)},
  year                                                 = {2026}
}
```
