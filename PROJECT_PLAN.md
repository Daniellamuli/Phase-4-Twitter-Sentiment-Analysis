# Project Plan: Twitter Sentiment Analysis

## 📅 4-Day Sprint (Start Date: 17/04/2026)

### Day 1: Setup & Data Understanding
| Person | Primary File | Deliverable |
|--------|--------------|-------------|
| Daniella | `src/constants.py` | Shared variables, paths, labels |
| Eve | `src/load_data.py` | Load CSV, basic info, filtering |
| Verah | `src/clean_text.py` | Text cleaning pipeline |
| Naomi | `src/visualize.py` | EDA plots, word clouds |

**End of Day Deliverable**: Cleaned data + visualizations

---

### Day 2: Feature Engineering
| Person | Primary File | Deliverable |
|--------|--------------|-------------|
| Daniella | `src/preprocess.py` | Full preprocessing pipeline |
| Eve | `src/vectorize.py` | TF-IDF, CountVectorizer |
| Verah | `src/features.py` | Length, mentions, hashtags |
| Naomi | `src/split_data.py` | Train/validation/test split |

**End of Day Deliverable**: Feature matrices ready for modeling

---

### Day 3: Model Training & Evaluation
| Person | Primary File | Deliverable |
|--------|--------------|-------------|
| Daniella | `src/train_binary.py` | Logistic Regression, Naive Bayes |
| Eve | `src/train_multiclass.py` | Random Forest, SVM |
| Verah | `src/evaluate.py` | Metrics, confusion matrices |
| Naomi | `src/compare_models.py` | Model comparison, best model selection |

**End of Day Deliverable**: Trained models + performance metrics

---

### Day 4: Final Assembly & Presentation
| Person | Primary File | Deliverable |
|--------|--------------|-------------|
| Daniella | `src/pipeline.py` | End-to-end pipeline |
| Eve | `notebooks/final_presentation.ipynb` | Demo notebook |
| Verah | `docs/README.md` | Complete documentation |
| Naomi | `docs/presentation_slides.md` | Final presentation slides |

**End of Day Deliverable**: Complete project + presentation

---

## 📁 File Structure (What We'll Build)
Phase-4-Twitter-Sentiment-Analysis/  
│  
├── src/ # 15 Python files (one per person per day)  
│ ├── constants.py # Day 1 - Daniella  
│ ├── load\_data.py # Day 1 - Eve  
│ ├── clean\_text.py # Day 1 - Verah  
│ ├── visualize.py # Day 1 - Naomi  
│ ├── preprocess.py # Day 2 - Daniella  
│ ├── vectorize.py # Day 2 - Eve  
│ ├── features.py # Day 2 - Verah  
│ ├── split\_data.py # Day 2 - Naomi  
│ ├── train\_binary.py # Day 3 - Daniella  
│ ├── train\_multiclass.py # Day 3 - Eve  
│ ├── evaluate.py # Day 3 - Verah  
│ ├── compare\_models.py # Day 3 - Naomi  
│ ├── pipeline.py # Day 4 - Daniella  
│ └── init.py  
│  
├── notebooks/  
│ └── final\_presentation.ipynb # Day 4 - Eve (ONLY ONE!)  
│  
├── docs/  
│ ├── README.md # Day 4 - Verah  
│ └── presentation\_slides.md # Day 4 - Naomi  
│  
├── data/  
│ └── judge-1377884607\_tweet\_product\_company.csv  
│  
├── requirements.txt  
├── .gitignore  
└── PROJECT\_PLAN.md # THIS FILE

## Team Workflow Guide: How We Work Together Without Merge Conflicts

### The Golden Rule 

> **If your code needs to be used by others → put it in `src/` as a `.py` file**
> 
> **If your code is just for showing results → put it in a notebook (but only ONE person creates it)**

---

## TO NOTE:

- Everyone IMPORTS from `constants.py`
    - **Result**: When merged, ALL code uses the SAME variable names!
- We merge to `.py` files, NOT `.ipynb` notebooks!
- Only ONE notebook exists in the repository and it ONLY imports from `src/` (no code inside the notebook itself).
    - The notebook is just a SHOWCASE - it contains almost no code. All the real work is in `src/`.
- To ensure Merge Strategy (With No Conflicts!):
    - **The #1 Rule**: Each person owns their assigned file. No two people touch the same file on the same day.

---

## Quick Reference: 

| Question | Answer |
|----------|--------|
| Where does code go? | `src/` as `.py` files |
| Where do results go? | `notebooks/` as ONE `.ipynb` file |
| How many notebooks? | Only ONE (`final_notebook.ipynb`) |
| How do we share variables? | `src/constants.py` - everyone imports from it |
| How do we avoid merge conflicts? | Each person owns different files, no sharing |
| What's the daily merge process? | Branch → Work → Pull Request → Merge → Pull |
