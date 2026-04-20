# English-Tamil Translation Model (Unsloth)
## Lightweight Fine-Tuning of Llama 3.1 Using Unsloth + LoRA (4-bit)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![GPU](https://img.shields.io/badge/GPU-Required-important)



This project implements a **Tamil-to-English neural machine translation model** using a **quantized Llama 3.1 8B model**, fine‑tuned efficiently with **Unsloth**, **LoRA**, and **4-bit QLoRA optimization**.  
It is designed for **fast training on consumer GPUs (like T4)** while still achieving **high translation quality**, evaluated using BLEU and semantic similarity metrics.

---

## 🚀 Key Features

### **1. Efficient Fine-Tuning with Unsloth**
- Uses `unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- 4-bit quantization → **low VRAM usage**
- LoRA applied on key transformer layers for efficient adaptation

### **2. Dataset Pipeline**
The project builds a dataset using two sources:

#### ✔ **Real Data (from OPUS-100)**  
- Fetches parallel Tamil–English sentences  
- Filters by length and Tamil script detection  
- Used until 50% of target dataset size is filled

#### ✔ **Synthetic Phrases**  
- Adds common conversational phrases  
- Ensures dataset reaches target size (default: 2000 samples)

All samples are converted to:
```
Translate from Tamil to English.

Tamil: <text>
English: <text>
```

---

## 🧠 Training Process

### **Hyperparameters**
- **Model**: Llama 3.1 8B (4-bit)
- **Batch Size**: 2  
- **Effective Batch Size**: 16 (via gradient accumulation)
- **Learning Rate**: 1.5e‑4  
- **Max Steps**: 600  
- **Warmup**: 100  
- **LoRA Rank**: 64  

### **Optimizer**
- AdamW (8-bit)
- Linear LR scheduler

### **Callbacks**
- Early stopping (patience: 3)
- Model checkpointing

### **Trainer**
Uses `trl.SFTTrainer` for supervised fine-tuning.

---

## 📊 Evaluation Pipeline

Evaluation includes:

### **1. BLEU (Corpus-Level)**
- BLEU-1 to BLEU-4  
- Brevity Penalty  
- Final BLEU score

### **2. Semantic Similarity (Sentence-BERT)**
- Mean cosine similarity  
- Median similarity  
- Standard deviation  
- Min/Max similarity  

### **3. Translation Quality Buckets**
- Excellent (similarity > 0.7)
- Good (similarity > 0.5)
- Exact matches

### **4. Final Combined Score**
Weighted:
- 40% BLEU  
- 60% semantic similarity  

---

## 📦 Outputs Generated
The project saves:

### **1. Fine-Tuned Model**
```
./tamil-translation-model/
```

### **2. Evaluation Summary**
```
./evaluation_results.txt
```

---

## 📝 Example Translation
Input:
```
வணக்கம், நான் தமிழ் கற்கிறேன்
```
Output:
```
Hello, I am learning Tamil.
```

---

## 🛠 Project Structure
```
📂 tamil-translation/
│── train.ipynb                # Main notebook (training + evaluation)
│── tamil-translation-model/   # Saved LoRA fine-tuned model
│── evaluation_results.txt     # Detailed evaluation summary
│── README.md                  # (this file)
```

---

## 👥 Team Members
- **JEEVAKAMAL K R** – CB.AI.U4AID23115  
- **JEIESH J S** – CB.AI.U4AID23116  
- **SRI SOMESH S** – CB.AI.U4AID23141  
- **SAI CHAKRITH** – CB.AI.U4AID23143  
- **SURIYA DHARSAUN KG** – CB.AI.U4AID23144  

---

## 📘 How to Run

### **1. Install Dependencies**
```bash
pip install unsloth datasets transformers peft accelerate bitsandbytes
pip install sentence-transformers sacrebleu trl
```

### **2. Run Training Notebook**
Execute the notebook cell-by-cell.

### **3. Use the Model for Inference**
```python
translate("நான் பள்ளிக்கு செல்கிறேன்")
```

---

## 🎯 Summary
This project showcases an end-to-end **custom Tamil-to-English translation system**, optimized for **speed, efficiency, and solid translation accuracy**, using modern lightweight fine‑tuning techniques.
