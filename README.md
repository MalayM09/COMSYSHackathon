# COMSYSHackathon

FACECOM Challenge Submission

Malay Mishra
f20220116@pilani.bits-pilani.ac.in
+91 6351226206

Project Overview
This repository contains the dual-head learning solution for the FACECOM challenge, capable of the Task A, i.e Gender classification from face images and Task B, i.e. Face verifiation(identity matching) using ArcFace loss.
This model is based on a ConvNeXt backbone with dedicated heads for each task, trained and validated on the provided dataset.

Directory Structre
COMSYSHackathon/
├── README.md
├── requirements.txt
├── src/
│   ├── model_architecture.py
│   ├── config.py
│   ├── data_loading.py
│   ├── task_a_inference.py
│   ├── task_b_inference.py
├── models/
│   ├── best_gender.pth
│   ├── best_arcface.pth
│   ├── best_joint.pth
├── results/
│   ├── results_task_a.csv
│   ├── results_task_b.csv
│   └── screenshots/
|.       |_____SS01
|        |_____SS02
|   |__Technical Summary
|        |_____COMSYSReport.pdf
|        |_____ModelArchitecture.jpeg
├── dataset/
│   ├── Task_A/
│   │   └── val/
│   │       ├── male/
│   │       └── female/
│   └── Task_B/
│       └── val/
│           ├── 00001/
│           │   ├── *.jpg
│           │   └── distortion/
│           │       └── *.jpg
│           └── ...

This guide explains how to use this repository to run model inference on your own validation dataset and generate output CSVs for evaluation.
Step 1: Clone the Repository
bash
git clone https://github.com/<YourUsername>/COMSYSHackathon.git
cd COMSYSHackathon
Step 2: Prepare the Validation Dataset
Organize your validation data as follows (replace dataset/ with your actual data folder if needed):
text
dataset/
├── Task_A/
│   └── val/
│       ├── male/
│       │   └── *.jpg
│       └── female/
│           └── *.jpg
└── Task_B/
    └── val/
        ├── 00001/
        │   ├── *.jpg
        │   └── distortion/
        │       └── *.jpg
        └── ...
Task A: Place male and female images in their respective folders.
Task B: Each identity should have its own folder, with images and (optionally) a distortion/ subfolder.
Step 3: Install Dependencies
Make sure you have Python 3.8+ and pip installed.
Then, run:
bash
pip install -r requirements.txt
Step 4: Run Inference for Task A (Gender Classification)
bash
python -m src.task_a_inference \
  --weights models/best_gender.pth \
  --data_dir dataset/Task_A/val \
  --output_csv results/results_task_a.csv
You can also use models/best_joint.pth as --weights if you want to test the joint model.
Step 5: Run Inference for Task B (Face Verification)
bash
python -m src.task_b_inference \
  --weights models/best_arcface.pth \
  --data_dir dataset/Task_B/val \
  --output_csv results/results_task_b.csv
You can also use models/best_joint.pth as --weights if you want to test the joint model.
Step 6: Find the Output
The output CSVs will be saved in the results/ directory:
results/results_task_a.csv
results/results_task_b.csv
Each CSV will contain predictions and true labels for your validation data.

Output
Results are saved as csv files in results/directory
Each csv column contains:
For Task A: image, predicted_label, true_label
For Task B: image1, image2, predicted_label, true_label, similarity

Troubleshoot and FAQ:
ModuleNotFoundError: Make sure all dependencies are installed (pip install -r requirements.txt)
CUDA/Device errors: Check PyTorch installation and device availability
Directory not found: Ensure that the --data_dir matches the actual folder structure

This guide explains how to use this repository to run model inference on your own validation dataset and generate output CSVs for evaluation.
Step 1: Clone the Repository
bash
git clone https://github.com/<YourUsername>/COMSYSHackathon.git
cd COMSYSHackathon
Step 2: Prepare the Validation Dataset
Organize your validation data as follows (replace dataset/ with your actual data folder if needed):
text
dataset/
├── Task_A/
│   └── val/
│       ├── male/
│       │   └── *.jpg
│       └── female/
│           └── *.jpg
└── Task_B/
    └── val/
        ├── 00001/
        │   ├── *.jpg
        │   └── distortion/
        │       └── *.jpg
        └── ...
Task A: Place male and female images in their respective folders.
Task B: Each identity should have its own folder, with images and (optionally) a distortion/ subfolder.
Step 3: Install Dependencies
Make sure you have Python 3.8+ and pip installed.
Then, run:
bash
pip install -r requirements.txt
Step 4: Run Inference for Task A (Gender Classification)
bash
python -m src.task_a_inference \
  --weights models/best_gender.pth \
  --data_dir dataset/Task_A/val \
  --output_csv results/results_task_a.csv
You can also use models/best_joint.pth as --weights if you want to test the joint model.
Step 5: Run Inference for Task B (Face Verification)
bash
python -m src.task_b_inference \
  --weights models/best_arcface.pth \
  --data_dir dataset/Task_B/val \
  --output_csv results/results_task_b.csv
You can also use models/best_joint.pth as --weights if you want to test the joint model.
Step 6: Find the Output
The output CSVs will be saved in the results/ directory:
results/results_task_a.csv
results/results_task_b.csv
Each CSV will contain predictions and true labels for your validation data.

Contact:
For any issues or queries, please contact:
[f20220116@pilani.bits-pilani.ac.in]
Thank you for evaluating my submission!
