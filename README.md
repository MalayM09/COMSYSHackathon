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

Setup Instructions
1. Clone the repository: 
git clone https://github.com/YourUsername/COMSYSHackathon.git
cd COMSYSHackathon
2. Install Dependancies
pip install -r requirements.txt
3. How to run Inference:
3a. Task A Gender Classification:
python -m src.task_a_inference \
  --weights models/best_gender.pth \
  --data_dir dataset/Task_A/val \
  --output_csv results/results_task_a.csv
3b. Task B Face Verification:
python -m src.task_b_inference \
  --weights models/best_arcface.pth \
  --data_dir dataset/Task_B/val \
  --output_csv results/results_task_b.csv
To use the joint model: replace best_arcface.pth or best_gender.pth with best_joint.pth.

Output
Results are saved as csv files in results/directory
Each csv column contains:
For Task A: image, predicted_label, true_label
For Task B: image1, image2, predicted_label, true_label, similarity

Troubleshoot and FAQ:
ModuleNotFoundError: Make sure all dependencies are installed (pip install -r requirements.txt)
CUDA/Device errors: Check PyTorch installation and device availability
Directory not found: Ensure that the --data_dir matches the actual folder structure

Contact:
For any issues or queries, please contact:
[f20220116@pilani.bits-pilani.ac.in]
Thank you for evaluating my submission!
