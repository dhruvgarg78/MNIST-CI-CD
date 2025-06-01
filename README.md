# 🧪 MNIST CI/CD Pipeline

This project demonstrates a complete CI/CD pipeline for a simple convolutional neural network (CNN) trained on the MNIST dataset using PyTorch. It includes:

- ✅ Model training and saving
- ✅ Automatic deployment and testing via GitHub Actions
- ✅ Structural and performance validation
- ✅ Parameter budget enforcement (< 100,000)
- ✅ 10-class classification on 28x28 grayscale images

---

## 🚀 How It Works

1. **Train**: Runs `train.py`, trains a 3-layer CNN (2 conv, 1 FC), saves with timestamp.
2. **Deploy**: `deploy.py` renames the latest model to `model_latest.pt` for consistency.
3. **Test**: `test_model.py` checks:
   - Input compatibility (28x28 images)
   - Output classes = 10
   - Parameter count < 100,000
4. **Validate**: `validate_model.py` ensures accuracy > 80%.

---

## 🧪 CI/CD Pipeline: GitHub Actions

On every push to `main`, the following run:

- ✅ Install dependencies
- ✅ Train model for 1 epoch
- ✅ Run tests for structure and accuracy
- ✅ Print parameter breakdown
- ✅ Automatically fail if model breaks contract

See `.github/workflows/ci.yml` for configuration.

---

## 🖼️ Test Fail Screenshot (Old Model)

![Screenshot 2025-06-01 134609](https://github.com/user-attachments/assets/195b3a6f-a12f-4f42-87b5-02329586f296)
AssertionError: Too many parameters: 206922


---

## ✅ All Tests Passed Screenshot (Final Model)

![Screenshot 2025-06-01 140719](https://github.com/user-attachments/assets/4123089a-9bb8-4afb-abd9-3b7abf35c1a0)
Model passes shape and parameter tests.
Validation Accuracy: 98.12%


---

## 📂 Local Setup

```bash
pip install -r requirements.txt

python train.py
python deploy.py
python test_model.py
python validate_model.py
```
---

## 📁 Folder Structure

├── .github/workflows/ci.yml       # CI/CD config
├── train.py                       # Train model
├── deploy.py                      # Rename model for deployment
├── test_model.py                  # Structural tests
├── validate_model.py              # Accuracy check
├── model_utils.py                 # CNN architecture
├── requirements.txt               # pip dependencies
└── .gitignore                     # Ignore models, data, pycache

---

## 📊 Model Summary (Final)
Parameters: 55,338

Architecture: 2 Conv + 1 FC + Output

Input: 1x28x28 images

Output: 10 class logits

Accuracy: > 98% on test set
