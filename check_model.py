import torch
import joblib

print("Load model AI...")

svm = joblib.load("model/svm_classifier_stroberi.pkl")

print("SVM berhasil diload")