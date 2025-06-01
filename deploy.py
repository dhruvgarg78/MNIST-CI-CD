import os
import shutil
from glob import glob

latest_model = sorted(glob("model_*.pt"))[-1]
shutil.copy(latest_model, "model_latest.pt")
print(f"Copied {latest_model} to model_latest.pt")