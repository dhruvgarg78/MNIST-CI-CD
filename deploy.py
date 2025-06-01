import os
import shutil
from glob import glob

latest_model = sorted(glob("model_*.pt"))[-1]

# Only copy if it's not already model_latest.pt
if os.path.abspath(latest_model) != os.path.abspath("model_latest.pt"):
    shutil.copy(latest_model, "model_latest.pt")
    print(f"Copied {latest_model} to model_latest.pt")
else:
    print("Latest model is already named model_latest.pt. No copy needed.")
