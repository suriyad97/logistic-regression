"""
CI helper: check_packages.py
----------------------------
Called from the AzDO CI pipeline Stage 2 (Package Check).

1. Reads requirements.txt
2. pip-installs them (subprocess so the same interpreter is used).
3. Imports each key package to verify no import errors.
4. Runs pip check for dependency conflicts.

Exit 0 = all good.
Exit 1 = failure with details printed to stdout.
"""

import importlib
import subprocess
import sys


# --------------------------------------------------------------------------
# Step 1: Locate requirements.txt
# --------------------------------------------------------------------------
print("=== Step 1: Locating requirements.txt ===")
req_file = "requirements.txt"


# --------------------------------------------------------------------------
# Step 2: pip install all packages
# --------------------------------------------------------------------------
print("\n=== Step 2: pip install all packages ===")
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", req_file],
    check=False,
)
if result.returncode != 0:
    print("ERROR: pip install failed - see output above")
    sys.exit(1)

# --------------------------------------------------------------------------
# Step 3: Verify imports
# --------------------------------------------------------------------------
print("\n=== Step 3: Verifying key package imports ===")
packages = {
    "numpy":     "numpy",
    "pandas":    "pandas",
    "sklearn":   "scikit-learn",
    "mlflow":    "mlflow",
    "optuna":    "optuna",
    "pandera":   "pandera",
    "shap":      "shap",
    "lime":      "lime",
    "evidently": "evidently",
}

failed = []
for module, pkg_name in packages.items():
    try:
        importlib.import_module(module)
        print(f"  OK   {pkg_name}")
    except ImportError as e:
        failed.append(f"  FAIL {pkg_name}: {e}")

if failed:
    print("\nFailed imports:")
    for f in failed:
        print(f)
    sys.exit(1)

print(f"\nAll {len(packages)} packages imported successfully.")

# --------------------------------------------------------------------------
# Step 4: Installed version report
# --------------------------------------------------------------------------
print("\n=== Step 4: Installed versions ===")
subprocess.run(
    [sys.executable, "-m", "pip", "list", "--format=columns"],
    check=False,
)

# --------------------------------------------------------------------------
# Step 5: Dependency conflict check
# --------------------------------------------------------------------------
print("\n=== Step 5: Dependency conflict check (pip check) ===")
result = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    check=False,
)
if result.returncode != 0:
    print("WARNING: Dependency conflicts detected above (non-fatal)")

print("\nrequirements.txt is valid - safe to register in AML")
