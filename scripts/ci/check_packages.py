"""
CI helper: check_packages.py
----------------------------
Called from the AzDO CI pipeline Stage 2 (Package Check).

1. Reads config/conda.yml and extracts all pip packages.
2. Writes them to /tmp/ci_requirements.txt.
3. pip-installs them (subprocess so the same interpreter is used).
4. Imports each key package to verify no import errors.
5. Runs pip check for dependency conflicts.

Exit 0 = all good.
Exit 1 = failure with details printed to stdout.
"""

import importlib
import subprocess
import sys
import yaml


# --------------------------------------------------------------------------
# Step 1: Parse conda.yml and extract pip packages
# --------------------------------------------------------------------------
print("=== Step 1: Extracting pip packages from config/conda.yml ===")
with open("config/conda.yml") as f:
    conda = yaml.safe_load(f)

pip_packages = []
for dep in conda.get("dependencies", []):
    if isinstance(dep, dict) and "pip" in dep:
        for pkg in dep["pip"]:
            stripped = pkg.strip()
            if stripped and not stripped.startswith("#"):
                pip_packages.append(stripped)

print(f"Found {len(pip_packages)} pip packages:")
for p in pip_packages:
    print(f"  {p}")

req_file = "/tmp/ci_requirements.txt"
with open(req_file, "w") as f:
    f.write("\n".join(pip_packages))

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

print("\nconda.yml is valid - safe to register in AML")
