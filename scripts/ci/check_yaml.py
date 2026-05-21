"""
CI helper: check_yaml.py
Called from Stage 1 of the CI pipeline.
Checks that all YAML files in the repo parse without errors.
"""
import glob
import sys
import yaml

files = glob.glob("**/*.yml", recursive=True) + glob.glob("**/*.yaml", recursive=True)
errors = []

for f in files:
    try:
        with open(f, encoding="utf-8", errors="replace") as fh:
            yaml.safe_load(fh)
        print(f"  OK  {f}")
    except Exception as e:
        errors.append(f"  FAIL {f}: {e}")

if errors:
    print("\nYAML errors found:")
    for e in errors:
        print(e)
    sys.exit(1)

print(f"\nAll {len(files)} YAML files valid")
