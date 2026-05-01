import json
import os
from collections import OrderedDict
from pathlib import Path

from lib.json_validation import LocaleMap

# Define the standard file name
locale_dir = Path(__file__).resolve().parent / "locale"
standard_file = locale_dir / "zh_CN.json"

# Find all JSON files in the directory
languages = [
    locale_dir / f
    for f in os.listdir(locale_dir)
    if f.endswith(".json") and locale_dir / f != standard_file
]

# Load the standard file
with open(standard_file, "r", encoding="utf-8") as f:
    standard_data = LocaleMap.model_validate(json.load(f)).root

# Loop through each language file
for lang_file in languages:
    # Load the language file
    with open(lang_file, "r", encoding="utf-8") as f:
        lang_data = LocaleMap.model_validate(json.load(f)).root

    # Find the difference between the language file and the standard file
    diff = set(standard_data.keys()) - set(lang_data.keys())

    miss = set(lang_data.keys()) - set(standard_data.keys())

    # Add any missing keys to the language file
    for key in diff:
        lang_data[key] = key

    # Del any extra keys to the language file
    for key in miss:
        del lang_data[key]

    # Sort the keys of the language file to match the order of the standard file
    lang_data = OrderedDict(
        sorted(lang_data.items(), key=lambda x: list(standard_data.keys()).index(x[0]))
    )

    # Save the updated language file
    with open(lang_file, "w", encoding="utf-8") as f:
        json.dump(lang_data, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")
