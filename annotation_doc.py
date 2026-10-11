import pandas as pd
import os
import zipfile

# Load the samples
italian_sample = pd.read_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_sample_for_labeling.csv")
hungarian_sample = pd.read_csv("/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_sample_for_labeling.csv")

def create_inception_project(sample_df, language_name, output_path):

    # Create temporary directory structure
    project_dir = f"temp_{language_name}_project"
    source_dir = os.path.join(project_dir, "source")
    annotation_dir = os.path.join(project_dir, "annotation")

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    # Export texts to source directory
    for idx, row in sample_df.iterrows():
        filename = f"{language_name}_{idx}.txt"
        filepath = os.path.join(source_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            text = str(row['text']) if pd.notna(row['text']) else ""
            f.write(text)

    # Create the exportedproject.json file
    project_json = {
        "name": f"{language_name}_ner_project",
        "description": f"NER annotation project for {language_name} historical texts (1850-1899)",
        "version": 1
    }

    import json
    with open(os.path.join(project_dir, "exportedproject.json"), 'w', encoding='utf-8') as f:
        json.dump(project_json, f, indent=2)

    # Create ZIP file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(project_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, project_dir)
                zipf.write(file_path, arcname)

    # Clean up temporary directory
    import shutil
    shutil.rmtree(project_dir)

    print(f"Created {output_path}")

# Create INCEPTION project ZIPs
create_inception_project(
    italian_sample,
    "italian",
    "/Users/bauerkitti/Desktop/Uni/masterarbeit/italian_inception_project.zip"
)

create_inception_project(
    hungarian_sample,
    "hungarian",
    "/Users/bauerkitti/Desktop/Uni/masterarbeit/hungarian_inception_project.zip"
)