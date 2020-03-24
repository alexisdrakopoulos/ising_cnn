import subprocess
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation

# Read config file for paths
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")

def print_info(id):

    print(f"""
========================================
        Running Model Predictions {id}
========================================
        """)


def find_models():
	
    p = Path(config["Paths"]["models"])
    print("Looking for models")

    model_files = []
    for i in p.glob('*.h5'):
        print(i.name)
        model_files.append(i.name)

    return model_files


models = find_models()
print(models)

for model in models:
    print_info(model.split("_")[1])
    subprocess.run(["python", "predictions.py", model])