"""
To run the code on gpu, simply pass the GPU ID as the argument, for example "python main_gpu.py 0".
The models to run should be in data/models.csv with the columns as shown below in args comment.

For example a csv might be:
0,1,regression,relu,False,False,before,16,0.0001,False

where 0 is the GPU, 1 is the model iteration and so on.
"""


import subprocess
import csv
from sys import argv

GPU = argv[1]


def print_info(id):

    print(f"""
================================
        Running Model {id}
================================
        """)

# Args:
#    1 - GPU ID
#    2 - Model Iteration
#    3 - Model Type
#    4 - Activation Function
#    5 - Dropout true or false
#    6 - Batchnorm true or false
#    7 - Batchnorm order
#    8 - batch size
#    9 - learning rate
#    10 - sparse date true or false


models = []
with open('data/models.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == GPU:
            models.append(row)

for model in models:
    command = ["python", "code/model.py"] + model
    print_info(model[1])
    subprocess.run(command)

    print("Deleting final model")
    if model[2] == "regression":
        m_type = "reg"
    else:
        m_type = "clas"
    #subprocess.run(["rm", f"data/models/fmodel_{model[1]}_{m_type}.h5"])
