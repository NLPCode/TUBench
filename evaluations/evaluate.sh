# This script evaluates the UCR and UVQA datasets using the evaluate.py script.
python evaluate.py --dataset UCR --filename ${prediction_filename}
python evaluate.py --dataset UVQA --filename ${prediction_filename}
