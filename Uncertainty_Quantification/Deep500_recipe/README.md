#Deep500 Recipe for Uncertainty Quantification

[Deep500](https://www.deep500.org)  is A modular benchmarking infrastructure for 
high-performance deep learning — from the single operator 
to distributed training. In this directory, we  reproduce the result of
*uncertainty quantification* task using deep500 recipe.


## Usage

To run the code, you need to install Deep500 using the following line:

```
pip install deep500
```


Now, you can train the model using the following:

```
python recipe_UQ.py  --data_directory YOUR/DATA/DIRECTORY   --std_folder YOUR/STD&MEAN_MAP/FOLDER  --batch_size 2 --epochs 6  --fix_split True
```
