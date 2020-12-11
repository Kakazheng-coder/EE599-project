# EE599-project

The main code file is "aws1.py" which needs modules from "SGLA.py" and "SENET.py"
Thus, to run the entire model "aws1.py" needs to be run.
"aws1.py" and "SENET.py" together make up the global feature extraction part of the model
"SGLA.py" has the code for local feature extract.
Together all three files form the complete SGLA network.

# Dataset:

Link: https://www.kaggle.com/c/ifood-2019-fgvc6/data

Use the following command to extract the dataset from kaggle:

$ pip install kaggle

copy the kaggle.json file to the home directory
(Downloading kaggle.json: https://github.com/Kaggle/kaggle-api)

$ kaggle (This shd give an error along with the path(p) to move the kaggle.json in)

$ mv kaggle.json p (p is tye path shown in the error of previous command)

$ kaggle competitions download -c ifood-2019-fgvc6
After this unzip all the files and run "aws1.py"
