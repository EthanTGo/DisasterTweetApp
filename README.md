# DisasterTweetApp
This is a disaster tweet classification pipeline project that utilizes NLP. The webapp is built on top of Flask

# How to run

After download/cloning the project, you need to navigate to the app directory and run the following command:
- `pip install`

Then navigate up back to the main folder and run this command
- `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Finally, cd back to the app folder and run
- `python run.py`

The first command installs the requirements.txt file which includes the require libraries.

The second command creates the pkl file used to train the model. Since the pkl file would be at least 1 GB it would be too large to be stored in Github and has to be run manually.

The final command runs the web app. By default, it should be set to port 3001. This can be changed in the run.py file under the function main.

# Updating the database or model

This web app uses a pre-made database and pre-trained model to run the calculations in the backend. If you were to modify the database or model, you may do so.
The model is a Pipeline that processes text data and creates a Random Tree classifier. It is located at the models folder. Similarly, the database is located under 
the dataset folder. You would need to run the following commands: 

- `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

The first command creates a new database name DisasterResponse.db under the data folder. The name is important and if you choose to create one with a different name, you would need to update the file using the database in models/train_classifiers.py. 

The second command creates a new model based of a pipeline. The default model consists of a basic pipeline that uses CountVectorizer, TfidfTransformer and RandomForest. When naming the file it's important to keep the same name or you will need to change the name of the fill in the front end. 

# File structure

The file is structured into the following hierarchies

```
data
└── categories.csv
└── disaster_messages.csv
└── disaster_process_data.py
└── DisasterResponse.db
notebook
└── ETL Pipeline.ipynb
└── ML Pipeline.ipyn
model
└── train_classifier.py
└── classifier.pkl
app
└── run.py
└── templates
    └── go.html
    └── master.html
```
Here, the notebook contains the development environment where the function of the codes with in train_classifier.py and process_data.py is first designed. Looking at the notebook will give an easier and intuitive understanding on hoe the dataset was extracted and transformed. It also shows how the model was designed.

The app folder contains the Flask file that runs the web applications

The model contains the train_classifier.py which is used to create classifier.pkl. This is the pickle file used for the web application

The dataset contains the original training/testing datasets and database produced through the process_data.py

