# DisasterTweetApp
This is a disaster tweet classification pipeline project that utilizes NLP. The webapp is built on top of Flask

# How to run

After download/cloning the project, you only need to navigate to the app directory and run the following command:
- `python run.py`

By default, it should be set to port 3001. This can be changed in the run.py file under the function main.

# Updating the database or model

This web app uses a pre-made database and pre-trained model to run the calculations in the backend. If you were to modify the database or model, you may do so.
The model is a Pipeline that processes text data and creates a Random Tree classifier. It is located at the models folder. Similarly, the database is located under 
the dataset folder. You would need to run the following commands: 

- `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

