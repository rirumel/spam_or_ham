from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

app = Flask(__name__)

model_file = 'F:/data_glacier_internship/week4/model/trained_model.pkl'
filename = 'F:/data_glacier_internship/week4/model/tfidf_vectorizer.pkl'
tfidf = TfidfVectorizer()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    with open(model_file, 'rb') as f:
        loaded_model = pickle.load(f)
    loaded_tfidf = pickle.load(open(filename, 'rb'))
    
    if request.method == 'POST':
        text = request.form['text']
        data = [text]
        data = np.asarray(data)
        data = loaded_tfidf.transform(data).toarray()

        # # Get the number of features in the original array
        # n = vect.shape[1]

        # # reshape the data
        # vect = np.pad(vect, ((0,0),(0,7051-n)), mode='constant', constant_values=0)
        my_prediction = loaded_model.predict(data)
        print(my_prediction[0])
        return render_template('result.html', prediction = my_prediction[0])



if __name__ == "__main__":
    app.run(debug=True)