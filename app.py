import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
from flask_mail import Mail

app = Flask(__name__)

app.config.update(
    MAIL_SERVER = 'smtp.gmail.com',
    MAIL_PORT = '465',
    MAIL_USE_SSL = True,
    MAIL_USERNAME = "patel567praveen@gmail.com",
    MAIL_PASSWORD = "1664Papa@58"
)
mail = Mail(app)


model = pickle.load(open('Cancer_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact', methods = ["GET", "POST"])
@cross_origin()
def contact():
    if request.method =="POST":
        firstname = request.form.get('firstname')
        surname = request.form.get('name')
        email = request.form.get('email')
        mobile = request.form.get('phone')
        message = request.form.get('message')

        print(firstname)

        mail.send_message('New message from ' + firstname, sender = email, recipients = ["patel567praveen@gmail.com"], body = message + "\n" + mobile)

        return render_template("contact.html")
    else:
        return render_template("contact.html")

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method =="POST":

        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
        
        features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension']
        
        df = pd.DataFrame(features_value, columns=features_name)
        # print(df)
        output = model.predict(df)[0]
        # print(output)
        # print("asbhs")
            
        if output == 'M':
            res_val = " breast cancer "
        else:
            res_val = "no breast cancer"
            

        return render_template('predict.html', prediction_text='Patient has {}'.format(res_val))

    else:
        return render_template('predict.html')

if __name__ == "__main__":
    app.run()
