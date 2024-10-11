from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import pandas as pd

# Load the trained model from file
with open('D:\Project\LoanAI\models\model_logistic_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data and convert it into the required numeric format
        gender = int(request.form['gender'])
        married = int(request.form['married'])
        dependents = int(request.form['dependents'])
        education = int(request.form['education'])
        self_employed = int(request.form['self_employed'])
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_amount_term = float(request.form['loan_amount_term'])
        credit_history = float(request.form['credit_history'])
        # Handle Property Area selection as One-Hot Encoding
        property_area = request.form['property_area']
        rural = 1 if property_area == '0' else 0
        semiurban = 1 if property_area == '1' else 0
        urban = 1 if property_area == '2' else 0

        # Create feature array for prediction
        features = np.array([[gender, married, dependents, education, self_employed,
                              applicant_income, coapplicant_income, loan_amount,
                              loan_amount_term, credit_history, rural, semiurban, urban]])

        # ตั้งชื่อคอลัมน์ให้ตรงกับที่ใช้ในตอนฝึกสอนโมเดล
        column_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                        'Loan_Amount_Term', 'Credit_History', 'Rural', 'Semiurban', 'Urban']

        # สร้าง DataFrame ที่มีชื่อคอลัมน์จาก features
        features_df = pd.DataFrame(features, columns=column_names)

       # ทำการพยากรณ์โดยใช้ DataFrame แทนที่จะเป็น NumPy array

        prediction = model.predict(features_df.values)

        # Check the prediction result and redirect to the appropriate page
        if prediction[0] == 1:
            return redirect(url_for('approved'))
        else:
            return redirect(url_for('not_approved'))

    return render_template('index.html')

@app.route('/approved')
def approved():
    return render_template('approved.html')

@app.route('/not_approved')
def not_approved():
    return render_template('not_approved.html')

if __name__ == '__main__':
    app.run(debug=True)
