from flask import Flask, request, render_template_string
import pickle as pkl
import pandas as pd

app = Flask(__name__)

with open('loan_model.pkl', 'rb') as f:
    model = pkl.load(f)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Loan Eligibility Pro</title>
    <style>
        body { font-family: sans-serif; background-color: #f4f7f6; padding: 40px; }
        .container { max-width: 500px; margin: auto; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h2 { color: #2c3e50; text-align: center; }
        input, select { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
        button { width: 100%; background: #27ae60; color: white; padding: 12px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .result { margin-top: 20px; text-align: center; font-weight: bold; font-size: 1.2em; color: #27ae60; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🏦 Loan Risk Predictor</h2>
        <form action="/predict" method="post">
            <select name="Gender"><option>Male</option><option>Female</option></select>
            <select name="Married"><option>Yes</option><option>No</option></select>
            <input type="number" name="ApplicantIncome" placeholder="Applicant Income" required>
            <input type="number" name="CoapplicantIncome" placeholder="Coapplicant Income" required>
            <input type="number" name="LoanAmount" placeholder="Loan Amount (thousands)" required>
            <select name="Credit_History"><option value="1">Good Credit</option><option value="0">Bad Credit</option></select>
            <select name="Property_Area"><option>Urban</option><option>Semiurban</option><option>Rural</option></select>
            <button type="submit">Check Eligibility</button>
        </form>
        {% if result %}
            <div class="result">Status: {{ result }}</div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    # convert to dataframe as pipeline expects datframe
    data = {
        'Gender': [request.form['Gender']],
        'Married': [request.form['Married']],
        'Dependents': ['0'],     # hardcoded for now for simplicity
        'Education': ['Graduate'],
        'Self_Employed': ['No'],
        'ApplicantIncome': [float(request.form['ApplicantIncome'])],
        'CoapplicantIncome': [float(request.form['CoapplicantIncome'])],
        'LoanAmount': [float(request.form['LoanAmount'])],
        'Loan_Amount_Term': [360.0],
        'Credit_History': [float(request.form['Credit_History'])],
        'Property_Area': [request.form['Property_Area']]
    }

    df_input = pd.DataFrame(data)
    pred = model.predict(df_input)
    status = "Approved!" if pred[0] == 1 else "Rejected!"

    return render_template_string(HTML_TEMPLATE, result=status)

if __name__ == '__main__':
    app.run(debug=True)

