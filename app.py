from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle
import requests

clf=pickle.load(open('model3.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        company=request.form.get('company')
        year=request.form.get('year')
        fuel_type=request.form.get('fuel_type')
        kms_driven=request.form.get('kms')
        company=company.lower()
        fuel_type.lower()
        company=company.capitalize()
        kms_driven=kms_driven.strip('kms')
        kms_driven = kms_driven.strip('km')
        def change(text):
            if text == 'petrol':
                return 1
            elif text == 'diesel':
                return 2
            else:
                return 3
        fuel=change(fuel_type)

        d = {'company': company, 'year': year, 'kms_driven': kms_driven, 'fuel_type': fuel}
        data = pd.DataFrame(d, index=[1])
        new = pd.DataFrame(columns=['kms_driven', 'company_Audi', 'company_BMW', 'company_Chevrolet',
                                    'company_Datsun', 'company_Fiat', 'company_Force', 'company_Ford',
                                    'company_Hindustan', 'company_Honda', 'company_Hyundai',
                                    'company_Jaguar', 'company_Jeep', 'company_Land',
                                    'company_Mahindra', 'company_Maruti', 'company_Mercedes',
                                    'company_Mini', 'company_Mitsubishi', 'company_Nissan',
                                    'company_Renault', 'company_Skoda', 'company_Tata',
                                    'company_Toyota', 'company_Volkswagen', 'company_Volvo',
                                    'fuel_type_1', 'fuel_type_2', 'fuel_type_3', 'year_1995',
                                    'year_2000', 'year_2001', 'year_2002', 'year_2003', 'year_2004',
                                    'year_2005', 'year_2006', 'year_2007', 'year_2008', 'year_2009',
                                    'year_2010', 'year_2011', 'year_2012', 'year_2013', 'year_2014',
                                    'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019'])
        F = pd.get_dummies(data, columns=['company', 'fuel_type', 'year'])
        new[F.columns.values] = F[F.columns.values]
        new = new.replace(np.nan, 0)
        x = new.iloc[:, :].values
        pred=round(clf.predict(x)[0],2)

        return render_template('home.html', pred=pred, label=1)

    except:

        return render_template('home.html', label=0)


if __name__=="__main__":
    app.run(debug=True)