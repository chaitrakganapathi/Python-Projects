from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

from pcod_prediction_team5_final import RandomForest,RandomForestClassifier,ExtraTreeClassifier,TreeNode


app = Flask(__name__)

# Load the pickle file
pickleModel = pickle.load(open('pickleFinal.pkl', 'rb'))



# Method to display form
@app.route('/')
def startWebApp():
    return render_template("pcosDiagnosisForm.html")


# Method to process form data
@app.route('/form_diagnosis', methods=['POST', 'GET'])
def predict_pcod():
    flag = 1
    # Store all the values from the form page in a list
    features = [x for x in request.form.values()]
    for val in features:
        # Check if all the text fields have values
        if val == '':
            info = 'Enter values for all the fields'
            flag = 0
            break

    if flag:
        features = [int(x) for x in request.form.values()]
        featuresArr = np.array(features)
        featuresArray = pd.DataFrame([featuresArr])
        # Pass values to trained model to predict output
        predictedVal = pickleModel.predict(featuresArray)

        if predictedVal[0] >= 0.5:
            info = 'You have significant chances of having PCOS. Please check with your doctor as soon as ' \
                   'possible.'
        else:
            info = 'You do not have significant chances of having PCOS. However,if you are experiencing some other ' \
                   'symptoms or feeling that you need a diagnosis, it is good to bring it up at your next doctor’s ' \
                   'appointment.'

    return render_template('pcosDiagnosisForm.html', info=info)


if __name__ == '__main__':
    app.run()
