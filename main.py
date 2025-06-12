import pandas as pd
import joblib
from flask import Flask,render_template,request,redirect

app=Flask(__name__)
preprocessing_pipeline = joblib.load('pipeline.joblib')
model = joblib.load('rf.joblib')
print(preprocessing_pipeline.named_steps)

df = pd.read_csv("FINAL.csv", parse_dates=['date'])


def create_lagged_features(df, n_lags):
    lagged_data = []
    for col in df.columns:
        for lag in range(1, n_lags + 1):
            lagged_data.append(df[col].shift(lag).rename(f'{col}_lag{lag}'))
    lagged_data = pd.concat(lagged_data, axis=1)
    return lagged_data

def create_multistep_targets(df, target_variables, n_steps):
    target_data = []
    for variable in target_variables:
        for step in range(1, n_steps + 1):
            target_data.append(df[variable].shift(-step).rename(f'{variable}_step{step}'))
    target_data = pd.concat(target_data, axis=1)
    return target_data


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/",methods=["GET","POST"])
def predict():
    if request.method=="POST":
        user_input=pd.DataFrame([{
            "date":pd.Timestamp.now(),
            "co":request.form["co"],
            "no":request.form["no"],
            "no2":request.form["no2"],
            "o3":request.form["o3"],
            "so2":request.form["so2"],
            "pm2_5":request.form["pm2_5"],
            "pm10":request.form["pm10"],
            "nh3":request.form["nh3"]
        }])
        print(user_input)

        df_combined = pd.concat([df, user_input], ignore_index=True)

        # Generate lagged features and targets
        n_lags = 24
        df_lagged = create_lagged_features(df_combined, n_lags)
        df_lagged = df_lagged.dropna()

        n_steps = 24
        target_variables = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        df_targets = create_multistep_targets(df_combined, target_variables, n_steps)
        df_targets = df_targets.dropna()

        # Combine lagged features and targets
        df_combined_latest = pd.concat([df_lagged, df_targets], axis=1).dropna()
        print(df_combined_latest)
        # Ensure the feature names in df_combined_latest match those used in the pipeline
        # This is an example; modify according to your specific pipeline and data

        feature_names = preprocessing_pipeline.named_steps['std_scaler'].feature_names_in_
        df_combined_latest_selected = df_combined_latest[feature_names]
        preprocessed_data = preprocessing_pipeline.transform(df_combined_latest_selected)

        print(f"preprocessed Data\n{preprocessed_data}")
        prediction = model.predict(preprocessed_data)
        print("Predictions shape:", prediction.shape)
        print("First few predictions:\n", prediction[:5])

        pollutant_index = 0  # 'co' is the first pollutant, so its starting index is 0
        n_steps = 24  # Number of steps

        # Extract all step predictions for 'co'
        co_predictions = prediction[:, pollutant_index:pollutant_index + n_steps]
        print("CO Predictions for all steps:\n", co_predictions)

        co_step1_index = 0  # Assuming 'co_step1' is at index 0
        co_step2_index = 1  # Assuming 'co_step2' is at index 1 (or adjust based on actual order)

        # Extract the specific predictions
        co_step1_prediction = prediction[-1, co_step1_index]  # Get the last entry, which corresponds to the user input
        co_step2_prediction = prediction[-1, co_step2_index]

        print(f"Prediction for 'co' at Step 1: {co_step1_prediction}")
        print(f"Prediction for 'co' at Step 2: {co_step2_prediction}")
        co_scaler = preprocessing_pipeline.named_steps['std_scaler']  # or 'min_max_scaler'

        feature_names = [
            'co_step1', 'no_step1', 'no2_step1', 'o3_step1', 'so2_step1', 'pm2_5_step1', 'pm10_step1', 'nh3_step1',
            'co_step2', 'no_step2', 'no2_step2', 'o3_step2', 'so2_step2', 'pm2_5_step2', 'pm10_step2', 'nh3_step2',
            # Continue for all steps...
        ]

        # Initialize a dictionary to hold the actual predictions
        actual_predictions = {}

        # Perform inverse transformation for all predictions
        predictions_actual = co_scaler.inverse_transform(prediction)

        # Assuming there are 8 pollutants and each pollutant has predictions for multiple steps

        def extract_predictions(predictions_actual):
            steps = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7', 'step8', 'step9', 'step10',
                     'step11', 'step12', 'step13', 'step14', 'step15', 'step16', 'step17', 'step18', 'step19', 'step20',
                     'step21', 'step22', 'step23', 'step24']
            pollutants = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']

            results = {}
            for i, pollutant in enumerate(pollutants):
                results[pollutant] = {step: predictions_actual[:, i * 24 + j] for j, step in enumerate(steps)}

            return results

        predictions_dict = extract_predictions(predictions_actual)

        return render_template('result.html', predictions=predictions_dict)

    return render_template("air.html")

if __name__=="__main__":
    app.run(debug=True)