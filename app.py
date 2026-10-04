
from flask import Flask, render_template_string, request
import pandas as pd
import joblib

app = Flask(__name__)

# --------------------------------------------------
# LOAD DATASET AND MODEL
# --------------------------------------------------

df = pd.read_csv("crop_recommendation_dataset.csv")
model = joblib.load("crop_recommendation_model.pkl")


# --------------------------------------------------
# DASHBOARD HTML
# --------------------------------------------------

HTML = """

<!DOCTYPE html>

<html>

<head>

    <title>Crop Recommendation System</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f7f5;
            color: #222;
        }

        .header {
            background: #1b5e20;
            color: white;
            padding: 30px 50px;
        }

        .header h1 {
            margin: 0;
            font-size: 34px;
        }

        .header p {
            margin-top: 10px;
            font-size: 17px;
        }

        .container {
            max-width: 1200px;
            margin: 30px auto;
            padding: 0 20px;
        }

        .cards {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        }

        .card h3 {
            margin: 0;
            color: #555;
            font-size: 15px;
        }

        .card .value {
            font-size: 28px;
            font-weight: bold;
            margin-top: 10px;
            color: #1b5e20;
        }

        .section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        }

        .section h2 {
            margin-top: 0;
            color: #1b5e20;
        }

        .crop-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .crop {
            background: #e8f5e9;
            color: #1b5e20;
            padding: 8px 14px;
            border-radius: 20px;
            font-size: 14px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 7px;
        }

        input, select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 7px;
            font-size: 15px;
        }

        button {
            margin-top: 25px;
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 8px;
            background: #2e7d32;
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #1b5e20;
        }

        .result {
            margin-top: 25px;
            padding: 25px;
            border-radius: 10px;
            background: #e8f5e9;
            text-align: center;
        }

        .result h2 {
            color: #1b5e20;
        }

        .result .crop-name {
            font-size: 32px;
            font-weight: bold;
            color: #2e7d32;
        }

        .table-container {
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background: #e8f5e9;
        }

        .footer {
            text-align: center;
            padding: 25px;
            color: #666;
        }

        @media(max-width: 800px) {

            .cards {
                grid-template-columns: repeat(2, 1fr);
            }

            .form-grid {
                grid-template-columns: 1fr;
            }

        }

    </style>

</head>


<body>


<div class="header">

    <h1>🌱 Random Forest Based Crop Recommendation</h1>

    <p>
        A machine learning system that recommends suitable crops
        based on agricultural and soil conditions.
    </p>

</div>


<div class="container">


    <!-- DATASET OVERVIEW -->

    <div class="cards">

        <div class="card">
            <h3>Total Records</h3>
            <div class="value">{{ total_records }}</div>
        </div>

        <div class="card">
            <h3>Input Features</h3>
            <div class="value">9</div>
        </div>

        <div class="card">
            <h3>Crop Categories</h3>
            <div class="value">{{ crop_count }}</div>
        </div>

        <div class="card">
            <h3>Soil Categories</h3>
            <div class="value">{{ soil_count }}</div>
        </div>

    </div>


    <!-- DATASET INFORMATION -->

    <div class="section">

        <h2>📊 Dataset Overview</h2>

        <p>
            The dataset contains agricultural and soil parameters
            used to recommend suitable crops.
        </p>

        <p>
            <b>Target Variable:</b> Crop
        </p>

        <p>
            <b>Algorithm:</b> Random Forest Classifier
        </p>

    </div>


    <!-- CROPS -->

    <div class="section">

        <h2>🌾 Available Crops</h2>

        <div class="crop-list">

            {% for crop in crops %}

                <span class="crop">{{ crop }}</span>

            {% endfor %}

        </div>

    </div>


    <!-- PREDICTION FORM -->

    <div class="section">

        <h2>🌱 Crop Recommendation</h2>

        <p>
            Enter the agricultural conditions to predict the most suitable crop.
        </p>


        <form method="POST">

            <div class="form-grid">


                <div>

                    <label>Temperature</label>

                    <input
                        type="number"
                        step="any"
                        name="Temperature"
                        value="{{ values.get('Temperature', '') }}"
                        required
                    >

                </div>


                <div>

                    <label>Humidity</label>

                    <input
                        type="number"
                        step="any"
                        name="Humidity"
                        value="{{ values.get('Humidity', '') }}"
                        required
                    >

                </div>


                <div>

                    <label>Rainfall</label>

                    <input
                        type="number"
                        step="any"
                        name="Rainfall"
                        value="{{ values.get('Rainfall', '') }}"
                        required
                    >

                </div>


                <div>

                    <label>Soil pH</label>

                    <input
                        type="number"
                        step="any"
                        name="PH"
                        value="{{ values.get('PH', '') }}"
                        required
                    >

                </div>


                <div>

                    <label>Nitrogen</label>

                    <input
                        type="number"
                        step="any"
                        name="Nitrogen"
                        value="{{ values.get('Nitrogen', '') }}"
                        required
                    >

                </div>


                <div>

                    <label>Phosphorous</label>

                    <input
                        type="number"
                        step="any"
                        name="Phosphorous"
                        value="{{ values.get('Phosphorous', '') }}"
                        required
                    >

                </div>


                <div>

                    <label>Potassium</label>

                    <input
                        type="number"
                        step="any"
                        name="Potassium"
                        value="{{ values.get('Potassium', '') }}"
                        required
                    >

                </div>


                <div>

                    <label>Carbon</label>

                    <input
                        type="number"
                        step="any"
                        name="Carbon"
                        value="{{ values.get('Carbon', '') }}"
                        required
                    >

                </div>


                <div>

                    <label>Soil Type</label>

                    <select name="Soil" required>

                        {% for soil in soils %}

                            <option value="{{ soil }}"
                            {% if values.get('Soil') == soil %}
                            selected
                            {% endif %}>
                                {{ soil }}
                            </option>

                        {% endfor %}

                    </select>

                </div>


            </div>


            <button type="submit">
                🌱 Recommend Crop
            </button>


        </form>


        {% if prediction %}

        <div class="result">

            <h2>Recommended Crop</h2>

            <div class="crop-name">
                🌾 {{ prediction }}
            </div>

            <p>
                Based on the agricultural and soil conditions provided.
            </p>

        </div>

        {% endif %}


    </div>


    <!-- FEATURES -->

    <div class="section">

        <h2>📋 Model Input Parameters</h2>

        <div class="table-container">

            <table>

                <tr>
                    <th>Parameter</th>
                    <th>Description</th>
                </tr>

                <tr>
                    <td>Temperature</td>
                    <td>Environmental temperature</td>
                </tr>

                <tr>
                    <td>Humidity</td>
                    <td>Moisture level in the environment</td>
                </tr>

                <tr>
                    <td>Rainfall</td>
                    <td>Amount of rainfall</td>
                </tr>

                <tr>
                    <td>PH</td>
                    <td>Soil acidity/alkalinity</td>
                </tr>

                <tr>
                    <td>Nitrogen</td>
                    <td>Nitrogen content in soil</td>
                </tr>

                <tr>
                    <td>Phosphorous</td>
                    <td>Phosphorous content in soil</td>
                </tr>

                <tr>
                    <td>Potassium</td>
                    <td>Potassium content in soil</td>
                </tr>

                <tr>
                    <td>Carbon</td>
                    <td>Carbon content in soil</td>
                </tr>

                <tr>
                    <td>Soil</td>
                    <td>Type of soil</td>
                </tr>

            </table>

        </div>

    </div>


</div>


<div class="footer">

    Random Forest Based Crop Recommendation |
    Machine Learning Project

</div>


</body>

</html>

"""


# --------------------------------------------------
# HOME + PREDICTION
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None

    values = {}

    crops = sorted(df["Crop"].dropna().unique())

    soils = sorted(df["Soil"].dropna().unique())


    if request.method == "POST":

        values = request.form.to_dict()


        input_data = pd.DataFrame({

            "Temperature": [float(values["Temperature"])],

            "Humidity": [float(values["Humidity"])],

            "Rainfall": [float(values["Rainfall"])],

            "PH": [float(values["PH"])],

            "Nitrogen": [float(values["Nitrogen"])],

            "Phosphorous": [float(values["Phosphorous"])],

            "Potassium": [float(values["Potassium"])],

            "Carbon": [float(values["Carbon"])],

            "Soil": [values["Soil"]]

        })


        prediction = model.predict(input_data)[0]


    return render_template_string(

        HTML,

        total_records=len(df),

        crop_count=df["Crop"].nunique(),

        soil_count=df["Soil"].nunique(),

        crops=crops,

        soils=soils,

        prediction=prediction,

        values=values

    )


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8501,
        debug=False
    )
