# The machine learning regressor

Starting with v0.9.0, a new framework is proposed within EMHASS. It provides a machine learning module to predict values from a CSV file using different regression models.

This API provides two main methods:

- **fit**: To train a model with the passed data. This method is exposed with the `regressor-model-fit` endpoint.

- **predict**: To obtain a prediction from a pre-trained model. This method is exposed with the `regressor-model-predict` endpoint.

## A basic model fit

To train a model use the `regressor-model-fit` end point.

Some parameters can be optionally defined at runtime:

- `csv_file`: The name of the csv file containing your data.

- `features`: A list of features, you can provide new values for this.

- `target`: The target, the value that has to be predicted.

- `model_type`: Define the name of the model regressor that this will be used for. For example: `heating_hours_degreeday`. This should be a unique name if you are using multiple custom regressor models.

- `regression_model`: The regression model that will be used. For now, only these options are possible: `LinearRegression`, `RidgeRegression`, `LassoRegression`, `RandomForestRegression`, `GradientBoostingRegression` and `AdaBoostRegression`.

- `timestamp`: If defined, the column key that has to be used for timestamp.

- `date_features`: A list of 'date_features' to take into account when fitting the model. Possibilities are `year`, `month`, `day_of_week` (monday=0, sunday=6), `day_of_year`, `day`(day_of_month) and `hour`

### Examples: 
```yaml
runtimeparams = {
    "csv_file": "heating_prediction.csv",
    "features": ["degreeday", "solar"],
    "target": "heating_hours",
    "regression_model": "RandomForestRegression",
    "model_type": "heating_hours_degreeday",
    "timestamp": "timestamp",
    "date_features": ["month", "day_of_week"]
    }
```

A correct `curl` call to launch a model fit can look like this:

```bash
curl -i -H "Content-Type:application/json" -X POST -d  '{"csv_file": "heating_prediction.csv", "features": ["degreeday", "solar"], "target": "hour", "regression_model": "RandomForestRegression", "model_type": "heating_hours_degreeday", "timestamp": "timestamp", "date_features": ["month", "day_of_week"], "new_values": [12.79, 4.766, 1, 2] }' http://localhost:5000/action/regressor-model-fit
```

A Home Assistant `rest_command` can look like this:

```yaml
fit_heating_hours:
  url: http://127.0.0.1:5000/action/regressor-model-fit
  method: POST
  headers:
    content-type: application/json
  payload: >-
    {
    "csv_file": "heating_prediction.csv",
    "features": ["degreeday", "solar"],
    "target": "hours",
    "regression_model": "RandomForestRegression",
    "model_type": "heating_hours_degreeday",
    "timestamp": "timestamp",
    "date_features": ["month", "day_of_week"]
    }
```
After fitting the model the following information is logged by EMHASS:

    2024-04-17 12:41:50,019 - web_server - INFO - Passed runtime parameters: {'csv_file': 'heating_prediction.csv', 'features': ['degreeday', 'solar'], 'target': 'heating_hours', 'regression_model': 'RandomForestRegression', 'model_type': 'heating_hours_degreeday', 'timestamp': 'timestamp', 'date_features': ['month', 'day_of_week']}
    2024-04-17 12:41:50,020 - web_server - INFO -  >> Setting input data dict
    2024-04-17 12:41:50,021 - web_server - INFO - Setting up needed data
    2024-04-17 12:41:50,048 - web_server - INFO -  >> Performing a machine learning regressor fit...
    2024-04-17 12:41:50,049 - web_server - INFO - Performing a MLRegressor fit for heating_hours_degreeday
    2024-04-17 12:41:50,064 - web_server - INFO - Training a RandomForestRegression model
    2024-04-17 12:41:57,852 - web_server - INFO - Elapsed time for model fit: 7.78800106048584
    2024-04-17 12:41:57,862 - web_server - INFO - Prediction R2 score of fitted model on test data: -0.5667567505914477

## The predict method

To obtain a prediction using a previously trained model use the `regressor-model-predict` endpoint.

The list of parameters needed to set the data publish task is:

- `mlr_predict_entity_id`: The unique `entity_id` to be used.

- `mlr_predict_unit_of_measurement`: The `unit_of_measurement` to be used. (Defaults to `W`)

- `mlr_predict_device_class`: The `device_class` for the sensor to be used. (Defaults to `power`). See the Home Assistant documentation [here](https://www.home-assistant.io/integrations/sensor#device-class) for a list of available device_classes.

- `mlr_predict_friendly_name`: The `friendly_name` to be used.

- `new_values`: The new values for the features (in the same order as the features list). Also when using date_features, add these to the new values.

- `model_type`: The model type that has to be predicted

### Examples: 
```yaml
runtimeparams = {
    "mlr_predict_entity_id": "sensor.mlr_predict",
    "mlr_predict_unit_of_measurement": None,
    "mlr_predict_friendly_name": "mlr predictor",
    "new_values": [8.2, 7.23, 2, 6],
    "model_type": "heating_hours_degreeday"
}
```

Pass the correct `model_type` like this:
```bash
curl -i -H "Content-Type:application/json" -X POST -d '{"new_values": [8.2, 7.23, 2, 6], "model_type": "heating_hours_degreeday" }' http://localhost:5000/action/regressor-model-predict
```
or
```bash
curl -i -H "Content-Type:application/json" -X POST -d  '{"mlr_predict_entity_id": "sensor.mlr_predict", "mlr_predict_unit_of_measurement": "h", "mlr_predict_device_class": "duration","mlr_predict_friendly_name": "mlr predictor", "new_values": [8.2, 7.23, 2, 6], "model_type": "heating_hours_degreeday" }' http://localhost:5000/action/regressor-model-predict
```

A Home Assistant `rest_command` can look like this:

```yaml
predict_heating_hours:
  url: http://localhost:5001/action/regressor-model-predict
  method: POST
  headers:
    content-type: application/json
  payload: >-
   {
    "mlr_predict_entity_id": "sensor.predicted_hours",
    "mlr_predict_unit_of_measurement": "h",
    "mlr_predict_device_class": "duration",
    "mlr_predict_friendly_name": "Predicted hours",
    "new_values": [8.2, 7.23, 2, 6],
    "model_type": "heating_hours_degreeday"
    }
```
After predicting the model the following information is logged by EMHASS:

```
2024-04-17 14:25:40,695 - web_server - INFO - Passed runtime parameters: {'mlr_predict_entity_id': 'sensor.predicted_hours', 'mlr_predict_unit_of_measurement': 'h', 'mlr_predict_friendly_name': 'Predicted hours', 'new_values': [8.2, 7.23, 2, 6], 'model_type': 'heating_hours_degreeday'}
2024-04-17 14:25:40,696 - web_server - INFO -  >> Setting input data dict
2024-04-17 14:25:40,696 - web_server - INFO - Setting up needed data
2024-04-17 14:25:40,700 - web_server - INFO -  >> Performing a machine learning regressor predict...
2024-04-17 14:25:40,715 - web_server - INFO - Performing a prediction for heating_hours_degreeday
2024-04-17 14:25:40,750 - web_server - INFO - Successfully posted to sensor.predicted_hours = 3.716600000000001
```
The predict method will publish the result to a Home Assistant sensor.


## Storing CSV files  

### Docker container - how to mount a .csv files in data_path folder
If running EMHASS with the Docker method, you will need to volume mount a folder to be the `data_path`, or mount a single .csv file inside `data_path`

Example of mounting a folder as data_path *(.csv files stored inside)*
```bash
docker run -it --restart always -p 5000:5000 -e LOCAL_COSTFUN="profit" -v ./data:/data -v ./config_emhass.yaml:/app/config_emhass.yaml -v ./secrets_emhass.yaml:/app/secrets_emhass.yaml --name DockerEMHASS <REPOSITORY:TAG>
```
Example of mounting a single CSV file
```bash
docker run -it --restart always -p 5000:5000 -e LOCAL_COSTFUN="profit" -v ./data/heating_prediction.csv:/app/heating_prediction.csv -v ./config_emhass.yaml:/app/config_emhass.yaml -v ./secrets_emhass.yaml:/app/secrets_emhass.yaml --name DockerEMHASS <REPOSITORY:TAG>
```

### Add-on - How to store data in a CSV file from Home Assistant

#### Change data_path
If running EMHASS-Add-On, you will likely need to change the `data_path` to a folder your Home Assistant can access. 
To do this, set the `data_path` to `/share/` in the addon *Configuration* page. 

#### Store sensor data to CSV

Notify to a file
```yaml
notify:
  - platform: file
    name: heating_hours_prediction
    timestamp: false
    filename: /share/heating_prediction.csv
```
Then you need an automation to notify to this file
```yaml
alias: "Heating csv"
id: 157b1d57-73d9-4f39-82c6-13ce0cf42
trigger:
  - platform: time
    at: "23:59:32"
action:
  - service: notify.heating_hours_prediction
    data:
      message: >
        {% set degreeday = states('sensor.degree_day_daily') |float %}
        {% set heating_hours = states('sensor.heating_hours_today') |float | round(2) %}
        {% set solar = states('sensor.solar_daily') |float | round(3) %}
        {% set time = now() %}

          {{time}},{{degreeday}},{{solar}},{{heating_hours}}
```
