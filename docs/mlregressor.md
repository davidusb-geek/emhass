# The machine learning regressor

Starting with v0.9.0, a new framework is proposed within EMHASS. It provides a machine learning module to predict values from a csv file using different regression models.

This API provides two main methods:

- fit: To train a model with the passed data. This method is exposed with the `regressor-model-fit` end point.

- predict: To obtain a prediction from a pre-trained model. This method is exposed with the `regressor-model-predict` end point.

## A basic model fit

To train a model use the `regressor-model-fit` end point.

Some paramters can be optionally defined at runtime:

- `csv_file`: The name of the csv file containing your data.

- `features`: A list of features, you can provide new values for this.

- `target`: The target, the value that has to be predicted.

- `model_type`: Define the name of the model regressor that this will be used for. For example: `heating_hours_degreeday`. This should be an unique name if you are using multiple custom regressor models.

- `regression_model`: The regression model that will be used. For now only this options are possible: `LinearRegression`, `RidgeRegression`, `LassoRegression`, `RandomForestRegression`, `GradientBoostingRegression` and `AdaBoostRegression`.

- `timestamp`: If defined, the column key that has to be used for timestamp.

- `date_features`: A list of 'date_features' to take into account when fitting the model. Possibilities are `year`, `month`, `day_of_week` (monday=0, sunday=6), `day_of_year`, `day`(day_of_month) and `hour`

```
runtimeparams = {
    "csv_file": "heating_prediction.csv",
    "features":["degreeday", "solar"],
    "target": "heating_hours",
    "regression_model": "RandomForestRegression",
    "model_type": "heating_hours_degreeday",
    "timestamp": "timestamp",
    "date_features": ["month", "day_of_week"]
    }
```

A correct `curl` call to launch a model fit can look like this:

```
curl -i -H "Content-Type:application/json" -X POST -d '{}' http://localhost:5000/action/regressor-model-fit
```

After applying the `curl` command to fit the model the following information is logged by EMHASS:

    2023-02-20 22:05:22,658 - __main__ - INFO - Training a LinearRegression model
    2023-02-20 22:05:23,882 - __main__ - INFO - Elapsed time: 1.2236599922180176
    2023-02-20 22:05:24,612 - __main__ - INFO - Prediction R2 score: 0.2654560762747957

## The predict method

To obtain a prediction using a previously trained model use the `regressor-model-predict` end point.

```
curl -i -H "Content-Type:application/json" -X POST -d '{}' http://localhost:5000/action/regressor-model-predict
```

If needed pass the correct `model_type` like this:

```
curl -i -H "Content-Type:application/json" -X POST -d '{"model_type": "load_forecast"}' http://localhost:5000/action/regressor-model-predict
```

It is possible to publish the predict method results to a Home Assistant sensor.

The list of parameters needed to set the data publish task is:

- `mlr_predict_entity_id`: The unique `entity_id` to be used.

- `mlr_predict_unit_of_measurement`: The `unit_of_measurement` to be used.

- `mlr_predict_friendly_name`: The `friendly_name` to be used.

- `new_values`: The new values for the features (in the same order as the features list). Also when using date_features, add these to the new values.

- `model_type`: The model type that has to be predicted

```
runtimeparams = {
    "mlr_predict_entity_id": "sensor.mlr_predict",
    "mlr_predict_unit_of_measurement": None,
    "mlr_predict_friendly_name": "mlr predictor",
    "new_values": [8.2, 7.23, 2, 6],
    "model_type": "heating_hours_degreeday"
}
```
