# The forecast module

EMHASS will basically need 4 forecasts to work properly:

- PV power production forecast (internally based on the weather forecast and the characteristics of your PV plant). This is given in Watts.

- Load power forecast: how much power your house will demand on the next 24h. This is given in Watts.

- Load cost forecast: the price of the energy from the grid on the next 24h. This is given in EUR/kWh.

- PV production selling price forecast: at what price are you selling your excess PV production on the next 24h. This is given in EUR/kWh.

## PV power production forecast

The default method for PV power forecast is the scrapping of weather forecast data from the https://clearoutside.com/ website. This site proposes detailed forecasts based on Lat/Lon locations. This method seems quite stable but as with any scrape method it will fail if any changes are made to the webpage API.

This may change in the future to direct API's of weather forecast models as GFS or ECMWF, see:

- GFS: https://github.com/jagoosw/getgfs

- ECMWF: https://pypi.org/project/ecmwf-api-client/

## Load power forecast

The default method for load forecast is a naive method, also called persistence. This method simply assumes that the forecast for a future period will be equal to the observed values in a past period. The past period is controlled using parameter `delta_forecast` and the default value for this is 24h.

This is presented graphically here:

![](./images/naive_forecast.png)

## Passing your own forecast data

For all the needed forecasts in EMHASS two other methods allows the user to provide their own forecast value. This may be used to provide a forecast provided by a more powerful and accurate forecaster. The two methods are: `csv` and `list`.

For the `csv` method you should push a csv file to the `data` folder. The CSV file should contain no header and the timestamped data should have the following format:
    
    2021-04-29 00:00:00+00:00,287.07
    
    2021-04-29 00:30:00+00:00,274.27
    
    2021-04-29 01:00:00+00:00,243.38

...

For the list method you just have to add the data as a list of values to a data dictionnary during the call to `emhass` using the `runtimeparams` option.

The possible dictionnary keys to pass data are:

- `pv_power_forecast` for the PV power production forecast.

- `load_power_forecast` for the Load power forecast.

- `load_cost_forecast` for the Load cost forecast.

- `prod_price_forecast` for the PV production selling price forecast.

For example if using the add-on or the standalone docker installation you can pass this data as list of values to the data dictionnary during the `curl` POST:
```
curl -i -H "Content-Type: application/json" -X POST -d '{"pv_power_forecast":[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}' http://localhost:5000/action/dayahead-optim
```

## Now/current values in forecasts

When implementing MPC applications with high optimization frequencies it can be interesting if at each MPC iteration the forecast values are updated with the real now/current values measured from live data. This is useful to improve the accuracy of the short-term forecasts. 

A simple integration of current/now values for PV and load forecast is implemented using a mixed one-observation presistence model and the one-step-ahead forecasted values from the current passed method. 

This can be represented by the following equation at time $t=k$: 

$$
P^{mix}_{PV} = \alpha \hat{P}_{PV}(k) + \beta P_{PV}(k-1)
$$

Where $P^{mix}_{PV}$ is the mixed power forecast for PV prodduction, $\hat{P}_{PV}(k)$ is the current first element of the original forecast data, $P_{PV}(k-1)$ is the now/current value of PV production and $\alpha$ and $\beta$ are coefficients that can be fixed to reflect desired dominance of now/current values over the original forecast data or viceversa.

The `alpha` and `beta` values can be passed in the dictionnary using the `runtimeparams` option during the call to `emhass`. If not passed they will both take the default 0.5 value.