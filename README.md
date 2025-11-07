This Repo is used to present our Assignment for SAIL Dashboard Challenge. The objective was to develop an informative dashboard for crowdmanagers of the SAIL event to monitor the crowdiness at multiple locations, including an prediction algorithm and visualisation methods showing these predictions in an clear and user-friendly way. 


To use the application:


1 Download, to the same Directory, the Q_Max, sensor_data_modified and sensor_location_ines CSV files.

2 Download to this same directory the model_xgb.ipynb and Final_Code_6.py [and also the sensor_forecast_model folder]

3 [REDACTED] Run All cells in the ipynb [this part is now unnecessary if you downloaded the sensor_forecast_model folder}

      ++          This will take some 8 minutes.
      
      ++          A subdirectory with the model shall appear
      
4 Open anaconda prompt

5 type in "cd PATH-TO-THAT-DIRECTORY" 

6 type in "streamlit run Final_Code_6.py

      ++          your browser will open with the application.
      

The Sarimax ipynb was included because it is a very good model, and achieved amazing MAE (as per illustrated by the graphs, if you run the whole ipynb).
The XGBoost model was a last-minute change, because the SARIMAX models were too heavy to upload on github. The Final Code was updated to work with the XGB model,
thus probably would not work with the Sarimax models.


This research was done by:
- Ines Blanes
- Kevin Verbakel
- Zake Marin Domit
- Michiel Pater
- Emeline Neuteboom
