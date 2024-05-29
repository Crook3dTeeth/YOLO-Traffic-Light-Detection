# YOLO Traffic Light Detection

Traffic light detection for COSC428 assingment.


## YOLO Models

Updated table of included models and their results.  Some models were trained on different size images which will affect performance (Smaller image = faster).  Will be updated as more training is done

| model | Precision | Recall | IMG Size
| --- | --- | --- | --- |
| 23.pt | 0.967 | 0.879 | 640p |
| 40.pt | 0.956 | 0.983 | 1280p | 


## Install with requirements.txt

Install any prerequisites using pip.  Newer versions of ultralytics can be used which may improve performance but is currently untested
```
pip install -r requirements.txt
```

## Usage

The main funciton in detectTrafficLight.py contains basic config information such as which model is used, input source and target FPS.  Defaults to 23.pt model and runs using drive.mp4

Windows:
```bash
python.exe .\detectTrafficLight.py
```
Linux:
```bash
python3 detectTrafficLight.py
```