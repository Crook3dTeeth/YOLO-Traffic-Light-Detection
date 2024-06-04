# YOLO Traffic Light Detection

Traffic light detection for COSC428 assingment.


## YOLO Models

Updated table of included models and their results.  Some models were trained on different size images which will affect performance (Smaller image = faster).  Will be updated as more training is done

| model | Precision | Recall | IMG Size
| --- | --- | --- | --- |
| 23.pt | 0.967 | 0.879 | 640p |
| 40.pt | 0.956 | 0.983 | 1280p | 


## Install with requirements.txt

Install any prerequisites using pip.  Newer versions of ultralytics can be used but it is currently untested
```
pip install -r requirements.txt
```

## Usage

The main funciton in detectTrafficLight.py contains basic config information such as which model is used, input source and target FPS.  To change any settings, edit the variables at the top of the detectTrafficLight.py file.  Defaults to 23.pt model and drive.mp4.  

Windows:
```bash
python.exe .\detectTrafficLight.py
```
Linux:
```bash
python3 detectTrafficLight.py
```

## Training

Annotating a dataset was done using https://app.roboflow.com/ as it can automatically split the data set into training validation and testing folders aswell as creating the yaml file for the dataset.  After a dataset is annotated using configure the [training file](https://github.com/Crook3dTeeth/YOLO-Traffic-Light-Detection/blob/main/training/yolov8/main.py) with the .yaml file and how many epochs, img size etc. More info on yolov8 and training can be found [here] (https://github.com/ultralytics/ultralytics)