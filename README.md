# Label editor
Edit labels from DeepLabCut analysis outputs and save as a separate file in both HDF and MATLAB file formats.


## Installation
Install [Anaconda Python 3.7](https://www.anaconda.com/distribution/#download-section)

Install dependencies by using the provided file to create the *labeleditor* conda environment:

`conda env create -f environment.yaml`

## Running
Open Anaconda Prompt and activate the conda environment:

`activate labeleditor`

In Anaconda Prompt, change the directory to **dlc-label-editor/** and add the label and video filepaths as the first and second arguments:

`python editor.py labelfile videofile`

Example:

`python editor.py SampleData\reachingvideo1DLC_resnet50_ReachingAug30shuffle1_200000.h5 SampleData/reachingvideo1.avi`

(Note: After you have edited once, you can use the generated file ***_Fixed.h5** as the first argument to work on a previously edited file)

![alt text](https://github.com/jonperdomo/LabelEditor/blob/master/Images/TrackedDLCExampleData.PNG)

## Editing labels

Change the active label and scroll through frames by using the sliders or the **<>** keys (not the arrow pad).

Left-click the corrected position and the HDF file will be automatically saved with extension ***_Fixed.h5**

Place points faster by holding the left mouse button while scrolling with **<>**

Press the **ESC** key to exit.


### Changing the marker appearance
The three parameters after the filepaths are for marker appearance:

* color (red, green, or blue; default=blue)
* size (default=40)
* thickness (default=2)

Example:

`python editor.py SampleData\reachingvideo1DLC_resnet50_ReachingAug30shuffle1_200000.h5 SampleData/reachingvideo1.avi red 100 3`

![alt text](https://github.com/jonperdomo/LabelEditor/blob/master/Images/TrackedDLCExampleData2.PNG)

## MATLAB file format
The file will be saved in ***MATLAB (*.MAT)*** format after you exit the editor as a structure with X, Y, and likelihood data.

## Known issues
Pressing the window close button closes the UI but the video stays open. This is due to the limited functionality of OpenCV HighGUI. If this occurs just press **ESC** to exit.
