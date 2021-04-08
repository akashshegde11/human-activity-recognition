Recognizing Set of Human Activities from Video Dataset using Machine Learning
---------------------------------------------------------------------------------------------


Software Requirements:
----------------------------
These software have to be installed in the system before running the project - 
1. Python 3.6.4 (x64).
	Relevant link - https://www.python.org/downloads/release/python-360/
2. Pycharm Community Edition 2017.2.3.
	Relevant link - https://www.jetbrains.com/pycharm/download/
3. TensorFlow 1.4.0.
	Relevant link - https://www.tensorflow.org/install/
4. Keras 2.1.5.
	Relevant link - https://keras.io/
5. NumPy 1.14.0.
	Relevant link - https://www.scipy.org/scipylib/download.html
6. FFmpeg 4.0.
	Relevant link - https://ffmpeg.zeranoe.com/builds/
7. OpenCV 3.4.0.12.
	Relevant link - https://opencv.org/releases.html
8. Matplotlib 2.2.0.
	Relevant link - https://matplotlib.org/
9. Graphviz 0.8.2.
	Relevant link - https://graphviz.gitlab.io/download/
10. Pydot 1.2.4.
	Relevant link - https://github.com/erocarrera/pydot
11. h5py 2.7.1.
	Relevant link - http://docs.h5py.org/en/latest/build.html
12. Django 2.0.2.
	Relevant link - https://www.djangoproject.com/download/


Installation Procedure:
---------------------------
Step1 - The KTH Dataset can be downloaded from the following link - http://www.nada.kth.se/cvap/actions/
Step2 - Once downloaded, the videos should be placed under the main folder "KTH" and within that folder, six more folders for each activity have to be created, namely "boxing", "handclapping", "handwaving", "jogging", "running" and "walking."
Step3 - The text file containing the frame sequence information should be placed within the "KTH" folder. It is named "00sequences.txt" according to convention.
Step4 - The python script "Prepare_Data.py" is run to convert videos to frames using FFmpeg. If OpenCV is necessary to be run for comparison, the python script "PD_OpenCV.py" can be run instead.
Step5 - The python script "LSTM.py" is run to perform human activity recognition, after specifying the required parameters. If GRU is necessary to be run for comparison, the python script "GRU.py" can be run instead.
Step6 - Outputs are observed, in terms of confusion matrix, JSON string and HDF5 file.


Front End:
------------
Step1 - Open Command Prompt (or Windows PowerShell) in the folder "actual" where the python script "manage.py" is present.
Step2 - Type the command "python manage.py runserver" into the prompt and minimize the prompt.
Step3 - Open a browser window (Google Chrome preferred) and enter the address "http://127.0.0.1:8000" to reach the introduction page.
Step4 - Click "Enter" to read about team.
Step5 - Click "Next" to read about the project and the dataset.
Step6 - Click "Next" to arrive at the inputs page.
Step7 - Input the "Training Parameters" (range 1-25), "Testing Parameters" (range 1-25) and "Epochs" (range 10-200) and click "Submit."
Step8 - Open the command prompt to observe the back-end working of human activity recognition.
Step9 - Once the execution is complete, revisit browser window to see the output confusion matrix displayed on the screen.
