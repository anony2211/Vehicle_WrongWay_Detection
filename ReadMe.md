## IDENTIFY ROAD FLOW DIRECTION & WRONG WAY DETECTION USING DEEP LEARNING & OBJECT TRACKING ##

[![IMAGE ALT TEXT HERE](https://www.youtube.com/watch?v=224_xUMf_IQ/0.jpg)](https://www.youtube.com/watch?v=224_xUMf_IQ)

Install dependencies

```
pip3 install -r requirements.txt 

```

To run the script and view results of test.mkv a snippet from test video NVR_ch1_main_20200207140000_20200207143000.asf

```
python3 main.py

```
The output can be viewed in OpenCV window or written as mp4 by setting
write = True in main.py

Bounding Box and centroid colors notation

Green - vehicle moving in correct direction
Orange - vehicle is under observation for moving in incorrect direction
Red- vehicle is declared to be moving in incorrect direction

If training should be done from scratch upload the contents of colab folder to colaboratory along with
vehicle_detection.ipynb and start training in colaboratory.


