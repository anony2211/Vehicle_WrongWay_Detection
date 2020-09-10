## IDENTIFY ROAD FLOW DIRECTION & WRONG WAY DETECTION USING DEEP LEARNING & OBJECT TRACKING ##

[![WRONG WAY VEHICLE DETECTION YOUTUBE](https://www.youtube.com/watch?v=224_xUMf_IQ/0.jpg)](https://www.youtube.com/watch?v=224_xUMf_IQ)

Install dependencies

```
pip3 install -r requirements.txt 

```

To run the script and view results of test.mkv 

```
python3 main.py

```
The output can be viewed in OpenCV window or written as mp4 by setting
write = True in main.py

Bounding Box and centroid colors notation

Green - vehicle moving in correct direction
Orange - vehicle is under observation for moving in incorrect direction
Red- vehicle is declared to be moving in incorrect direction

### For Training in Colaboratory ###

Upload the contents of colab folder 
and start training with vehicle_detection.ipynb colab notebook.


