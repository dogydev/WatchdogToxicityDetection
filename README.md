# WatchdogToxicityDetection
A state-of-the-art discord bot which utilizes the XLNet model to detect toxicity

To execute the program,

1. Download and extract the Kaggle Toxic Comments Classification Dataset (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) to /data/

2. Download and install required libraries using 
``` pip install -r requirements.txt ```

3. Train the model using
``` python train.py ```

4. Insert your Discord logging channel ID  and Discord Bot token into main.py. After that execute
``` python main.py ``` 
