**Data**

Download this dataset and put all the files under data/raw

Run data_cleaner.py

**Install**
```
pip install -r requirements.txt
```

Look https://pytorch.org to install PyTorch

**Run**

Run train_emotion.py to train the emotion recognition machine (it will train and dump the machine in models folder)

Run test_emotion.py to test the emotion recognition machine

Run train_speaker.py to train the speaker recognition machine (it will train and dump the machine in models folder)

Run test_speaker.py to test the speaker recognition machine


**Notice**

Speaker is recognised with 70% accuracy

For emotion, it is 33% :-( - to understand how bad it is, a random guess without training would be 14%. 
   
In addition to my naivety, another big reason for this low accuracy is low volume data (only around 400).

Emotion is a complex phenomena - we need more and more data. Also, need to capture temporal features - maybe with Transformer, which is my next project. 
