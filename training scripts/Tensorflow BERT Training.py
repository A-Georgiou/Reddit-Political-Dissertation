import tensorflow_hub as hub
import tensorflow as tf
import bert
FullTokenizer = bert.bert_tokenization.FullTokenizer
from keras.models import Model       # Keras is the new high level API for TensorFlow
import math
import pandas as pd
from sklearn.model_selection import train_test_split 
from tqdm import tqdm
import numpy as np
from tensorflow import keras

import datetime
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
import os

import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train = pd.read_csv('./datasets/combined_output_newer_flat.csv', engine='python') x

plt.hist(train['sentiment'], color = 'green', edgecolor='black')
plt.title('Histogram of Star Ratings')
plt.xlabel('Star Rating')
plt.ylabel('Frequency')

training_data, testing_data = train_test_split(train, test_size=0.1, random_state=42, shuffle=True)

bert_model_name = "./models/uncased_L-12_H-768_A-12"  
bert_ckpt_dir = bert_model_name
bert_ckpt_file = os.path.join(bert_ckpt_dir, 'bert_model.ckpt')
bert_config_file = os.path.join(bert_ckpt_dir, 'bert_config.json')

"""
Class: Intent Detection
Function: Handles preparing and pre-processing text for dataset
"""
class IntentDetectionData:
  DATA_COLUMN = 'comment'
  LABEL_COLUMN = 'sentiment'

  def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len=192):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes

    ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

    print("max seq_len", self.max_seq_len)
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df):
    x, y = [], []

    for _, row in tqdm(df.iterrows()):
      text, label = str(row[IntentDetectionData.DATA_COLUMN]), row[IntentDetectionData.LABEL_COLUMN]
      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]

      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      self.max_seq_len = max(self.max_seq_len, len(token_ids))

      x.append(token_ids)
      y.append(self.classes.index(label))
    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)

class BertModel:
	"""
	Create Model
	Script to generate BERT model with additional Layers for handling output
	"""
	def create_model(max_seq_len, bert_config_file, bert_ckpt_file):

	  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
	      bc = StockBertConfig.from_json_string(reader.read())
	      bert_params = map_stock_config_to_params(bc)
	      bert_params.adapter_size = None
	      bert = BertModelLayer.from_params(bert_params, name="bert")

	  input_ids = keras.layers.Input(shape=(max_seq_len, ),dtype='int32',name="input_ids")
	  bert_output = bert(input_ids)

	  print("bert shape", bert_output.shape)
	  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
	  cls_out = keras.layers.Dropout(0.5)(cls_out)
	  logits = keras.layers.Dense(units=768, activation="relu")(cls_out)
	  logits = keras.layers.Dropout(0.5)(logits)
	  logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

	  model = keras.Model(inputs=input_ids, outputs=logits)
	  model.build(input_shape=(None, max_seq_len))

	  load_stock_weights(bert, bert_ckpt_file)

	  return model

"""
Main section of code handles training and evaluation of model
"""
if __name__ == "__main__":
	tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, 'vocab.txt'))

	classes = training_data.sentiment.unique().tolist()
	data = IntentDetectionData(training_data, testing_data, tokenizer, classes, max_seq_len=256)

	model = BertModel.create_model(data.max_seq_len, bert_config_file, bert_ckpt_file)

	model.compile(optimizer=keras.optimizers.Adam(2e-5),
	              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')]
	)

	log_dir = 'log/intent_detection'

	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

	history = model.fit(
	    x=data.train_x,
	    y=data.train_y,
	    validation_split=0.1,
	    batch_size=16,
	    shuffle=True,
	    epochs=10,
	    callbacks=[tensorboard_callback]
	)
	    
	"""
	Test sentences extracted from reddit used for prediction
	"""
	sentences = [
	        "Good after you’ve served your time you should have ALL your rights restored including the right to VOTE.",
	        "The United States civil war. Those who sought to preserve the union were later proven to be on the right side of history.",
	        "I believe donald is our president for good reason and shall remain that way",
	        "This is so spot on. In large part because Democrats actually do act like small spoiled children.",
	        "people need to stop being offended all the damn time, trump is making america great again",
	        "donald is making america great again, what are you doing about it commie",
	        "I believe the right to free healthcare is a given"
	        ]

	tokens = map(tokenizer.tokenize, sentences)
	tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], tokens)
	token_ids = list(map(tokenizer.convert_tokens_to_ids, tokens))

	#Pad rest of token id's with 0
	token_ids = map(lambda tids: tids + [0] * (data.max_seq_len - len(tids)), token_ids)
	token_ids = np.array(list(token_ids))

	predictions = model.predict(token_ids).argmax(axis=-1)

	for txt, label in zip(sentences, predictions):
	  print("Text: ", txt)
	  print("sentiment: ", classes[label])

	model.save('reddit-bot.h5')
