#import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split 
from tqdm import tqdm
import numpy as np
import datetime
import math
from transformers import TFDistilBertForSequenceClassification, DistilBertConfig
from transformers import DistilBertTokenizer
from sklearn.model_selection import GridSearchCV
#from tensorboard.plugins.hparams import api as hp

distil_bert = 'distilbert-base-uncased'

FullTokenizer = DistilBertTokenizer.from_pretrained(distil_bert, do_lower_case=True, add_special_tokens=True, max_length=8, padding=True, Truncation=True)

print('setup bert tokenizer')

train = pd.read_csv('./datasets/combined_output_newer_flat.csv', engine='python')  # Update path as needed
print(train)
train['length'] = train.comment.str.len()
train = train[train.length > 50]
lengths = train['length']
training_data, testing_data = train_test_split(train, test_size=0.1, random_state=38, shuffle=True)

training_data_x = training_data['comment']
training_data_y = training_data['sentiment']

testing_data_x = testing_data['comment']
testing_data_y = testing_data['sentiment']

def tokenize(sentences, tokenizer):
        input_ids, input_masks, input_segments = [],[],[]
        for sentence in tqdm(sentences):
            inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=16, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])        
        #, np.asarray(input_segments, dtype='int32')
        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

class DistilBertModel:
    #split and assign each input to vocab id, include generating input mask
    def create_model(max_seq_len):    
        config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
        config.output_hidden_states = False
        
        transformer_model=TFDistilBertForSequenceClassification.from_pretrained(distil_bert, config =config) 
        
        input_ids = tf.keras.layers.Input(shape=(256,), name='input_token', dtype='int32')
        input_masks_ids = tf.keras.layers.Input(shape=(256,), name='masked_token', dtype='int32')
        
        X = transformer_model(input_ids, input_masks_ids)[0]
        
        X = tf.keras.layers.Dropout(0.2)(X)
        X = tf.keras.layers.Dense(2, activation='softmax')(X)
        model = tf.keras.Model(inputs=[input_ids, input_masks_ids], outputs = X)
        for layer in model.layers[:2]:
            layer.trainable = False

        model.build(input_shape=(None, max_seq_len))
        
        return model

    def create_learning_rate_scheduler(max_learn_rate=5e-5, end_learn_rate=1e-7, warmup_epoch_count=10, total_epoch_count=90):
        def lr_scheduler(epoch):
            if epoch < warmup_epoch_count:
                res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
            else:
                res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
            return float(res)
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

        return learning_rate_scheduler
    

distilBert = DistilBertModel()

count = 0

training_data_x = ["hello world, example tokenisation"]

token_train_x = tokenize(training_data_x, FullTokenizer)
print(token_train_x)
token_test_x = tokenize(testing_data_x, FullTokenizer)

total_epoch_count = 5

config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
config.output_hidden_states = False
    
#model = TFDistilBertForSequenceClassification.from_pretrained(distil_bert, num_labels=2) 

print('creating model')
model = distilBert.create_model(256)
print('created model')

model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='acc')])

model.summary()

log_dir = 'log/detect_the_intent/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
lr_schedule_callback = create_learning_rate_scheduler(max_learn_rate=1e-4, end_learn_rate=1e-7, warmup_epoch_count=0, total_epoch_count=5)

print('fitting models')

history = model.fit(
    x=token_train_x,
    y=training_data_y,
    validation_data=(token_test_x, testing_data_y),
    batch_size=32,
    epochs=total_epoch_count,
    callbacks=[lr_schedule_callback, tensorboard_callback]
    )

model.save_pretrained('/distilBertModelNew/my_model')


def text_preproc(x):
        x = x.lower()
        x = x.encode('ascii', 'ignore').decode()
        x = re.sub(r'https*\S+', ' ', x)
        x = re.sub(r'@\S+', ' ', x)
        x = re.sub(r'#\S+', ' ', x)
        x = re.sub(r'\'\w+', '', x)
        x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
        x = re.sub(r'\w*\d+\w*', '', x)
        x = re.sub(r'\s{2,}', ' ', x)
        return x

content = [
        "Good after you’ve served your time you should have ALL your rights restored including the right to VOTE.",
        "The United States civil war. Those who sought to preserve the union were later proven to be on the right side of history.",
        "I believe donald is our president for good reason and shall remain that way",
        "This is so spot on. In large part because Democrats actually do act like small spoiled children.",
        "people need to stop being offended all the damn time, trump is making america great again",
        "donald is making america great again, what are you doing about it commie",
        "I believe the right to free healthcare is a given",
        "trump is a con man he stole from america",
        "fools and their money are soon parted",
        "trumps america is a scam he is a con man who has ruined this country",
        "black lives matter is a racist movement",
        "bernie will fix this country",
        "bernie sanders was supposed to be the one to help this country get rid of the debt caused by education system",
        "you cant take my guns i dare you to try",
        "we need to seperate church and state once and for all",
        "ive been waiting all day for biden to start signing these orders i remember being heartbroken hearing about trumps first executive orders its nice to have a change in the right direction",
        "Voter id is racist, but I got IDd multiple times yesterday to get into Arlington National Cemetery.",
        "They likely don’t have the vaccine. Just hypocritical at every turn",
        "Liberal logic in a nutshell"]

sentences = []

for sent in content:
    sentences.append(text_preproc(sent))

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

model.save('fresh-reddit-newer-bot.h5')