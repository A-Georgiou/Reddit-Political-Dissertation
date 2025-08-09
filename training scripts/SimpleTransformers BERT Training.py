from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

class BertModel:
    def train_model():
         # Preparing train data
        train_df = pd.read_csv('./datasets/combined_output_newer_flat.csv') 
        
        train_df.columns = ["text", "labels"]
        train_df["labels"] = train_df["labels"].astype('int32')

        train_df, eval_df =  train_test_split(train_df, shuffle=True)
        print(train_df)
        
        # Optional model configuration
        model_args = ClassificationArgs(num_train_epochs=5)
        model_args.overwrite_output_dir = True
        model_args.reprocess_input_data = True
        model_args.max_seq_length = 256
        
        
        #Evaluation during training
        model_args.evaluate_during_training = True
        model_args.evaluate_during_training_verbose = True
        
        steps_per_epoch = int(np.ceil(len(train_df) / float(model_args.train_batch_size)))
        model_args.evaluate_during_training_steps = steps_per_epoch*2
        
        model_args.use_cached_eval_features = False
        
        # Create a ClassificationModel
        model = ClassificationModel("bert", "bert-base-uncased", num_labels=2, args=model_args, use_cuda=True)
        
        # Train the model
        model.train_model(train_df, eval_df=eval_df)
        
        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(eval_df)
        
        # Make predictions with the model
        predictions, raw_outputs = model.predict(["911 was an inside job"])
        print(predictions)

    def load_model():
        model = ClassificationModel("bert", "outputs/best_model/", num_labels=4, use_cuda=True)
        content = ["trump is a con man he stole from america",
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
        actuals, pred = model.predict(content)
        for i, pred in enumerate(actuals):
            print(content[i], pred)

if __name__ == "__main__":
    BertModel.train_model()
   