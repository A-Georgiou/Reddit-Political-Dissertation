import praw
import pandas as pd
import re
import string
import os
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer

# SECURITY NOTE: Reddit API credentials are now loaded from environment variables
# Set the following environment variables before running this script:
# export REDDIT_CLIENT_ID="your_reddit_app_client_id"
# export REDDIT_CLIENT_SECRET="your_reddit_app_client_secret" 
# export REDDIT_USERNAME="your_reddit_username"
# export REDDIT_PASSWORD="your_reddit_password"
# 
# Alternatively, you can create a .env file with these variables and load them using python-dotenv

conservative_subreddits = ['LouderWithCrowder', 'Capitalism', 'AskThe_Donald', 'HillaryForPrison', 'tucker_carlson', 'anarcho_capitalism', 'republicans', 'walkaway', 'NEWPOLITIC', 'SocialJusticeInAction','conservative', 'conservatives']
liberal_subreddits = ['ShitLiberalsSay', 'LateStageCapitalism', 'Socialism_101', 'BreadTube', 'socialism', 'communism', 'Communism101', 'Anarchism', 'Anarchy101', 'COMPLETEANARCHY', 'DemocraticSocialism', 'DemocraticSocialism', 'ToiletPaperUSA', 'accidentallycommunist', 'antifastonetoss', 'TheRightCantMeme' , 'AntifascistsofReddit', 'LateStageImperialism', 'capitalism_in_decay', 'SocialistRA', 'ABoringDystopia', 'ENLIGHTENEDCENTRISM', 'DankLeft', 'Liberal']
centre_left_subs = ['JoeBiden', ]
centre_left_subs = ['politics', 'neoliberal']

"""
Class: DataFinder
Usage: Used to pull and pre-process data from Reddit for dataset
Input (chosen_subr): The subreddit to pull data from
"""
class DataFinder:
    def __init__(self, chosen_subr):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID", "YOUR_CLIENT_ID_HERE"),  # Replace with your Reddit app client ID
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE"),  # Replace with your Reddit app client secret
            password=os.getenv("REDDIT_PASSWORD", "YOUR_REDDIT_PASSWORD_HERE"),  # Replace with your Reddit account password
            user_agent="Political Analysis Data",
            username=os.getenv("REDDIT_USERNAME", "YOUR_REDDIT_USERNAME_HERE"),  # Replace with your Reddit username
             check_for_async=False 
            )
        self.subreddit_list = chosen_subr
    
    #Pulls top data from past year from reddit given set comment guidelines reached
    def extract_comment(self, subreddit):
        comments_arr = []    
        subreddit_data = self.reddit.subreddit(subreddit)
        hot_data = subreddit_data.top("year", limit = None)
        count = 0
        for submission in hot_data:       
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                if(comment.score > 10 and len(comment.body) > 15 and len(comment.body) < 500):
                    comment_clean = self.text_preproc(comment.body)
                    comments_arr.append(comment_clean)
            count += 1
            print(count, len(comments_arr))
        return comments_arr
     
    #Pre-process data stream coming in from Reddit, removes a lot of reddit specific syntax, utilises custom stopwords for politics
    def text_preproc(self, x):
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
    
    #Data combiner used to combine and extract comments from csv file
    def pull_data(self):
        comment_list = []
        for subreddit in self.subreddit_list:
            print(subreddit)
            comments = self.extract_comment(subreddit)
            comment_list.extend(comments)
            print(subreddit, len(comments))
            dataFrame = pd.DataFrame(comments, columns=['comment'])
            dataFrame.to_csv(subreddit+'.csv', index=False)

"""
Class: DataCombiner
Usage: Used to generate CSV Files and combine CSV files, mainly for processing and flattening data once it has been collected

"""

class DataCombiner:
    def __init__(self, folders):
        self.folders = folders
    
    def generateFolders(self):
        count = 0
        data = pd.DataFrame()
        for folder in self.folders:
            for filename in os.listdir(folder):
                if filename.endswith(".csv"): 
                    file = pd.read_csv(folder+'\\'+filename)
                    file['sentiment'] = count
                    print(file)
                    data = data.append(file)
                    continue
                else:
                    continue
            count += 1
        data.to_csv('combined_output_newer.csv', index=False)
        
    def viewData(self):
        data = pd.read_csv('combined_output_newer.csv')
        df = data.groupby('sentiment').nunique()
        print(df)
        min_index = np.argmin(df['comment'])
        for i in range(len(df)):
            if(i != min_index):
                to_remove = np.random.choice(data[data['sentiment']==i].index, size=(df['comment'][i] - df['comment'][min_index]), replace=False)
                data = data.drop(to_remove)
        df = data.groupby('sentiment').nunique()
        data.to_csv('combined_output_newer_flat.csv', index=False)
    
    def generateFoldersData(self):
        count = 0
        data = pd.DataFrame()
        for folder in self.folders:
            for filename in os.listdir(folder):
                if filename.endswith(".csv"): 
                    file = pd.read_csv(folder+'\\'+filename)
                    file['sentiment'] = count
                    data = data.append(file)
                    continue
                else:
                    continue
            count += 1
        return data

#additional_politics_english_stop = ['www', 'things', 'does', 'x200b', 'amp', 'want', 'watch', 'just', 'like', 'https', 'com', 'people', 'says', 'say', 'did', 'this', 'conservative', 'conservatives' ]
#additional_english_stop = ['www', 'things', 'does', 'x200b', 'amp', 'just', 'like', 'https', 'com', 'watch', 'want','says', 'say', 'did', 'this']

personal_stopwords = ["people", "don't", "dont", "a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]

class DataReader:
    def __init__(self, dataset):
        self.file = dataset
    
    #Generate Word Distribution List
    def generateWordDist(self):
        #Tokenizer to split words
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = stopwords.words('english')
        #Append additional stopwords which are not features in nltk library
        stop_words.extend(personal_stopwords)
        tokens = []
        for line in self.file['comment']:
            toks = tokenizer.tokenize(str(line))
            toks = [t.lower() for t in toks if t.lower() not in stop_words] #Create list with words not in stop_words list
            tokens.extend(toks)
        return tokens


#Test simple functionality of PRAW API scraping data for different subreddits
def testFunctionality():
	#df = DataFinder(conservative_subreddits)
    #comments = df.pullData()
    #print(comments)
    
    files = ['E:\HuggingFaceTraining\FarLeftWingData', 'E:\HuggingFaceTraining\FarRightWingData']
    dc = DataCombiner(files)
    dc.generateFolders()
    data = dc.viewData()
  
    
    data = pd.read_csv("E:\HuggingFaceTraining\datasets\combined_output_newer_flat.csv")
    
    
    right_data = data[data.sentiment == 1]
    left_data = data[data.sentiment == 0]
    
    dr_right = DataReader(right_data)
    output_tokens = dr_right.generateWordDist()
    
    dr_left = DataReader(left_data)
    output_tokens_left = dr_left.generateWordDist()
    
    pos_freq_left = nltk.FreqDist(output_tokens_left)
    most_common_left = pos_freq_left.most_common(1500)
    
    pos_freq_right = nltk.FreqDist(output_tokens)
    most_common_right = pos_freq_right.most_common(1500)
    
    output_list = []
    print(len(most_common_right))
    print(len(most_common_left))
    
    for i, word in enumerate(most_common_right):
        output_list.append([word, 1])
        output_list.append([most_common_left[i], 0])
    
    output = pd.DataFrame(output_list)
    output.to_csv('word_freq_new.csv', index=False)

if __name__ == '__main__':
    testFunctionality()

    
    
    
    
    
    
    