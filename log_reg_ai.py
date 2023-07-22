#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install google-api-python-client oauth2client gspread


# In[2]:


import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set up the credentials
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name('/Users/benedictadadson/Downloads/central-octane-393606-73538ceb126c.json', scope)
client = gspread.authorize(credentials)

sheet_url = 'https://docs.google.com/spreadsheets/d/1tw8AYqIJTSDJan2cgVSrzXwdSei7ToK7Y8WJmc8SDPc/edit#gid=860282713'
sheet = client.open_by_url(sheet_url)


# In[3]:


import pandas as pd
import numpy as np

# Select the worksheet you want to read data from
worksheet = sheet.get_worksheet(0)  # Replace 0 with the index of the worksheet you want to read from (0-indexed)

# Read the data from the worksheet
data = worksheet.get_all_values()

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data[1:], columns=data[0])  # Assuming the first row contains column headers
# Adding the 'ID' column with values corresponding to the row numbers starting with "1"
df['ID'] = np.arange(1, len(df) + 1)

df.head()


# In[4]:


# Convert the 'created_utc' column to datetime objects
df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')

# Format the 'created_utc' column as mm/dd/yy
df['created_utc'] = df['created_utc'].dt.strftime('%m/%d/%y')

print(df.head())


# In[5]:


df.head()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[7]:


print(df.shape)


# In[8]:


# Convert 'created_utc' to datetime format
df['created_utc'] = pd.to_datetime(df['created_utc'])

# Extract the year from 'created_utc' and save it to a new column 'year'
df['year'] = df['created_utc'].dt.year
#df['ID'] = np.arange(1, len(df) + 1)

# Display the DataFrame with the new 'year' column
df.head()


# In[9]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download the required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Assuming you have a DataFrame called 'df' containing your data
# Combine 'title' and 'selftext' into a single text column
df['text'] = df['title'] + ' ' + df['selftext']

# Drop the original 'title' and 'selftext' columns
df.drop(columns=['title', 'selftext'], inplace=True)

# Define the text preprocessing function
def preprocess_text(text):
    # Remove punctuation using regular expressions
    import re
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text into individual words
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the lemmatized words back into a single text string
    processed_text = ' '.join(words)
    return processed_text

# Apply text preprocessing to the 'text' column
df['text'] = df['text'].apply(preprocess_text)

# Now the 'text' column contains the preprocessed text data
print(df['text'])


# In[10]:


df.head()


# In[11]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[12]:


nltk.download('vader_lexicon')


# In[13]:


# Function to classify sentiment as 0 (negative) or 1 (positive)
def get_sentiment(text):
    polarity_scores = sia.polarity_scores(text)
    return 1 if polarity_scores['compound'] >= 0 else 0

# Apply sentiment analysis to the 'text' column and create a new 'sentiment' column
df['sentiment'] = df['text'].apply(get_sentiment)

print(df)


# In[14]:


df.head()


# In[15]:


# Rename the column 
df.rename(columns={'sentiment': 'target'}, inplace=True)

df.head()


# In[16]:


df.columns


# In[17]:


df.info()


# In[18]:


import seaborn as sns
sns.countplot(x='target', data=df)


# In[19]:


data = df[['text','target']]


# In[20]:


data.head()


# In[21]:


data['target'] = data['target'].replace(4,1)


# In[22]:


data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]


# In[23]:


data_pos = data_pos.iloc[:int(20000)]
data_neg = data_neg.iloc[:int(20000)]


# In[24]:


dataset = pd.concat([data_pos, data_neg])


# In[25]:


#separate input features and labels
X = data.text
y = data.target


# In[26]:


# Separating the 95% data for training data and 5% for testing data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =42)


# In[27]:


vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))


# In[28]:


X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


# In[29]:


from sklearn.model_selection import cross_val_predict, cross_val_score

def model_Evaluate(model, X, y):
    # Perform k-fold cross-validation
    y_pred = cross_val_predict(model, X, y, cv=5)  # Use cv=5 or other desired number of folds

    # Print the evaluation metrics for the dataset.
    print("Classification Report:")
    print(classification_report(y, y_pred))
    
    # Calculate and print accuracy percentage
    accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Accuracy: {accuracy.mean()*100:.2f}%")

    # Compute precision and F1 score
    precision = cross_val_score(model, X, y, cv=5, scoring='precision')
    f1 = cross_val_score(model, X, y, cv=5, scoring='f1')

    # Print precision and F1 score
    print(f"Precision: {precision.mean():.2f}")
    print(f"F1 Score: {f1.mean():.2f}")

    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)

# Initialize and fit the model using the entire training data with stronger L2 regularization
LRmodel = LogisticRegression(C=0.02, max_iter=1000, n_jobs=-1, penalty='l2')  # Try different values of C
LRmodel.fit(X_train, y_train)

# Evaluate the model using cross-validation
model_Evaluate(LRmodel, X_train, y_train)

# Make predictions on the test set
y_pred = LRmodel.predict(X_test)


# In[ ]:




