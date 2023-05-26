import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
import pandas as pd
import random
import time
import string
from collections import Counter
import re
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
VOCAB_SIZE = 20000
DEVICE = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('reduced_dataset.csv')
data.columns = ['LYRICS', 'LABELS']
data.to_csv('reduced_dataset.csv', index=None)
data = pd.read_csv('reduced_dataset.csv')

# Training size: 2000
# Validation size: 500
# Testing size: 500

# Data cleaning!

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
data['LYRICS'] = data['LYRICS'].apply(remove_punctuation)

all_text = ' '.join(data['LYRICS'])
phrases = re.split(r'\W+', all_text)
phrase_counts = Counter(phrases)
common_phrases = phrase_counts.most_common(20)
common_phrases_set = set([phrase for phrase, count in common_phrases])

def remove_common_phrases(text):
    words = text.split()
    return ' '.join([word for word in words if word not in common_phrases_set])

data['LYRICS'] = data['LYRICS'].apply(remove_common_phrases)

# All punctuation and the 25 most common phrases (usually phrases like "I" or "there") are removed

# We now need to extract idioms from the data before we
# tokenize everything. And to do that, we need an idiom
# classifier. The first step is to import a set of
# labelled idioms. 

idioms = pd.read_csv('idioms.csv')

# create network to look at all idioms, identify them, and
# look for them specifically in each song instance

# In the song, could look at all idioms, average their
# tokenized values, and create a single sentiment vector
# for each song. Should be weighted against how many words
# are in the song/how much of the song is composed of the
# detected idioms

def extract_idioms():
    idiom_vector = []
    for song in data['LYRICS']:

        phrase_vector = []
        song_length = len(song)
        for phrase in idioms:
            if phrase[1] in song:
                phrase_count = song.count(phrase)
                phrase_vector.append[phrase[1]]
                # phrase_vector.append[phrase[1], phrase[3], phrase[4], phrase[5]]
                song.replace(phrase, ' ', phrase_count)
        '''
        new_length = len(song)
        len_diff_ratio = (song_length - new_length)/song_length
        phrase_vector[:2] = phrase_vector[:2]/()
        '''
        # Ideally we would have an algorithm that weighted
        # the volume of idioms in each song to how much
        # sentiment they contribute to the song overall

    idiom_vector.append(phrase_vector)
    assert len(idiom_vector) == len(data)
    data['IDIOMS'] = idiom_vector # Create new column to store data


nlp = English()
tokenizer = Tokenizer(nlp.vocab)

def tokenize_with_idioms(text):
    # Tokenize the text using spacy tokenizer
    tokens = [token.text for token in tokenizer(text)]
    
    # Add the extracted idioms to the token list
    tokens += extract_idioms(text)  # Replace extract_idioms with your idiom extraction function
    
    return tokens

LYRICS = torchtext.legacy.data.Field(
    tokenize=tokenize_with_idioms,
    tokenizer_language='en_core_web_sm'
)
LABELS = torchtext.legacy.data.LabelField(dtype=torch.long)

fields = [('LYRICS', LYRICS), ('LABELS', LABELS)]

data = torchtext.legacy.data.TabularDataset(
    path='dataset.csv', format='csv', 
    skip_header=True, fields=fields
)

LYRICS.build_vocab(data, max_size=VOCAB_SIZE)
LABELS.build_vocab(data)

train_data, valid_data, test_data = data.split(
    split_ratio=[4/6, 1/6, 1/6],
    random_state=random.seed(RANDOM_SEED)
)

batch_size = 128

train_dataloader, valid_dataloader, test_dataloader = \
    torchtext.legacy.data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=False,
    sort_key=lambda x: len(x.LYRICS),
    device=DEVICE
    )

# Define the LSTM network
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_lyrics = nn.Embedding(input_size, embedding_size)
        self.embedding_idioms = nn.Embedding(input_size, embedding_size)  # Add embedding layer for idioms
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # Multiply hidden_size by 2 to account for lyrics and idioms
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden_size by 2 to account for lyrics and idioms

    def forward(self, lyrics, idioms):
        embedded_lyrics = self.embedding_lyrics(lyrics)
        embedded_idioms = self.embedding_idioms(idioms)  # Embed the idioms

        out_lyrics, (hidden_lyrics, cell_lyrics) = self.lstm(embedded_lyrics)
        out_idioms, (hidden_idioms, cell_idioms) = self.lstm(embedded_idioms)  # Pass idioms through LSTM

        hidden_lyrics.squeeze_(0)
        hidden_idioms.squeeze_(0)

        combined_hidden = torch.cat((hidden_lyrics, hidden_idioms), dim=1)  # Concatenate lyrics and idioms

        out = self.batch_norm(combined_hidden)  # Apply batch normalization
        out = self.fc(out)
        return out

# Set hyperparameters
input_size = len(LYRICS.vocab)  # Number of input features (1 because we have one column)
output_size = 128  # Number of output values (1 because we have one target column)
hidden_size = 256
num_layers = 32
num_epochs = 20
learning_rate = 0.008
embedding_size = 256

# Create the LSTM network
torch.manual_seed(RANDOM_SEED)
model = LSTMNetwork(input_size, embedding_size, hidden_size, output_size)
model = model.to(DEVICE)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_func = nn.CrossEntropyLoss()

def train(model, train_dataloader, device):

    with torch.no_grad():

        correct_predict, num_examples = 0,0

        for i, (features, targets) in enumerate(train_dataloader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)

            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_predict += (predicted_labels == targets).sum()

        return correct_predict.float()/num_examples * 100


start_time = time.time()

for epoch in range(num_epochs):
    model.train()

    for batch_id, batch_data in enumerate(train_dataloader):
            
            lyrics = batch_data.LYRICS.to(DEVICE)
            labels = batch_data.LABELS.to(DEVICE)
            idioms = batch_data.IDIOMS.to(DEVICE)

            logits = model(lyrics, idioms)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            if not batch_id % 4:
            # Print the loss for monitoring
                print(f'Epoch [{epoch + 1:03d}/{num_epochs:03d}] | '
                      f'Batch {batch_id:03d} /{len(train_dataloader):03d} | '
                      f'Loss: {loss:.4f}')

    with torch.set_grad_enabled(False):
        print(f'Training Accuracy: {train(model, train_dataloader, DEVICE):.2f}%' 
              f'\nvalid accuracy: '
              f'{train(model, valid_dataloader, DEVICE):.2f}%')
        
    print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')

print(f'Total training time: {(time.time() - start_time)/60:.2f} min')
print(f'Test accuracy: {train(model, test_dataloader, DEVICE):.2f}%')
