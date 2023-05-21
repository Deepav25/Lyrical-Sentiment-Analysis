import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
import pandas as pd
import random
import time

torch.backends.cudnn.deterministic = True

RANDOM_SEED = 123
VOCAB_SIZE = 20000
DEVICE = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('Lyrical-Sentiment-Analysis/dataset.csv')
data.columns = ['LYRICS', 'LABELS']
data.to_csv('dataset.csv', index=None)
data = pd.read_csv('Lyrical-Sentiment-Analysis/dataset.csv') # Data now has better column names

LYRICS = torchtext.legacy.data.Field(
    tokenize='spacy',
    tokenizer_language='en_core_web_sm'
)
LABELS = torchtext.legacy.data.LabelField(dtype=torch.long)

fields = [('LYRICS', LYRICS), ('LABELS', LABELS)]

data = torchtext.legacy.data.TabularDataset(
    path='dataset.csv', format='csv', 
    skip_header=True, fields=fields
)

full_data, used_data = data.split(
    split_ratio=[0.9, 0.1],
    random_state=random.seed(RANDOM_SEED)
)

other_data, test_data = used_data.split(
    split_ratio=[0.9, 0.1],
    random_state=random.seed(RANDOM_SEED)
)

train_data, valid_data = other_data.split(
    split_ratio=[0.85, 0.15],
    random_state=random.seed(RANDOM_SEED)
)

LYRICS.build_vocab(train_data, max_size=VOCAB_SIZE)
LABELS.build_vocab(train_data)

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
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)

        out, (hidden, cell) = self.lstm(embedded)
        hidden.squeeze_(0) 
        out = self.fc(hidden)
        return out

# Set hyperparameters
input_size = len(LYRICS.vocab)  # Number of input features (1 because we have one column)
output_size = 128  # Number of output values (1 because we have one target column)
hidden_size = 128
num_layers = 2
num_epochs = 10
learning_rate = 0.001
embedding_size = 128

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

            logits = model(lyrics)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            if not batch_id % 10:
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
