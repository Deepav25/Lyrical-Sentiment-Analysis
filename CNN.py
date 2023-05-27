import pandas as pd
import jieba
from torch.utils import data as Data
import torch
import numpy as np
import torch.nn as nn
device = torch.device( "cpu")
import torch.optim as optim
from sklearn.model_selection import train_test_split

df = pd.read_csv('reduced_dataset.csv')
texts = df['Lyric'].values
labels = df['Label'].values
print(df['Label'].unique())
def cal_base_date():
    test=pd.DataFrame({'text':texts,
                      'label':labels})

    return test

def cal_clear_word(test):
    stoplist = [' ', '\n', '，']

    def function(a):
        word_list = [w for w in jieba.cut(a) if w not in list(stoplist)]
        return word_list

    test['text'] = test.apply(lambda x: function(x['text']), axis=1)
    return test

def cal_update_date(test, sequence_length):
    def prepare_sequence(seq):
        idxs = [w for w in seq]
        if len(idxs) >= sequence_length:
            idxs = idxs[:sequence_length]
        else:
            pad_num = sequence_length - len(idxs)
            for i in range(pad_num):
                idxs.append('UNK')
        return idxs

    test['text'] = test.apply(lambda x: prepare_sequence(x['text']), axis=1)
    return test

def cal_word_to_ix(test):
    word_to_ix = {}
    for sent, tags in test.values:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return idxs

    test['text'] = test.apply(lambda x: prepare_sequence(x['text'], word_to_ix), axis=1)
    return test, len(word_to_ix)

class MyDataset(Data.Dataset):
    def __init__(self,test):
        self.Data=test['text']
        self.Label=test['label']

    def __getitem__(self, index):
        #把numpy转换为Tensor
        txt=torch.from_numpy(np.array(self.Data[index])).float()
        label=torch.tensor(np.array(self.Label[index]))
        return txt,label

    def __len__(self):
        return len(self.Data)

class TextCNN(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.conv = nn.Sequential(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(1, 3, (2, embedding_size)),
            nn.BatchNorm2d(3,track_running_stats=False),
            nn.ReLU(),
            # pool : ((filter_height, filter_width))
            nn.MaxPool2d((2, 1)),

        )
        # fc
        self.fc = nn.Sequential(nn.Linear(147, num_classes))

    def forward(self, X):
        batch_size = X.shape[0]
        embedding_X = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(
            1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X)  # [batch_size, output_channel, 1, 1]
        flatten = conved.view(batch_size, -1)  # [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output



base_df = cal_base_date()
return_df = cal_clear_word(base_df)
sequence_length = 100
return_df = cal_update_date(return_df, sequence_length)
return_df, vocab_size = cal_word_to_ix(return_df)

data= return_df['text'].to_numpy()
label= return_df['label'].to_numpy() - 1



train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size=0.2, random_state=42)
traindata=pd.DataFrame({
    "text":train_data,
    "label":train_labels
})

testdata=pd.DataFrame({
    "text":test_data,
    "label":test_labels
})


Train=MyDataset(traindata)
Test=MyDataset(testdata)
batch_size = 32
train_loader = Data.DataLoader(Train,batch_size,shuffle=True)
test_loader = Data.DataLoader(Test,batch_size,shuffle=False)
embedding_size =2
num_classes = 10
model = TextCNN(embedding_size,num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
model.train()
for epoch in range(300):
    total=0
    correct=0
    total_loss=0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device).long(), batch_y.to(device).long()
        pred = model(batch_x)
        _, predicted = torch.max(pred.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        loss = criterion(pred, batch_y)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = 100 * correct / total
    if (epoch + 1) % 10 == 0:
        print(f"Train Epoch:{epoch + 1},Train Accuracy：{accuracy}% loss ={total_loss / len(train_loader):.6f}")
        with torch.no_grad():
            correct = 0
            total = 0
            total_loss = 0
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device).long(), batch_y.to(device).long()
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
            accuracy = 100 * correct / total

            print(f"Test Epoch:{epoch + 1},Test Accuracy：{accuracy}%,Test Loss:{total_loss / len(test_loader):.4f}")
