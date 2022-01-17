# Joint_Learning_Classification_Persian_Reviews-
Joint Learning for Aspect and Polarity Classification in Persian Reviews Using Multi-Task Deep Learning



# CNN

```python
embedding_size = 300
input_1 = Input(shape=(max_tokens,))
x=Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    name='embedding_layer',
            embeddings_initializer=initializer)(input_1)
x=Dropout(0.2,seed=sd)(x)
x=Conv1D(256,kernel_size=3,padding='same',activation='relu',strides=1
       ,kernel_initializer=initializer,use_bias=False)(x)
x=BatchNormalization()(x)
x=GlobalMaxPooling1D()(x)

x=Dense(200, activation='relu',
        kernel_initializer=initializer,use_bias=False)(x)
x=Dropout(0.2,seed=sd)(x)
output1 = Dense(3, activation='softmax',name='bazigar')(x)
output2 = Dense(3, activation='softmax',name='bazigari')(x)
output3 = Dense(3, activation='softmax',name='dastan')(x)
output4 = Dense(3, activation='softmax',name='dialog')(x)
output5 = Dense(3, activation='softmax')(x)
output6 = Dense(3, activation='softmax')(x)
output7 = Dense(3, activation='softmax')(x)
output8 = Dense(3, activation='softmax')(x)
output9 = Dense(3, activation='softmax')(x)
model = Model(inputs=input_1, outputs=[output1, output2, output3, output4, output5, output6,output7,output8,output9
                                      ])

optimizer =Nadam(learning_rate=1e-3)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,)
model.summary()
```
# LSTM

```python
embedding_size = 300
input_1 = Input(shape=(max_tokens,))
x=Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    name='embedding_layer',
            embeddings_initializer=initializer)(input_1)
x=Dropout(0.2,seed=sd)(x)
x=LSTM(units=256, return_sequences=True,kernel_initializer=initializer,use_bias=False)(x)
x=BatchNormalization()(x)
x=GlobalMaxPooling1D()(x)
x=Dense(200, activation='relu',
        kernel_initializer=initializer,use_bias=False)(x)
x=Dropout(0.2,seed=sd)(x)
output1 = Dense(3, activation='softmax',name='bazigar')(x)
output2 = Dense(3, activation='softmax',name='bazigari')(x)
output3 = Dense(3, activation='softmax',name='dastan')(x)
output4 = Dense(3, activation='softmax',name='dialog')(x)
output5 = Dense(3, activation='softmax')(x)
output6 = Dense(3, activation='softmax')(x)
output7 = Dense(3, activation='softmax')(x)
output8 = Dense(3, activation='softmax')(x)
output9 = Dense(3, activation='softmax')(x)
model = Model(inputs=input_1, outputs=[output1, output2, output3, output4, output5, output6,output7,output8,output9
                                      ])

optimizer =Nadam(learning_rate=1e-3)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,)
model.summary()
```

# GRU

```python
embedding_size = 300
input_1 = Input(shape=(max_tokens,))
x=Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    name='embedding_layer',
            embeddings_initializer=initializer)(input_1)
x=Dropout(0.2,seed=sd)(x)
x=GRU(units=256, return_sequences=True,kernel_initializer=initializer,use_bias=False)(x)
x=BatchNormalization()(x)
x=GlobalMaxPooling1D()(x)
x=Dense(200, activation='relu',
        kernel_initializer=initializer,use_bias=False)(x)
x=Dropout(0.2,seed=sd)(x)
output1 = Dense(3, activation='softmax',name='bazigar')(x)
output2 = Dense(3, activation='softmax',name='bazigari')(x)
output3 = Dense(3, activation='softmax',name='dastan')(x)
output4 = Dense(3, activation='softmax',name='dialog')(x)
output5 = Dense(3, activation='softmax')(x)
output6 = Dense(3, activation='softmax')(x)
output7 = Dense(3, activation='softmax')(x)
output8 = Dense(3, activation='softmax')(x)
output9 = Dense(3, activation='softmax')(x)
model = Model(inputs=input_1, outputs=[output1, output2, output3, output4, output5, output6,output7,output8,output9
                                      ])

optimizer =Nadam(learning_rate=1e-3)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,)
model.summary()
```

# Bi-LSTM

```python
embedding_size = 300
input_1 = Input(shape=(max_tokens,))
x=Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    name='embedding_layer',
            embeddings_initializer=initializer)(input_1)
x=Dropout(0.2,seed=sd)(x)
x=Bidirectional(LSTM(units=256, return_sequences=True,kernel_initializer=initializer,use_bias=False))(x)
x=BatchNormalization()(x)
x=GlobalMaxPooling1D()(x)
x=Dense(200, activation='relu',
        kernel_initializer=initializer,use_bias=False)(x)
x=Dropout(0.2,seed=sd)(x)
output1 = Dense(3, activation='softmax',name='bazigar')(x)
output2 = Dense(3, activation='softmax',name='bazigari')(x)
output3 = Dense(3, activation='softmax',name='dastan')(x)
output4 = Dense(3, activation='softmax',name='dialog')(x)
output5 = Dense(3, activation='softmax')(x)
output6 = Dense(3, activation='softmax')(x)
output7 = Dense(3, activation='softmax')(x)
output8 = Dense(3, activation='softmax')(x)
output9 = Dense(3, activation='softmax')(x)
model = Model(inputs=input_1, outputs=[output1, output2, output3, output4, output5, output6,output7,output8,output9
                                      ])

optimizer =Nadam(learning_rate=1e-3)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,)
model.summary()
```
# jacard (multi label accuracy)

```python
def jacard(y_test, predictions):
    accuracy = 0.0

    for i in range(y_test.shape[0]):
        intersection = 0.0
        union = 0.0
        for j in range(y_test.shape[1]):
            if int(y_test[i,j]) == 1 or int(predictions[i,j]) == 1:
                union += 1
            if int(y_test[i,j]) == 1 and int(predictions[i,j]) == 1:
                intersection += 1
            if int(y_test[i,j]) == 2 or int(predictions[i,j]) == 2:
                union += 1
            if int(y_test[i,j]) == 2 and int(predictions[i,j]) == 2:
                intersection += 1
            
        if union != 0:
            accuracy = accuracy + float(intersection/union)

    accuracy = float(accuracy/y_test.shape[0])

    return accuracy
```
# preprocess

```python
import re
def preprocess_text(sentence):
    # Removing multiple spaces
    sentence = re.sub(r'@\S+', '', sentence)
    sentence = re.sub(r'!\S+', '', sentence)
    sentence = re.sub(r'؟\S+', '', sentence)
    sentence = re.sub(r'[.]', ' ', sentence)
    sentence = re.sub(r'[/]', '', sentence)
    sentence = re.sub(r'[،]', ' ', sentence)
    sentence = re.sub(r'[؛]', '', sentence)
    sentence = sentence.split()
    sentence =[word for word in sentence if word not in stop_words]
    sentence = ' '.join(sentence)
    return sentence
def ReplacetwoMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1", s)
```

# شمارش تعداد کلمات منحصر به فرد

```python
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import FreqDist
all_words=' '.join(data)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
print ('number unique word:',num_unique_word)

```

# شمارش بزرگترین طول متن

```python
r_len=[]
for text in data:
    word=word_tokenize(text)
    l=len(word)
    r_len.append(l)   
MAX_REVIEW_LEN=np.max(r_len)
print('max len:',MAX_REVIEW_LEN)

```
