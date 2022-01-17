
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