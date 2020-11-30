import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels),  (test_data, test_labels) = data.load_data(num_words=88000)     # shrinking and loading data

word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# making the data only 251 words long
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index['<PAD>'], padding="post", maxlen=251)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding="post", maxlen=251)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# defining the model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))    # creating 10,000 word vectors for each word
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))    # rectified linear unit
model.add(keras.layers.Dense(1, activation="sigmoid"))  # sigmoid to get number between 0 - 1

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])    # loss function "binary_crossentropy" to calculate loss between 0-1

# splitting the data
x_validation = train_data[:10000]
x_train = train_data[10000:]
y_validation = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=51, batch_size=512, validation_data=(x_validation, y_validation), verbose=1)
# batch-size specifies how many reviews we want to load at a time

# saving a model
model.save("TextClassifier.h5")



def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


# model = keras.models.load_model("TextClassifier.h5")

with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace("'", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").strip()
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index['<PAD>'], padding="post", maxlen=251)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


