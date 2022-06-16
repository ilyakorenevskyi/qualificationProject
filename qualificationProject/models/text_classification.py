from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import model_selection, preprocessing
import tensorflow as tf
import pickle
import pandas
import csv
from keras import models
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input


def prepare_dataset(df):
    for x in range(len(df['article'])):
        df['article'][x] = df['article'][x].replace(u'\xa0', u' ')
        df['article'][x] = df['article'][x].replace(u'\n\n', u'\n')
        df['article'][x] = df['article'][x].replace(u'\n', u' ')
        df['article'][x] = df['article'][x].replace(u'\t', u' ')
        df['article'][x] = df['article'][x].replace(u'.', u'. ')
        df['article'][x] = df['article'][x].replace(u'  ', u' ')
    return df


def vectorize(train_x, valid_x, train_y):
    kwargs = {
        'ngram_range': (1, 2),
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': 'word',
        'min_df': 2,
    }

    vectorizer = TfidfVectorizer(**kwargs)
    tfidf_train_x = vectorizer.fit_transform(train_x)
    tfidf_valid_x = vectorizer.transform(valid_x)

    selector = SelectKBest(f_classif, k=min(30000, tfidf_train_x.shape[1]))
    selector.fit(tfidf_train_x, train_y)
    tfidf_train_x = selector.transform(tfidf_train_x).todense()
    tfidf_valid_x = selector.transform(tfidf_valid_x).todense()

    return tfidf_train_x, tfidf_valid_x, vectorizer, selector


def create_model(layers, units, dropout_rate, input_shape, classes_count):
    model = models.Sequential()
    model.add(Input(shape=input_shape))
    for _ in range(layers):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=classes_count, activation='softmax'))
    return model


def save_model(vectorizer, selector, model):
    pickle.dump(vectorizer.vocabulary_, open("feature.pkl", "wb"))
    pickle.dump(selector, open("selector.pkl", "wb"))
    model.save("classifier/")


df = pandas.read_csv('DataSet.csv', sep=";")
df = prepare_dataset(df)

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['article'], df['category'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

tfidf_train_x, tfidf_valid_x, vectorizer, selector = vectorize(train_x, valid_x, train_y)

learning_rate = 1e-3
loss = 'sparse_categorical_crossentropy'
epochs = 20
batch_size = 32

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model = create_model(2, 64, 0.2, tfidf_train_x.shape[1:], len(encoder.classes_))
model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]
history = model.fit(tfidf_train_x, train_y, epochs=epochs, callbacks=callbacks,
                    validation_data=(tfidf_valid_x, valid_y), verbose=2, batch_size=batch_size)
history = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
