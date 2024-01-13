import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r'E:\Bharat Intern\Task2\sms_spam_Collection.txt', sep='\t', names=['label', 'message'])

# Display the first few rows of the dataframe
print(df.head())

# Map labels to binary values
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split the data into features (X) and labels (y)
X = df['message'].values
y = df['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tokenize the text data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)
encoded_train = tokenizer.texts_to_sequences(X_train)
encoded_test = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
max_length = 20  
pad_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
pad_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')

# Get the vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Define the model architecture 
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    tf.keras.layers.GRU(32, return_sequences=False),  #  GRU
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model
model.fit(x=pad_train,
          y=y_train,
          epochs=10,
          validation_data=(pad_test, y_test),
          callbacks=[early_stop])

# Evaluate the model on the test set
preds = (model.predict(pad_test) > 0.5).astype("int32")
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {round(acc * 100, 2)}%")
#accuracy is 98.98%