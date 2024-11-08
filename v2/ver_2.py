# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:00.400040Z","iopub.execute_input":"2024-11-07T18:01:00.400501Z","iopub.status.idle":"2024-11-07T18:01:00.406307Z","shell.execute_reply.started":"2024-11-07T18:01:00.400450Z","shell.execute_reply":"2024-11-07T18:01:00.405366Z"}}
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:00.408293Z","iopub.execute_input":"2024-11-07T18:01:00.409167Z","iopub.status.idle":"2024-11-07T18:01:02.035195Z","shell.execute_reply.started":"2024-11-07T18:01:00.409121Z","shell.execute_reply":"2024-11-07T18:01:02.034201Z"}}
# Load the pre-trained model
model = load_model('/kaggle/input/lstm_txt_gen/keras/default/1/text_generator_model_v1.keras')

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.037349Z","iopub.execute_input":"2024-11-07T18:01:02.038143Z","iopub.status.idle":"2024-11-07T18:01:02.074297Z","shell.execute_reply.started":"2024-11-07T18:01:02.038097Z","shell.execute_reply":"2024-11-07T18:01:02.073443Z"}}
# Load the pre-trained tokenizer
with open('/kaggle/input/lstm_txt_gen/keras/default/1/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.075521Z","iopub.execute_input":"2024-11-07T18:01:02.075824Z","iopub.status.idle":"2024-11-07T18:01:02.088551Z","shell.execute_reply.started":"2024-11-07T18:01:02.075793Z","shell.execute_reply":"2024-11-07T18:01:02.087630Z"}}
# Load and preprocess your new text data
with open('/kaggle/input/text-data/poem.txt', 'r') as f:
    data = f.read()

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.090669Z","iopub.execute_input":"2024-11-07T18:01:02.090956Z","iopub.status.idle":"2024-11-07T18:01:02.121587Z","shell.execute_reply.started":"2024-11-07T18:01:02.090926Z","shell.execute_reply":"2024-11-07T18:01:02.120834Z"}}
tokenizer.fit_on_texts([data])
total_words = len(tokenizer.word_index) + 1

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.122809Z","iopub.execute_input":"2024-11-07T18:01:02.123166Z","iopub.status.idle":"2024-11-07T18:01:02.349691Z","shell.execute_reply.started":"2024-11-07T18:01:02.123127Z","shell.execute_reply":"2024-11-07T18:01:02.348690Z"}}
input_sequences = []
for line in data.split('\n'):
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    input_sequences.append(n_gram_sequence)

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.351086Z","iopub.execute_input":"2024-11-07T18:01:02.351931Z","iopub.status.idle":"2024-11-07T18:01:02.417793Z","shell.execute_reply.started":"2024-11-07T18:01:02.351885Z","shell.execute_reply":"2024-11-07T18:01:02.416906Z"}}
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.419038Z","iopub.execute_input":"2024-11-07T18:01:02.419452Z","iopub.status.idle":"2024-11-07T18:01:02.461488Z","shell.execute_reply.started":"2024-11-07T18:01:02.419408Z","shell.execute_reply":"2024-11-07T18:01:02.460689Z"}}
x , y = input_sequences[:,:-1] , input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.462752Z","iopub.execute_input":"2024-11-07T18:01:02.463430Z","iopub.status.idle":"2024-11-07T18:01:02.468342Z","shell.execute_reply.started":"2024-11-07T18:01:02.463385Z","shell.execute_reply":"2024-11-07T18:01:02.467434Z"}}
num_classes = y.shape[1]
print("Number of classes:", num_classes)

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.469778Z","iopub.execute_input":"2024-11-07T18:01:02.470352Z","iopub.status.idle":"2024-11-07T18:01:02.492262Z","shell.execute_reply.started":"2024-11-07T18:01:02.470293Z","shell.execute_reply":"2024-11-07T18:01:02.491600Z"}}
model.pop() 
model.add(Dense(num_classes, activation='softmax', name="output_layer"))

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.495370Z","iopub.execute_input":"2024-11-07T18:01:02.495647Z","iopub.status.idle":"2024-11-07T18:01:02.503583Z","shell.execute_reply.started":"2024-11-07T18:01:02.495619Z","shell.execute_reply":"2024-11-07T18:01:02.502771Z"}}

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:01:02.505084Z","iopub.execute_input":"2024-11-07T18:01:02.505485Z","iopub.status.idle":"2024-11-07T18:05:51.282759Z","shell.execute_reply.started":"2024-11-07T18:01:02.505445Z","shell.execute_reply":"2024-11-07T18:05:51.281769Z"}}
# Train the model and store the history
history = model.fit( x , y , epochs=30 , validation_data=(x , y))

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training & validation accuracy values if available
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:05:51.284174Z","iopub.execute_input":"2024-11-07T18:05:51.284513Z","iopub.status.idle":"2024-11-07T18:05:51.381731Z","shell.execute_reply.started":"2024-11-07T18:05:51.284480Z","shell.execute_reply":"2024-11-07T18:05:51.380909Z"}}
# Save the fine-tuned model
model.save('/kaggle/working/text_generator_finetuned.keras')


# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:05:51.382882Z","iopub.execute_input":"2024-11-07T18:05:51.383192Z","iopub.status.idle":"2024-11-07T18:05:51.396861Z","shell.execute_reply.started":"2024-11-07T18:05:51.383160Z","shell.execute_reply":"2024-11-07T18:05:51.396149Z"}}
with open ('/kaggle/working/tokenizer2.pickle' , 'wb') as f:
  pickle.dump(tokenizer , f )

# %% [code] {"execution":{"iopub.status.busy":"2024-11-07T18:05:51.397820Z","iopub.execute_input":"2024-11-07T18:05:51.398092Z","iopub.status.idle":"2024-11-07T18:05:51.418877Z","shell.execute_reply.started":"2024-11-07T18:05:51.398062Z","shell.execute_reply":"2024-11-07T18:05:51.418006Z"}}
model.summary()