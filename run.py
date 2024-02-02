import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import os
import argparse
from keras.preprocessing.sequence import pad_sequences

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trainortest', required=True)

args = vars(parser.parse_args())

trainortest = int(args['trainortest'])

#df = pd.DataFrame()
#df_test = pd.DataFrame()
# Example data
'''data = {
    'id': [1, 1, 1, 2, 2, 3, 3, 3],
    'event_id': [101, 102, 103, 201, 202, 301, 302, 303],
    'down_time': [4000, 5000, 6000, 4000, 10000, 4500, 5500, 7500],
    'other_feature': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    # Add other columns as needed
}

# Example label data
label_data = {
    'id': [1, 2, 3],
    'score': [3.5, 6.0, 7.4],
}'''
try:
    input_path = os.getcwd()
    train_logs_filename = input_path+"/train_logs.csv"
    train_scores_filename = input_path+"/train_scores.csv"
    test_logs_filename = input_path+"/test_logs.csv"
    model_path = input_path
    model_filename = model_path+"/lstmModel.h5"
except Exception as e:
    print(f"An unexpected error occurred setting paths: {e}")
# Create sequences for LSTM

# Function to add zero padding
def add_zero_padding(group, df):

    max_rows = df['id'].value_counts().max()
    current_rows = group.shape[0]
    if current_rows < max_rows:
        padding_rows = max_rows - current_rows
        zero_padding = pd.DataFrame([[group['id'].iloc[0]] + [0] * (df.shape[1] - 1)] * padding_rows, columns=df.columns)
        return pd.concat([group, zero_padding], ignore_index=True)
    else:
        return group

def add_zero_padding_to_test(group, df_temp):

    max_rows = df_temp['id'].value_counts().max()
    current_rows = group.shape[0]
    if current_rows < max_rows:
        padding_rows = max_rows - current_rows
        zero_padding = pd.DataFrame([[group['id'].iloc[0]] + [0] * (df_temp.shape[1] - 1)] * padding_rows, columns=df_temp.columns)
        return pd.concat([group, zero_padding], ignore_index=True)
    else:
        return group

def create_sequences(data, maximum_length, label_data=None):

    sequences, labels = [], []

    i = 0
    j = 0
    while i in range(len(data)):# - sequence_length):

        seq = data[i:i+maximum_length]
        sequences.append(seq)
        if label_data is not None:
            label = label_data.iloc[j]['score']
            labels.append(label)
        #print(list(data.iloc[i:i+1, ]['id'])[0])
        #print(data.iloc[i+1:i+2, :])
        #quit()
        '''seq._append(data.iloc[i:i+1, :])

        if (i == len(data) - 1) or (list(data.iloc[i+1:i+2, ]['id'])[0] != list(data.iloc[i:i+1, ]['id'])[0]):
            if len(seq) < maximum_length:
                seq = seq.reindex(range(i+1, maximum_length, 1), fill_value=0)
            sequences.append(seq)
            seq = pd.DataFrame()
        #label = label_data.iloc[i + sequence_length]['score']  # Assuming you want to predict 'event_id'
        '''
        i += maximum_length
        j += 1

    '''for i in range(len(label_data)):
        label = label_data.iloc[i:i+1, :]['score']
        labels.append(label)
    print(sequences)'''

    return np.array(sequences), np.array(labels)

def convert_str_to_num(word):

    sum1 = 0.0
    if word != 0:
        for ch in word:
            sum1 += ord(ch)

    return sum1
# Choose a sequence length
# Find the maximum sequence length for each id


#df = pd.DataFrame(data)
#df_1 = pd.DataFrame(label_data)

# Convert 'down_time' column to datetime if it's not already
#df['down_time'] = pd.to_datetime(df['down_time'])

# Sort the DataFrame by 'id' and 'down_time'

def training():

    df = pd.DataFrame()
    if not os.path.exists(os.getcwd()+"//train_logs_padded.csv"):
        print('reading logs...')
        df = pd.read_csv(train_logs_filename)
        print('df=', df)
        print('sorting logs by id...')
        df.sort_values(['id'], inplace=True)
        print('applying zero padding...')
        df = df.groupby('id', group_keys=False).apply(add_zero_padding, df)
        df.to_csv('train_logs_padded.csv', encoding='utf-8', index=False)
        print(df)
        
    else:
        print('reading padded logs...')
        df = pd.read_csv('train_logs_padded.csv')

    print(df)

    print('reading scores...')
    df_1 = pd.read_csv(train_scores_filename)
    print('df_1=', df_1)

    print('sorting scores by id...')
    df_1.sort_values(['id'], inplace=True)

    #quit()
    max_sequence_length = df.groupby('id', group_keys=False).size().max()
    print(max_sequence_length)
    #quit()
    '''print(df.groupby('id').value_counts())
    some_list = []
    for seq1 in df.groupby('id'):
        sm_df = pd.DataFrame(seq1)
        print("sm_df=",sm_df)
        #item = list(seq1)
        some_list.append(sm_df)
    
    print(some_list)
    numpy_array = np.array(some_list)
    print('numpy=', numpy_array[0][1][0])
    print(max_sequence_length)
    quit()
    '''
    #sequence_length = 3  # You can adjust this based on your requirements
    # dropping the column 'id'
    print('dropping column ID...')
    df = df.drop('id', axis=1)
    #df = df.drop('Unnamed: 0', axis=1)
    print(df)

    df_1 = df_1.drop('id', axis=1)
    # Scale numerical features if needed
    scaler = StandardScaler()
    print('scaling data...')
    df['event_id'] = scaler.fit_transform(df[['event_id']])
    df['down_time'] = scaler.fit_transform(df[['down_time']])
    df['up_time'] = scaler.fit_transform(df[['up_time']])
    df['action_time'] = scaler.fit_transform(df[['action_time']])

    df['activity'] = df['activity'].apply(convert_str_to_num)
    df['activity'] = scaler.fit_transform(df[['activity']])

    df['down_event'] = df['down_event'].apply(convert_str_to_num)
    df['down_event'] = scaler.fit_transform(df[['down_event']])

    df['up_event'] = df['up_event'].apply(convert_str_to_num)
    df['up_event'] = scaler.fit_transform(df[['up_event']])

    df['text_change'] = df['text_change'].apply(convert_str_to_num)
    df['text_change'] = scaler.fit_transform(df[['text_change']])

    df['cursor_position'] = scaler.fit_transform(df[['cursor_position']])
    df['word_count'] = scaler.fit_transform(df[['word_count']])
    #df['other_feature'] = scaler.fit_transform(df[['other_feature']])
    #df = scaler.fit_transform(df)
    df_1['score'] = scaler.fit_transform(df_1[['score']])
    # Create sequences
    print('creating sequences...')
    sequences, labels = create_sequences(df, max_sequence_length, df_1)

    print('sequences=',sequences)
    print('labels=',labels)
    #quit()

    # Pad sequences
    #padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre', dtype='float32', truncating='pre', value=0.0)

    #print(padded_sequences)

    # Split the data into training and testing sets
    print('splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    X_train = tf.constant(X_train, dtype=tf.float32)
    y_train = tf.constant(y_train, dtype=tf.float32)
    X_test = tf.constant(X_test, dtype=tf.float32)
    y_test = tf.constant(y_test, dtype=tf.float32)
    # Now X_train, X_test, y_train, y_test can be used to train and test your LSTM model

    #print(X_train)
    #print(y_train)
    #print(X_test)
    #print(y_test)

    # Assuming you have your training data (X_train) and corresponding labels (y_train)

    # Define the model
    print('creating model...')
    model = Sequential()

    # Add an LSTM layer with 50 units (adjust as needed), input_shape should match the shape of your input sequences
    print('X_train shape=', X_train.shape)

    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))

    # Add a Dense output layer with the number of classes (assuming event_id is categorical)
    model.add(Dense(units=1, activation='linear'))  # Adjust activation based on your problem

    # Compile the model
    print('compiling model...')
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())  # Adjust the optimizer and loss function based on your problem

    # Print the model summary
    print('model summary...')
    model.summary()

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 2e-3 * 0.90 ** x)

    # Train the model
    print('training model...')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[reduce_lr])  # Adjust epochs and batch_size as needed

    print('evaluating model...')
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    print('saving the model')
    model.save(model_filename)
    model.summary()

def testing():
    try:
        # loading the test logs
        print('loading test logs...')
        df_test = pd.read_csv(test_logs_filename)
        print('df_test=\n', df_test)
        print('sorting test logs by id...')
        df_test.sort_values(['id'], inplace=True)
        print('applying zero padding to test logs...')
        print(df_test.groupby('id').size())
        df_test = df_test.groupby('id', group_keys=False).apply(add_zero_padding_to_test, df_test)
        df_test_ids = df_test['id'].unique()
        print('unique ids = \n', df_test_ids)

        max_test_sequence_length = df_test.groupby('id').size().max()
        print(max_test_sequence_length)

        print('dropping test column id')
        df_test = df_test.drop('id', axis=1)
        print(df_test)
        #quit()
        # Scale numerical features if needed
        scaler = StandardScaler()
        print('scaling test data...')
        df_test['event_id'] = scaler.fit_transform(df_test[['event_id']])
        df_test['down_time'] = scaler.fit_transform(df_test[['down_time']])
        df_test['up_time'] = scaler.fit_transform(df_test[['up_time']])
        df_test['action_time'] = scaler.fit_transform(df_test[['action_time']])

        df_test['activity'] = df_test['activity'].apply(convert_str_to_num)
        df_test['activity'] = scaler.fit_transform(df_test[['activity']])

        df_test['down_event'] = df_test['down_event'].apply(convert_str_to_num)
        df_test['down_event'] = scaler.fit_transform(df_test[['down_event']])

        df_test['up_event'] = df_test['up_event'].apply(convert_str_to_num)
        df_test['up_event'] = scaler.fit_transform(df_test[['up_event']])

        df_test['text_change'] = df_test['text_change'].apply(convert_str_to_num)
        df_test['text_change'] = scaler.fit_transform(df_test[['text_change']])

        df_test['cursor_position'] = scaler.fit_transform(df_test[['cursor_position']])
        df_test['word_count'] = scaler.fit_transform(df_test[['word_count']])

        # Create sequences
        print('creating sequences...')
        test_sequences, test_labels = create_sequences(df_test, max_test_sequence_length)

        print(test_sequences)

        X_test = np.asarray(test_sequences, dtype=np.float32)
        print(X_test.shape[0])
        print(X_test.shape[1])
        print(X_test.shape[2])
        print('loading the model')
        savedModel = tf.keras.models.load_model(model_filename)
        savedModel.summary()

        # apply for loop over X_test [i,:,:]...
        predictions_list = []
        print('running predictions...')
        #y_predictions = tf.constant(tf.zeros(shape=(X_test.shape[0]*X_test.shape[1], 0)), dtype=tf.float32)
        x_input = np.array(np.zeros([12876, 10]), dtype=np.float64)
        x_input = np.expand_dims(x_input, axis=0)
        print('printed shape xinput: ', x_input.shape)

        for i in range(X_test.shape[0]):
            x_input[0, 0:2, :] = X_test[i, :, :]
            #x_input[0:2,:] = X_test[0,:,:]
            #X_reshaped = tf.reshape(X_test, shape=(3, 12876, 10))

            y_predictions = savedModel.predict(x_input)
            print('predictions=', y_predictions)
            predictions_list.append(y_predictions[0][0])


        print(predictions_list)

        print('unscaling predictions score...')
        scaled_column = np.array(predictions_list).reshape(1, -1)
        predictions_unscaled = scaler.inverse_transform(scaled_column)
        #print(predictions_unscaled[0].reshape(-1,1))
        pred_numpy = predictions_unscaled[0].reshape(1, -1).flatten()
        #pred_numpy[1] = 2.1234
        print(pred_numpy)
        print('appending score column with ids...')
        #df_test_ids = df_test_ids.drop('', axis=1)
        new_df_score = pd.DataFrame({'score': pred_numpy})
        new_df_score = new_df_score.round(1)
        print(new_df_score)
        new_df_ids = pd.DataFrame({'id': df_test_ids})
        df_test_final = pd.concat([new_df_ids, new_df_score], axis=1)

        print(df_test_final)

        print('saving scores to csv...')
        df_test_final.to_csv('test_scores.csv', index=False)
    except Exception as e:
        print(f"An unexpected error while testing: {e}")


def main():
    if trainortest == 0:
        training()
    elif trainortest == 1:
        testing()

    return


if __name__=="__main__":
    main()