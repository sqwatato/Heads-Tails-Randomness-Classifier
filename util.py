import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from tensorflow import lite
random.seed(1)

# Helper Functions

def change_freq(response:str) -> int:
    """Counts how many times a sequence changes outcomes

    Args:
        response (str): sequence to count

    Returns:
        int: times the sequence changed
    """
    c = 0
    for i in range(1,20):
        if response[i-1] != response[i]:
            c += 1
    return c

def isValid(response:str) -> bool:
    """Checks to see if sequence data is valid

    Args:
        response (str): sequence to check

    Returns:
        bool: True: if sequence valid False: if not valid
    """
    return response.count("H") + response.count("T") == 20

def convert(str:str) -> list:
    """Converts sequence to a numpy array of 1s and 0s

    Args:
        str (str): sequence to convert

    Returns:
        list: converted sequence
    """
    return [1 if str[i] == "h" else 0 for i in range(20)]

def convert_RNN(str:str) -> list:
    """Change input to array of 1/0 to represent H/T for rnn model

    Args:
        str (str): sequence to convert

    Returns:
        list: converted sequence
    """
    return [[1] if str[i] == "h" else [0] for i in range(20)]

def random_coinflip() -> str:
    """Randomly generates a sequence of heads and tails

    Returns:
        str: sequence generated
    """
    random.seed(None)
    str = ""
    for _ in range(20):
        if random.randint(0,1) == 1:
            str += "h"
        else:
            str += "t"
    random.seed(1)
    return str

def get_streaks(sequence:str) -> list[int]:
    """Return an array of streaks of heads or tails in a row

    Args:
        sequence (str): sequence to parse

    Returns:
        list[int]: array of streaks of heads or tails in a row
    """
    streaks = [0] * 20
    c = 1
    previous = sequence[0]
    for flip in sequence[1:]:
        if flip != previous:
            streaks[c-1] += 1
            c = 0
            previous = flip
        c += 1
    streaks[c-1] += 1
    return streaks
            
            

def predict_nn(str:str, model:Sequential) -> None:
    """Prints model prediction and confidence of prediction

    Args:
        str (str): sequence to predict
        model (Sequential): nn model trained for prediction
    """
    prediction = model.predict((convert(str),))[0,0]
    if round(prediction) == 1:
        print(f"Neural Network model's prediction of " + str + ": %.2f%% Human" % (prediction*100))
    else:
        print(f"Neural Network model's prediction of " + str + ": %.2f%% Computer" % ((1-prediction)*100))
        
def predict_rnn(str:str, model:Sequential, model2:Sequential, model3:Sequential) -> None:
    """Prints model prediction and confidence of prediction

    Args:
        str (str): sequence to predict
        model (Sequential): rnn model trained for prediction
        model2 (Sequential): lstm model trained for prediction
        model3 (Sequential): gru model trained for prediction
    """
    prediction = model.predict((convert_RNN(str),))[0,0]
    if round(prediction) == 1:
        print(f"Recurrent Neural Network model's prediction of " + str + ": %.2f%% Human" % (prediction*100))
    else:
        print(f"Recurrent Neural Network model's prediction of " + str + ": %.2f%% Computer" % ((1-prediction)*100))
    prediction = model2.predict((convert_RNN(str),))[0,0]
    if round(prediction) == 1:
        print(f"LSTM model's prediction of " + str + ": %.2f%% Human" % (prediction*100))
    else:
        print(f"LSTM model's prediction of " + str + ": %.2f%% Computer" % ((1-prediction)*100))
    prediction = model3.predict((convert_RNN(str),))[0,0]
    if round(prediction) == 1:
        print(f"GRU model's prediction of " + str + ": %.2f%% Human" % (prediction*100))
    else:
        print(f"GRU model's prediction of " + str + ": %.2f%% Computer" % ((1-prediction)*100))
        
def predict_logistic(str:str, model:Sequential) -> None:
    """Prints model prediction and confidence of prediction

    Args:
        str (str): sequence to predict
        model (Sequential): logistic model trained for prediction
    """
    prediction = model.predict((get_streaks(str),))[0,0]
    if round(prediction) == 1:
        print(f"Logistic model's prediction of " + str + ": %.2f%% Human" % (prediction*100))
    else:
        print(f"Logistics model's prediction of " + str + ": %.2f%% Computer" % ((1-prediction)*100))

def predict_nn_2(str:str, model:Sequential) -> None:
    """Prints model prediction and confidence of prediction

    Args:
        str (str): sequence to predict
        model (Sequential): nn model trained for prediction
    """
    prediction = model.predict((get_streaks(str),))[0,0]
    if round(prediction) == 1:
        print(f"Neural Network model with streaks's prediction of " + str + ": %.2f%% Human" % (prediction*100))
    else:
        print(f"Neural Network model with streaks's prediction of " + str + ": %.2f%% Computer" % ((1-prediction)*100))
    
def predict_all(str:str, logistic:Sequential, nn:Sequential, nn_2:Sequential, rnn:Sequential, lstm:Sequential, gru:Sequential) -> None:
    """Runs prediction for all models

    Args:
        str (str): sequence to predict
        logistic (Sequential): logisti regression model
        nn (Sequential): neural network model with sequence
        nn_2 (Sequential): neural network model with sequence streaks
        rnn (Sequential): recurrent neural network model
        lstm (Sequential): lstm model
        gru (Sequential): gru model
        
    """
    predict_logistic(str, logistic)
    predict_nn(str, nn)
    predict_nn_2(str, nn_2)
    predict_rnn(str, rnn, lstm, gru)
    

# Loading Data Functions

def load_data() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Combines data from two csv files and joins them into one pandas dataframe

    Returns:
        tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: general dataset, human dataset, computer dataset
    """
    # Load Form Data
    df1 = pd.read_csv("forms.csv")
    df1 = df1.drop(columns=['Timestamp',])

    # Cut off responses at length 20
    df1.apply(lambda x: x[:20])

    # Filter out invalid responses
    df1 = df1[df1['combo'].apply(isValid)]
    df1 = df1.reset_index()

    # Load Head Tail Data
    df2 = pd.read_csv("headstails.csv")

    # Split responses at length 20 
    split_reponses = []
    for response in df2["sequence"]:
        for i in range(len(response)//20):
            split_reponses.append(response[i*20:i*20+20])
    df2 = pd.DataFrame(split_reponses,columns=["combo"])
    
    # Combine into one dataset of human data
    human_data = pd.concat([df1,df2])
    human_data = human_data.drop(columns=['index'])
    human_data.reset_index()
    
    # Add column class for computer or human data
    human_data['class'] = ['human'] * len(human_data.index)
    
    
    # Create computer data
    computer_string = []
    for _ in range(len(human_data.index)):
        tmp = ""
        for _ in range(20):
            tmp += "h" if random.randint(0,1) == 1 else "t"
        computer_string.append(tmp)
    
    # Create Dataframe    
    computer_data = pd.DataFrame(computer_string, columns=["combo"])
    computer_data["class"] = ["computer"] * len(computer_data)
    
    data = pd.concat([human_data,computer_data])
    data = data.sample(frac=1,random_state=69)
    data = data.reset_index()
    data = data.drop(columns="index")

    return (data, human_data, computer_data)

def load_analysis_data() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Takes datasets and adds a column with the count of the times a sequence changed

    Returns:
        tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: general dataset, human dataset, computer dataset
    """
    
    # Load Data in mixed data, human, and computer data
    data, human, computer = load_data()
    
    # Add Column for Frequency to data DataFrame
    data['change_freq'] = data['combo'].apply(change_freq)
    
    # Add Column for Frequency to human and computer DataFrames
    human['change_freq'] = human['combo'].apply(change_freq)
    computer['change_freq'] = computer['combo'].apply(change_freq)
    
    return data, human, computer

def load_nn_data() -> tuple[np.array, np.array, np.array, np.array]:
    """Creates train and test data for neural network

    Returns:
        tuple[np.array, np.array, np.array, np.array]: X_train, y_train, X_test, y_test
    """
    
    data, bap, bapo = load_analysis_data()
    del bap
    del bapo
    # Create input and output
    X = data.iloc[:,0].values
    y = data.iloc[:,1:2].values
    
    # Change input to array of 1/0 to represent H/T
    X = np.array(list(map(convert,X)))
    
    # Change output from strings to 1 (human) or 0 (computer)
    y = np.array(list(map(lambda x: 1 if x == "human" else 0, y)))
    
    # Split data into train test stuff
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
    
    return X_train, y_train, X_test, y_test

def load_rnn_data() -> tuple[np.array, np.array, np.array, np.array]:
    """Creates train and test data for recuurent neural network

    Returns:
        tuple[np.array, np.array, np.array, np.array]: X_train, y_train, X_test, y_test
    """
    
    X_train, y_train, X_test, y_test = load_nn_data()
    
    # Changing data to fit rnn model
    X_train = np.array([[[i] for i in seq] for seq in X_train])
    X_test = np.array([[[i] for i in seq] for seq in X_test])
    
    return X_train, y_train, X_test, y_test


def load_logistic_regression_dataframe() -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Adds 20 columns for coinflip streaks

    Returns:
        tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]: general dataset, human dataset, computer dataset
    """
    
    data, human, computer = load_data()
    
    streaks = [[] for _ in range(20)]

    for sequence in data['combo'].values:
        tmp = get_streaks(sequence)
        for i in range(20):
            streaks[i].append(tmp[i])
    for i in range(20):
        data["streak_" + str(i+1)] = streaks[i]
        
    streaks = [[] for _ in range(20)]
    for sequence in human['combo'].values:
        tmp = get_streaks(sequence)
        for i in range(20):
            streaks[i].append(tmp[i])
    for i in range(20):
        human["streak_" + str(i+1)] = streaks[i]
        
    streaks = [[] for _ in range(20)]
    for sequence in computer['combo'].values:
        tmp = get_streaks(sequence)
        for i in range(20):
            streaks[i].append(tmp[i])
    for i in range(20):
        computer["streak_" + str(i+1)] = streaks[i]
    
    return data, human, computer

def load_logistic_regression_data() -> tuple[np.array, np.array, np.array, np.array]:
    """Creates train and test data for logistic regression with new data columns

    Returns:
        tuple[np.array, np.array, np.array, np.array]: X_train, y_train, X_test, y_test
    """
    
    data, bap, bapo = load_logistic_regression_dataframe()
    del bap
    del bapo
    # Create input and output
    X = data.iloc[:,2:].values
    y = data.iloc[:,1].values
    
    # Change output from strings to 1 (human) or 0 (computer)
    y = np.array(list(map(lambda x: 1 if x == "human" else 0, y)))
    
    # Split data into train test stuff
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
    
    return X_train, y_train, X_test, y_test

def load_nn_2_data() -> tuple[np.array, np.array, np.array, np.array]:
    """Creates train and test data for nn  with new data columns

    Returns:
        tuple[np.array, np.array, np.array, np.array]: X_train, y_train, X_test, y_test
    """
    return load_logistic_regression_data()

def convert_models() -> None:
    """Converts saved Keras models into tensorflow lite models
    """
    # Load Models
    logistic = load_model('./models/logistic')
    nn = load_model('./models/nn')
    nn_2 = load_model('./models/nn_2')
    rnn = load_model('./models/rnn')
    
    models = [("logistic", logistic), ("nn", nn), ("nn_2", nn_2), ("rnn", rnn)]

    for name, model in models:
        # Convert models
        converter = lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS, lite.OpsSet.SELECT_TF_OPS]
        
        # Save models
        lite_model = converter.convert()
        with open('lite/' + name + '.tflite', 'wb') as f:
            f.write(lite_model)
    
    