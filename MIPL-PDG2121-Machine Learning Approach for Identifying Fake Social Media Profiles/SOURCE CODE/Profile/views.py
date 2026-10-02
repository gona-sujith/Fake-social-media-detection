from django.shortcuts import render
import pandas as pd
import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import class_weight

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import backend as K

# GLOBALS
model = None
scaler = None
input_dim = None
graph = None
tf_session = None

# ------------------ BASIC PAGES ------------------

def index(request):
    return render(request, 'index.html')

def User(request):
    return render(request, 'User.html')

def Admin(request):
    return render(request, 'Admin.html')

# ------------------ ADMIN LOGIN ------------------

def AdminLogin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username == 'admin' and password == 'admin':
            return render(request, 'AdminScreen.html', {'data': 'Welcome ' + username})
        else:
            return render(request, 'Admin.html', {'data': 'Login Failed'})

# ------------------ LOAD DATA ------------------

def importdata():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(BASE_DIR, 'dataset.txt')

    if not os.path.exists(file_path):
        raise Exception("dataset.txt file not found!")

    data = pd.read_csv(file_path, sep=',', skip_blank_lines=True)

    expected_columns = ['Account_Age','Gender','User_Age','Link_Desc','Status_Count',
                        'Friend_Count','Location','Location_IP','Status']
    if list(data.columns) != expected_columns:
        raise Exception(f"Dataset columns must be: {expected_columns}")

    return data

# ------------------ SPLIT DATA ------------------

def splitdataset(data):
    global scaler, input_dim

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    input_dim = X.shape[1]

    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    train_x, test_x, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    return train_x, test_x, train_y, test_y

# ------------------ TRAIN MODEL ------------------

def train_model():
    global model, graph, tf_session

    K.clear_session()
    tf_session = K.get_session()
    graph = tf.get_default_graph()

    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)

    y_labels = np.argmax(train_y, axis=1)
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_labels),
        y=y_labels
    )
    class_weights = dict(enumerate(weights))
    with graph.as_default():
        with tf_session.as_default():
            model = Sequential()
            model.add(Dense(64, input_dim=train_x.shape[1], activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(train_y.shape[1], activation='softmax'))

            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            model.fit(
                train_x,
                train_y,
                validation_split=0.2,
                epochs=30,
                batch_size=10,
                class_weight=class_weights,
                callbacks=[EarlyStopping(patience=5)],
                verbose=2
            )

            loss, acc = model.evaluate(test_x, test_y)

    return acc


def GenerateModel(request):
    try:
        acc = train_model()

        return render(request, 'AdminScreen.html', {
            'data': f'ANN Accuracy: {acc * 100:.2f}%'
        })

    except Exception as e:
        return render(request, 'AdminScreen.html', {'data': str(e)})

# ------------------ USER PREDICTION ------------------

def UserCheck(request):
    global model, scaler, input_dim, graph, tf_session

    if request.method == 'POST':
        try:
            if model is None or graph is None or tf_session is None:
                train_model()

            data = request.POST.get('t1')
            lines = [line.strip() for line in data.splitlines() if line.strip()]
            if not lines:
                return render(request, 'User.html', {'data': 'Please enter account details'})

            rows = []
            for row in lines:
                if row.startswith('Account_Age'):
                    continue

                values = list(map(float, row.split(',')))

                if len(values) == input_dim + 1:
                    values = values[:-1]

                if len(values) != input_dim:
                    return render(request, 'User.html', {
                        'data': f'Expected {input_dim} values, got {len(values)} in row: {row}'
                    })

                rows.append(values)

            if not rows:
                return render(request, 'User.html', {'data': 'Please enter at least one data row'})

            input_data = pd.DataFrame(rows)
            input_scaled = scaler.transform(input_data)

            with graph.as_default():
                with tf_session.as_default():
                    K.set_session(tf_session)
                    pred = model.predict(input_scaled, verbose=0)
            pred_classes = np.argmax(pred, axis=1)

            results = []
            for index, pred_class in enumerate(pred_classes, start=1):
                label = "GENUINE" if pred_class == 0 else "FAKE"
                results.append(f"Row {index}: {label}")

            if len(pred_classes) == 1:
                msg = "GENUINE" if pred_classes[0] == 0 else "FAKE"
            else:
                msg = " | ".join(results)

            return render(request, 'User.html', {'data': msg})

        except Exception as e:
            return render(request, 'User.html', {'data': str(e)})

# ------------------ VIEW DATA ------------------

def ViewTrain(request):
    try:
        data = importdata()
        html = "<table border=1>"
        html += "<tr>" + "".join(f"<th>{col}</th>" for col in data.columns) + "</tr>"
        for _, row in data.iterrows():
            html += "<tr>" + "".join(f"<td>{val}</td>" for val in row) + "</tr>"
        html += "</table>"
        return render(request, 'ViewData.html', {'data': html})
    except Exception as e:
        return render(request, 'ViewData.html', {'data': str(e)})

