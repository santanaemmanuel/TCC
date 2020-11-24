from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras import Input
import time
import matplotlib as plt
import tensorflow as tf
import pandas as pd
import numpy as np

def change_names(ds, city):
    "This functions change the name of the city columns"
    column_indices = [2]
    old_names = ds.columns[column_indices]
    new_names = old_names + '_' + city
    ds.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    return ds


def prepare_data(f, f_weather):
    #read files
    df = pd.read_csv(f, low_memory=False, parse_dates=True)
    df_weather = pd.read_csv(f_weather, low_memory=False, parse_dates=True)
    #select the relevant data
    df = df[['time', 'total load actual', 'price actual']]
    df_weather = df_weather[['dt_iso', 'city_name', 'temp']]
    #fills missing values
    df = df.interpolate(method='linear')
    #Drop duplicate lines
    df_weather = df_weather.rename(columns={'dt_iso': 'time'})
    df_weather.drop_duplicates(keep='last', inplace=True)
    #Separate the temperature data of each city
    valencia = df_weather.loc[df_weather['city_name'] == 'Valencia']
    bilbao = df_weather.loc[df_weather['city_name'] == 'Bilbao']
    madrid = df_weather.loc[df_weather['city_name'] == 'Madrid']
    barcelona = df_weather.loc[df_weather['city_name'] == ' Barcelona']
    seville = df_weather.loc[df_weather['city_name'] == 'Seville']

    #Name each temperature as the name of its city
    cities = ['valencia', 'bilbao', 'madrid', 'barcelona', 'seville']
    valencia = change_names(valencia, cities[0])
    bilbao = change_names(bilbao, cities[1])
    madrid = change_names(madrid, cities[2])
    barcelona = change_names(barcelona, cities[3])
    seville = change_names(seville, cities[4])

    #clean the dataset
    valencia = valencia.drop(['city_name'], axis=1)
    bilbao = bilbao.drop(['city_name'], axis=1)
    madrid = madrid.drop(['city_name'], axis=1)
    barcelona = barcelona.drop(['city_name'], axis=1)
    seville = seville.drop(['city_name'], axis=1)

    #Merge both energy and weather datasets
    df = pd.merge(df, valencia, on='time', how='left')
    df = pd.merge(df, bilbao, on='time', how='left')
    df = pd.merge(df, seville, on='time', how='left')
    df = pd.merge(df, madrid, on='time', how='left')
    df = pd.merge(df, barcelona, on='time', how='left')

    #Creates a new column with the temperature avarage
    summary_ave_data = df.iloc[:, 2:]
    summary_ave_data['average'] = summary_ave_data.mean(numeric_only=True, axis=1)
    df = df.iloc[:, :3]
    df['temp'] = summary_ave_data['average']

    #Resets the whole index of the final dataset
    df = df.reset_index(drop=True)
    df = df.drop(['time'], axis=1)

    return df


def data_to_input(dt, n_horizon, n_steps):
    "This funcionts transforms the dataset to a tuple X and Y"
    total_size = (len(dt)- n_horizon - n_steps)
    if total_size > 0:
        l = dt['total load actual']
        p = dt['price actual']
        t = dt['temp']
        load = np.zeros((total_size, n_steps, ))
        price = np.zeros((total_size, n_steps,))
        temp = np.zeros((total_size, n_steps,))
        targets = np.zeros((total_size, n_horizon, ))
        for i in range(total_size):
            load[i] =l[i :i +n_steps].values
            price[i] = p[i:i + n_steps].values
            temp[i] = t[i:i + n_steps].values
            targets[i] =l[i + n_steps:i + n_horizon + n_steps].values
    else:
        return None, None, None, None
    return load, price, temp, targets

def data_to_variable(df, k):
    "Function designed to prepare the dataset for K interactions. All non target data is scaled base on training dataset"
    #Separates validadion partition
    val_load, val_price, val_temp, val_targets = data_to_input(df[k * fold: (k + 1) * fold], n_horizon, n_steps)
    #Separates training partition in two sets
    load1, price1, temp1, targ1 = data_to_input(df[:k * fold], n_horizon, n_steps)
    load2, price2, temp2, targ2 = data_to_input(df[(k + 1) * fold:], n_horizon, n_steps)
    #Once the data is split in X and Ys, we can merge both in a single variable
    if (np.ndim(load1) != 0) and (np.ndim(load2) != 0):
        load = np.concatenate((load1, load2))
        price = np.concatenate((price1, price2))
        temp = np.concatenate((temp1, temp2))
        targets = np.concatenate((targ1, targ2))
    elif (np.ndim(load1) != 0):
        load = load1
        temp = temp1
        price = price1
        targets = targ1
    else:
        load = load2
        temp = temp2
        price = price2
        targets = targ2

    #A scaler is instatiated for each variable
    scl_load = MinMaxScaler()
    scl_temp = MinMaxScaler()
    scl_price = MinMaxScaler()
    scl_targets = MinMaxScaler()
    #The scale is fitted on training data and then applied on the dataset
    load = scl_load.fit_transform(load)
    temp = scl_temp.fit_transform(temp)
    price = scl_price.fit_transform(price)
    #Validation dataset is also scaled by trainning data
    val_load = scl_load.transform(val_load)
    val_price = scl_price.transform(val_price)
    val_temp = scl_temp.transform(val_temp)

    return load, price, temp, targets, val_load, val_price, val_temp, val_targets, scl_load, scl_price, scl_temp, scl_targets


def data_generator(load, price, temp, targets, batch_size, steps, variables, models, shuffle = True):
    "This is the generator that will feed the neural network"
    i = 0
    while True:
        #Resets the counter at the end of each epoch
        if i == steps:
            i = 0
        #Shuffles the data at the beginning of a new epoch
        if i == 0:
            if shuffle:
                #All datasets are shuffled by same random factor
                index = np.arange(len(load))
                np.random.shuffle(index)
                load = load[index]
                price = price[index]
                temp = temp[index]
                targets = targets[index]

        #Selects the correct variables to feed the network
        if len(variables) == 1:
            x1 = load[i * batch_size: (i + 1) * batch_size]
            y = targets[i * batch_size: (i + 1) * batch_size]
            i += 1
            #DNN models have diferent input shapes
            if models == 'DNN':
                yield x1, y
            else:
                x1 = np.reshape(x1, (batch_size, n_steps, 1))
                yield x1, y

        if 'temp' in variables:
            x1 = load[i * batch_size: (i + 1) * batch_size]
            x2 = temp[i * batch_size: (i + 1) * batch_size]
            y = targets[i * batch_size: (i + 1) * batch_size]
            i += 1
            if models == 'DNN':
                yield [x1, x2], y
            else:
                x1 = np.reshape(x1, (batch_size, n_steps, 1))
                x2 = np.reshape(x2, (batch_size, n_steps, 1))
                yield [x1, x2], y

        if 'price' in variables:
            x1 = load[i * batch_size: (i + 1) * batch_size]
            x2 = price[i * batch_size: (i + 1) * batch_size]
            y = targets[i * batch_size: (i + 1) * batch_size]
            i += 1
            if models == 'DNN':
                yield [x1, x2], y
            else:
                x1 = np.reshape(x1, (batch_size, n_steps, 1))
                x2 = np.reshape(x2, (batch_size, n_steps, 1))
                yield [x1, x2], y

def get_model(models, variables):
    "this function selects the appropriate model based on start configurations"
    if 'DNN' in models and len(variables) == 1:
        model = DNN_1var()
        return model
    if 'DNN' in models and len(variables) == 2:
        model = DNN_2var()
        return model
    if 'CNN' in models and len(variables) == 1:
        model = CNN_1var()
        return model
    if 'CNN' in models and len(variables) == 2:
        model = CNN_2var()
        return model
    if 'CNN_Mult' in models and len(variables) == 1:
        model = CNN_Mult_Filter_1var()
        return model
    if 'CNN_Mult' in models and len(variables) == 2:
        model = CNN_Mult_Filter_2var()
        return model

def metrics (model, input1, output, n_steps, input2, models, variables):
    "calculate all the metrics"
    #Predicts Y values based on X test
    if models == 'DNN':
        if len(variables) == 1:
            predict = model.predict(input1)
        else:
            predict = model.predict([input1, input2])
    else:
        if len(variables) == 1:
            input1 = np.reshape(input1, (len(input1), n_steps, 1))
            predict = model.predict(input1)
        else:
            input1 = np.reshape(input1, (len(input1), n_steps, 1))
            input2 = np.reshape(input2, (len(input2), n_steps, 1))
            predict = model.predict([input1, input2])

    #Calculates each metrics based on the values predicted
    abs_error = abs(predict - output)
    mean_error = np.mean(abs_error)
    std_error = np.std(abs_error)

    return mean_error, std_error


def get_validation_set(variables, val_temp, val_price):
    "Returns the proper validation set"
    if 'temp' in variables:
        return val_temp
    else:
        print('b')
        return val_price


def DNN_1var():
    x1 = Input(shape=(n_steps, ), name='x1')
    dense1 = layers.Dense(256, activation=activation)(x1)
    dense2 = layers.Dense(256, activation='relu')(dense1)
    dropout = layers.Dropout(.5)(dense2)
    y = layers.Dense(24)(dropout)

    model = Model(x1, y)
    model.compile(optimizer=opt, loss=loss, metrics=['mean_absolute_error'])
    model.summary()

    return model

def DNN_2var():
    x1 = Input(shape=(n_steps, ), name='x1')
    dense1_x1 = layers.Dense(256, activation=activation)(x1)
    dense2_x1 = layers.Dense(256, activation='relu')(dense1_x1)

    x2 = Input(shape=(n_steps,), name='x2')
    dense1_x2 = layers.Dense(256, activation=activation)(x2)
    dense2_x2 = layers.Dense(256, activation='relu')(dense1_x2)

    concat = layers.concatenate([dense2_x1, dense2_x2], axis=1)
    dropout = layers.Dropout(.5)(concat)
    y = layers.Dense(24)(dropout)

    model = Model([x1,x2], y)
    model.compile(optimizer=opt, loss=loss, metrics=['mean_absolute_error'])
    model.summary()

    return model

def CNN_1var():
    x1 = Input(shape=(n_steps, 1), name='x1')
    conv11_x1 = layers.Conv1D(98, kernel_size=10, activation=activation)(x1)
    max11_x1 = layers.MaxPooling1D(2)(conv11_x1)
    conv21_x1 = layers.Conv1D(98, kernel_size=10, activation=activation)(max11_x1)
    max21_x1 = layers.MaxPooling1D(2)(conv21_x1)
    flatten_x1 = layers.Flatten()(max21_x1)
    dense1 = layers.Dense(256, activation='relu')(flatten_x1)
    dropout = layers.Dropout(.5)(dense1)
    y = layers.Dense(24)(dropout)

    model = Model(x1, y)
    model.compile(optimizer=opt, loss=loss, metrics=['mean_absolute_error'])

    model.summary()

    return model

def CNN_2var():
    x1 = Input(shape=(n_steps, 1), name='x1')
    conv11_x1 = layers.Conv1D(98, kernel_size=10, activation=activation)(x1)
    max11_x1 = layers.MaxPooling1D(2)(conv11_x1)
    conv11_x1 = layers.Conv1D(98, kernel_size=10, activation=activation)(max11_x1)
    max21_x1 = layers.MaxPooling1D(2)(conv11_x1)
    flatten_x1 = layers.Flatten()(max21_x1)
    dense1_x1 = layers.Dense(256, activation='relu')(flatten_x1)

    x2 = Input(shape=(n_steps, 1), name='x2')
    conv11_x2 = layers.Conv1D(98, kernel_size=10, activation=activation)(x2)
    max11_x2 = layers.MaxPooling1D(2)(conv11_x2)
    conv11_x2 = layers.Conv1D(98, kernel_size=10, activation=activation)(max11_x2)
    max21_x2 = layers.MaxPooling1D(2)(conv11_x2)
    flatten_x2 = layers.Flatten()(max21_x2)
    dense1_x2 = layers.Dense(256, activation='relu')(flatten_x2)

    concat = layers.concatenate([dense1_x1, dense1_x2], axis=1)

    dropout = layers.Dropout(.5)(concat)
    y = layers.Dense(24)(dropout)

    model = Model([x1, x2], y)
    model.compile(optimizer=opt, loss=loss, metrics=['mean_absolute_error'])
    model.summary()

    return model

def CNN_Mult_Filter_1var():
    x1 = Input(shape=(n_steps, 1), name='x1')

    conv11_x1 = layers.Conv1D(98, kernel_size=24 * 5, activation=activation)(x1)
    max11_x1 = layers.MaxPooling1D(2)(conv11_x1)
    conv12_x1 = layers.Conv1D(98, kernel_size=24 * 2, dilation_rate=2, activation=activation)(max11_x1)
    max12_x1 = layers.MaxPooling1D(2)(conv12_x1)

    conv21_x1 = layers.Conv1D(98, kernel_size=24 * 3, activation=activation)(x1)
    max21_x1 = layers.MaxPooling1D(2)(conv21_x1)
    conv22_x1 = layers.Conv1D(98, kernel_size=22, dilation_rate=2, activation=activation)(max21_x1)
    max22_x1 = layers.MaxPooling1D(2)(conv22_x1)

    conv31_x1 = layers.Conv1D(64, kernel_size=15 * 5, activation=activation)(x1)
    max31_x1 = layers.MaxPooling1D(2)(conv31_x1)
    conv32_x1 = layers.Conv1D(64, kernel_size=15 * 2, dilation_rate=2, activation=activation)(max31_x1)
    max32_x1 = layers.MaxPooling1D(2)(conv32_x1)

    conv41_x1 = layers.Conv1D(64, kernel_size=7 * 5, activation=activation)(x1)
    max41_x1 = layers.MaxPooling1D(2)(conv41_x1)
    conv42_x1 = layers.Conv1D(64, kernel_size=7 * 2, dilation_rate=2, activation=activation)(max41_x1)
    max42_x1 = layers.MaxPooling1D(2)(conv42_x1)

    flatten1_x1 = layers.Flatten()(max12_x1)
    flatten2_x1 = layers.Flatten()(max22_x1)
    flatten3_x1 = layers.Flatten()(max32_x1)
    flatten4_x1 = layers.Flatten()(max42_x1)

    flatten_x1 = layers.Flatten()(x1)

    concatenated = layers.concatenate([flatten1_x1, flatten2_x1, flatten3_x1, flatten4_x1, flatten_x1], axis=1)
    dropout1 = layers.Dropout(.3)(concatenated)
    dense1 = layers.Dense(1024, activation='relu')(dropout1)
    dropout2 = layers.Dropout(.3)(dense1)
    y = layers.Dense(24)(dropout2)
    model = Model(x1, y)
    model.compile(optimizer=opt, loss=loss, metrics=['mean_absolute_error'])
    model.summary()

    return model



def CNN_Mult_Filter_2var():
    x1 = Input(shape=(n_steps, 1), name='x1')
    x2 = Input(shape=(n_steps, 1), name='x2')

    conv11_x1 = layers.Conv1D(98, kernel_size=24 * 5, activation=activation)(x1)
    max11_x1 = layers.MaxPooling1D(2)(conv11_x1)
    conv12_x1 = layers.Conv1D(98, kernel_size=24 * 2, dilation_rate=2, activation=activation)(max11_x1)
    max12_x1 = layers.MaxPooling1D(2)(conv12_x1)

    conv21_x1 = layers.Conv1D(98, kernel_size=24 * 3, activation=activation)(x1)
    max21_x1 = layers.MaxPooling1D(2)(conv21_x1)
    conv22_x1 = layers.Conv1D(98, kernel_size=22, dilation_rate=2, activation=activation)(max21_x1)
    max22_x1 = layers.MaxPooling1D(2)(conv22_x1)

    conv31_x1 = layers.Conv1D(64, kernel_size=15 * 5, activation=activation)(x1)
    max31_x1 = layers.MaxPooling1D(2)(conv31_x1)
    conv32_x1 = layers.Conv1D(64, kernel_size=15 * 2, dilation_rate=2, activation=activation)(max31_x1)
    max32_x1 = layers.MaxPooling1D(2)(conv32_x1)

    conv41_x1 = layers.Conv1D(64, kernel_size=7 * 5, activation=activation)(x1)
    max41_x1 = layers.MaxPooling1D(2)(conv41_x1)
    conv42_x1 = layers.Conv1D(64, kernel_size=7 * 2, dilation_rate=2, activation=activation)(max41_x1)
    max42_x1 = layers.MaxPooling1D(2)(conv42_x1)

    flatten1_x1 = layers.Flatten()(max12_x1)
    flatten2_x1 = layers.Flatten()(max22_x1)
    flatten3_x1 = layers.Flatten()(max32_x1)
    flatten4_x1 = layers.Flatten()(max42_x1)
    flatten_x1 = layers.Flatten()(x1)

    concatenated_x1 = layers.concatenate([flatten1_x1, flatten2_x1, flatten3_x1, flatten4_x1, flatten_x1], axis=1)
    dropout1_x1 = layers.Dropout(.3)(concatenated_x1)
    dense1_x1 = layers.Dense(1024, activation='relu')(dropout1_x1)


    conv11_x2 = layers.Conv1D(98, kernel_size=24 * 5, activation=activation)(x2)
    max11_x2 = layers.MaxPooling1D(2)(conv11_x2)
    conv12_x2 = layers.Conv1D(98, kernel_size=24 * 2, dilation_rate=2, activation=activation)(max11_x2)
    max12_x2 = layers.MaxPooling1D(2)(conv12_x2)

    conv21_x2 = layers.Conv1D(98, kernel_size=24 * 3, activation=activation)(x2)
    max21_x2 = layers.MaxPooling1D(2)(conv21_x2)
    conv22_x2 = layers.Conv1D(98, kernel_size=22, dilation_rate=2, activation=activation)(max21_x2)
    max22_x2 = layers.MaxPooling1D(2)(conv22_x2)

    conv31_x2 = layers.Conv1D(64, kernel_size=15 * 5, activation=activation)(x2)
    max31_x2 = layers.MaxPooling1D(2)(conv31_x2)
    conv32_x2 = layers.Conv1D(64, kernel_size=15 * 2, dilation_rate=2, activation=activation)(max31_x2)
    max32_x2 = layers.MaxPooling1D(2)(conv32_x2)

    conv41_x2 = layers.Conv1D(64, kernel_size=7 * 5, activation=activation)(x2)
    max41_x2 = layers.MaxPooling1D(2)(conv41_x2)
    conv42_x2 = layers.Conv1D(64, kernel_size=7 * 2, dilation_rate=2, activation=activation)(max41_x2)
    max42_x2 = layers.MaxPooling1D(2)(conv42_x2)

    flatten1_x2 = layers.Flatten()(max12_x2)
    flatten2_x2 = layers.Flatten()(max22_x2)
    flatten3_x2 = layers.Flatten()(max32_x2)
    flatten4_x2 = layers.Flatten()(max42_x2)

    flatten_x2 = layers.Flatten()(x2)

    concatenated_x2 = layers.concatenate([flatten1_x2, flatten2_x2, flatten3_x2, flatten4_x2, flatten_x2], axis=1)
    dropout1_x2 = layers.Dropout(.3)(concatenated_x2)
    dense1_x2 = layers.Dense(1024, activation='relu')(dropout1_x2)

    concat = layers.concatenate([dense1_x1, dense1_x2], axis=1)
    dropout2 = layers.Dropout(.3)(concat)
    y = layers.Dense(24)(dropout2)
    model = Model([x1, x2], y)
    model.compile(optimizer=opt, loss=loss, metrics=['mean_absolute_error'])

    model.summary()

    return model


#Setting the datasets adress
f = r'F:\Dataset\energy_dataset.csv'
f_weather = r'F:\Dataset\weather_features.csv'
#Prepare the dataset
df = prepare_data(f, f_weather)

#Define hyperparameters used
n_steps = 24 * 15
n_horizon = 24
batch_size = 32
lr = 3e-4
kfolds = 5
size = len(df)
fold = size // kfolds
loss = tf.keras.losses.Huber()
opt = tf.keras.optimizers.Adam(lr=lr)
epochs = 1

#Setup test parameters
activation = 'tanh'
#Can be ['load'], ['load', 'temp'] or ['load', 'price']
variables = ['load', 'price']
#Can be 'DNN', 'CNN' or 'CNN_Mult'
models = 'CNN'


#Setting up the variables to store metrics
all_mae_histories = []
all_mae = []
all_std = []
all_time = []

for k in range(kfolds):
    print('processing fold #', k)
    #Orga
    load, price, temp, targets, val_load, val_price, val_temp, val_targets, scl_load, scl_price, scl_temp, scl_targets = data_to_variable(df, k)
    #calculates steps per epoch
    steps = len(load)//batch_size
    val_steps = len(val_load)//batch_size
    #Prepare the data generators
    train_gen = data_generator(load, price, temp, targets, batch_size, steps, variables, models, shuffle = True)
    val_gen = data_generator(val_load, val_price, val_temp, val_targets, batch_size, val_steps, variables, models, shuffle = True)
    #Get the chosen model
    model = get_model(models, variables)
    #Get the start training time
    start = time.time()
    #Fits the model
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, steps_per_epoch=steps, validation_steps=val_steps, verbose=1)
    #Get the end training time
    end = time.time()
    #Selects the second input
    val_x2 = get_validation_set(variables, val_temp, val_price)

    mean_error, std_error = metrics(model, input1=val_load, output=val_targets, n_steps=n_steps, input2=val_x2, variables=variables, models=models)
    all_mae.append(mean_error)
    all_std.append(std_error)
    all_time.append(end - start)

    print('o erro medio absoluto e:', mean_error)
    print('o desvio medio padrao e:',std_error)
    print('o tempo de treinamento foi de:', end-start)


print('o desvio medio padrao de cada iteração K foi:', all_std)
print('o erro medio absoluto de cada iteração K foi:', all_mae)
print('o tempo de treinamento de cada iteração K foi:', all_time)














