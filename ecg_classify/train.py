import numpy as np
from keras import optimizers
from ecg_classify.constants import NUMBER_OF_CLASSES
from ecg_classify.feature import get_samples, get_labels
from ecg_classify.model import build_cnn_model
from ecg_classify.resnet import resnet_v1
# import pandas as pd
# from sklearn.metrics import confusion_matrix, accuracy_score
# from keras.callbacks import ModelCheckpoint
# from keras.models import load_model


def prepare_data():
    x = get_samples(True)
    x_test = get_samples(False)
    y = get_labels(True)
    y_test = get_labels(False)
    x = np.expand_dims(x, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    return x, y, x_test, y_test


def shuffle_data(x, y):
    if x.shape[0] != y.shape[0]:
        raise Exception("Invalid input, x and y should be same length in dimension 0")
    # np.random.seed(7)
    order = np.random.permutation(np.arange(x.shape[0]))
    x = x[order]
    y = y[order]
    return x, y


def one_hot_to_number(y):
    size = np.shape(y)[0]
    res = np.zeros(size)
    for idx in range(size):
        max_value = max(y[idx])
        res[idx] = list(y[idx]).index(max_value)
    return res.astype(np.int)


def number_to_one_host(y, number_of_classes):
    size = np.shape(y)[0]
    res = np.zeros((size, number_of_classes))
    for idx in range(size):
        dummy = np.zeros(number_of_classes)
        dummy[int(y[idx])] = 1
        res[idx, :] = dummy
    return res


def train_model(model_type='cnn', n=3, version=1):
    x, y, x_test, y_test = prepare_data()
    x, y = shuffle_data(x, y)
    x_test, y_test = shuffle_data(x_test, y_test)

    print('Total training size is ', x.shape[0])
    y = number_to_one_host(y, NUMBER_OF_CLASSES)
    y_test = number_to_one_host(y_test, NUMBER_OF_CLASSES)

    if model_type == 'resnet':
        if version == 1:
            depth = n * 6 + 2
        elif version == 2:
            depth = n * 9 + 2
        model_name = 'ResNet%dv%d' % (depth, version)
        print(model_name)
        model = resnet_v1(x[0].shape, depth, NUMBER_OF_CLASSES)
    elif model_type == 'cnn':
        print(model_type)
        model = build_cnn_model(x[0].shape, NUMBER_OF_CLASSES)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=1e-3),
                  metrics=['acc'])
    model.fit(x, y, epochs=5, batch_size=64)
    print(model.evaluate(x_test, y_test))
    return model


# model = train_model()
# history = LossHistory()
# cnn_model.fit(x, y, epochs=5, batch_size=64)
# return cnn_model.evaluate(x_test, y_test)
#
# # 创建一个实例history
#
# checkpointer = ModelCheckpoint(filepath='models/Best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
# split_point = int(train_X.shape[0] * 0.8)
# X_valid = train_X[split_point:]
# label_valid = label[split_point:]
# hist = model.fit(train_X[0: split_point], label[0: split_point],  validation_data=(X_valid, label_valid), batch_size=32, epochs=80, verbose=2, callbacks=[checkpointer, history],
#                  shuffle=True)
# pd.DataFrame(hist.history).to_csv(path_or_buf='models/History.csv')
# model = load_model('models/Best_model.h5')
# predictions = model.predict(test_X)
# score = accuracy_score(change(label_test), change(predictions))
# print('Test score is:', score)
# df = pd.DataFrame(change(predictions))
# df.to_csv(path_or_buf='models/Preds_' + str(format(score, '.4f')) + '.csv', index=None, header=None)
# pd.DataFrame(confusion_matrix(change(label_test), change(predictions))).to_csv(
#     path_or_buf='models/Result_Conf' + str(format(score, '.4f')) + '.csv', index=None, header=None)
#
# # 绘制acc-loss曲线
# history.loss_plot('epoch')
