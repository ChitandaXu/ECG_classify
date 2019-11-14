# -*- coding: utf-8 -*-

"""Console script for ecg_classify."""
import sys
import os
import click
from sklearn.metrics import confusion_matrix
from ecg_classify.train import train_model, one_hot_to_number, create_model, prepare_data


@click.command()
def main(args=None):
    """Console script for ecg_classify."""
    os.chdir('C:/Users/Xuexi/PycharmProjects/ECG_classify/ecg_classify')
    x, y, x_test, y_test = prepare_data(intra=False, raw=False, expand_dim=False)
    print('Total training size is:', x.shape[0])
    model_type = 'cnn'
    dimension = x[0].shape
    model = create_model(dimension, model_type)
    train_model(model, x, y)
    print(model.evaluate(x_test, y_test))
    y_pred = model.predict(x_test)
    conf = confusion_matrix(one_hot_to_number(y_test), one_hot_to_number(y_pred))
    print(conf)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
