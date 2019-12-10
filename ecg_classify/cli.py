# -*- coding: utf-8 -*-

"""Console script for ecg_classify."""
import sys
import os
import click


@click.command()
def main(args=None):
    """Console script for ecg_classify."""
    os.chdir('C:/Users/Xuexi/PycharmProjects/ECG_classify/ecg_classify')
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
