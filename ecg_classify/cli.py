# -*- coding: utf-8 -*-

"""Console script for ecg_classify."""
import sys
import click
from ecg_classify.test_model import *


@click.command()
def main(args=None):
    """Console script for ecg_classify."""
    inter_patient()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
