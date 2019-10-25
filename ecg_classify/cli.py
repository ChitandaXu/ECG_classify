# -*- coding: utf-8 -*-

"""Console script for ecg_classify."""
import sys
import click

from ecg_classify.train import train_model


@click.command()
def main(args=None):
    """Console script for ecg_classify."""

    click.echo("Replace this message by putting your code into "
               "ecg_classify.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    test_loss, test_acc = train_model()
    print(test_acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
