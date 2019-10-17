# -*- coding: utf-8 -*-

"""Console script for ecg_classify."""
import sys
import click

from ecg_classify.wfdb_io import generate_sample_by_heartbeat, NormalBeat, DataSetType


@click.command()
def main(args=None):
    """Console script for ecg_classify."""

    click.echo("Replace this message by putting your code into "
               "ecg_classify.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set] = generate_sample_by_heartbeat(NormalBeat(), DataSetType.TRAINING)
    print(r_loc_set[0: 100])
    print(prev_r_loc_set[0: 100])
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
