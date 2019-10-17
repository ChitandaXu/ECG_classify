from ecg_classify.wfdb_io import *


def get_rr_interval(heartbeat, data_set_type):
    if not isinstance(heartbeat, (NormalBeat, LBBBBeat, RBBBBeat, APCBeat, VPCBeat)):
        raise Exception("Heart beat type invalid.")
    if not isinstance(data_set_type, DataSetType):
        raise Exception("Data type is invalid, please specify 'TRAINING' or 'TEST'.")

    [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set] = generate_sample_by_heartbeat(heartbeat, DataSetType.TRAINING)
    prev_rr_interval = r_loc_set - prev_r_loc_set
    next_rr_interval = next_r_loc_set - r_loc_set
    return [prev_rr_interval, next_rr_interval]

