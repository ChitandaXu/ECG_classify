from ecg_classify.wfdb_io import *


def get_rr_interval(heartbeat, data_set_type):
    if not isinstance(heartbeat, (NormalBeat, LBBBBeat, RBBBBeat, APCBeat, VPCBeat)):
        raise Exception("Heart beat type invalid.")
    if not isinstance(data_set_type, DataSetType):
        raise Exception("Data type is invalid, please specify 'TRAINING' or 'TEST'.")

    [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set] = generate_sample_by_heartbeat(heartbeat, data_set_type)

    # RR interval
    rr_interval = r_loc_set - prev_r_loc_set

    # P region
    p_start = r_loc_set - (r_loc_set - prev_r_loc_set) * 0.35
    p_end = r_loc_set - 22

    # T region
    t_start = r_loc_set + 22
    t_end = r_loc_set + (next_r_loc_set - r_loc_set) * 0.65

    # kurtosis
    # kurtosis = count_kurtosis()

