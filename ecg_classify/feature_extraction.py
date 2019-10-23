from ecg_classify.wfdb_io import generate_sample_by_heartbeat


def get_rr_interval(heartbeat_symbol, data_set_type):
    [data_set, r_loc_set, prev_r_loc_set, next_r_loc_set, number_set] = \
        generate_sample_by_heartbeat(heartbeat_symbol, data_set_type)

    # RR interval
    rr_interval = r_loc_set - prev_r_loc_set

    # P region
    p_start = r_loc_set - int((r_loc_set - prev_r_loc_set) * 0.35)
    p_end = r_loc_set - 22

    # T region
    t_start = r_loc_set + 22
    t_end = r_loc_set + int((next_r_loc_set - r_loc_set) * 0.65)



