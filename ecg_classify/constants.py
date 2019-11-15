class NormalBeat:
    beat_type = 'N'
    training_set_dict = {
        101: 1000,
        103: 1000,
        108: 1000,
        112: 1000
    }
    test_set_dict = {
        100: 1000
    }


class LBBBBeat:
    beat_type = 'L'
    training_set_dict = {
        109: 1500,
        111: 1500,
        207: 1000
    }
    test_set_dict = {
        214: 1000
    }


class RBBBBeat:
    beat_type = 'R'
    training_set_dict = {
        118: 1500,
        124: 1500,
        231: 1000
    }
    test_set_dict = {
        212: 1000
    }


class APCBeat:
    beat_type = 'A'
    training_set_dict = {
        101: 3,
        108: 4,
        112: 2,
        118: 96,
        124: 2,
        200: 30,
        207: 106,
        209: 383,
        232: 1374
    }
    test_set_dict = {
        100: 33,
        201: 30,
        202: 36,
        205: 3,
        213: 25,
        220: 94,
        222: 207,
        223: 72
    }


class VPCBeat:
    beat_type = 'V'
    training_set_dict = {
        106: 520,
        116: 109,
        118: 16,
        119: 443,
        124: 47,
        200: 825,
        203: 444,
        208: 991,
        215: 164,
        221: 79,
        228: 362
    }
    test_set_dict = {
        105: 41,
        201: 198,
        205: 71,
        214: 256,
        219: 64,
        223: 370
    }


def heartbeat_factory(symbol):
    if symbol == 'N':
        return NormalBeat()
    elif symbol == 'L':
        return LBBBBeat()
    elif symbol == 'R':
        return RBBBBeat()
    elif symbol == 'A':
        return APCBeat()
    elif symbol == 'V':
        return VPCBeat()
    else:
        raise Exception('Invalid heartbeat type')


LABEL_LIST = ['N', 'L', 'R', 'A', 'V']
CLASS_NUM = 5
FEATURE_NUM = 9
DROP_LIST = ['8']
FREQUENCY = 360
