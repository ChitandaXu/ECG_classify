normal_beat = {
    'train_dict': {
        101: 1000,
        103: 1000,
        108: 1000,
        112: 1000
    },
    'test_dict': {
        100: 1000
    }
}

lbbb_beat = {
    'train_dict': {
        109: 1500,
        111: 1500,
        207: 1000
    },
    'test_dict': {
        214: 1000
    }
}

rbbb_beat = {
    'train_dict': {
        118: 1500,
        124: 1500,
        231: 1000
    },
    'test_dict': {
        212: 1000
    }
}

apc_beat = {
    'train_dict': {
        101: 3,
        108: 4,
        112: 2,
        118: 95,
        124: 2,
        200: 30,
        207: 104,
        209: 383,
        232: 1377  # 1380
    },
    'test_dict': {
        100: 33,
        201: 30,
        202: 36,
        205: 3,
        213: 25,
        220: 94,
        222: 207,  # 208
        223: 72
    }
}

vpc_beat = {
    'train_dict': {
        106: 518,
        116: 109,
        118: 16,
        119: 443,
        124: 47,
        200: 825,
        203: 444,
        208: 990,
        215: 140,  # 164
        221: 108,  # 396
        228: 360   # 361
    },
    'test_dict': {
        105: 41,
        201: 198,
        205: 71,
        214: 256,
        219: 64,
        223: 370   # 473
    }
}

unknown_beat = {

}


def __heartbeat_factory(symbol):
    if symbol == 'N':
        return normal_beat
    elif symbol == 'L':
        return lbbb_beat
    elif symbol == 'R':
        return rbbb_beat
    elif symbol == 'A':
        return apc_beat
    elif symbol == 'V':
        return vpc_beat
    else:
        raise Exception('Invalid heartbeat type')


def heartbeat_factory(symbol, is_training):
    heartbeat = __heartbeat_factory(symbol)
    if is_training:
        return heartbeat['train_dict']
    else:
        return heartbeat['test_dict']


LABEL_LIST = ['N', 'L', 'R', 'A', 'V']
CLASS_NUM = 5
DIM = 31
# DIM = 553
FEATURE_NUM = DIM - 1
FREQUENCY = 360
