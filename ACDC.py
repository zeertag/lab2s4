import numpy as np


def dc_coding(num):
    num = int(num)

    def category(val):
        return 0 if val == 0 else val.bit_length()

    def bits(val):
        cat = category(val)
        if cat == 0:
            return ""
        if val < 0:
            val = ((1 << cat) - 1) ^ (abs(val))
        return bin(val)[2:].zfill(cat)

    return category(num), bits(num)


def ac_coding(ac):
    def category(val):
        return 0 if val == 0 else val.bit_length()

    def bits(val):
        cat = category(val)
        if cat == 0:
            return ""
        if val < 0:
            val = ((1 << cat) - 1) ^ (abs(val))
        return bin(val)[2:].zfill(cat)

    result = []
    run_length = 0
    for val in ac:
        val = int(val)
        if val == 0:
            run_length += 1
        else:
            size = category(val)
            bits_val = bits(val)
            while run_length > 15:
                result.append((15, 0, ''))
                run_length -= 16
            result.append((run_length, size, bits_val))
            run_length = 0
    if run_length > 0:
        result.append((0, 0, ''))
    return result


def decode_coefficients(dc_list, ac_list):
    def decode_bits(bits, cat):
        if cat == 0:
            return 0
        value = int(bits, 2)
        if bits[0] == '0':
            value = -((1 << cat) - 1 - value)
        return value

    result = []
    for dc_entry, ac_entries in zip(dc_list, ac_list):
        dc_cat, dc_bits = dc_entry
        dc_val = decode_bits(dc_bits, dc_cat)  # УБРАН prev_dc

        block = [dc_val]
        for run, size, bits in ac_entries:
            if (run, size) == (0, 0):  # EOB
                block.extend([0] * (64 - len(block)))
                break
            block.extend([0] * run)
            val = decode_bits(bits, size)
            block.append(val)
        while len(block) < 64:
            block.append(0)
        result.append(np.array(block, dtype=np.int16))
    return np.array(result)


def decode_blocks(data):
    dc = [d[0] for d in data]
    ac = [d[1] for d in data]
    return decode_coefficients(dc, ac)


def encode_blocks(blocks):
    result = []
    for block in blocks:
        dc = block[0]
        ac = block[1:]
        dc_encoded = dc_coding(dc)
        ac_encoded = ac_coding(ac)
        result.append((dc_encoded, ac_encoded))
    return result
