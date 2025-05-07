class Node():
    def __init__(self, symbol=None, counter=None, left=None, right=None, parent=None):
        self.symbol = symbol
        self.counter = counter
        self.left = left
        self.right = right


def Huffman(data):
    alphabet = {}
    for i in data:
        if i in alphabet:
            alphabet[i] += 1
        else:
            alphabet[i] = 1
    tree_help = [Node(value, count) for value, count in alphabet.items()]
    while len(tree_help) > 1:
        tree_help.sort(key=lambda node: node.counter)
        elem1 = tree_help.pop(0)
        elem2 = tree_help.pop(0)
        parent = Node(None, elem1.counter + elem2.counter, elem1, elem2)
        tree_help.append(parent)
    root = tree_help[0]
    codes = Huffman_codes(root)
    text = coded_text(data, codes)
    k = 8 - len(text) % 8
    text += k * "0"
    bytes_string = b""
    for i in range(0, len(text), 8):
        s = text[i:i + 8]
        x = string_binary_to_int(s)
        bytes_string += x.to_bytes(1, "big")

    letters_info = b""
    for i in alphabet:
        letters_info += i.to_bytes(4, "big")
        letters_info += alphabet[i].to_bytes(4, "big")
    bytes_string = len(alphabet).to_bytes(2, "big") + letters_info + bytes_string + (8 - k).to_bytes(1, "big")
    return bytes_string


def Huffman_codes(root):
    codes = {}

    def make_codes(node, cur_code=""):
        if node is not None:
            if node.symbol is not None:
                codes[node.symbol] = cur_code
            make_codes(node.left, cur_code + "0")
            make_codes(node.right, cur_code + "1")

    make_codes(root)
    return codes


def coded_text(data, codes):
    text = ""
    for i in data:
        text += codes[i]
    return text


def string_binary_to_int(s):
    X = 0
    for i in range(8):
        if s[i] == "1":
            X = X + 2 ** (7 - i)
    return X


def Huffman_decode(c_data):
    original = bytearray()
    len_letters = int.from_bytes(c_data[:2], "big")
    alphabet = {}
    i = 2
    while i < len_letters * 8 + 2:
        symbol = int.from_bytes(c_data[i:i + 4], "big")
        count = int.from_bytes(c_data[i + 4:i + 8], "big")
        alphabet[symbol] = count
        i += 8

    tree_help = [Node(value, count) for value, count in alphabet.items()]
    while len(tree_help) > 1:
        tree_help.sort(key=lambda node: node.counter)
        elem1 = tree_help.pop(0)
        elem2 = tree_help.pop(0)
        parent = Node(None, elem1.counter + elem2.counter, elem1, elem2)
        tree_help.append(parent)
    root = tree_help[0]
    codes = Huffman_codes(root)

    real = c_data[-1]
    bin_txt = ""
    for j in range(i, len(c_data[:-2])):
        bin_txt += format(c_data[j], '08b')
    bin_txt += format(c_data[j + 1], '08b')[:real]

    i = 0
    word = ''
    while i < len(bin_txt):
        word += bin_txt[i]
        if word in codes.values():
            for j in codes:
                if codes[j] == word:
                    original.append(j)
                    break
            word = ''
        i += 1
    return original


def Huffman_JPEG_encode(encoded_blocks):
    img_size = encoded_blocks[0]
    symbols = []

    for block in encoded_blocks[1:]:
        dc_size, dc_code = block[0]
        symbols.append(dc_size)
        # Если dc_code пустой, сохраняем 0
        symbols.append(int(dc_code, 2) if dc_code != '' else 0)

        for run, size, code in block[1]:
            symbols.append(run)
            symbols.append(size)
            # Если code пустой, сохраняем 0
            symbols.append(int(code, 2) if code != '' else 0)

    encoded_data = Huffman(symbols)

    size_bytes = img_size[0].to_bytes(2, 'big') + img_size[1].to_bytes(2, 'big')
    return size_bytes + encoded_data


def Huffman_JPEG_decode(byte_data):
    # 1) Считываем размер изображения
    height = int.from_bytes(byte_data[0:2], 'big')
    width = int.from_bytes(byte_data[2:4], 'big')
    original_size = (height, width)

    # 2) Декодируем остальное с помощью обычного Хаффмана
    decoded_bytes = Huffman_decode(byte_data[4:])

    # 3) Преобразуем обратно в структуру [(dc_size, dc_code), [(run, size, code), ...]]
    decoded = list(decoded_bytes)
    i = 0
    blocks = []
    while i < len(decoded):
        dc_size = decoded[i]
        dc_val = decoded[i + 1]
        dc_code = bin(dc_val)[2:].zfill(dc_size) if dc_size > 0 else ''

        i += 2
        ac_list = []
        while i + 2 < len(decoded):
            run = decoded[i]
            size = decoded[i + 1]
            val = decoded[i + 2]
            i += 3

            if run == 0 and size == 0:
                ac_list.append((0, 0, ''))
                break

            bits = bin(val)[2:].zfill(size) if size > 0 else ''
            ac_list.append((run, size, bits))

        blocks.append(((dc_size, dc_code), ac_list))

    return [original_size] + blocks
