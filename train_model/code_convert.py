def convert_float_to_754(number):  # 传入的是一个浮点数
    number = float(number)
    if number == 0.0:
        return "00000000000000000000000000000000"
    if number < 0:
        sign = '1'  # 符号位
    else:
        sign = '0'
    integer_part_int = int(abs(number))

    integer_part_str = bin(integer_part_int)[2:]
    decimal_part_float = float(str(number)[str(number).index('.'):])
    tmp_length = 23 - (len(integer_part_str) - 1)
    decimal_part_str = ''
    tmp = decimal_part_float
    while tmp_length > 0:
        tmp = tmp * 2
        if tmp > 1:
            decimal_part_str += '1'
            tmp = tmp - 1.0
        elif tmp < 1:
            decimal_part_str += '0'
        else:
            decimal_part_str += '1'
            tmp = 0.0
        tmp_length -= 1
    if integer_part_int == 0:
        ex = -(decimal_part_str.index('1') + 1)  # 若整数部分的二进制是零，那么小数点是要向右移动的
        mantissa = decimal_part_str[decimal_part_str.index('1') + 1:]  # 尾数部分
        add_num_of_0 = abs(ex)
        for i in range(add_num_of_0):
            mantissa = mantissa + '0'
    else:
        mantissa = integer_part_str[1:] + decimal_part_str  # 尾数部分
        ex = len(integer_part_str) - 1
    exponet = bin(127 + ex)[2:]
    while len(exponet) < 8:
        exponet = '0' + exponet
    return sign + exponet + mantissa


# print(convert_float_to_754(1234))
# print(convert_float_to_754(1234)[1:9])


def convert_754_to_float(number_binary_str):
    if number_binary_str[0] == '0':

        sign = 1
    else:
        sign = -1

    ex = int(number_binary_str[1:9], 2) - 127  # 移动的位数
    if ex > 0:

        integer_part_str = '1' + number_binary_str[9:9 + ex]
        decimal_part_str = number_binary_str[9 + ex:]
    elif ex < 0:
        integer_part_str = '0'
        decimal_part_str = number_binary_str[9:]
        decimal_part_str = '1' + decimal_part_str
        for i in range(abs(ex) - 1):
            decimal_part_str = '0' + decimal_part_str
    else:
        integer_part_str = '1'
        decimal_part_str = number_binary_str[9:]
    integer_part_int = int(integer_part_str, 2)
    decimal_part_float = 0.0
    for index in range(len(decimal_part_str)):
        if decimal_part_str[index] == '1':
            decimal_part_float += 2 ** (-(index + 1))

    return sign * (integer_part_int + decimal_part_float)

# floa1 = 3.1
# floa2 = 2.79999999999999
#
# print(convert_float_to_754(0))
# print(convert_754_to_float('00000000000000000000000000000000'))



