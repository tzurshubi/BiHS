import os
from utils.utils import *
from types import SimpleNamespace

os.system('cls' if os.name == 'nt' else 'clear')

# 2d
d2 = SimpleNamespace()
d2.longest_cib_len = 4
d2.longest_cib = [0, 1, 3, 2]
d2.longest_cib_on_bits = [[], [1], [1, 2], [2]]

# 3d
d3 = SimpleNamespace()
d3.longest_cib_len = 6
d3.longest_cib = [0, 1, 3, 7, 6, 4]
d3.longest_cib_on_bits = [[], [1], [1, 2], [1, 2, 3], [2, 3], [3]]

# 4d
d4 = SimpleNamespace()
d4.longest_cib_len = 8
d4.longest_cib = [0, 1, 3, 7, 6, 14, 10, 8]
d4.longest_cib_on_bits = [[], [1], [1, 2], [1, 2, 3], [2, 3], [2, 3, 4], [2, 4], [4]]

# 5d
d5 = SimpleNamespace()
d5.longest_cib_len = 14
d5.longest_cib = [0, 1, 3, 7, 6, 14, 12, 13, 29, 31, 27, 26, 18, 16]
d5.longest_cib_on_bits = [[], [1], [1, 2], [1, 2, 3], [2, 3], [2, 3, 4], [3, 4], [1, 3, 4], [1, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 4, 5], [2, 4, 5], [2, 5], [5]]

# 6d
d6 = SimpleNamespace()
d6.longest_cib_len = 26
d6.longest_cib = [0, 1, 3, 7, 15, 31, 29, 25, 24, 26, 10, 42, 43, 59, 51, 49, 53, 37, 45, 44, 60, 62, 54, 22, 20, 4]
d6.longest_cib_on_bits = [[], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 3, 4, 5], [1, 4, 5], [4, 5], [2, 4, 5], [2, 4], [2, 4, 6], [1, 2, 4, 6], [1, 2, 4, 5, 6], [1, 2, 5, 6], [1, 5, 6], [1, 3, 5, 6], [1, 3, 6], [1, 3, 4, 6], [3, 4, 6], [3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 5, 6], [2, 3, 5], [3, 5], [3]]

# 7d
d7 = SimpleNamespace()
d7.longest_cib_len = 48
d7.longest_cib = [0, 1, 3, 7, 15, 13, 12, 28, 30, 26, 27, 25, 57, 56, 40, 104, 72, 73, 75, 107, 111, 110, 46, 38, 36, 52, 116, 124, 125, 93, 95, 87, 119, 55, 51, 50, 114, 98, 66, 70, 68, 69, 101, 97, 113, 81, 80, 16]

# 8d
d8 = SimpleNamespace()
d8.longest_cib_len = 96
d8.longest_cib = [0, 1, 3, 7, 6, 14, 12, 13, 29, 31, 27, 26, 18, 50, 54, 62, 60, 56, 57, 49, 53, 37, 101, 69, 68, 196, 132, 133, 149, 151, 150, 158, 156, 220, 92, 94, 86, 87, 119, 115, 123, 122, 250, 254, 255, 191, 187, 179, 163, 167, 231, 230, 226, 98, 66, 74, 202, 200, 136, 137, 139, 143, 207, 205, 237, 173, 172, 174, 170, 42, 43, 47, 111, 110, 108, 104, 105, 73, 89, 217, 219, 211, 195, 193, 225, 241, 245, 244, 116, 112, 80, 208, 144, 176, 160, 32]

ds = [{}, d2, d3, d4, d5, d6]

# Play
dim = 4
d = ds[dim-1]
print(len(coil_dim_crossed_to_vertices("012031041203523120324650760410316713052307206234260275371601261560127026012054713415420725765475")))
print("---------------------------")
print_bits_pattern(d.longest_cib_on_bits)
print("---------------------------")
print_bit_statistics(d.longest_cib_on_bits)
print("---------------------------")
print(bits_to_moves(d.longest_cib_on_bits))
