import sys

arg_length = len(sys.argv)
print(" - argv Length : ", arg_length)

print(' - argv : ', sys.argv)

for i in range(arg_length):
    print("argv[%d] : "%(i), sys.argv[i])