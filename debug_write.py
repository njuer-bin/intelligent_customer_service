import os
with open('debug_out.txt', 'w') as f:
    f.write('hello from python\n')
    f.write(str(os.listdir('.')))