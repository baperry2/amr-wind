import os
import sys

if len(sys.argv) > 1:
    dirname = sys.argv[1]
else:
    dirname = ""

prefix =  '${AMRW_SRC_DIR}/'

def get_whitespace(recursion_level):
    return '  ' * recursion_level

def read_file(dirname, srcs_list, dirs_list, level = 0):
    if dirname not in dirs_list:
        dirs_list.append(dirname)
    finame = os.path.join(dirname,"CMakeLists.txt")
    print(get_whitespace(level), 'start reading ', finame)
    with open(os.path.join(dirname,"CMakeLists.txt") ) as fi:
        READING_SOURCES = False
        for line in fi:
            if 'target_sources' in line:
                if ')' not in line:
                    print(get_whitespace(level), 'start reading for sources at line ',line.strip())
                    READING_SOURCES = True
                else :
                    print(get_whitespace(level), 'skip reading sources at line ',line.strip())

            elif READING_SOURCES:
                if ')' in line:
                    READING_SOURCES = False
                    print(get_whitespace(level), 'stop reading for sources at line ', line.strip())

                testline = line.replace(')','').split('#')[0].strip()
                if testline != 'PRIVATE':
                    if len(testline) > 0:
                        srcs_list.append(prefix+os.path.join(dirname, testline))

            if 'add_subdirectory' in line:
                subdirname = line.split('(')[1].split(')')[0]
                pathname = os.path.join(dirname, subdirname)
                print(get_whitespace(level), 'entering subdirectory ', pathname)
                read_file(pathname, srcs_list, dirs_list, level+1)
    print(get_whitespace(level), 'finish reading ', finame)

srcs = []
dirs = []
read_file(dirname, srcs, dirs)
print('')
print('SRCS')
for src in srcs:
    print('')
    print(src, flush=True)

#print('')
#print('DIRS')
#dirs = [prefix + ddir for ddir in dirs]
#for ddir in dirs:
#    print('')
#    print('{}'.format(ddir), flush=True)
