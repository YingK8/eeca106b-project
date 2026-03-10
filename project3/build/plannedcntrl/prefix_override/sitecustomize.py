import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cc/ee106b/sp26/class/ee106b-aac/eeca106b-project/project3/install/plannedcntrl'
