import os
import sys

sys.path.append('/homes/ugrads/ball4018')
sys.path.append('/homes/ugrads/ball4018/.local/lib/python2.7/site-packages/')


# find . -type f -name pitjob\* -exec rm {} \;
#os.system('cd ~/br8;find . -type f -name pitjob\* -exec rm {} \;')

EXECUTION_TEMPLATE = '''cd ~/br8
$HOME = /homes/ugrads/ball4018
cd ~/br8
python detection.py %d'''

EXEC_PATH = 'Runners/pitjob%d.sh'

def run_n(a, b):
    for j in xrange(a, b):
        run(j)

def run(j):
    path = EXEC_PATH%j
    with open(path, 'wb') as f:
        f.write(EXECUTION_TEMPLATE % j)
    print os.system("qsub %s" % path)

run_n(1, 2)