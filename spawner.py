import os
import sys

sys.path.append('/homes/ugrads/ball4018')

# find . -type f -name pitjob\* -exec rm {} \;
#os.system('cd ~/br8;find . -type f -name pitjob\* -exec rm {} \;')

EXECUTION_TEMPLATE = '''cd ~/br8
$HOME = /homes/ugrads/ball4018
python detection.py %d'''

EXEC_PATH = 'Runners/pitjob%d.sh'

def run_n(n):
    for j in xrange(n):
        run(j)

def run(j):
    path = EXEC_PATH%j
    with open(path, 'wb') as f:
        f.write(EXECUTION_TEMPLATE % j)
    print os.system("qsub %s" % path)

run(0)