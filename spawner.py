import os


# find . -type f -name pitjob\* -exec rm {} \;
#os.system('cd ~/br8;find . -type f -name pitjob\* -exec rm {} \;')

EXECUTION_TEMPLATE = '''cd ~/br8
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