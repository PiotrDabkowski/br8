import os
import sys
from utils import to_bool_arr
import numpy as np
from PIL import Image
try:
    import tifffile
except:
    print 'Cant import tifffile'

sys.path.append('/homes/ugrads/ball4018')
sys.path.append('/homes/ugrads/ball4018/.local/lib/python2.7/site-packages/')


# cd ~/br8;find . -type f -name pitjob\* -exec rm {} \;
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

def compose_n(a, b):
    res_path = 'Results/res%d.tif'
    arrs = []
    for j in xrange(a, b):
        arrs.append(to_bool_arr(Image.open(res_path%j)).reshape((1, 1024, 1024)))
    tifffile.imsave('main_result.tif', (np.concatenate(tuple(arrs))*255).astype(np.uint8))

def evidence_n(a, b):
    from detection import evidence_to_result
    for j in xrange(a, b):
        evidence_to_result(num=j, overwrite=None, save=True)

def compare_n(a, b):
    from detection import compare_results
    for j in xrange(a, b):
        compare_results(j)




if __name__=='__main__':
    typ = sys.argv[1]
    start = stop = 0
    if len(sys.argv)==4:
        _, _, start, stop = sys.argv
        start, stop = int(start), int(stop)
    if typ=='run':
        run_n(start, stop)
    elif typ=='clear':
        os.system('cd ~/br8;find . -type f -name pitjob\* -exec rm {} \;')
    elif typ=='compose':
        compose_n(start, stop)
    elif typ=='evidence':
        evidence_n(start, stop)
    elif typ=='compare':
        compare_n(start, stop)
    else:
        print 'Invalid args, check source code'