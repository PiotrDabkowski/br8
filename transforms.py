from PIL import Image
from gradient_calc import find_grad, from_grad_res, gradient_show

HV_MOVES = ((0,1), (0,-1), (1,0),(-1,0))
SMALL_AREA = 15



#         Retina Regions :
#
#                F F F
#              E D D D E
#            F D C B C D F
#            F D B A B D F
#            F D C B C D F
#              E D D D E
#                F F F
#
# Works best when ABC are Inner and DEF are outer.  Tried different combinations but this works best.
# Formula:  (InnerNum*OuterSum - OuterNum*InnerSum) / (InnerNum + OuterNum)
A_PX = {(0,0)}
B_PX = set(HV_MOVES)
C_PX = {(-1,-1), (1,1),(-1,1),(1,-1)}
E_PX = {(-2,-2), (2,2),(-2,2),(2,-2)}
D_PX = set([(x,y) for x in xrange(-2,3) for y in xrange(-2,3)]) - A_PX - B_PX - C_PX - E_PX
F_PX = {(1, 3), (-3, 1), (3, -1), (-3, 0), (-1, 3), (-3, -1), (3, 1), (3, 0), (-1, -3), (0, -3), (1, -3), (0, 3)}



IN_PIXELS = A_PX | B_PX | C_PX
OUT_PIXELS =  E_PX | D_PX | F_PX



IN_PIXELS = tuple(IN_PIXELS)
OUT_PIXELS = tuple(OUT_PIXELS)

NOT_LOOP_DA_LOPS = set([])
def is_small_closed_loop_da_loop(im):
    '''Has to start with black dot. Searches whether its a small closed black area  (less than SMALL_AREA)
    Can only walk horiz and vert so no diag

    I am using previously found loop da loops to make that faster!'''
    if im[0,0]:
        return 0
    cx, cy = im.current_px
    if (cx-1, cy) in NOT_LOOP_DA_LOPS or (cx, cy-1) in NOT_LOOP_DA_LOPS:
        NOT_LOOP_DA_LOPS.add(im.current_px)
        return 0  # cant be loop da loop
    # new unconnected black dot. have to run exhaustive search
    found_dots = set(((0,0)))
    current_area = 1
    new_dots = [(0,0)]
    while new_dots:
        found_here = []
        for sx, sy in new_dots:
            for ox, oy in HV_MOVES:
                dot = (sx + ox, sy + oy)
                if dot in found_dots or im[dot]:
                    continue
                found_here.append(dot)
                found_dots.add(dot)
                current_area += 1
                if current_area>SMALL_AREA:
                    NOT_LOOP_DA_LOPS.add(im.current_px)
                    return 0
        new_dots = found_here
    return 255


def retina(im):
    val = (len(IN_PIXELS)*sum(im[px] for px in OUT_PIXELS) - len(OUT_PIXELS)*sum(im[px] for px in IN_PIXELS))/(len(IN_PIXELS)+len(OUT_PIXELS))
    if val<0:
        return 0
    cand = min(abs(val), 255)
    return cand

def avg(a, b):
    return (a[0,0] + b[0,0])/2


def thresh(im):
    if im[0,0]>135:
        return 0
    return 255

def nthresh(im):
    if im[0,0]<135:
        return 0
    return 255

def show_only_certain(im):
    g, c, f = from_grad_res(im[0,0])
    if c<1 or f!=0:
        return 0
    return im[0,0]

def reverse(im):
    return -im[0,0]%256

def to_proper(path):
    Image.open(path).convert('L').convert('LA').save(path)




