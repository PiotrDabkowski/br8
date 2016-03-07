'''Functions used to calculate the gradient of boundary pixels

Takes image and performs operation on small NxN square where N is an odd number.

Uses least DISTANCE (actual distance from point to line not y distance) square fit to determine the gradient. Result is:
GGGG CC FF
FF: 0 to 3 for y offset  (0 centered, 1  more than 12% left. 2 more than 12% right. 3 more than 25% either direction (crap))
CC: Certainty information from 0 to 3. (0 least certain, 3 most)
GGGG: 0 to 11 for gradient. This gives 15 degree resolution 0 means 0deg 11 means 165deg. 0 looks right.

'''
import math
SQ_SIZE = 15
HSQ_SIZE = SQ_SIZE/2.0
POINTS = tuple((x,y) for x in xrange(-SQ_SIZE/2, SQ_SIZE/2+1) for y in xrange(-SQ_SIZE/2, SQ_SIZE/2+1))
G_SCALER = math.pi/12

def to_res(grad, cert, offset):
    return grad*16 + cert*4 + offset

def from_grad_res(res):
    offset = res%4
    res = res/4
    cert = res%4
    return res/4, cert, offset

def get_offset(A, B):
    offset = abs(B/(A**2 + 1)**0.5)
    return min(int(round(3*offset/HSQ_SIZE)), 3)

def get_grad(A):
    return int(round(A/G_SCALER)) % 12






def calc_grad(points):
    '''Returns A, B, E/Emax.  Ax+B'''
    Vx = 0
    Vy = 0
    Mxy = 0
    Mx = 0
    My = 0
    num_points = float(len(points))
    for point in points:
        x, y = point
        Vx += x*x
        Vy += y*y
        Mxy += x*y
        Mx += x
        My += y
    # Normalise params
    Vx /= num_points
    Vy /= num_points
    Mxy /= num_points
    Mx /= num_points
    My /= num_points
    # now the formula for optimal a is c2*a^2 + c1*a -c2 = 0
    c2 = Mxy - Mx*My
    if not c2:
        return math.pi/2, 0, 0
    c1 = (Vx - Mx*Mx) - (Vy - My*My)
    b = c1/float(c2)
    s = -b/2
    d = (b**2+4)**0.5 / 2
    A1 = s - d
    B1 = My - A1*Mx
    E1 = (A1**2*Vx + Vy + B1**2 + 2*A1*B1*Mx - 2*B1*My - 2*A1*Mxy) / (A1**2 + 1)
    A2 = s + d
    B2 = My - A2*Mx
    E2 = (A2**2*Vx + Vy + B2**2 + 2*A2*B2*Mx - 2*B2*My - 2*A2*Mxy) / (A2**2 + 1)
    if E1<E2:
        A, B = A1, B1
        E = E1
        EM = E2
    else:
        A, B = A2, B2
        E = E2
        EM = E1
    return math.atan(A), B, E/EM


def find_grad(im):
    '''see logbook for proof'''
    MIN_CONST = 10
    ps = []
    for point in POINTS:
        if im[point]:
            ps.append(point)

    A, B, ER = calc_grad(ps)
    grad = get_grad(A)
    if len(ps)>=MIN_CONST:
        return to_res(grad, 3-int(round(3*ER)), get_offset(A, B))
    return to_res(grad, 0, 3)


 # num_points = float(num_points)
 #    Vx /= num_points
 #    Vy += num_points
 #    Mxy += num_points
 #    Mx += num_points
 #    My += num_points






def gradient_show(im, val, expand=5):
    if not val:
        return
    expand = expand/2
    grad = math.tan(from_grad_res(val)[0]*G_SCALER)
    trans = False
    if abs(grad)>1:
        trans = True
        grad = 1/grad
    for x in xrange(-expand, expand+1):
        y = int(round(grad*x))
        if trans:
            im[y, x] = 255
        else:
            im[x, y] = 255

