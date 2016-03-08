from utils import get_point_spans, point_inside_polygon

class Box:
    '''Can be ANY polygon hehe. Much room for improvement if I decide to track the membrane instead of dumb rect box'''
    def __init__(self, points):
        self.points = points
        self.xspan, self.yspan = get_point_spans(points)


    def iter_int_points(self):
        # dumb but easy approach. fast enough when compared to neural network anyway :)
        for x in xrange(*self.xspan):
            for y in xrange(*self.yspan):
                if point_inside_polygon(x, y, self.points):
                    yield (x,y)

    def count_in_box(self, matrix):
        '''Counts number of true entries in matrix that are inside the box'''
        count = 0
        for point in self.iter_int_points():
            try:
                if matrix[point]:
                    count += 1
            except:
                print point
        return count

