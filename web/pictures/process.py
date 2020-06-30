
from pictures.map_seq import ccl, map_segment

def ImgProcess(code):

    print(111111)
    map_segment(r"media/out/%s/1.png" % code, "media/out/%s/thinning.png" % code)
    print(222222)
    ccl("media/out/%s/1.png" % code,"media/out/%s/ccl.png" % code, True)
    print(3333333)
    return
if __name__ == '__main__':
    ImgProcess('YrEE')
