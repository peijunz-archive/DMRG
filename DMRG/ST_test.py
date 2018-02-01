from . import ST
s=''
def A(k):
    def _A():
        global s
        s+="A({})".format(k)
    return _A
def B(k):
    def _B():
        global s
        s+="B({})".format(k)
    return _B

def test_ST2():
    global s
    s=''
    ST.ST2((A, B), 5)
    s0 = 'A(0.1)B(0.2)A(0.2)B(0.2)A(0.2)B(0.2)A(0.2)B(0.2)A(0.2)B(0.2)A(0.1)'
    assert s==s0

def test_ST1():
    global s
    s=''
    ST.ST1((A, B), 5)
    s0 = 'A(0.2)B(0.2)A(0.2)B(0.2)A(0.2)B(0.2)A(0.2)B(0.2)A(0.2)B(0.2)'
    assert s==s0
