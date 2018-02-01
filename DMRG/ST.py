'''Suzuki Trotter Expansion'''
def nothing(k):
    return lambda:0

def ST2(f_list, n):
    if len(f_list)<2:
        f_list = list(f_list) + [nothing]
    delta = 1./n
    half = [f(delta/2) for f in f_list[:-1]]
    full = [f_list[0](delta), f_list[-1](delta)]
    def middle():
        for h in half[1:]:
            h()
        full[-1]()
        for h in half[1:][::-1]:
            h()

    half[0]()
    for i in range(n-1):
        middle()
        full[0]()
    middle()
    half[0]()

def ST1(f_list, n):
    delta = 1./n
    full = [f(delta) for f in f_list]
    for i in range(n):
        for f in full:
            f()
