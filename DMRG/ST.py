'''Suzuki Trotter Expansion'''

def streamline(tasks):
    '''Notice aliasing'''
    L = len(tasks)
    i=-1
    for j in range(L):
        if tasks[i][0] == tasks[j][0] and j:
            tasks[i][1] += tasks[j][1]
        else:
            i+=1
            tasks[i] = tasks[j].copy()
    return [tuple(t) for t in tasks[:i+1]]

def do_tasks(tasks, funs):
    tasks = streamline(tasks)
    ready = {t:funs[t[0]](t[1]) for t in set(tasks)}
    for task in tasks:
        ready[task]()


def ST1_tasks(L, n):
    return [(i, 1./n) for i in range(L)]*n

def ST1(funs, n):
    do_tasks(ST1_tasks(len(funs), n), funs)

def ST2_tasks(L, n):
    l= [[i, 0.5/n] for i in range(L)]
    l += l[::-1]
    return l*n

def ST2(funs, n):
    do_tasks(ST2_tasks(len(funs), n), funs)
