'''Suzuki Trotter Expansion'''

def streamline(tasks):
    L = len(tasks)
    j=0
    for i in range(1, L):
        if tasks[i][0] == tasks[j][0]:
            tasks[j][1] += tasks[i][1]
        else:
            j+=1
            tasks[j] = list(tasks[i])
    return [tuple(i) for i in tasks[:j+1]]

def do_tasks(tasks, funs):
    tasks = streamline(tasks)
    todo = {t:funs[t[0]](t[1]) for t in set(tasks)}
    for task in tasks:
        todo[task]()


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
