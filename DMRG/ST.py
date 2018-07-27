'''Suzuki Trotter Expansion'''

from typing import Tuple, List, Callable

# A list of which task should do how long time sequentially
Taskline = List[Tuple[int, float]]

# A list of callable. Every callable element generates
# functions to apply operators
Pipeline = List[Callable[[int], Callable]]


def streamline(tasks: Taskline) -> Taskline:
    L = len(tasks)
    i = -1
    for j in range(L):
        if tasks[i][0] == tasks[j][0] and j:
            tasks[i][1] += tasks[j][1]
        else:
            i += 1
            tasks[i] = tasks[j].copy()
    return [tuple(t) for t in tasks[:i+1]]


def do_tasks(tasks: Taskline, funs: Pipeline)->None:
    tasks = streamline(tasks)
    '''Generate a list of callables from funs'''
    ready = {t: funs[t[0]](t[1]) for t in set(tasks)}
    for task in tasks:
        ready[task]()


def ST1_tasks(L: int, n: int)->Taskline:
    '''First Order Suzuki Trotter Expansion

    Args
    ----
        L:
            Number of operators
        n:
            1/n is the step size of expansion
    Returns
    ---
        tasks: Taskline
    '''
    return [[i, 1./n] for i in range(L)]*n


def ST2_tasks(L: int, n: int)->Taskline:
    '''Second Order Suzuki Trotter Expansion

    Args
    ----
        L:int
            Number of operators
        n:int
            Number of time slices
    Returns
    ----
        tasks: Taskline
    '''
    l = [[i, 0.5/n] for i in range(L)]
    l += l[::-1]
    return l*n


def ST1(funs: Pipeline, n: int)->None:
    '''Do functions by ST1 expansion

    Args
    ----
        funs: Pipeline
        n: int
            Number of time slices
    '''
    do_tasks(ST1_tasks(len(funs), n), funs)


def ST2(funs: Pipeline, n: int)->None:
    '''Do functions by ST2 expansion

    Args
    ----
        funs: Pipeline
        n: int
            Number of time slices
    '''
    do_tasks(ST2_tasks(len(funs), n), funs)
