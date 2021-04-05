import math
import psutil
import os,sys


def findsquares(squares):
    winsquarenums = set()
    perrow = int(math.sqrt(squares))
    for s in range(squares-perrow-1):
        if s % perrow != perrow-1:
            winsquarenums.add(frozenset({s,s+1,s+perrow,s+perrow+1}))
    return winsquarenums

def remove_useless_wsn(winsquarenums):
    discardos = set()
    for ws1 in winsquarenums:
        for ws2 in winsquarenums:
            if ws1!=ws2 and ws1.issubset(ws2):
                discardos.add(ws2)
    for d in discardos:
        winsquarenums.discard(d)
def findfivers(squares):
    winsquarenums = set()
    perrow = int(math.sqrt(squares))
    for s in range(squares):
        if perrow - (s % perrow) >= 5:
            winsquarenums.add(frozenset({s,s+1,s+2,s+3,s+4}))
            if perrow - (s // perrow) >= 5:
                winsquarenums.add(frozenset({s,s+perrow+1,s+2*(perrow+1),s+3*(perrow+1),s+4*(perrow+1)}))
        if perrow - (s // perrow) >= 5:
            winsquarenums.add(frozenset({s,s+perrow,s+2*perrow,s+3*perrow,s+4*perrow}))
            if (s % perrow) >= 4:
                winsquarenums.add(frozenset({s,s+perrow-1,s+2*(perrow-1),s+3*(perrow-1),s+4*(perrow-1)}))
    return winsquarenums

def resources_avaliable():
    memory = psutil.virtual_memory()
    if memory.percent > 97:
        return False
    return True

room_num = 0
def provide_room_num():
    global room_num
    room_num+=1
    if room_num > 1e7:
        room_num = 0
    return room_num