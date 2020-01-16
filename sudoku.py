import numpy as np
import numba
from numba import vectorize,guvectorize,int32
import random
import zlib
import math
import json
from multiprocessing.pool import Pool

MASK_ANY = (1<<9) -1 # (1<<0) | (1<<1) | ... |  (1<<8) -- any possible digit
BIT_COUNTS = np.zeros(1<<9, dtype=np.int32) # Map from mask to number of bits on
for i in range(1<<9): BIT_COUNTS[i] = sum([(i&(1<<x))==(1<<x) for x in range(9)])
FIRST_BIT = np.zeros(1<<9, dtype=np.int32) # Map from mask to first bit that's on
for i in range(1,1<<9): FIRST_BIT[i] = min([x for x in range(9) if (i&(1<<x))==(1<<x)])

def mask_str(bitmask):
    return ",".join([str(d) for d in range(9) if (bitmask&(1<<d))==(1<<d)])

@numba.njit
def unset_bits(a,b):
    "Remove bits b from a, skipping any elements where only b bits are on"
    for i,val in np.ndenumerate(a):
        if val & b and not (val & b) == val:
            a[i] = (val | b) ^ b

solved_func = np.vectorize(lambda a: any([a==(1<<x) for x in range(9)]))

@numba.njit
def eliminate_tuples_from_slice(mask_slice):
    "Given a slice, e.g. mask[r,:], find any n-tuples that only occur n times and eliminate them from other positions"
    changed = False
    for bitmask in range(1, 1<<9):
        count = ((mask_slice & ~bitmask)==0).sum() # number of cells that contain only options within bitmask
        inv_count = ((mask_slice & bitmask)==0).sum() # number of cells that contain only options within bitmask
        if count == BIT_COUNTS[bitmask] and not (count + inv_count == 9):
            # We found one!
            #print(f"bitmask {mask_str(bitmask)} occurs exactly {count} times.")
            # Eliminate bitmask from the remaining cells
            unset_bits(mask_slice, bitmask)
            changed = True
    return changed

def count_solutions(board, cache={}, clone=True):
    board_hash = board.hash()
    if board_hash in cache:
        return 1

    if clone: # Avoids messing with the boards state
        board = board.clone()

    while board.eliminate(): pass

    solved_mask = solved_func(board.mask)
    if solved_mask.all():
        cache[board_hash] = board.hash()
        return 1

    solutions = set()
    for k,v in cache.items():
        solutions.add(v)
    if len(solutions) >= 2:
        return 2

    for r in range(9):
        for c in range(9):
            if solved_mask[r,c]: continue
            d = FIRST_BIT[board.mask[r,c]]
            copy = board.clone()
            copy.set_value(r,c,d)
            s=count_solutions(copy, cache=cache, clone=False)
            if s>2:
                return s

    solutions = set()
    for k,v in cache.items():
        solutions.add(v)
    return len(solutions)

class Board(object):
    def __init__(self, clone=None):
        "Creates a blank board."
        if clone:
            # If clone is passed in, we clone the board passed in
            self.mask = clone.mask.copy()
        else:
            self.mask = np.ones((9,9), dtype=np.int32) * MASK_ANY

    def clone(self):
        "Return a cloned board"
        return Board(clone=self)

    def hash(self):
        "Gets a hash of the current state"
        return zlib.adler32(self.mask.tobytes())


    def eliminate(self):
        changed = False
        for r in range(0,9):
            changed |= eliminate_tuples_from_slice(self.mask[:,r:r+1])
            changed |= eliminate_tuples_from_slice(self.mask[r:r+1,:])
        for sr in range(0,3):
            for sc in range(0,3):
                changed |= eliminate_tuples_from_slice(self.mask[3*sr:(3*sr + 3), 3*sc:(3*sc+3)])
        return changed


    def is_solved_valid(self):
        "Returns true if and only if the board is completely solved and the solution is valid."
        def is_valid_slice(mask):
            return all([(mask==(1<<d)).sum() == 1 for d in range(9)])

        for r in range(9):
            if not is_valid_slice(self.mask[:,r]): return False
            if not is_valid_slice(self.mask[r,:]): return False
        for sr in range(0,3):
            for sc in range(0,3):
                if not is_valid_slice(self.mask[3*sr:(3*sr + 3), 3*sc:(3*sc+3)]): return False
        return True

    def print(self):
        "Print the board in a user friendly way"
        for r in range(9):
            row_str = ""
            for c in range(9):
                row_str += " "
                opts = [str(d) for d in range(9) if (self.mask[r,c]&(1<<d))==(1<<d)]
                if len(opts) > 4:
                    row_str += "***".ljust(8)
                else:
                    row_str += ",".join(opts).ljust(8)
                row_str += " |"
                if c==2 or c== 5:
                    row_str += "|"

            print(row_str)
            if r == 2 or r==5:
                print("-"*len(row_str))
        print()

    def print_slice(self, sliced_mask):
        "Print one slice of the mask"
        sliced_mask = sliced_mask.ravel()
        for r in range(9):
            opts = [str(d) for d in range(9) if (sliced_mask[r]&(1<<d))==(1<<d)]
            print(",".join(opts))
        print()


    def set_value(self, r,c,val):
        "Set one value on the board.  Note that you pass in the value, not the bitmask"
        self.mask[r,c]=1<<val

    def rand_move(self):
        solved_mask = solved_func(self.mask)

        pos_moves = []
        for i,val in np.ndenumerate(solved_mask):
            if not val:
                pos_moves.append(i)

        if not len(pos_moves):
            return True

        r,c = random.choice(pos_moves)

        vals = self.mask[r,c]
        opts = []
        for d in range(9):
            if vals & (1<<d):
                opts.append(d)
        if len(opts):
            opt = random.choice(opts)
            self.mask[r,c] = 1 << opt
            return False
        else:
            print("No options for (%d,%d)" %(r,c))
            print(self.mask)
            raise Exception("Here")


    def solve_rand(self):
        while not self.rand_move():
            while self.eliminate():
                pass

    def to_json(self):
        lookup = {(1<<d):(d+1) for d in range(9)}

        return [[lookup.get(j,0) for j in i] for i in self.mask]


def gen_sudoku(*args):
    x=Board()
    x.solve_rand()
    if not x.is_solved_valid():
        # Failed to generate a random board
        # skip the invalid result
        return


    solved = x.to_json()

    pos=[(random.randint(0,8), random.randint(0, 8)) for i in range(50)] + [(r,c) for r in range(9) for c in range(9)]


    for r,c in pos:
        clone = x.clone()
        clone.mask[r,c]=MASK_ANY
        while clone.eliminate():
            pass
        if clone.is_solved_valid():
            x.mask[r,c]=MASK_ANY

    solved_mask = ~solved_func(x.mask)
    unsolved = x.to_json()
    print({"solved": solved, "unsolved": unsolved})
    return

from multiprocessing import Pool
with Pool(10) as p:
    p.map(gen_sudoku, range(100))



