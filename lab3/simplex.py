from scipy.optimize import linprog
import numpy as np
from functools import cmp_to_key

def lex(v1, v2):
    res = v1[0] - v2[0]
    for i in res:
        if abs(i) < 1e-5:
            return False if i > 0 else True
    return False

class SimplexAlgorithm:
    table = None
    lr = None
    lc = None
    glob_row_id = 0
    glob_col_id = 0

    def gen_matrix(self, var, cons):
        self.table = np.zeros((cons+1, var+cons+2), dtype=float)
        self.lr = len(self.table[:, 0])
        self.lc = len(self.table[0, :])



    # checks the furthest right column for negative values ABOVE the last row. If negative values exist, another pivot is required.
    def iterate_column(self):
        m = min(self.table[:-1, -1])
        return (False if m >= 0 else True)

    # checks that the bottom row, excluding the final column, for negative values. If negative values exist, another pivot is required.
    def iterate_row(self):
        m = min(self.table[self.lr - 1, :-1])
        return (False if m >= 0 else True)


    def neg_row_id(self):
        m = min(self.table[:-1, self.lc - 1])
        n = None
        if m <= 0:
            n = np.where(self.table[:-1, self.lc - 1] == m)[0][0]
        else:
            n = None
        return n

    def neg_col_id(self):
        m = min(self.table[self.lr - 1, :-1])
        if m <= 0:
            n = np.where(self.table[self.lr - 1, :-1] == m)[0][0]
        else:
            n = None
        return n

    # locates pivot element in table to remove the negative element from the furthest right column.
    def col_pivot(self):
            total = []
            r = self.neg_row_id()
            row = self.table[r,:-1]
            m = min(row)
            c = np.where(row == m)[0][0]
            col = self.table[:-1,c]
            #smallest positive ratio
            
            for i, b in zip(col,self.table[:-1,-1]):
                if i**2 > 0 and b / i > 0:
                    total.append(b / i)
                else:
                    total.append(0)
            element = max(total)
            for t in total:
                if t > 0 and t < element:
                    element = t
                else:
                    continue

            index = total.index(element)
            return (index, c)

    # similar process, returns a specific array element to be pivoted on.
    def row_pivot(self):
        if self.iterate_row():
            n = self.neg_col_id()
            vecs = []
            for r in range(0, self.lr - 1):
                qis = self.table[r][n]
                if qis > 0:
                    vecs.append([np.array([self.table[r][j] / qis for j in range(self.lc)]), r])
            vecs = sorted(vecs, key=cmp_to_key(lex))
            return (vecs[0][1], n)

    def convert_min(self):
        self.table[-1,:-2] = [-1*i for i in self.table[-1,:-2]]
        self.table[-1,-1] = -1*self.table[-1,-1]

    def gen_var(self):
        var = self.lc - self.lr -1
        v = []
        for i in range(var):
            v.append('x'+str(i+1))
        return v

    # pivots the table such that negative elements are purged from the last row and last column
    def pivot(self, idx):
        row, col = idx
        t = np.zeros((self.lr, self.lc))
        pr = self.table[row,:]
        if self.table[row,col] ** 2 > 0:
            e = 1 / self.table[row,col]
            r = pr*e
            for i in range(len(self.table[:,col])):
                k = self.table[i,:]
                c = self.table[i,col]
                if list(k) == list(pr):
                    continue
                else:
                    t[i,:] = list(k-r*c)
            t[row,:] = list(r)
            return t
        else:
            print('Zero entry')

    def constrain(self, eq_lhs, eq_rhs):
        var = self.lc - self.lr - 1
        j = 0
        while j < self.lr:
            row_check = self.table[j,:]
            total = 0
            for i in row_check:
                total += float(i**2)
            if total == 0:
                row = row_check
                break
            j += 1
        for i in range(len(eq_lhs)):
            row[i] = eq_lhs[i]
        row[-1] = eq_rhs
        # add slack variable according to location in table.
        row[var + j] = 1

    def obj(self, eq):
        row = self.table[self.lr - 1,:]
        for i in range(len(eq)):
            row[i] = eq[i]*-1
        row[-2] = 1

    def minz(self):
        self.convert_min()
        while self.iterate_column():
            self.table = self.pivot(self.col_pivot())
        while self.iterate_row():
            self.table = self.pivot(self.row_pivot())
        var = self.lc - self.lr - 1
        i = 0
        val = {}
        for i in range(var):
            col = self.table[:,i]
            s = sum(col)
            m = max(col)
            if float(s) == float(m):
                loc = np.where(col == m)[0][0]
                val[self.gen_var()[i]] = self.table[loc, -1]
            else:
                val[self.gen_var()[i]] = 0
        val['min'] = self.table[-1,-1] * -1
        for k,v in val.items():
            val[k] = round(v,6)

        return val

    def linprog(self, f_obj, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq):
        self.gen_matrix(len(f_obj), len(lhs_ineq) + 2 * len(lhs_eq))
        for i in range(len(lhs_ineq)):
            self.constrain(lhs_ineq[i], rhs_ineq[i])
        for i in range(len(lhs_eq)):
            self.constrain(lhs_eq[i], rhs_eq[i])
            self.constrain([-j for j in lhs_eq[i]], -rhs_eq[i])
        self.obj(f_obj)
        return self.minz()


def simple_test(f_obj, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq):    
    opt = linprog(c=f_obj, A_ub=lhs_ineq, b_ub = rhs_ineq, A_eq = lhs_eq, b_eq = rhs_eq, method = 'revised simplex')
    algo = SimplexAlgorithm()
    res = algo.linprog(f_obj, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq)
    assert(abs(opt.fun - res['min']) < 1e-5)


def simple_test_eq(f_obj, lhs_eq, rhs_eq):    
    opt = linprog(c=f_obj, A_eq = lhs_eq, b_eq = rhs_eq, method = 'revised simplex')
    algo = SimplexAlgorithm()
    res = algo.linprog(f_obj, [], [], lhs_eq, rhs_eq)
    assert(abs(opt.fun - res['min']) < 1e-5)

def runAllTests():
    #task 8:
    f_obj = [1, 1, -2, -3]
    lhs_eq = [[2, 1, 1, 0], [-1, 2, 0, 1]]
    rhs_eq = [1, 1]
    simple_test_eq(f_obj, lhs_eq, rhs_eq)
    #it is the minimum

    #testing
    #1
    f_obj = [-6, -1, -4, 5]
    lhs_eq = [[3, 1, -1, 1], [5, 1, 1, -1]]
    rhs_eq = [4, 4]
    simple_test_eq(f_obj, lhs_eq, rhs_eq)
    #2
    f = [-1, -2, -3, 1]
    l = [[1, -3, -1, -2], [1, -1, 1, 0]]
    r = [-4, 0]
    simple_test_eq(f, l, r)
    #3
    f_obj = [-1, -2]
    lhs_ineq = [[ 2,  1], 
                [-4,  5], 
                [ 1, -2]]
    rhs_ineq = [20, 10, 2]
    lhs_eq = [[-1, 5]]
    rhs_eq = [15]
    simple_test(f_obj, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq)
    #4
    f = [-1, -1, -1, 1, -1]
    leq = [[1, 1, 2, 0, 0], [0, -2, -2, 1, -1]]
    req = [4, -6]
    l_ineq = [[-1, 1, -6, -1, -1]]
    r_ineq = [0]
    simple_test(f, l_ineq, r_ineq, leq, req)
    #5
    f = [-1, 4, -3, 10]
    leq = [[1, 1, -1, -10], [1, 14, 10, -10]]
    req = [0, 11]
    l_ineq = [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    r_ineq = [0, 0, 0, 0]
    simple_test(f, l_ineq, r_ineq, leq, req)
    #6
    f = [-1, 5, 1, -1]
    ineq_l = [[1, 3, 3, 1], [2, 0, 3, -1]]
    ineq_r = [3, 4]
    simple_test(f, ineq_l, ineq_r, [[0, 0, 0, 0]], [0])
    #7
    f = [-1, -1, 1, -1, 2]
    eq_l = [[3, 1, 1, 1, -2], [6, 1, 2, 3, -4], [10, 1, 3, 6, -7]]
    eq_r =[10, 20, 30]
    simple_test_eq(f, eq_l, eq_r)
    print("All tests passed.")