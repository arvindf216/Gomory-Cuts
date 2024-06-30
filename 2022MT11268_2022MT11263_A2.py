import json
import sys
import numpy as np
from math import *
from fractions import Fraction as fr

if (len(sys.argv)>1):
    file_name = sys.argv[1]
else:
    file_name = "input_ilp.txt"

# ------------------------------------------------------------------------------------------------------------------------------

#File Handling, obtaining dct dictionary
def file_handling(filename,dct):
    with open(filename, 'r') as file:
        # Read all lines into a list
        lines = [line.strip() for line in file.readlines()]
        motive=''
        for line in lines:
            if line=='':
                continue
            elif line[0]=='[':
                motive = line
            else:
                if motive=='[A]' or motive=='[b]' or motive=='[c]':
                    lst = json.loads('[' + line + ']')
                    dct[motive].append(lst)
                else:
                    dct[motive].append(line)

    var = 'x'
    dct['vars']=[var+str(i) for i in range(len(dct['[A]'][0]))]

    n_initial = len(dct['vars'])
    n_vars = len(dct['[A]'][0])

    #adding slack variables            
    _ = 0
    for constraint in dct['[constraint_types]']:
        if constraint=='<=':
            dct['vars'].append(var + str(n_vars))
            n_vars+=1
            dct['[constraint_types]'][_] = '='
            for i in range(len(dct['[A]'])):
                if i==_:
                    dct['[A]'][i].append(1)
                else:
                    dct['[A]'][i].append(0)
            dct['[c]'][0].append(0)
        elif constraint=='>=':
            dct['vars'].append(var + str(n_vars))
            n_vars+=1
            dct['[constraint_types]'][_] = '='
            for i in range(len(dct['[A]'])):
                if i==_:
                    dct['[A]'][i].append(-1)
                else:
                    dct['[A]'][i].append(0)
            dct['[c]'][0].append(0)
        else:
            _+=1
            continue
        _+=1

    # changing maximize to minimize
    if (dct['[objective]'][0]=='maximize'):
        dct['[c]'][0]=[(-1)*dct['[c]'][0][i] for i in range(len(dct['[c]'][0]))]

    for i in range(len(dct['[A]'])):
        for j in range(len(dct['[A]'][0])):
            dct['[A]'][i][j] = fr(dct['[A]'][i][j])
        
    for i in range(len(dct['[b]'])):
        dct['[b]'][i][0] = fr(dct['[b]'][i][0])

    for i in range(len(dct['[c]'][0])):
        dct['[c]'][0][i] = fr(dct['[c]'][0][i]) 

    # conversion to numpy array

    for key in dct:
        if (key!='vars' and key!='[objective]'):
            dct[key] = np.array(dct[key],dtype=fr)

    # removing constraint types from dct, no longer required since all constraints have been converted to <= type
    dct.pop('[constraint_types]')    

    return n_initial

# ----------------------------------------------------------------------------------------------------------------------------

# Two Phase Simplex for obtaining initial solution
def two_phase(dct,bss,n_initial):

    ans = {'initial_tableau': None, 'final_tableau': None, 'solution_status': '', 'optimal_solution': None, 'optimal_value': None}

    # making b>=0
    n_original = len(dct['[A]'][0])
    unbdd = False

    for i in range(len(dct['[b]'])):
        if dct['[b]'][i][0]<fr(0):
            dct['[A]'][i] = fr(-1)*dct['[A]'][i]
            dct['[b]'][i] = fr(-1)*dct['[b]'][i]

    # checking if it is required to add artificial variables

    nb = 0 # number of basis vectors obtained
    ids = []  

    for i in range(len(dct['[A]'][0])):
        # iterating over all the variables
        if (nb==len(dct['[A]'])):  #basis has m variables
            break
        column_vector = dct['[A]'][:, i]
        cond1 = (np.sum(column_vector==0) == (len(column_vector) - 1))
        is_condition_met = False
        if (cond1):
            for j in range(len(column_vector)):
                if (column_vector[j]>fr(0)):
                    ids.append(j)
                    is_condition_met = True
                    break
        if (is_condition_met):
            nb+=1
            bss['idx'].append(i)
            bss['vars'].append(dct['vars'][i])

    flag = True  #whether artificial simplex is required or not
    if (nb==len(dct['[A]'])):   # if we have m many basis variables, then artificial simplex not required
        flag = False

    if (flag):
        #perform tableau method for artificial simplex and drive artificial variables out of basis
        m = len(dct['[A]'])
        n = len(dct['[A]'][0])

        #new cost vector for phase 1
        c_new = np.full(dct['[c]'].shape, fr("0"), dtype = fr)

        #adding artificial varibles
        var = 'y'
        art = 0
        for i in range(m):
            if i not in ids:
                bss['vars'].insert(i,var+str(art))
                bss['idx'].insert(i,len(dct['[A]'][0]))
                dct['vars'].append(var+str(art))
                col = np.full((m, 1), fr("0"), dtype = fr)
                col[i][0]=1
                dct['[A]'] = np.hstack((dct['[A]'],col))
                c_new = np.append(c_new, [[fr("1")]], axis=1) 
                art+=1    

        m = len(dct['[A]'])
        n = len(dct['[A]'][0]) 

        #initializing tableau
        dct['tableau']=np.full((m+1, n+1), fr("0"), dtype = fr)
        bss['mtx'] = dct['[A]'][:,bss['idx']]
        B = bss['mtx']
        B_1 = np.copy(bss['mtx'])
        np.fill_diagonal(B_1, fr(1) / np.diag(B))

        # B_1b
        xB = B_1@dct['[b]']
        dct['tableau'][1:,0]= (B_1@dct['[b]']).reshape(m,)

        # B-1 A
        h = B_1@dct['[A]']
        dct['tableau'][1:,1:] = h

        # basic costs
        cB = c_new[:,bss['idx']]
        # reduced costs
        dct['tableau'][0,1:] = (c_new +fr(-1)*cB@h).reshape(n,)

        # negative of current cost
        dct['tableau'][0,0] = (fr(-1)*cB@xB)[0][0]

        # applying tableau algorithm
        dct['[c_new]'] = c_new
        ans['initial_tableau'] = np.copy(dct['tableau'][1:])

        while True:
            found = False
            j = 0
            for idx in range(1,n+1):
                if (dct['tableau'][0,idx]<fr("0")):
                    found = True
                    j = idx  #entering column
                    break

            if not found:  # all reduced costs are non-negative
                break  #hence we have achieved optimal cost

            u = dct['tableau'][1:,j]

            if (np.all(u<=fr("0"))):
                unbdd = True  # optimal cost is -inf
                break

            ratios = []

            for i in range(len(u)):
                if (u[i]>fr("0")):
                    ratios.append(dct['tableau'][i+1,0]/u[i])

            mini = min(ratios)

            l = 0  # minimizing index, exiting column
            for i in range(len(u)):
                if (u[i]>fr("0")):
                    if dct['tableau'][i+1,0]/u[i] == mini:
                        l = i+1
                        break

            dct['tableau'][l] = dct['tableau'][l]/dct['tableau'][l][j]  # making pivot element = 1

            for k in range(m+1):  # making all other entries of pivot column = 0
                if k!=l:
                    dct['tableau'][k] = dct['tableau'][k] +fr(-1)*dct['tableau'][k][j]*dct['tableau'][l]

            #updating basis indices and basis varibles vector
                    
            bss['vars'][l-1] = dct['vars'][j-1]
            bss['idx'][l-1] = j-1

        if (unbdd):
            ans['final_tableau'] = dct['tableau'][1:]
            ans['solution_status'] = 'unbounded'
            return ans

        if (abs(dct['tableau'][0][0])>fr("0")):
            ans['final_tableau'] = dct['tableau'][1:]
            ans['solution_status'] = 'infeasible'
            return ans 

        drive_out = []
        for k in range(len(bss['vars'])):
            if bss['vars'][k][0]=='y':
                drive_out.append(k)

        popy = []
        while (len(drive_out)!=0):  # aritficial variables are there in the basis
            l = drive_out.pop()
            all_zero = True
            for j in range(n_original):
                if dct['tableau'][l+1][j+1]!=0:
                    all_zero=False

                    dct['tableau'][l+1] = dct['tableau'][l+1]/dct['tableau'][l+1][j+1]  # making pivot element = 1
                    for k in range(m+1):  # making all other entries of pivot column = 0
                        if k!=l+1:
                            dct['tableau'][k] = dct['tableau'][k] +fr(-1)* dct['tableau'][k][j+1]*dct['tableau'][l+1]

                    bss['vars'][l] = dct['vars'][j]
                    bss['idx'][l] = j

            if (all_zero):
                popy.append(l+1)

        for id in popy:
            dct['[A]'] = np.delete(dct['[A]'] ,id-1, axis = 0)
            dct['[b]'] = np.delete(dct['[b]'] ,id-1, axis =0)
            dct['tableau'] = np.delete(dct['tableau'] ,id, axis=0)
            bss['vars'].pop(id-1)
            bss['idx'].pop(id-1)

        bss['mtx'] = dct['[A]'][:,bss['idx']]
        dct.pop('[c_new]')

        start_idx = 0 ### index where artificial variables begin

        for k in range(len(dct['vars'])):
            if (dct['vars'][k][0]=='y'):
                start_idx=k
                break

        if (k!=n+1):
            tableau = np.delete(dct['tableau'] , np.s_[start_idx+1:], axis=1)
            dct['tableau'] = tableau
            A = np.delete(dct['[A]'], np.s_[start_idx:], axis=1)
            dct['[A]'] = A
            del dct['vars'][start_idx:]

        n = len(dct['[A]'][0])
        h = dct['tableau'][1:,1:] 
        xB = dct['tableau'][1:,0]

        # basic costs
        cB = dct['[c]'][:,bss['idx']]

        # reduced costs
        dct['tableau'][0,1:] = (dct['[c]'] +fr(-1)*cB@h).reshape(n,)

        # negative of current cost
        dct['tableau'][0,0] = (fr(-1)*cB@xB)[0]

    else: 

        # artificial simplex not required, initialize tableau for normal simplex
        m = len(dct['[A]'])
        n = len(dct['[A]'][0])

        #initializing tableau

        dct['tableau']=np.full((m+1, n+1), fr("0"), dtype = fr)
        bss['mtx'] = dct['[A]'][:,bss['idx']]
        B = bss['mtx']
        B_1 = np.copy(bss['mtx'])
        np.fill_diagonal(B_1, 1 / np.diag(B))

        # B_1b
        xB = B_1@dct['[b]']
        dct['tableau'][1:,0]= (B_1@dct['[b]']).reshape(m,)

        # B-1 A
        h = B_1@dct['[A]']
        dct['tableau'][1:,1:] = h

        # basic costs
        cB = dct['[c]'][:,bss['idx']]

        # reduced costs
        dct['tableau'][0,1:] = (dct['[c]'] + fr(-1)*cB@h).reshape(n,)

        # negative of current cost
        dct['tableau'][0,0] = (fr(-1)*cB@xB)[0][0]

        ans['initial_tableau'] = np.copy(dct['tableau'][1:])

    ######### phase-2 begins ###############
        
    m = len(dct['[A]'])
    n = len(dct['[A]'][0])
    unbdd = False

    while True:
        found = False
        j = 0
        for idx in range(1,n+1):
            if (dct['tableau'][0,idx]<fr("0")):
                found = True
                j = idx  #entering column
                break

        if not found:  # all reduced costs are non-negative
            break  #hence we have achieved optimal cost

        u = dct['tableau'][1:,j]

        if (np.all(u<=fr("0"))):
            unbdd = True  # optimal cost is -inf
            break

        ratios = []

        for i in range(len(u)):
            if (u[i]>fr("0")):
                ratios.append(dct['tableau'][i+1,0]/u[i])

        mini = min(ratios)
        l = 0  # minimizing index, exiting column
        for i in range(len(u)):
            if u[i]>fr("0") and dct['tableau'][i+1,0]/u[i] == mini:
                l = i+1
                break

        dct['tableau'][l] = dct['tableau'][l]/dct['tableau'][l][j]  # making pivot element = 1

        for k in range(m+1):  # making all other entries of pivot column = 0
            if k!=l:
                dct['tableau'][k] = dct['tableau'][k] + fr(-1)*dct['tableau'][k][j]*dct['tableau'][l]

        #updating basis indices and basis varibles vector
                
        bss['vars'][l-1] = dct['vars'][j-1]
        bss['idx'][l-1] = j-1

    if (unbdd):
        ans['final_tableau'] = dct['tableau'][1:]
        ans['solution_status'] = 'unbounded'
        return ans
    
    #### optimal solutions reach this point #####
    ans['final_tableau'] = dct['tableau'][1:]
    ans['solution_status'] = 'optimal'
    if (dct['[objective]'][0]=='minimize'):
        ans['optimal_value'] = fr("-1")*dct['tableau'][0][0]
    else:
        ans['optimal_value'] = dct['tableau'][0][0]

    vec = {}
    var = 'x'
    for i in range(n_initial):
        vec[var + str(i)] = fr("0")

    for i in range(len(bss['vars'])):
        if (bss['vars'][i]) in vec:
            vec[bss['vars'][i]] = dct['tableau'][i+1][0]

    ans['optimal_solution'] = []

    for i in range(n_initial):
        ans['optimal_solution'].append(fr(vec[var + str(i)]))

    return ans  #returning answer dictionary

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------

# Dual Simplex Function, which accepts dual feasible tableau and applies dual simplex on it
def dual_simplex(dct,bss, n_initial):
    ans = {'initial_tableau': None, 'final_tableau': None, 'solution_status': '', 'optimal_solution': None, 'optimal_value': None}
    ans['initial_tableau'] = np.copy(dct['tableau'])

    m = len(dct['tableau']) - 1
    unbdd = False

    while True:
        idx = 0
        found = False
        for i in range(1,m+1):
            if (dct['tableau'][i,0] < fr("0")):
                idx = i-1  #exiting variable's index in basis
                found = True
                break

        if not found:  # all basis variables are non-negative
            break  #hence we have achieved optimal cost

        v = dct['tableau'][idx+1,1:]    # v vector

        if (np.all(v>=0)):
            unbdd = True  # optimal cost is inf
            break

        ratios = []
        for i in range(len(v)):
            if (v[i]<fr(0)):
                ratios.append(dct['tableau'][0,i+1]/abs(v[i]))
        mini = min(ratios)

        l = 0  # minimizing index, entering variable
        minis = []  # list of all minimizing indices
        for i in range(len(v)):
            if (v[i]<fr(0)) and ((dct['tableau'][0,i+1]/abs(v[i])) == mini):
                minis.append(i+1)

        # applying lexicographic pivoting if more than one minimizing index
        if len(minis)>1:
            all_cols = np.copy(dct['tableau'][:,minis])
            for j in range(len(minis)):  # dividing all clashing columns with respective v_i
                all_cols[:,j] = all_cols[:,j]/abs( v[ minis[j]-1 ] ) 
            comp = [list(lst) for lst in all_cols.T]       # finding lexicographically smallest column
            lst = min(comp)
            l = minis[comp.index(lst)]
        else:
            l = minis[0]

        # entering column, i.e., Aj (tab[:,l]) is now obtained
        # pivot element is tab[idx+1,l]
        dct['tableau'][idx+1] = dct['tableau'][idx+1]/dct['tableau'][idx+1,l]  # making pivot element = 1

        for k in range(m+1):  # making all other entries of pivot column = 0
            if k!=idx+1:
                dct['tableau'][k] = dct['tableau'][k] + fr("-1")*dct['tableau'][k,l]*dct['tableau'][idx+1]

        #updating basis indices and basis varibles vector
        bss['vars'][idx] = dct['vars'][l-1]
        bss['idx'][idx] = l-1

    if (unbdd):
        ans['final_tableau'] = np.copy(dct['tableau'])
        ans['solution_status'] = 'unbounded'
        return ans
    
    #### optimal solutions reach this point #####
    ans['final_tableau'] = np.copy(dct['tableau'])
    ans['solution_status'] = 'optimal'
    if (dct['[objective]'][0]=='minimize'):
        ans['optimal_value'] = fr(-1)*dct['tableau'][0][0]
    else:
        ans['optimal_value'] = dct['tableau'][0][0]

    vec = {}
    var = 'x'
    for i in range(n_initial):
        vec[var + str(i)] = fr("0")

    # for i in range(len(bss['vars'])):
    #     if (bss['vars'][i]) in vec:
    #         vec[bss['vars'][i]] = dct['tableau'][i+1][0]

    vec.update({var: dct['tableau'][i+1][0] for i, var in enumerate(bss['vars']) if var in vec})
    ans['optimal_solution'] = []

    for i in range(n_initial):
        ans['optimal_solution'].append(fr(vec[var + str(i)]))

    return ans  #returning answer dictionary

# -----------------------------------------------------------------------------------------------------------------------------

# considering a number to be integral if the difference between the number and its nearest integer is less than  1e-6
def is_integer(number):
    return (number.denominator == 1)

# function to calculate fractional part of a number
def fractional_part(number):
    return number % 1

# -----------------------------------------------------------------------------------------------------------------------------

def gomory_cut_algo_fun(filename = file_name):

    dct = {'[objective]':[],'[A]':[],'[b]':[],'[constraint_types]':[],'[c]':[]}   
    bss = {'vars':[] , 'idx': []}
    ans = {'initial_solution': None, 'final_solution': None, 'solution_status': '', 'number_of_cuts': 0, 'optimal_value': None}

    # File reading
    n_initial = file_handling(filename,dct)

    # obtaining initial solution from two phase simplex, the optimal solution to Relaxed ILP
    ans_initial = two_phase(dct,bss,n_initial)
    ans['initial_solution'] = np.copy(ans_initial['optimal_solution'])
    ans['final_solution'] = np.copy(ans_initial['optimal_solution'])  # keeping track of current solution in 'final_solution'

    #checking for infeasibile or unbounded solutions
    if (ans_initial['solution_status']!='optimal'):
        ans['solution_status'] = ans_initial['solution_status']
        return ans
    
    # the slack variables which we will add during fractional dual
    var = 's'
    cnt = 0

    while True:
        # checking if all entries of current solution are integers
        check = True
        for i in range(len(ans['final_solution'])):   # if any co-ordinate is non-integer, make check false (we are yet to reach optimum)
            if (not is_integer(ans['final_solution'][i])):
                check = False
                break
        
        if (check):    # all co-ordinates of final_solution are integers, hence final_solution is optimal
            ans['solution_status'] = 'optimal'
            ans['optimal_value'] = dct['[c]'][0][:n_initial]@ans['final_solution']
            if (dct['[objective]'][0]=='maximize'):
                ans['optimal_value'] = fr(-1)*ans['optimal_value']
            return ans
        else:
            ans['number_of_cuts']+=1 # increase number of gomory cuts by one
            # print(ans['number_of_cuts'],float(dct['tableau'][0,0]))

            ind = 0  # index of fractional co-ordinate

            # first fractional co-ordinate pivoting rule, find the first frac co-ord and then add gomory cut of corres constr

            for i in range(len(dct['tableau'])):
                if (not is_integer(dct['tableau'][i,0])):
                    ind = i
                    break

            # alternate pivoting rule, add the constraint corresponding to largest fractional co-ordinate

            # indis = []
            # for i in range(len(dct['tableau'])):
            #     if (not is_integer(dct['tableau'][i,0])):
            #         indis.append(dct['tableau'][i,0])
            # m = max(indis)
            # for i in range(len(dct['tableau'])):
            #     if (dct['tableau'][i,0] == m):
            #         ind = i
            #         break

            # choose row ind as the source row, add corresponding constraint
                    
            #create the new row
            new_row = np.full(dct['tableau'][ind].shape, fr("0"), dtype = fr)
            for i in range(len(dct['tableau'][ind])):
                new_row[i] = fr((-1)*fractional_part(dct['tableau'][ind][i]))

            #attach the new row at the end of the tableau
            dct['tableau'] = np.vstack((dct['tableau'],new_row))

            #create the new column
            new_col = np.full((len(dct['tableau']),1), fr("0"), dtype = fr)
            new_col[-1][0] = fr("1")

            #attach the new column at the end of the tableau
            dct['tableau'] = np.hstack((dct['tableau'],new_col)) 

            #book keeping
            dct['vars'].append(var + str(cnt))
            cnt+=1
            bss['idx'].append(len(dct['vars'])-1)
            bss['vars'].append(dct['vars'][-1])
 
            # applying dual simplex and updating final_solution
            ans_dual = dual_simplex(dct,bss,n_initial)
            ans['final_solution'] = np.copy(ans_dual['optimal_solution'])

            # ryan wala optimisation

            keep_row = [0]
            popy_bss = []
            popy_var = []
            bool_col = [1 for i in range(len(dct['tableau'][0]))]
            for i in range(len(bss['vars'])):
                if (bss['vars'][i][0]!='s'):   # removing rows and columns associated with slack variables
                    keep_row.append(i+1)
                else:
                    bool_col[bss['idx'][i] + 1] = 0
                    popy_bss.append(i)
                    popy_var.append(bss['idx'][i])

            keep_col = []
            for i in range(len(bool_col)):
                if (bool_col[i]):
                    keep_col.append(i)

            # for i in range(len(popy_var)-1,-1,-1):
            #     dct['vars'].pop(popy_var[i])

            # for i in range(len(popy_bss)-1,-1,-1):
            #     bss['vars'].pop(popy_bss[i])         
            
            # if (len(popy_bss)>0):
            #     bss['idx'] = []
            #     for var in bss['vars']:
            #         bss['idx'].append(dct['vars'].index(var))

            if (len(popy_bss)>0):
                # Convert popy_var to a set for efficient membership checking
                popy_set = set(popy_var)

                # Filter dct['vars'] to keep only elements whose index is not in popy_set
                dct['vars'] = [elem for idx, elem in enumerate(dct['vars']) if idx not in popy_set]

                # Convert popy_bss to a set for efficient membership checking
                popy_set_2 = set(popy_bss)
                # Filter bss['vars'] to keep only elements whose index is not in popy_set_2
                bss['vars'] = [elem for idx, elem in enumerate(bss['vars']) if idx not in popy_set_2]   

                # Create a dictionary to map variables to their indices in dct['vars']
                vars_indices_map = {var: idx for idx, var in enumerate(dct['vars'])}
                # Update bss['idx'] using the dictionary
                bss['idx'] = [vars_indices_map[var] for var in bss['vars']]

                dct['tableau'] = dct['tableau'][keep_row,:]
                dct['tableau'] = dct['tableau'][:,keep_col]

            #checking for infeasibile or unbounded solutions
            if (ans_dual['solution_status']=='unbounded'):
                ans['solution_status'] = 'infeasible'
                return ans

def gomory_cut_algo():
    dict = gomory_cut_algo_fun()
    for key in dict:
        if (key[-1]=='n' and key[0]=='i'):
            try:
                print(key,end=': ')
                for i in range(len(dict[key])-1):
                    print(float(dict[key][i]),end=", ")
                print(float(dict[key][-1]))
            except:
                print(dict[key])
        elif (key[-1]=='n' and key[0]=='f'):
            try:
                print(key,end=': ')
                for i in range(len(dict[key])-1):
                    print(int(dict[key][i]),end=", ")
                print(int(dict[key][-1]))
            except:
                print(dict[key]) 
        elif key[-1]=='e':
            print(key,end=": ")
            try:
                print(int(dict[key]))
            except:
                print(dict[key])        
        else:
            print(key+":",dict[key])

# ""All the variables given in the test cases will be non-negative.""
# Calling the function will print information in this order: initial_solution, final_solution, solution_status, number_of_cuts, optimal_value
gomory_cut_algo()