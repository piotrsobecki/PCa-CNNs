import os
import shutil
import ast


def next_available_file(fname,sep='_', i=0):
    try_fname = fname
    if i > 0:
        try_fname += sep+str(i)
    if os.path.isfile(try_fname) or os.path.isdir(try_fname):
        return next_available_file(fname,sep, i + 1)
    else:
        return try_fname

def next_available_dir(fname,sep='_', i=0):
    try_fname = os.path.dirname(fname)
    if i > 0:
        try_fname += sep+str(i)
    if os.path.isfile(try_fname) or os.path.isdir(try_fname):
        return next_available_dir(fname,sep, i + 1)
    else:
        return try_fname

def getfunc(modulefunc):
    bydots = modulefunc.split(".")
    prev_module = bydots[0]
    module = locals().get(prev_module) or globals().get(prev_module) or __import__(prev_module)
    for idx in range(1,len(bydots)):
        module = getattr(module,bydots[idx])
    return module

def backup_file(starting):
    if os.path.isfile(starting):
        shutil.move(starting, next_available_file(starting))



def args_map(args):
    argsdict = {}
    for farg in args:
        if farg.startswith('--'):
            (arg,val) = farg.split("=")
            arg = arg[2:]
            val = ast.literal_eval(val)
            if arg in argsdict:
                if type(argsdict[arg]) == list:
                    argsdict[arg].append(val)
                else:
                    argsdict[arg] = [argsdict[arg],val]
            else:
                argsdict[arg] = val
    return argsdict