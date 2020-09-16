#### Mostly Cython related logs / errors etc 

- Own Code - below line of code throws further below - warnings etc in terminal
> cdef int[:] max_val # FOOBAR_CyThon - Not in the .py file , now TYPED as an ARRAY
        # CyThon Error == undeclared name not builtin: max_val

#

'''
(pytorch_venv) dhankar@dhankar-1:~/temp/pytorch/PyTorch_1/cython$ python setup.py build_ext --inplace
Compiling fromScratch_cy.pyx because it changed.
[1/1] Cythonizing fromScratch_cy.pyx
/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/Cython/Compiler/Main.py:369: FutureWarning: Cython directive 'language_level' not set, using 2 for now (Py2). This will change in a later release! File: /home/dhankar/temp/pytorch/PyTorch_1/cython/fromScratch_cy.pyx
  tree = Parsing.p_module(s, pxd, full_module_name)
warning: fromScratch_cy.pyx:164:12: local variable 'max_val' referenced before assignment
warning: fromScratch_cy.pyx:169:52: local variable 'max_val' referenced before assignment
warning: fromScratch_cy.pyx:169:60: Index should be typed for more efficient access
running build_ext
building 'fromScratch_cy' extension
gcc -pthread -B /home/dhankar/anaconda3/envs/pytorch_venv/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/dhankar/anaconda3/envs/pytorch_venv/include/python3.8 -c fromScratch_cy.c -o build/temp.linux-x86_64-3.8/fromScratch_cy.o
In file included from /usr/include/numpy/ndarraytypes.h:1809:0,
                 from /usr/include/numpy/ndarrayobject.h:18,
                 from /usr/include/numpy/arrayobject.h:4,
                 from fromScratch_cy.c:623:
/usr/include/numpy/npy_1_7_deprecated_api.h:15:2: warning: #warning "Using deprecated NumPy API, disable it by " "#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
 #warning "Using deprecated NumPy API, disable it by " \
  ^~~~~~~
gcc -pthread -shared -B /home/dhankar/anaconda3/envs/pytorch_venv/compiler_compat -L/home/dhankar/anaconda3/envs/pytorch_venv/lib -Wl,-rpath=/home/dhankar/anaconda3/envs/pytorch_venv/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.8/fromScratch_cy.o -o /home/dhankar/temp/pytorch/PyTorch_1/cython/fromScratch_cy.cpython-38-x86_64-linux-gnu.so
(pytorch_venv) dhankar@dhankar-1:~/temp/pytorch/PyTorch_1/cython$ 
'''