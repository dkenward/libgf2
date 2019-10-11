'''
Utility functions and classes

Copyright 2013-2017 Jason M. Sachs

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import numbers
from weakref import WeakKeyDictionary

class LazyProperty(object):
    def __init__(self, deferred_computation, attrkey=None, 
                 cacheClass=WeakKeyDictionary):
        self.deferred_computation = deferred_computation
        self.cache = cacheClass()
        self.attrkey = attrkey
    def __get__(self, instance, type=None):
        if instance is None:
            return self
        key = (instance if self.attrkey is None 
               else getattr(instance, self.attrkey))
        if key in self.cache:
            return self.cache[key]
        else:
            result = self.deferred_computation(instance)
            self.cache[key] = result
            return result
    @staticmethod
    def decorate(**kwargs):
        def decorator(func):
            return LazyProperty(func, **kwargs)
        return decorator

def bit_lsb_first_iterator(bits, min_bit_count=0):
    """
    Yields individual bits of a number, first from lsb.
    Continues for at least min_bit_count bits or until the number is out of bits.
    
    Example:
     bits = 19, min_bit_count = 0
     
     yields 1, 1, 0, 0, 1
    """
    if bits < 0:
        raise ValueError("bit_lsb_first_iterator accepts only nonnegative numbers")
    while bits > 0 or min_bit_count > 0:
        min_bit_count -= 1
        bit = bits & 1
        bits >>= 1
        yield bit

def cumulative_bits_iterator(bits, n=None, msb_first=True):
    """
    Yields partial bits of an input string.
    The input string can either be an iterable of bits (or any values that can be tested as true or false),
    or a number. If a number, the bits are interpreted LSB first.
    
    Example:
     bits = 19, n=8
     yields (in binary):
     
       1
       11
       110
       1100
       11001
       110010
       1100100
       11001000
       
    The identical string is produced if bits = [1,1,0,0,1]
       
    """       
    
    output = 0
    n = float('inf') if n is None else n
    try:
        iter(bits)
    except TypeError:
        bits = bit_lsb_first_iterator(bits, n)
    for bit in bits:
        n -= 1
        if n <= 0:
            break
        output <<= 1
        output |= bit
        yield output

def parity(bitsx):
    '''
    parity of the bits contained in bitsx
    Execution time is logarithmic in bit length.
    This is the preferred general-purpose parity calculation,
    unless there are special circumstances. 
    '''
    if bitsx < 0:
        raise ValueError("Parity defined only for nonnegative integers")
    k = 1
    while True:
        y = bitsx >> k
        if y == 0:
            break
        bitsx ^= y
        k <<= 1
    return bitsx & 1
    
def parity_sparse(bitsx):
    '''
    parity of the bits contained in bitsx
    Execution time is linear in the number of 1 bits
    '''
    if bitsx < 0:
        raise ValueError("Parity defined only for nonnegative integers")
    p = 0    
    while bitsx > 0:
        bitsx &= bitsx - 1    # clear least significant 1
        p ^= 1
    return p
    
def parity_sparse_is_faster(bitsx):
    '''
    Select an appropriate parity algorithm for computing
    the parity of bitsx; values with sparsely distributed 1s
    should use parity_sparse in most cases.
    
    This effectively runs both algorithms, which seems silly,
    except that for determining the parity of (x & mask)
    with a fixed mask numerous times, calculation of
    parity_sparse_is_faster(mask) will tell whether
    parity_sparse(x & mask) is a better choice than parity(x & mask).
    in most cases.       
    '''
    if bitsx < 0:
        raise ValueError("Parity defined only for nonnegative integers")
    x_sparse = bitsx
    k = 1    
    while True:
        x_sparse &= x_sparse - 1 # clear least significant 1
        if x_sparse == 0:
            return True
        if (bitsx >> k) == 0:
            return False
        k <<= 1    

def reverse_bits(x, n=0):
    y = 0
    while x > 0 or n > 0:
        n -= 1
        y <<= 1
        y |= x & 1
        x >>= 1
    return y

def berlekamp_massey(bits, N=None, verbose=False):
    '''
    compute minimal polynomial of LFSR that produces the specified bit sequence
    
       bits:    sequence of bits to match
       N:       stop after N bits
       verbose: True to print something
    
    https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Massey_algorithm
    '''
        
    b = 1
    c = 1
    L = 0
    m = -1
    n = -1
    history = 0
    
    for bit in bits:
        n += 1
        if N is not None and n >= N:
            break
        history = (history << 1) | bit  
        # history = the first n bits, oldest = most significant
        mask = (1 << (L+1)) - 1
        d = parity(mask & history & c)
        if verbose:
            print('n={0:d}, m={1:d}, history={2:b}, c={3:b}, b={4:b}, L={5:d}'.format(n, m, history, c, b, L))
        if d == 1:
            t = c
            c ^= b << (n-m)
            if 2*L <= n:
                L = n+1 - L
                m = n
                b = t
    # c is now in reversed order
    return (reverse_bits(c,L+1), L)
    
def state_from_reversed_output(polycoeffs, rbits):
    '''
    Compute LFSR state from polynomial coefficients 
    and a reversed sequence of output bits
    
      polycoeff: polynomial coefficient in bit vector form
      rbits:     any iterable that represents a reversed sequence of output bits 
    '''
    s = 0
    # run backwards until the state has no unknown bits
    for bit in rbits:
        if bit != 0:
            s ^= polycoeffs
        s >>= 1
    return s  

def coset_representative(x,n):
    """ 
    Find the coset representative of x,
    which is the bitwise rotation of x over n bits
    that has the smallest value.
    """
    r = x
    for k in range(n):
        if x&1:
            x |= 1<<n
        x >>= 1
        if x < r:
            r = x
    return r

if __name__ == '__main__':
    pass