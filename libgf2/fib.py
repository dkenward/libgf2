"""
Objects and functions for working with Fibonacci representation
in GF(2)[x]/p(x).

Copyright 2017 Jason M. Sachs

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

from libgf2.util import parity

def _fibmulxmod(u, poly, mask=-1):
    '''
    Multiplies by x in the Fibonacci representation
    
    u: input, in Fibonacci representation
    poly: characteristic polynomial of a quotient ring, 
          in Fibonacci form (reversed from Galois)
    mask: bitmask of significant bits 
          (-1 to retain upper bits that are not significant)  
          
    returns: Fibonacci representation of u*x
    '''
    y = u << 1
    y |= parity(y & poly)
    return y & mask
    
def _gf2_to_fib(u, poly, mask=-1):
    '''
    Converts from Galois to Fibonacci representation
    
    u: input, in Galois representation
    poly: characteristic polynomial of a quotient ring, 
          in Fibonacci form (reversed from Galois)
    mask: bitmask of significant bits 
          (-1 to retain upper bits that are not significant)  
          
    returns: Fibonacci representation of u     
    '''    
    xk = 1
    y = u & 1
    while u > 1:
        u >>= 1
        xk = _fibmulxmod(xk, poly, mask)
        y ^= (u&1)*xk
    return y
    
def _fibmulxnmod(u, n, poly, mask=-1):
    '''
    Multiplies by x^n in the Fibonacci representation
    
    u: input, in Fibonacci representation
    poly: characteristic polynomial of a quotient ring, 
          in Fibonacci form (reversed from Galois)
    mask: bitmask of significant bits 
          (-1 to retain upper bits that are not significant)  
          
    returns: Fibonacci representation of u*(x^n)      
    '''
    y = u
    for _ in range(n):
        y = _fibmulxmod(y, poly, mask)
    return y     
    
def _fib_to_gf2(u, polycomp, msb):
    '''
    Converts from Fibonacci to Galois representation
    
    u: input, in Fibonacci representation
    polycomp: x^(n-1) in Fibonacci representation
              (obtain from _fibmulxmod(1,n-1,poly); no mask needed)
    msb:      1 in bit n-1 (= 1<<(n-1))
    
    returns: Galois representation of u
    ''' 
    y = 0
    while polycomp > 0:
        if u & msb:
            y |= msb
            u ^= polycomp
        polycomp >>= 1
        msb >>= 1
    return y
        
        
                
        
    