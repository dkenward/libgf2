"""
Objects and functions for working with polynomials in GF(2).

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

"""

import numpy as np
from operator import itemgetter
import libgf2.util as util
#import primefac

try:
    # Speedup if Cython is installed.
    import pyximport; pyximport.install()
    import fastgf2mulmod
except:
    fastgf2mulmod = None
    
# ---------- Basic arithmetic     
    
def _gf2mod(a,m):
    """
    Computes `a` mod `m`.
    
    Parameters
    ----------
    a, m : integer
        Polynomial coefficient bit vectors.
        
    Returns
    -------
    integer
        Polynomial coefficient bit vectors of `a` mod `m`.    
    """
    m2 = m
    i = 0
    while m2 < a:
        m2 <<= 1
        i += 1
    while i >= 0:
        anew = a ^ m2
        if anew < a:
            a = anew
        m2 >>= 1
        i -= 1
    return a    

def _gf2mul(a,b):
    """
    Computes ``a * b``.
    
    Parameters
    ----------
    a, b : integer
        Polynomial coefficient bit vectors.
        
    Returns
    -------
    c : integer
        Polynomial coefficient bit vector of ``c = a * b``.    
    """

    c = 0
    while a > 0:
        if (a & 1) != 0:
            c ^= b
        b <<= 1
        a >>= 1
    return c

def _gf2bitlength_linear(a):
    """
    Computes the length of a polynomial coefficient bit vector = degree + 1. 
    
    Parameters
    ----------
    a : integer
        Polynomial coefficient bit vector.
        
    Returns
    -------
    n : integer
        length of  polynomial `a`.    
    """     
    n = 0
    while a > 0:
        n += 1
        a >>= 1
    return n
    
def _gf2bitlength(a):
    """
    Computes the length of a polynomial coefficient bit vector = degree + 1. 
    
    Parameters
    ----------
    a : integer
        Polynomial coefficient bit vector.
        
    Returns
    -------
    n : integer
        length of  polynomial `a`.    
    """     
    if a == 0:
        return 0
    n = 1
    while (a >> n) > 0:
        n <<= 1
    n >>= 1
    b = n
    while b > 0:
        b >>= 1
        nnew = n ^ b
        if (a >> nnew) > 0:
            n = nnew
    return n + 1

def _gf2mulmod(a,b,m):
    """
    Computes ``a * b mod m``.
    *NOTE*: Does *not* check whether `a` and `b` are both
    smaller in degree than `m`.
    
    Parameters
    ----------
    a, b, m : integer
        Polynomial coefficient bit vectors.
        Polynomials `a` and `b` should be smaller degree than `m`.
        
    Returns
    -------
    c : integer
        Polynomial coefficient bit vector of ``c = a * b mod m``.    
    """
    c = 0
    while a > 0:
        if (a & 1) != 0:
            c ^= b
        b <<= 1
        b2 = b ^ m
        if b2 < b:
            b = b2
        a >>= 1
    return c
    
def _gf2squaremod(a,N,squarecache):
    """
    Computes ``a * a mod m``.
    *NOTE*: Does *not* check whether `a` is
    smaller in degree than `m`.
    
    Parameters
    ----------
    a : integer
        Polynomial coefficient bit vector.
        Polynomial `a` should be smaller than `N`.
    N : degree of m
    squarecache : iterable
        cached list of x**k mod m (as polynomial coefficient bit vectors)
        for k between N2+1 and N2*2 with N2 = (N+1)/2
        
    Returns
    -------
    c : integer
        Polynomial coefficient bit vector of ``c = a * a mod m``.    
    """
    c = 0
    N2 = (N+1)//2
    ah = a >> N2
    c ^= a&1
    c ^= (ah&1)*squarecache[0]
    for i in range(1,N2):
        a >>= 1
        ah >>= 1
        c ^= (a&1) << (2*i)
        c ^= (ah&1) * squarecache[i]
    return c 
    
def _gf2mulxmod(a,m):
    """
    Computes ``a * x mod m``.
    *NOTE*: Does *not* check whether `a` is smaller in degree than `m`.
    
    Parameters
    ----------
    a, m : integer
        Polynomial coefficient bit vectors.
        Polynomial `a` should be smaller degree than `m`.
        
    Returns
    -------
    c : integer
        Polynomial coefficient bit vector of ``c = a * x mod m``.    
    """
    c = a << 1
    c2 = c^m
    if c2 < c:
        c = c2
    return c
    
def _gf2mulxinvmod(a,m):
    """
    Computes ``a * x^(-1) mod m``.
    *NOTE*: Does *not* check whether `a` is smaller in degree than `m`.
    
    Parameters
    ----------
    a, m : integer
        Polynomial coefficient bit vectors.
        Polynomial `a` should be smaller degree than `m`.
        
    Returns
    -------
    c : integer
        Polynomial coefficient bit vector of ``c = a * x^(-1) mod m``.    
    """
    c = (a ^ ((a&1)*m)) >> 1
    return c             

def _gf2power(a,k):
    """
    Computes `a` to the `k`th power.
    
    Parameters
    ----------
    a : integer
        Polynomial coefficient bit vector.
    k : integer     
        Exponent.
        
    Returns
    -------
    c : integer
        Polynomial coefficient bit vector of `a` raised to the `k`th power.
    """    
    c = 1
    while k > 0:
        if (k & 1) != 0:
            c = _gf2mul(c,a)
        a = _gf2mul(a,a)
        k >>= 1
    return c

def _gf2powmod(a, k, m):
    """
    Computes `a` to the `k`th power, in the quotient ring GF(2)[x]/m(x).
    *NOTE*: Does *not* check whether k is nonnegative.
    
    Parameters
    ----------
    a : integer
        Polynomial coefficient bit vector.
    k : integer
        Nonnegative exponent.       
    m : integer
        Polynomial coefficient bit vector. 
           
    Returns
    -------
    c : integer 
        Polynomial coefficient bit vector of `a` raised to the `k`th power, 
        mod `m`. 
    """
    c = 1
    while k > 0:
        if (k & 1) != 0:
            c = _gf2mulmod(c,a,m)
        a = _gf2mulmod(a,a,m)
        k >>= 1
    return c 

def _gf2divmodvect(avec,dvec):
    """
    Computes `q` = `a`/`d` and `r` = `a`%`d` in a vectorized form.
    
    Parameters
    ----------
    avec : array_like
        Array of dividends, each of which should be a polynomial coefficient bit vector
    dvec : array_like
        Array of divisors, each of which should be a polynomial coefficient bit vector     
        
    Returns
    -------
    q : array_like
        Array of quotients, each of which is a polynomial coefficient bit vector
    r : array_like 
        Array of quotients, each of which is a polynomial coefficient bit vector
    """
    na = _gf2bitlength(avec[0])
    nd = _gf2bitlength(dvec[0])
    i = na - nd
    q = 0
    test = 1 << (na-1)
    while i >= 0:
        if (avec[0] & test) != 0:
            avec = [a ^ (d << i) for (a,d) in zip(avec,dvec)]
            q |= (1 << i)
        i -= 1
        test >>= 1
    r = avec    
    return (q,r)

def _gf2exteuc(a,b):
    r"""
    Computes the extended Euclidean algorithm using Blankinship's method.
    
    Returns :math:`(g,u,v)` such that :math:`g = \gcd(a,b)` and :math:`g = au+bv` in :math:`GF(2)`.
    
    Parameters
    ----------
    a, b : integer
        Polynomial coefficient bit vectors.
        
    Returns
    -------
    g : integer
        Polynomial coefficient bit vector :math:`g = \gcd(a,b)`
    u, v : integer    
        Polynomial coefficient bit vectors where :math:`g = au+bv`
    """
    arow = [a,1,0]
    brow = [b,0,1]
    while True:
        (_,rrow) = _gf2divmodvect(arow, brow)
        if rrow[0] == 0:
            break
        arow = brow
        brow = rrow
    return tuple(brow)
    
def _gf2polyvalmod(u,g,m):
    """
    Evaluates polynomial `g(u)` in the quotient ring GF(2)[x]/m(x).
    
    Uses Paterson + Stockmeyer's technique  
    in O(sqrt(N)) quotient ring multiplications, as in 
    "On the number of nonscalar multiplications necessary
    to evaluate polynomials" SIAM J Comping 2(1973)
    
    Parameters
    ----------
    u : integer
        Polynomial coefficient bit vector.
    g : integer
        Polynomial coefficient bit vector.       
    m : integer
        Polynomial coefficient bit vector. 
           
    Returns
    -------
    y : integer 
        Polynomial coefficient bit vector of `g(u)` mod `m`. 
    """
    L = g.bit_length()
    upowers = [1,u]    
    k = 1
    uk = u
    while (k+1)*(k+1) <= L:
        k += 1
        uk = _gf2mulmod(uk,u,m)
        upowers.append(uk)
    upowers = upowers[k-1::-1]
    # powers from u^(k-1) to u^0 in descending order

    # nrow = ceil(L/k) so that k*nrow >= L
    nrow = (L+k-1)//k
    
    y = 0
    mask = 1 << (nrow*k - 1)
    for _ in range(nrow):
        y = _gf2mulmod(y,uk,m)
        for j in range(k):
            if g & mask:
                y ^= upowers[j]
            mask >>= 1    
    return y

class _GF2(object):
    """
    Internal helper class that encapsulates
    basic arithmetic in GF(2).
    Why use a class? Because it seemed right at the time.
    """
    mod = staticmethod(_gf2mod)
    mul = staticmethod(_gf2mul)
    bitlength = staticmethod(_gf2bitlength)
    mulmod = staticmethod(_gf2mulmod)
    power = staticmethod(_gf2power)
    powmod = staticmethod(_gf2powmod)
    divmodvect = staticmethod(_gf2divmodvect)
    exteuc = staticmethod(_gf2exteuc)
    mulxmod = staticmethod(_gf2mulxmod)
    mulxinvmod = staticmethod(_gf2mulxinvmod)
    polyvalmod = staticmethod(_gf2polyvalmod)
    
class GF2Polynomial(object):
    """
    A class of (mostly)-immutable instances representing an element of the polynomial ring with coefficients in GF(2).
    
    Parameters
    ----------
    coeffs : integer
        Polynomial coefficient bit vector
    """
    __slots__ = ('__weakref__','coeffs','bitlength')
    def __new__(cls, coeffs):
        result = object.__new__(cls)
        object.__setattr__(result, "coeffs", coeffs)
        object.__setattr__(result, "bitlength", _GF2.bitlength(coeffs))
        # cache the bit length
        return result
    def __setattr__(self, *args):
        raise AttributeError("Immutable object")
    def __delattr__(self, *args):
        raise AttributeError("Immutable object")
    def __repr__(self):
        return '{0}(0b{1:b})'.format(self.__class__.__name__, self.coeffs)
    @staticmethod
    def getcoeffs(p):
        if isinstance(p, GF2Polynomial):
            return p.coeffs
        else:
            return p
    def __rmod__(self, a):
        a = GF2Polynomial.getcoeffs(a)
        return GF2Polynomial(_GF2.mod(a, self.coeffs))
    def __mod__(self, m):
        m = GF2Polynomial.getcoeffs(m)
        return GF2Polynomial(_GF2.mod(self.coeffs, m))
    @property
    def degree(self):
        return self.bitlength - 1
    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.coeffs == other.coeffs
    def __hash__(self):
        return self.coeffs.__hash__()        

class GF2QuotientRing(GF2Polynomial):
    """
    A class representing the ring of polynomials with coefficients in GF(2),
    mod p(x), where p(x) is the characteristic polynomial having coefficients in GF(2).
    """
    __slots__ = ('_squares')
    def __init__(self,*args,**kwargs):
        N2 = (self.degree+1)//2
        squares = []
        u = 1
        for i in range(1,2*N2):
            u = self.lshiftraw1(u)
            u = self.lshiftraw1(u)
            if i >= N2:
                squares.append(u) 
        object.__setattr__(self, "_squares", tuple(squares))
    @staticmethod
    def cast(p):
        """
        Converts the input `p` to an instance of `GF2QuotientRing`.
        
        Parameters
        ----------
        p : integer or GF2Polynomial or GF2QuotientRing
            Polynomial coefficient bit vector or a polynomial object
        
        Returns
        -------
        GF2QuotientRing
            Corresponding instance of `GF2QuotientRing`
        """          
        if isinstance(p, GF2QuotientRing):
            return p
        try:
            return GF2QuotientRing(p.coeffs)
        except:
            return GF2QuotientRing(p)
    def wrap(self,a):
        """
        Returns an element in this quotient ring.
        
        Parameters
        ----------
        a : integer
            Polynomial coefficient bit vector
        
        Returns
        -------
        GF2Element
            Corresponding instance of `GF2Element`
        """    
        return GF2Element(a, self)
    def unwrap(self,a):
        """
        Returns the coefficient bit vector of an element in this quotient ring.
        
        Parameters
        ----------
        a : GF2Element or integer
            Instance of `GF2Element`, or a coefficient bit vector itself
        
        Returns
        -------
        integer 
            Coefficient bit vector of `a`
        """    
        try:
            f = a.field
            if f != self:
                raise ValueError('Field mismatch: expected %s, found %s' % (self,f))  
        except AttributeError:
            pass
        try:
            return a.coeffs
        except AttributeError:
            return a                                
    def add(self,a,b):
        """
        Adds two elements.
        
        Parameters
        ----------
        a, b : GF2Element or integer
        
        Returns
        -------
        GF2Element 
            Element ``c = a + b`` 
        """    
        a = self.unwrap(a)
        b = self.unwrap(b)
        return self.wrap(a^b)
    def mul(self,a,b):        
        """
        Multiplies two elements in GF(2^n).
        
        Parameters
        ----------
        a, b : GF2Element or integer
            Operands.
        
        Returns
        -------
        GF2Element 
            Element ``c = a * b`` 
        """    
        a = self.unwrap(a)
        b = self.unwrap(b)
        return self.wrap(self.mulraw(a,b))
    def mulraw(self,a,b):
        """
        Multiplies two elements in GF(2^n).
        
        Parameters
        ----------
        a, b : integer
            Polynomial coefficient bit vectors.
        
        Returns
        -------
        integer 
            ``c = a * b`` 
        """ 
        return _GF2.mulmod(a,b,self.coeffs)
    def square(self,a):        
        """
        Squares an element in GF(2^n).
        
        Parameters
        ----------
        a : GF2Element or integer
            Operand.
        
        Returns
        -------
        GF2Element 
            Element ``c = a * a`` 
        """    
        a = self.unwrap(a)
        return self.wrap(self.squareraw(a))
    def squareraw(self, a):
        """
        Squares an element in GF(2^n).
        
        Parameters
        ----------
        a : integer
            Polynomial coefficient bit vector.
        
        Returns
        -------
        integer 
            ``c = a * a``         
        """
        return _gf2squaremod(a,self.degree,self._squares)     
    def power(self, a, k):
        """
        Raises an element in GF(2^n) to the `k`th power.
        
        Parameters
        ----------
        a : GF2Element or integer
            Operand.
        k : integer
            Exponent.       
        
        Returns
        -------
        GF2Element 
            Element `a` raised to the `k`th power. 
        """
        a = self.unwrap(a)
        return self.wrap(self.powraw(a,k))
    def powraw(self, a, k):
        """
        Raises an element in GF(2^n) to the `k`th power.
        
        Parameters
        ----------
        a : integer
            Polynomial coefficient bit vector.
        k : integer
            Exponent.       
        
        Returns
        -------
        c : integer 
            Element `a` raised to the `k`th power, 
            as a polynomial coefficient bit vector. 
        """
        c = 1
        while k > 0:
            if (k & 1) != 0:
                c = _gf2mulmod(c,a,self.coeffs)
            a = self.squareraw(a)
            k >>= 1
        return c
    def powvect(self, a, kvect):
        """
        Raises an element in GF(2^n) to the ``k``th power,
        for each ``k`` in `kvect`.
        
        Parameters
        ----------
        a : GF2Element or integer
            Element, or a polynomial coefficient bit vector.
        kvect : array_like 
            Exponent vector.       
        
        Returns
        -------
        cvect : GF2Element 
            Element `a` raised to the `k`th power
        """    
        a = self.unwrap(a)
        return [self.wrap(c) for c in self.powvectraw(a, kvect)]
    def powvectraw(self, a,kvect):
        """
        Raises an element in GF(2^n) to the ``k``th power,
        for each ``k`` in `kvect`.
        
        Parameters
        ----------
        a : integer
            Polynomial coefficient bit vector.
        kvect : array_like 
            Exponent vector.       
        
        Returns
        -------
        cvect : integer 
            Element `a` raised to the `k`th power, 
            as a polynomial coefficient bit vector. 
        """

        nk = len(kvect)
        cvect = [1]*nk
        alldone = False
        kvecttmp = [k for k in kvect]
        while not alldone:
            alldone = True
            for i in range(nk):
                k = kvecttmp[i]
                if (k & 1) != 0:
                    cvect[i] = _GF2.mulmod(cvect[i],a,self.coeffs)
                k >>= 1
                if k > 0:
                    alldone = False
                kvecttmp[i] = k
            a = self.squareraw(a)
        return cvect
    def polyvalraw(self, u, g):
        """
        Evaluates polynomial `g(u)` in the quotient ring GF(2)[x]/m(x).
    
        Uses Paterson + Stockmeyer's technique  
        in O(sqrt(N)) quotient ring multiplications, as in 
        "On the number of nonscalar multiplications necessary
        to evaluate polynomials" SIAM J Comping 2(1973)
    
        Parameters
        ----------
        u : integer
            Polynomial coefficient bit vector.
        g : integer
            Polynomial coefficient bit vector.       

           
        Returns
        -------
        y : integer 
            Polynomial coefficient bit vector of `g(u)` mod m. 
        """
        y = _GF2.polyvalmod(u,g,self.coeffs)
        return y
    def polyval(self, u, g):
        """
        Evaluates polynomial `g(u)` in the quotient ring GF(2)[x]/m(x).
    
        Uses Paterson + Stockmeyer's technique  
        in O(sqrt(N)) quotient ring multiplications, as in 
        "On the number of nonscalar multiplications necessary
        to evaluate polynomials" SIAM J Comping 2(1973)
    
        Parameters
        ----------
        u : GF2Element or integer
            Polynomial coefficient bit vector.
        g : integer
            Polynomial coefficient bit vector.       

           
        Returns
        -------
        y : GF2Element 
            Polynomial coefficient bit vector of `g(u)` mod m. 
        """
        u = self.unwrap(u)
        y = self.wrap(self.polyvalraw(u,g))
        return y
    def _lshifthelper(self, k):
        """
        return 2^k in GF(2) for nonnegative k
        TODO: works for primitive polynomials only for negative shift counts
        """  
        return self.powraw(2,k)
    def _rshifthelper(self, k):
        """
        return 2^k in GF(2) for nonpositive k
        TODO: works for primitive polynomials only for negative shift counts
        """  
        r = self.coeffs >> 1
        return self.powraw(r,k)        
    def lshiftraw1(self, a):
        return _GF2.mulxmod(a,self.coeffs)    
    def rshiftraw1(self, a):
        return _GF2.mulxinvmod(a,self.coeffs) 
    def lshiftraw(self, a, k):
        """
        Returns `a` << `k` mod m in GF(2)
        
        Parameters
        ----------
        a : integer
            Polynomial coefficient bit vector.
        k : integer
            Shift count.    
                
        Returns
        -------
        integer
            `a` << `k` mod m in GF(2), as a polynomial coefficient bit vector
        
        Notes
        -----
        TODO: works for primitive polynomials only for negative shift counts
        """
        if k >= 0:
            if k > self.degree:
                return self.mulraw(a,self._lshifthelper(k))
            else:
                for _ in range(k):
                    a = self.lshiftraw1(a)
                return a        
        else: # k < 0
            if k < -self.degree:
                return self.mulraw(a,self._rshifthelper(-k))
            else:
                for _ in range(-k):
                    a = self.rshiftraw1(a)
                return a    
    def lshift(self, a, k):
        """
        Returns `a` << `k` mod m in GF(2)
        
        Parameters
        ----------
        a : GF2Element or integer
            Polynomial coefficient bit vector.
        k : integer
            Shift count.    
                
        Returns
        -------
        GF2Element
            `a` << `k` mod m in GF(2) 
        
        Notes
        -----
        TODO: works for primitive polynomials only for negative shift counts
        """
        a = self.unwrap(a)        
        return self.wrap(self.lshiftraw(a, k))    
    def rshiftraw(self, a, k):
        """
        Returns `a` >> `k` mod m in GF(2)
        
        Parameters
        ----------
        a : integer
            Polynomial coefficient bit vector.
        k : integer
            Shift count.    
                
        Returns
        -------
        integer
            `a` >> `k` mod m in GF(2), as a polynomial coefficient bit vector
        
        Notes
        -----
        TODO: works for primitive polynomials only for positive shift counts
        """
        if k >= 0:
            if k > self.degree:
                return self.mulraw(a,self._rshifthelper(k))
            else:
                for _ in range(k):
                    a = self.rshiftraw1(a)
                return a        
        else: # k < 0
            if k < -self.degree:
                return self.mulraw(a,self._lshifthelper(-k))
            else:
                for _ in range(-k):
                    a = self.lshiftraw1(a)
                return a
    def rshift(self, a, k):
        """
        Returns `a` >> `k` mod m in GF(2)
        
        Parameters
        ----------
        a : GF2Element or integer
            Polynomial coefficient bit vector.
        k : integer
            Shift count.    
                
        Returns
        -------
        GF2Element
            `a` >> `k` mod m in GF(2) 
        
        Notes
        -----
        TODO: works for primitive polynomials only for positive shift counts
        """
        a = self.unwrap(a)
        return self.wrap(self.rshiftraw(a, k))    

    def invraw(self, a):
        """
        Returns multiplicative inverse mod m
        
        Uses Blankinship's Algorithm to compute `y` such that `a` * `y` = 1 mod m in GF(2)
        
        Parameters
        ----------
        a : integer
            Polynomial coefficient bit vector. 
            
        Returns
        -------
        y : integer
            Multiplicative inverse of a, as a polynomial coefficient bit vector.   
            
        Raises
        ------
        ValueError            
            If `a` and the characteristic polynomial have a common factor.
        """

        (r,y,_) = _GF2.exteuc(a,self.coeffs)
        if r != 1:
            raise ValueError('%x and %x are not relatively prime but have a common factor of %x' % (a,self.coeffs,r))
        return y
    def inv(self, a):
        """
        Returns multiplicative inverse mod m
        
        Uses Blankinship's Algorithm to compute `y` such that `a` * `y` = 1 mod m in GF(2)
        
        Parameters
        ----------
        a : GF2Element or integer
            Polynomial coefficient bit vector. 
            
        Returns
        -------
        y : GF2Element
            Multiplicative inverse of a.   
            
        Raises
        ------
        ValueError            
            If `a` and the characteristic polynomial have a common factor.
        """
        a = self.unwrap(a)    
        return self.wrap(self.invraw(a))
    def divraw(self, a, b):
        """
        Returns `a` / `b` defined as `a` * ``b.inv``. 
        
        Parameters
        ----------
        a,b : integer
            Polynomial coefficient bit vectors. 
            
        Returns
        -------
        integer
            Polynomial coefficient bit vector d, such that d*`b` == `a`.   
            
        Raises
        ------
        ValueError            
            If `b` and the characteristic polynomial have a common factor.
        """
        return self.mulraw(a,self.invraw(b))
    def div(self, a, b):
        """
        Returns `a` / `b` defined as `a` * ``b.inv``. 
        
        Parameters
        ----------
        a,b : GF2Element or integer
            Polynomial coefficient bit vectors. 
            
        Returns
        -------
        GF2Element
            Element d, such that d*`b` == `a`.   
            
        Raises
        ------
        ValueError            
            If `b` and the characteristic polynomial have a common factor.
        """
        a = self.unwrap(a)
        b = self.unwrap(b)
        return self.wrap(self.divraw(a,b))

    @util.LazyProperty    
    def lograw(self):
        loghelper = GF2DiscreteLog(self.coeffs)
        def lograw(u): 
            """
            Returns the discrete logarithm of u
        
            Parameters
            ----------
            u : integer
                Polynomial coefficient bit vector
            
            Returns
            -------
            integer
                k such that x^k = u    
            
            Raises
            ------
            ValueError
                If u is zero
            """
            if u == 0:
                raise ValueError('Domain error: log of zero not defined')
            return loghelper.log(u)  
        return lograw          
    def log(self, u):
        """
        Returns the discrete logarithm of u
        
        Parameters
        ----------
        u : GF2Element or integer
            Polynomial coefficient bit vector
            
        Returns
        -------
        integer
            k such that x^k = u    
            
        Raises
        ------
        ValueError
            If u is zero
        """
        return self.lograw(self.unwrap(u))
    @util.LazyProperty
    def trace_constants(self):
        """
        Computes constants w,winv,trace useful for trace calculations
        
        Returns 
        -------
        
        (w,winv,trace)
        w : integer
        winv : integer
        trace: GF2TracePattern
        
           trace(u) = Tr(u) = highbit(winv*u)
           winv*w = 1
        """    
        w, winv = GF2TracePattern._constants_from_field(self)
        mask = GF2TracePattern._mask_from_lfsr_initial_state(self, winv)
        trace = GF2TracePattern.from_mask(self, mask)
        return (w,winv,trace)
    @property
    def trace(self):
        return self.trace_constants[2]              

def _bitlenlt(x,y):
    xy = x | y
    return (x << 1) < xy

def _gf2GaussJordan(A,b):
    """
    solves for x, where Ax = b, in GF2.
    A is an n x n matrix; b is an n-element vector or an n x m matrix.
    """
    n = A.shape[0]
    assert n == A.shape[1]
    try:
        s = b.shape
    except:
        s = (len(b),1)
    assert n == s[0]
    m = s[1]
    C = np.zeros((n,n+m),int)
    C[:,:n] = A
    if m == 1:
        C[:,n] = b
    else:
        C[:,n:] = b

    fails = []
    for j in range(n):
        # Find pivot
        p = None
        for i in range(j,n):
            if C[i,j] == 1:
                p = i
                break
        if p is None:
            fails.append(j)
            continue
        if p != j:
            C[p,:],C[j,:] = np.copy(C[j,:]),np.copy(C[p,:])
        for i in range(n):
            if i == j:
                continue
            if C[i,j] != 0:
                C[i,:] ^= C[j,:]
    if len(fails) > 0:
        raise ValueError('singular matrix: missing indices = %s' % fails)
    x = C[:,n:]
    return x

def _bitlencmp(x,y):
    xy = x | y
    if (x << 1) < xy:
        return -1
    elif (y << 1) <= x:
        return 1
    else:
        return 0    

def _bitsOf(p,n=None):
    def helper(p,n):
        if n is None:
            while p != 0:
                yield p & 1
                p >>= 1
        else:
            for _ in range(n):
                yield p & 1
                p >>= 1
    return tuple(helper(p,n))

def _gatherBits(v):
    x = 0
    p2 = 1
    for b in v:
        if b == 1:
            x |= p2
        p2 <<= 1
    return x
        



_gf2mulmod_purepython = _gf2mulmod
_gf2squaremod_purepython = _gf2squaremod
if fastgf2mulmod is not None:
    _gf2mulmod = fastgf2mulmod._gf2mulmod
    _gf2squaremod = fastgf2mulmod._gf2squaremod

def _gf2divmod(a,d):
    na = _GF2.bitlength(a)
    nd = _GF2.bitlength(d)
    i = na - nd
    q = 0
    while i >= 0:
        anew = a ^ (d << i)
        if anew < a:
            q |= (1 << i)
            a = anew
        i -= 1
    return (q,a)

        


            

def _exteuc(a,b):
    """
    based on Blankenship's algorithm
    return (g,u,v) such that g = gcd(a,b) and g = au+bv
    """
    arow = [a,1,0]
    brow = [b,0,1]
    while True:
        (q,r) = divmod(arow[0],brow[0])
        if r == 0:
            break
        rrow = [r,arow[1]-q*brow[1],arow[2]-q*brow[2]]
        arow = brow
        brow = rrow
    return tuple(brow)


def _modinv(x,m):
    (r,y,_) = _exteuc(x,m)
    if r != 1:
        raise ValueError('%d and %d are not relatively prime but have a common factor of %d' % (x,m,r))
    return y

def _calculateCofactors(factors, multiplicity=None):
    """
    given a vector of factors V, and multiplicity R
    calculate vector V' such that the element-by-element
    product of V and V' is a vector of equal elements P 
    where P is the product of all factors V^R.
    
    In other words, each element of V' is the product of
    all elements of V^R, except for the element in the corresponding position,
    where V^(R-1) is used. 
    
    The multiplicity vector, if not provided, is set to all ones. 
    
    for example: if V = [2,3,5,7] and R = [1,1,1,1]
                 then V' = [105, 70, 42, 30] and P = 210. 
    
    If V = [2,3,5,7] and R = [2,1,2,1]
    then V' = [1050, 700, 420, 300] and P = 2100.             
    """
    n = len(factors)
    if multiplicity is None:
        multiplicity = [1]*n
    cofactors = [1]*n
    for (i,factor) in enumerate(factors):
        r = multiplicity[i]
        fdiag = factor**(r-1)
        fnondiag = fdiag*factor
        cofactors = [x * (fnondiag if i != j else fdiag) 
                          for (j,x) in enumerate(cofactors)]    
    return tuple(cofactors) 

def _pullFactor(x, testFactor):
    """return tuple (n,r,y) such that
    x = r * y
    where y = testFactor^n
    and n is the largest possible integer
    such that r is not divisible by testFactor
    """ 
    n = 0
    fpower = 1
    while True:
        (q,r) = divmod(x, testFactor)
        if r != 0:
            break
        n += 1
        fpower *= testFactor
        x = q
    return (n,x,fpower)

def _calculatePolynomialFactors(poly):
    """
    calculate the factors of (2^n) - 1 where n is the degree of
    the desired polynomial
    """
    n = _GF2.bitlength(poly)-1
    period = (1 << n) - 1
    m = period
    e1 = GF2Element(1, poly)
    if (e1 << m).coeffs != 1:
        raise ValueError('%s not in primitive polynomial' % e1)
    return factorize(m)

def factorize_two_n_minus_1(m, n=None):
    # recover n
    if n is None:
        mcopy = m + 1
        n = 0
        while mcopy > 1: 
            mcopy >>= 1
            n += 1
    if (n & 1) == 1:
        return factorize(m, returnMultiplicity=True, specialFactor=False)
    if n == 0:
        return (), ()
    if n == 2:
        return (3,), (1,)          
    # m = 2^2k - 1 for even n -> m = (2^k+1)*(2^k-1)    
    sqrtm = 1 << (n//2)    
    f1,m1 = factorize(sqrtm + 1, True)
    f2,m2 = factorize_two_n_minus_1(sqrtm-1, n//2)
    # now merge factors
    result = {}
    for fx,mx in ((f1,m1),(f2,m2)):
        for fi, mi in zip(fx,mx):
            if fi in result:
                result[fi] += mi
            else:
                result[fi] = mi
    factors = []
    multiplicity = []
    for f in sorted(result.keys()):
        factors.append(f)
        multiplicity.append(result[f])
    return factors, multiplicity           

def factorize(m, returnMultiplicity=False, specialFactor=True):
    if returnMultiplicity and specialFactor and m > 2 and m&(m+1) == 0:
        result = factorize_two_n_minus_1(m)
        if result is not None:
            return result 
    factors = []
    multiplicity = []
    def testFactors():
        for f in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]:
            yield f
        f = 53
        while True:
            yield f
            f += 2
    for f in testFactors():
        if m < f*f:
            break
        (n,m,fpower) = _pullFactor(m,f)
        if fpower > 1:
            multiplicity.append(n)
            factors.append(f if returnMultiplicity else fpower)
    # Whatever factor is remaining must be a prime,
    # since we've already tested it with all the primes
    # up to sqrt(m)         
    if m > 1:
        factors.append(m)
        multiplicity.append(1)
    return (factors, multiplicity) if returnMultiplicity else factors

def product(v):
    return reduce(lambda x,y: x*y, v, 1)
                
class GF2DiscreteLog(object):
    """
    Facilitates computation of discrete logarithms, 
    after Clark and Weng (1994)
    "Maximal and Near-Maximal Shift Register Sequences:
    Efficient Event Counters and Easy Discrete Logarithms" 
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.44.118
    """
    def __init__(self, poly, factors=None, maxtablesize=65536):
        self.field = GF2QuotientRing.cast(poly)
        if factors is None:
            factors = _calculatePolynomialFactors(self.field.coeffs)
        factors = tuple(factors)
        self.factors = factors
        cofactors = _calculateCofactors(factors)
        self.cofactors = cofactors
        # verify factors
        e2 = self.field.wrap(1)
        period = cofactors[0]*factors[0]
        self.period = period
        assert (e2 << period).coeffs == 1
        lookup = []
        for (factor, cofactor) in zip(factors,cofactors):
            if factor > maxtablesize:
                raise ValueError('Factor %d exceeds maximum table size %d' % (factor, maxtablesize))
            g = e2 << cofactor
            gx = g
            glog = {1: 0}
            assert g.coeffs != 1
            for i in range(1,factor):
                glog[gx.coeffs] = i
                gx = gx * g
            assert gx.coeffs == 1
            v = _modinv(cofactor, factor)
            lookup.append({'factor':factor, 'cofactor':cofactor, 'g':g, 'logtable':glog, 'v':v})
        self.lookup = lookup
    @staticmethod
    def _rem(y,item,period=None):
        cofactor = item['cofactor']
        r = item['logtable'][y]
        if period is None:
            return r*cofactor*item['v']
        else:
            return (((r*cofactor)%period)*item['v'])%period
    def log(self, x):
        if isinstance(x, GF2Element):
            if x.field != self.field:
                raise ValueError('Element %s has a different polynomial than %x' % (x, self.field))
        else:
            x = self.field.wrap(x)
        #yvect = [(x ** k).coeffs for k in self.cofactors]
        yvect = self.field.powvectraw(x.coeffs, self.cofactors)     # this is faster, reuses powers of x
        r = [GF2DiscreteLog._rem(y,item,self.period) for (y,item) in zip(yvect,self.lookup)]
        return sum(r)%self.period 

def checkPeriod(poly, n, cofactors=None):
    """
    Verify whether the sequence x^0, x^1, x^2, ... is periodic with period n.
    
    Parameters
    ----------
    poly : integer
        Polynomial coefficient bit vector.
    n : integer
        Expected period.
    cofactors : array of integer, optional
        Array of n/p for prime factors p; 
        will be calculated for you if not provided.
    """                
    e2 = GF2Element(1,poly)
    if (e2 << n).coeffs != 1:
        return 0
    if cofactors is None:    
        factors, multiplicity = factorize(n, returnMultiplicity=True)
        cofactors = _calculateCofactors(factors, multiplicity)
    for cf in cofactors:
        if (e2 << cf).coeffs == 1:
            return cf
    return n
            
class GF2(object):
    """
    classdocs
    """
    def __init__(self, x):
        self.value = x
    def __repr__(self):
        return 'GF2(0b{0:b})'.format(self.value)
    def __add__(self, other):
        return GF2(self.value ^ other.value)
    def __sub__(self, other):
        return GF2(self.value ^ other.value)
    def __mul__(self, other):
        return GF2(_GF2.mul(self.value, other.value))
    def __mod__(self, other):
        return GF2(_GF2.mod(self.value, other.value))
    def __eq__(self, other):
        return self.value == other.value
    def __ne__(self, other):
        return self.value != other.value
    def expmod(self, k, m):
        return GF2(_GF2.powmod(self.value, k, m.value))
    
class GF2Element(tuple):
    """
    An immutable element in the field GF(2^n), which is represented as a quotient ring of polynomials. 
    
    Parameters
    ----------
    coeffs : integer
        Polynomial coefficient bit vector.
        
    field : GF2QuotientRing or integer
        The field in which the element is an element
        
    Notes
    -----
    The following standard Python operators are supported:
    
    - addition
    - subtraction
    - multiplication
    - division
    - raising to a power
    - left shift
    - right shift    
    """
    def __new__(cls, coeffs, field):
        field = GF2QuotientRing.cast(field)
        if (coeffs >> field.degree) != 0:
            raise ValueError("Elements must be smaller in degree than that of the characteristic polynomial")     
        return tuple.__new__(cls, (coeffs, field))
    coeffs = property(itemgetter(0))
    field = property(itemgetter(1))    

    def _wrapraw(self, a):   
        return GF2Element(a, self.field)
    def __add__(self, other):
        return self.field.add(self, other)
    def __radd__(self, other):
        return self.field.add(other, self)        
    def __sub__(self, other):
        return self.field.add(self, other)
    def __rsub__(self, other):
        return self.field.add(other, self) 
    def __mul__(self, other):
        return self.field.mul(self, other)
    def __rmul__(self, other):
        return self.field.mul(other, self)        
    def __div__(self, other):
        return self.field.div(self, other)
    def __rdiv__(self, other):
        return self.field.div(other, self)        
    def __pow__(self, k):
        return self.field.power(self, k)
    def __lshift__(self, k):
        return self.field.lshift(self, k)
    def __rshift__(self, k):
        return self.field.rshift(self, k)
    def __repr__(self):
        return 'GF2Element(0b{0:0{2}b},0x{1:x})'.format(self.coeffs, self.field.coeffs, self.field.degree)
    def polyval(self, g):
        """Evaluates g(u) where g is a polynomial and u is this element
        Parameters
        ----------
        g : integer
            Polynomial coefficient bit vector.
            
        Returns
        -------
        g(u) : GF2Element
        """
        return self.field.polyval(self, g)
    @property
    def inv(self):
        """Computes and returns the inverse element."""
        return self._wrapraw(self.field.invraw(self.coeffs))
    @property
    def log(self):
        """Returns the discrete logarithm of this element"""
        return self.field.lograw(self.coeffs)      
    @property
    def trace(self):
        """Returns the field trace of this element"""
        return self.field.trace(self.coeffs)      

class GF2TracePattern(tuple):
    '''
    Immutable base class for computing the trace parity of a value.
    
    This is defined mathematically as 
    
    Tr(beta,u) = z + z^2 + z^4 + z^8 + ... + z^(n-1)
    with z = beta*u. The value of beta is the trace pattern and is fixed.
    
    Instances of this class have the following immutable fields or properties:
    
    - field              : a GF2Element
    - mask               : bitmask used for calculating trace parity in practice
    - lfsr_initial_state : value of s (see below description)
    - pattern            : value of beta
    
    The values of mask and pattern are essentially conversions between
    different views of looking at beta. There is a third view,
    lfsr_initial_state, which is the value s such that y[k] = Tr(beta,x^k) is 
    the coefficient of x^n-1 in the reduced polynomial representation
    of the GF2Element s*x^k; y[k] is also the output of an LFSR
    which has been initialized with state bits corresponding to s.
    
    Any of these three quantities (mask, pattern, lfsr_initial_state)
    can be used to derive the other two.
    
    The pattern and lfsr_initial_state are related as beta = w*s
    where w and its inverse winv are constants
    related to a particular field's characteristic polynomial,
    cached in the trace_constants property of a GF2QuotientRing instance. 
    '''
    __slots__ = ()
    def __new__(cls, field, mask, lfsr_initial_state):
        return tuple.__new__(cls, (field, mask, lfsr_initial_state))
    field = property(itemgetter(0))
    mask = property(itemgetter(1))
    lfsr_initial_state = property(itemgetter(2))
    calculate_parity = staticmethod(util.parity)
    def __repr__(self):
        return '{0:s}({1:s},0x{2:x},0x{3:x}'.format(
            self.__class__.__name__,
            self.field,
            self.mask,
            self.lfsr_initial_state)
    @classmethod
    def create(cls, field, mask, lfsr_initial_state):
        if util.parity_sparse_is_faster(mask):
            cls = GF2TracePatternSparse
        return cls(field, mask, lfsr_initial_state)         
    @classmethod 
    def from_mask(cls, field, mask):
        s = cls._lfsr_initial_state_from_mask(field, mask)
        return cls.create(field, mask, s)
    @classmethod
    def _mask_from_lfsr_initial_state(cls, field, s):
        s = field.unwrap(s) 
        n = field.degree        
        mask = 0
        for k in range(n):
            # loop invariant: s = w^(-1)*x^k
            mask |= (s >> (n-1)) << k
            s = field.lshiftraw1(s)         
        return mask
    @classmethod
    def _lfsr_initial_state_from_mask(cls, field, mask):
        n = field.degree
        e = 1
        bits = []
        for _ in range(n):
            bits.append(util.parity(e&mask))
            e = field.lshiftraw1(e)
        return util.state_from_reversed_output(field.coeffs, bits[::-1])
    @property
    def pattern(self):
        w, winv, _ = self.field.trace_constants
        return self.field.mulraw(w, self.lfsr_initial_state)       
    @classmethod
    def _constants_from_field(cls, field):
        n = field.degree
        e = 1
        bits = []
        for _ in range(n):
            bits.append(e>>(n-1)) 
            e = field.lshiftraw1(e)
            e = field.lshiftraw1(e)
        v2 = util.state_from_reversed_output(field.coeffs, bits[::-1])
        winv = field.mulraw(v2,v2)
        w = field.invraw(winv)
        return w, winv        
    @classmethod 
    def from_pattern(cls, field, pattern):
        w, winv, _ = field.trace_constants    
        pattern = field.unwrap(pattern)
        s = field.mulraw(winv,pattern)
        mask = cls._mask_from_lfsr_initial_state(field, s) 
        return cls.create(field, mask, s)
    @classmethod
    def from_lfsr_initial_state(cls, field, s):   
        s = field.unwrap(s)
        mask = cls._mask_from_lfsr_initial_state(field, s) 
        return cls.create(field, mask, s)
    def __call__(self, u):
        u = self.field.unwrap(u)
        return self.calculate_parity(u&self.mask)
    def __mul__(self, u):    
        """
        return another GF2TracePattern instance which
        represents multiplication by u
        (equivalent to the trace pattern and lfsr_initial_state
        both multiplying by u)
        """
        u = self.field.unwrap(u)
        s2 = self.field.mulraw(self.lfsr_initial_state, u)
        return self.from_lfsr_initial_state(self.field, s2)
    def delay(self, k):
        """
        return another GF2TracePattern instance which
        represents multiplication by x^-k
        (equivalent to the trace pattern and lfsr_initial_state
        both multiplying by x^-k)
        When trace parity is viewed as the output of an LFSR,
        this can be used to delay that output by k steps.
        """
        s2 = self.field.rshiftraw(self.lfsr_initial_state, k)
        return self.from_lfsr_initial_state(self.field, s2)
        
class GF2TracePatternSparse(GF2TracePattern):
    '''
    Optimization of GF2TracePattern for sparse masks
    '''
    calculate_parity = staticmethod(util.parity_sparse)

def bit_vector_rank(vectors, copy=True):
    '''
    Determines the GF(2) rank of a 2-D matrix 
    represented by a 1-D array of bit vectors.
    Each bit vector represents one row of the matrix.
    '''
    n = len(vectors)
    if copy:
        vectors = vectors.copy()
        # make a copy since we are going to change some of the vectors
    rank = 0
    for k in range(n):
        if vectors[k] == 0:
            continue
        rank += 1
        v = vectors[k]
        lsb = v & -v
        # add v to every following row that shares the same lsb
        # (this clears that same lsb)
        for j in range(k+1,n):
            if vectors[j] & lsb:
                vectors[j] ^= v
    return rank

if __name__ == '__main__':
    x1 = GF2(0b101101)
    x2 = GF2(0b110110)
    x3 = GF2(0b101)
    print(x1+x2)
    print(x2)
    print(x3)
    print(x2*x3)
    e1 = GF2Element(0b110, 137)
    print(e1*e1)
    print(e1 << 6)
    print(e1 << 127)
    print(e1 >> 125)
    e2 = GF2Element(0b100, 137)
    print(e2 ** 2)
    print(e2 ** 3)
    print(e2 ** 127)
    
    b = 0b11010011
    a = 0b101101
    (g,x,y) = _GF2.exteuc(a,b)
    print(g)
    print(x,y)
    print(_GF2.mul(a,x) ^ _GF2.mul(b,y))
    
    dlog5a = GF2DiscreteLog(0x23, [3,7])
    dlog5 = GF2DiscreteLog(0x25, [31])
    dlog8 = GF2DiscreteLog(0x11d, [3,5,17])
    dlog14 = GF2DiscreteLog(0x402b, [3,43,127])
    dlog16 = GF2DiscreteLog(0x1002d, [3,5,17,257])
    for dlog in [dlog5,dlog5a,dlog14]:
        e1 = dlog.field.wrap(1)
        for i in range(25):
            x = e1 << i
            logx = dlog.log(x)
            print('log %s = %d' % (x, logx) )
    for dlog in [dlog14,dlog16]:
        e1 = dlog.field.wrap(1)
        for i in range(0,3000,33):
            x = e1 << i
            logx = dlog.log(x)
            print('%d: log %s = %d' % (i, x, logx))
            assert i == logx 
    e1 = dlog8.field.wrap(1)
    print(e1)
    for i in range(9):
        print('1 << %d == %s' % (i,e1<<i))
    for i in range(100,109):
        print('1 << %d == %s' % (i,e1<<i))
    for i in range(200,209):
        print('1 << %d == %s' % (i,e1<<i))
    
    for p in [0x402b, 0x100000000065, 0x10000000000b7]:
        n = _GF2.bitlength(p)-1
        dlog = GF2DiscreteLog(p)
        f = dlog.factors
        print('degree %d polynomial %x has factors %s' % (n,p,f))
    
    A = np.matrix([[1,0,1,0,1],[0,1,1,0,1],[0,1,0,0,1],[1,0,0,0,0],[0,0,1,1,0]], int)
    b = _bitsOf(0b10010)
    x = _gf2GaussJordan(A, b)
    print(b)
    print(x)
    print(((A*np.matrix(x))&1).transpose())

    b2 = np.matrix([[1,0,0,1,0],[0,1,1,1,1],[1,1,0,0,0]]).transpose()
    x2 = _gf2GaussJordan(A, b2)
    print(x2)
    print(((A*np.matrix(x2))&1))

    I5 = np.matrix(np.eye(5,dtype=int))
    Ainv = _gf2GaussJordan(A,I5)
    print(Ainv)
    print(A*Ainv&1)
    
    np.random.seed(123)
    I7 = np.matrix(np.eye(7,dtype=int))
    A7 = np.random.permutation(I7)
    Ainv = np.matrix(_gf2GaussJordan(A7,I7))
    print(Ainv)
    print(A7)
    print(Ainv*A7&1)
    
    poly = 0x20007
    qr = GF2QuotientRing(poly)
    import random
    r = random.Random(22)
    for i in range(1000):
        a = r.getrandbits(16)
        b = r.getrandbits(16)
        y1 = qr.mul(a,b)
        y2 = _GF2.mul(a,b) % qr
        print(y1)
        print(y2)
        assert y1 == y2
    print("Yay!")