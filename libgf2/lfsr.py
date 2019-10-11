'''
Objects and functions for working with linear feedback shift registers

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

from libgf2.gf2 import GF2Element, GF2DiscreteLog, GF2Polynomial, GF2QuotientRing
from libgf2.util import berlekamp_massey, coset_representative, parity
import random

class LFSRBase(object):
    def __init__(self, field):
        self.field = GF2QuotientRing.cast(field);
    @property
    def degree(self):
        return self.field.degree
    def _stateFromReverseIterator(self, it, atEnd=True):
        '''returns the state of an LFSR with default mask that produces the given output.
        if atEnd is True, the state in question is the state at the end of the output.
        Otherwise it is the state at the beginning of the output.
        '''
        n = self.degree
        s = 0
        # run backwards until the state has no unknown bits
        for bit in it:
            if bit != 0:
                s ^= self.field.coeffs
            s >>= 1
        # then run forwards again, if desired
        if atEnd:
            for i in range(n-1):
                s <<= 1
                if (s >> n) & 1 == 1:
                    s ^= self.field.coeffs
        return s
    
    def stateFromOutput(self, output, atEnd=True):
        '''returns the state of an LFSR with default mask that produces the given output.
        if atEnd is True, the state in question is the state at the end of the output.
        Otherwise it is the state at the beginning of the output.
        '''
        n = self.degree
        return self._stateFromReverseIterator(((output >> i & 1) for i in range(n)), atEnd)
    def stateFromOutputBits(self, output, atEnd=True):
        '''returns the state of an LFSR with default mask that produces the given output.
        if atEnd is True, the state in question is the state at the end of the output.
        Otherwise it is the state at the beginning of the output.
        '''
        n = self.degree
        return self._stateFromReverseIterator(output[n-1::-1], atEnd)

class LFSR(LFSRBase):
    '''
    Linear feedback shift register, with an output mask:
    at any given instant, the output is the parity of the state ANDed with the output mask.
    If no mask, then we just take the previous high bit.
    LFSR state s[k] = initstate * x^k 
    '''
    def __init__(self, field, initstate=1, mask=None):
        LFSRBase.__init__(self, field)
        self.testmask = 1 << self.degree
        self.coefficientsMask = mask
        self.state = initstate
    def __iter__(self):
        return self
    def next(self):
        s = self.state << 1
        (b,nextState) = (0,s) if ((s & self.testmask) == 0) else (1,s^self.field.coeffs)
        if self.coefficientsMask is not None:
            b = parity(self.state & self.coefficientsMask)
        self.state = nextState            
        return b
    def __lshift__(self, k):
        newstate = self.field.lshiftraw(self.state, k)
        return LFSR(self.field, initstate=newstate, mask=self.coefficientsMask)
    def __ilshift__(self, k):
        self.state = self.field.lshiftraw(self.state, k)
        return self
    def __rshift__(self, k):
        newstate = self.field.rshiftraw(self.state, k)
        return LFSR(self.field, initstate=newstate, mask=self.coefficientsMask)
    def __irshift__(self, k):
        self.state = self.field.rshiftraw(self.state, k)
        return self
    def __repr__(self):
        return 'LFSR({0:x},{1:x})'.format(self.field.coeffs,self.state)

class LFSRAnalyzer(LFSRBase):
    def __init__(self, field, factors=None):
        LFSRBase.__init__(self, field)
        self.dlog = GF2DiscreteLog(self.field, factors)
    def lookaheadCoefficients(self, k):
        '''returns coefficients a_j such that the sum of a_j*state[j] = output[n+k],
        for an LFSR with default mask'''
        e1 = GF2Element(1,self.field)
        s = e1 << k
        return self.coefficientsFromState(s.coeffs)
    def stateFromCoefficients(self, c):
        '''returns initial state of an LFSR with default mask, such that the output matches 
        the output of a masked LFSR with given coefficients and initial state of 1'''
         
        n = self.degree
        e1 = GF2Element(1,self.field)
        out = 0
        for i in range(n):
            out <<= 1
            out |= parity((e1<<i).coeffs & c)
        return self.stateFromOutput(out, atEnd=False)
    def timeshiftFromCoefficients(self, c):
        '''inverse of lookaheadCoefficients'''
        s = self.stateFromCoefficients(c)
        return self.dlog.log(s)
    def timeshiftFromState(self, s):
        return self.dlog.log(s)
    def coefficientsFromState(self, s):
        '''Inverse of stateFromCoefficients'''
        n = self.degree
        c = 0
        for i in range(n):
            b = (s >> (n-1)) << i
            c |= b
            s <<= 1
            if (s >> n) & 1 == 1:
                s ^= self.field.coeffs
        return c        

class PolyRingOverField(object):
    """
    Polynomial ring F[y], where F is a field
    of the type GF(2)[x]/p(x)
    
    This class encapsulates calculation behavior
    but not state; its methods take in lists of integers
    representing polynomials in this ring, with coefficients
    in the field F in order of lowest degree first. 
    
    For example, suppose the field is
    H25 = GF2QuotientRing(0x25) with p(x) = x^5 + x^2 + 1,
    and we are operating on the list [9,6,1]; this represents
    the polynomial in y of (x^3 + 1) + (x^2 + x)y + y^2.
    """
    def __init__(self, field):
        self.field = field
    def _mulraw(self, c1, c2):
        """ 
        helper function to multiply coefficients
        c1(x) and c2(x) in the field F.
        
        This tends to be the computational bottleneck,
        so we make it easy for subclasses to optimize.
        """
        return self.field.mulraw(c1,c2)

    def add(self, p1, p2):
        """
        Add p1(y) + p2(y).
        Since F has characteristic 2, this is just an XOR.
        """
        result = list(p1)
        n = len(result)
        for k,c in enumerate(p2):
            if k >= n:
                n += 1
                result.append(0)
            result[k] ^= c
        return result
    def sub(self, p1, p2):
        return self.add(p1,p2)
    def mul(self, p1, *args):
        """
        Multiply p1(y) by an arbitrary number of other polynomials
        """
        result = p1[:]
        for p2 in args:
            n1 = len(result)
            p1 = result         
            n2 = len(p2)
            result = [0]*(n1+n2-1)
            for k1,c1 in enumerate(p1):
                for k2,c2 in enumerate(p2):
                    result[k1+k2] ^= self._mulraw(c1,c2)
        return result
    def normalize(self, p):
        """ truncate terms of highest degree that are zeros """
        lastnonzero = 0
        for k,c in enumerate(p):
            if c != 0:
                lastnonzero = k
        return p[:lastnonzero+1]
    def divmod(self, p1, p2):
        """
        Compute quotient and remainder 
        
        Returns q,r such that p1(y) = q(y)p2(y) + r(y) 
        """
        n1 = len(p1)
        n2 = len(p2)
        if n1 < n2:
            return [0], p1
        nq = n1-n2+1
        q = [0]*nq
        r = p1[:]
        p2a = p2[n2-1]
        for k in range(nq-1,-1,-1):
            q[k] = self.field.divraw(r[n2+k-1],p2a)
            for j in range(n2):
                r[k+j] ^= self._mulraw(q[k],p2[j])
        r = self.normalize(r)
        assert len(r) <= n2, "len(r)=%d, n2=%d" % (len(r),n2)
        return q,r
    def mulmod(self, p1, p2, m):
        """ Computes p1(y)p2(y) mod m(y) """
        p1p2 = self.mul(p1,p2)
        q,r = self.divmod(p1p2,m)
        return r
    def powmod(self, p, m, k):
        """ Computes p(y)^k mod m(y) """
        a = p
        y = [1]
        while k > 0:
            if k & 1:
                y = self.mulmod(y,a,m)
            a = self.mulmod(a,a,m)
            k >>= 1
        return y
    def gcd(self, p1, p2):
        """
        Computes greatest common divider g(y)
        of p1(y) and p2(y)
        """
        a = p1
        b = p2
        result = [0]
        while b != [0]:
            q,r = self.divmod(a,b)
            result = b
            a,b = b,r
        r1 = result[-1]
        r1inv = self.field.invraw(r1)
        result = [self._mulraw(r1inv,c) for c in result]
        return result
    def sqrt(self, p):
        """
        Computes square root of a polynomial p(y)
        """
        assert not any(p[1::2]), "not a perfect square"
        n = self.field.degree
        d = 1 << (n-1)
        return [self.field.powraw(c,d) for c in p[::2]]
    def randpoly(self, n, r=None):
        """
        Generates random polynomial of degree n.
        Optional argument r can be a random.Random object
        (or any other object that has a getrandbits method)
        """
        if r is None:
            r = random
        d = self.field.degree
        return [int(r.getrandbits(d)) for _ in range(n+1)]
        
def squarefree_factor(prof, p):
    """
    returns (q,a) such that p = q*a*a
    """
    formalderivative = p[1:]
    for k in range(1,len(p)-1,2):
        formalderivative[k] = 0
    if not any(formalderivative):
        # derivative is zero: it's a perfect square
        return [1], prof.sqrt(p)
    formalderivative = prof.normalize(formalderivative)
    g = prof.gcd(p, formalderivative)
    q, r = prof.divmod(p,g)
    assert r == [0]
    return q, prof.sqrt(g)

def distinct_degree_factor(prof, p, linear_only=False):
    """
    returns a list of (a,b) pairs such that ...
    """
    a = [0,1]
    p = p[:]
    i = 0
    result = []
    n = prof.field.degree
    while True:
        i += 1
        if len(p) < 2*i:
            break
        for _ in range(n):
            a = prof.mulmod(a,a,p)
        y = a[:]
        y[1] ^= 1
        if not any(y):
            result.append((p,i))
            if linear_only and i == 1:
                return result
            p = [1]
            break
        else:
            g = prof.gcd(y,p)
        if g != [1]:
            result.append((g,i))
            if linear_only and i == 1:
                return result
        q,r = prof.divmod(p,g)
        assert r == [0]
        p = q
    if p != [1]:
        result.append((p,len(p)-1))
    if not result:
        result = [(p,1)]
    return result

def _czsub(prof, p, d, imax=None, only_one_factor=False):
    """
    Portion of Cantor-Zassenhaus algorithm
    
    see also https://math.stackexchange.com/questions/1636518/how-do-i-apply-the-cantor-zassenhaus-algorithm-to-mathbbf-2
    http://blog.fkraiem.org/2013/12/01/polynomial-factorisation-over-finite-fields-part-3-final-splitting-cantor-zassenhaus-in-odd-characteristic
    """
    n = len(p)-1
    if n == d:
        return [p],[]
    unfactored = [p]
    factored = []
    while unfactored and imax != 0:
        if imax is not None:
            imax -= 1
        p = unfactored.pop()
        h = prof.randpoly(n-1)
        hpow = h[:]
        hsum = h[:]
        for k in range(prof.field.degree * d - 1):
            hpow = prof.mulmod(hpow,hpow,p)
            hsum = prof.add(hsum,hpow)
        hsum = prof.normalize(hsum)
        if not any(hsum) or hsum == [1]:
            unfactored.append(p)
            continue
        try:
            g = prof.gcd(p, hsum)
        except:
            print("hsum", hsum)
            #raise Exception('')
        q, r = prof.divmod(p,g)
        assert r == [0]
        
        # Handle trivial factors (this iteration failed)
        if g == [1]:
            unfactored.append(q)
            continue
        if q == [1]:
            unfactored.append(g)
            continue
            
        if len(g) < len(q):
            g,q = q,g
        # now len(g) >= len(q);
        # this puts the shorter subfactor on the unfactored list
        # if it's not a completely factored subfactor
        for factor in [g,q]:
            target = factored if len(factor) == d+1 else unfactored
            target.append(factor)
        if factored and only_one_factor:
            break
    return factored, unfactored
    
def cantor_zassenhaus(prof, p, max_iter=1000):
    """
    Cantor-Zassenhaus algorithm
    """
    sqfree, sq = squarefree_factor(prof, p)
    ddflist = distinct_degree_factor(prof, sqfree)
    factors = []
    for p,d in ddflist:
        factored, unfactored = _czsub(prof, p, d, max_iter)
        assert not unfactored
        factors += factored
    return factors, sq

def find_linear_factor(prof, p, max_iter=1000):
    """
    Adaptation of Cantor-Zassenhaus to find a single linear factor
    """
    sqfree, sq = squarefree_factor(prof, p)
    if sqfree == [1]:
        raise ValueError('Missing squarefree component')
    ddflist = distinct_degree_factor(prof, sqfree, linear_only=True)
    for p,d in ddflist:
        if d == 1:
            factored, unfactored = _czsub(prof, p, d, max_iter, only_one_factor=True)
            return factored[0]
    raise ValueError('No linear factors found')

def decimate(field,j):
    e = field.wrap(1)
    u = e
    n = field.degree 
    m = e << j
    bits = []
    for k in range(2*n):
        bits.append(u.coeffs >> (n-1))
        u *= m
    return GF2QuotientRing(berlekamp_massey(bits)[0])      
    
def undecimate(field, original_field, profclass=PolyRingOverField):
    field = GF2QuotientRing.cast(field)
    original_field = GF2QuotientRing.cast(original_field)
    prof = profclass(original_field)
    p = [(field.coeffs >> k) & 1 for k in range(field.degree+1)]
    f = find_linear_factor(prof, p)
    j = original_field.lograw(f[0])
    return coset_representative(j, field.degree)          

if __name__ == '__main__':
    print(parity(0b1101))
    print(parity(0b11011000010101101))
    print(parity(0b11010000010101101))
    sr = LFSR(0b1101)
    for i in range(17):
        b = sr.next()
        print(i,b,sr)
    print('sr >> 17 = %s' % (sr >> 17))
    sr = LFSR(0x10000000000b7)
    print(sr << 17)
    print(sr << 19)
    print(sr << 123)
    sr <<= 1
    sr <<= 123
    print(sr)
    print(sr >> 124)
    print(sr << 12)
    print(sr << 13)
    poly = 0x8003
    poly = 0x9091
    lfsrAnalyzer = LFSRAnalyzer(poly)
    n = GF2Polynomial(poly).degree
    e1 = GF2Element(1,poly)
    for k in [100,110,200,556,9171]:
        o = 0
        for i in range(n):
            y = e1 << (i+k)
            o = (o << 1) | (y.coeffs >> n-1)
            print('%d: %s' % (i+k, y))
        print('output = %s' % bin(o))
        print('state  = 0b{0:b} (expected 0b{1:b})'.format(lfsrAnalyzer.stateFromOutput(o,True), (e1 << (k+n-1)).coeffs))
        c = lfsrAnalyzer.lookaheadCoefficients(k)
        print(bin(c))
        print('{0:b} --> timeshift={1:d} (expected {2:d})'.format(c,lfsrAnalyzer.timeshiftFromCoefficients(c),k))
        S = [e1 << j for j in range(30,50)]
        xexact = [(e1 << j).coeffs >> (n-1) for j in range(30+k,50+k)]
        xpredict = [parity(s.coeffs & c) for s in S]
        print('k=%d\n    xexact=%s\n  xpredict=%s' % (k,xexact,xpredict))

    ecomp = e1 << ((1 << n)-n)
    for c in [123, 942, 1000, 2107, 12280, 15092, 21038, 16384, 32767]:
        s = lfsrAnalyzer.stateFromCoefficients(c)
        c2 = lfsrAnalyzer.coefficientsFromState(s)
        print('c=%d, s=%s, c2=%d' % (c,s,c2))
        
   