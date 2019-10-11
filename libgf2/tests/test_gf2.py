'''

Copyright 2015-2017 Jason M. Sachs

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
import pytest
import numpy as np
import random

from libgf2.gf2 import (GF2Element, GF2Polynomial, GF2QuotientRing,
                        GF2TracePattern,
                        _gf2bitlength, _gf2power, _gf2mul,
                        _gf2mulmod, _gf2mulxmod, _gf2mulxinvmod,
                        checkPeriod,
                        bit_vector_rank)
                        

class Test_miscellaneous(object):
    def test_bitlength(self):
        for n in xrange(256):
            assert _gf2bitlength(1<<n) == n+1
        for n in xrange(6,256):
            assert _gf2bitlength((1<<n)+30) == n+1

class Test_Poly(object):
    def test_power(self):
        p = 5
        y = 1
        for k in xrange(77):
            assert _gf2power(p,k) == y
            y = _gf2mul(p,y)

class Test_Poly8(object):
    def setup(self):
        self.field = GF2QuotientRing(0x187)
    def test_value(self):
        for p in [0x3, 0x87, 0x0, 0x123, 0x12345]:
            assert GF2Polynomial(p).coeffs == p
    def test_mod1(self):
        assert 0x100 % self.field == GF2Polynomial(0x87)
        assert 0x187 % self.field == GF2Polynomial(0)
        assert 0x200 % self.field == GF2Polynomial(0x89)
    def test_mod2(self):
        assert self.field % 0x33 == GF2Polynomial(0x1f)
        assert self.field % self.field == GF2Polynomial(0)
    def test_degree(self):
        assert self.field.bitlength == 9
        assert self.field.degree == 8

class Test_Element(object):
    FIELD = GF2QuotientRing(0x187)
    def setup(self):
        self.e = GF2Element(1, self.FIELD)
    def test_toolarge(self):
        with pytest.raises(ValueError):
            GF2Element(0x100, self.FIELD)
        with pytest.raises(ValueError):
            e1 = GF2Element(0x1, self.FIELD)
            e1 * 0x100 
        with pytest.raises(ValueError):
            e1 = GF2Element(0x1, self.FIELD)
            e1 + 0x1111
            
    def test_lshift(self):
        e = self.e
        assert e.coeffs == 1
        assert e.field == self.FIELD
        assert (e << 1).coeffs == 2
        assert (e << 2).coeffs == 4
        assert (e << 40).coeffs == 0x62
        assert (e << 255).coeffs == 1
        assert (e << 254).coeffs == 0xc3
        assert (e << 253).coeffs == 0xa2
        assert (e << -1).coeffs == 0xc3
        assert (e << -2).coeffs == 0xa2
        assert (e << -215).coeffs == 0x62
    def test_rshift(self):
        e = self.e
        assert (e >> 1).coeffs == 0xc3
        assert (e >> 2).coeffs == 0xa2
        assert (e >> 215).coeffs == 0x62
        assert (e >> 253).coeffs == 4
        assert (e >> 254).coeffs == 2
        assert (e >> 255).coeffs == 1
        assert (e >> -1).coeffs == 2
        assert (e >> -2).coeffs == 4
        assert (e >> -40).coeffs == 0x62
    def test_mul(self):
        e = self.e
        e2 = GF2Element(2, self.FIELD)
        e3 = GF2Element(3, self.FIELD)
        assert (e2*e3).coeffs == 6
        assert (e2*3).coeffs == 6
        assert (2*e3).coeffs == 6
    def test_add(self):
        e = self.e
        e2 = GF2Element(2, self.FIELD)
        e3 = GF2Element(3, self.FIELD)
        assert (e2+e3).coeffs == 1
        assert (e2+3).coeffs == 1
        assert (2+e3).coeffs == 1
    def test_pow(self):
        e3 = GF2Element(3, self.FIELD)
        assert (e3 ** 2).coeffs == 0x5 
        assert (e3 ** 4).coeffs == 0x11
        assert (e3 ** 5).coeffs == 0x33 
        assert (e3 ** 100).coeffs == 0xa7
    def test_pow2(self):
        f2 = GF2QuotientRing(0x211)
        e = f2.wrap(73)
        ey = e
        for k in xrange(1,200):
            assert (e ** k) == ey
            ey *= e
    def test_powvect(self):
        e = self.e
        assert self.FIELD.powvect(e,[2,4,5,100]) == [e**2, e**4, e**5, e**100]
    def test_square(self):
        f1 = self.e.field
        f2 = GF2QuotientRing(0x211)
        for f in [f1,f2]:
            for k in xrange(3,250,7):
                assert f.mulraw(k,k) == f.squareraw(k)
                e = f.wrap(k)
                assert e*e == f.square(e)
    def test_inv(self):
        e = self.e
        e23 = GF2Element(23, self.FIELD)
        e23inv = e23.inv
        assert e23inv.coeffs == 0xa5
        assert e23inv*e23 == e         
        with pytest.raises(ValueError):
            GF2Element(0, self.FIELD).inv  
        for k in xrange(1,1<<self.FIELD.degree): 
            ek = GF2Element(k, self.FIELD)
            assert (ek.inv)*ek == e    
        F8b = GF2QuotientRing(0x180)
        with pytest.raises(ValueError):
            GF2Element(2, F8b).inv               
    def test_div(self):
        e2 = GF2Element(2, self.FIELD)
        e3 = GF2Element(3, self.FIELD)    
        eq = e2/e3   
        assert (eq).coeffs == 0x83
        assert e2 == e3*eq
        assert e2/3 == eq
        assert 2/e3 == eq
    def test_log(self):
        e=self.e
        for k in xrange(1,100):
            u = e << k
            assert u.log == k
        with pytest.raises(ValueError):
            self.FIELD.wrap(0).log
    def test_mulx(self):
        x = 2
        y = 1
        for k in xrange(1,100):
            y2 = _gf2mulxmod(y,self.FIELD.coeffs)
            y = self.FIELD.mulraw(x,y)
            assert y == y2
    def test_mulxinv(self):
        x = 2
        xinv = self.FIELD.invraw(x)
        y = 1
        for k in xrange(1,100):
            y2 = _gf2mulxinvmod(y,self.FIELD.coeffs)
            y = self.FIELD.mulraw(xinv,y)
            assert y == y2    
    def test_trace(self):
        w, winv, trace = self.FIELD.trace_constants
        assert w == 0x6
        assert winv == 0x41
        u = self.FIELD.wrap(1)
        u2 = u
        assert u.trace == self.FIELD.degree & 1
        for k in xrange(100):
            u <<= 1
            u2 <<= 2
            assert u.trace == (winv*u).coeffs >> (self.FIELD.degree - 1)
            assert u.trace == u2.trace
    def test_polyval(self):
        e23 = GF2Element(23, self.FIELD)
        assert (e23**3 + e23**2 + 1) == self.FIELD.polyval(e23, 0b1101)
        assert (e23**9 + e23**4 + e23**3) == self.FIELD.polyval(e23, 0b1000011000) 
        assert e23.polyval(0b11101) == e23**4 + e23**3 + e23**2 + 1
    def test_polyval2(self):
        def plain_polyval(u,g,qr):
            y = 0
            uk = 1
            while g > 0:
                if g & 1:
                    y ^= uk
                g >>= 1
                uk = qr.mulraw(uk,u)
            return y
        r = random.Random()
        r.seed(123)
        qr = GF2QuotientRing((1<<48) + 0xb7)
        p = (1<<48)-1
        assert checkPeriod(qr,p)==p
        for i in xrange(100):
            u = r.getrandbits(48)
            g = r.getrandbits(48)
            assert qr.polyvalraw(u,g) == plain_polyval(u,g,qr)
        e123 = qr.wrap(123)
        assert e123.polyval(0b100101) == e123**5 + e123**2 + 1

class Test_TracePattern(object):
    FIELD = GF2QuotientRing(0x187)
    def setup(self):
        self.e = GF2Element(1, self.FIELD)            
    def test_0(self):
        for w in xrange(1,60):
            tr1 = GF2TracePattern.from_pattern(self.FIELD, w)
            tr2 = GF2TracePattern.from_mask(self.FIELD, tr1.mask)
            assert tr1.pattern == w
            assert tr2 == tr1         
    def test_1(self):
        w,winv,tr0 = self.FIELD.trace_constants
        tr1 = GF2TracePattern.from_mask(self.FIELD, 1 << (self.FIELD.degree - 1))
        assert tr1.pattern == w
        u = 1
        # now make that tr1 acts like an LFSR output
        for k in xrange(100):
            u = self.FIELD.lshiftraw1(u)
            assert tr1(u) == (u >> (self.FIELD.degree - 1)) 
    def test_2(self):
        tr2 = GF2TracePattern.from_pattern(self.FIELD, 1)
        assert tr2 == self.FIELD.trace
    def test_3(self):
        for w in [1,2,3,4,7,81,self.FIELD.coeffs >> 1]:
            tr = GF2TracePattern.from_pattern(self.FIELD, w)
            for u in xrange(100):
                assert tr(u) == self.FIELD.mul(w,u).trace
    def test_delay(self):
        tr0 = GF2TracePattern.from_mask(self.FIELD, 0x80)
        bits0 = []
        tr1 = tr0.delay(4)
        bits1 = []
        tr2 = tr0.delay(9)
        bits2 = []
        u = 1
        for k in xrange(100):
            bits0.append(tr0(u))
            bits1.append(tr1(u))
            bits2.append(tr2(u))
            u = self.FIELD.lshiftraw1(u)   
        assert bits0[:-4] == bits1[4:]
        assert bits0[:-9] == bits2[9:]
    def test_mul(self):
        tr0 = GF2TracePattern.from_mask(self.FIELD, 0x80)
        assert (tr0 * 32) == tr0.delay(-5)

def to_bit_vector(B):
    return np.array([sum(b<<k for k,b in enumerate(row)) for row in B], dtype=np.uint64)
             
def test_bit_vector_rank(N=48, seed=None):
    r = np.random.RandomState(seed)
    for k in xrange(1,N):
        # makes use of the fact that if A=Nxk and B=kxN are matrices of full rank,
        # with k <= N, then AB has known rank k 
        while True:
            A = r.randint(0,2,(N,k))
            if bit_vector_rank(to_bit_vector(A)) < k:
                continue
            B = r.randint(0,2,(k,N))
            if bit_vector_rank(to_bit_vector(B)) == k:
                break
        C = np.dot(A,B) & 1
        v = to_bit_vector(C)
        assert bit_vector_rank(v) == k
                           
                                      