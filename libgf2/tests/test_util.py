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

from libgf2.util import (parity, parity_sparse, parity_sparse_is_faster,
                         berlekamp_massey, coset_representative) 
from libgf2.gf2 import GF2QuotientRing

class Test_parity(object):
    def test(self):
        for p in [parity, parity_sparse]:
            for bits in [0,6,9,258]:
                assert p(bits) == 0
            for bits in [1,7,254]:
                assert p(bits) == 1
            with pytest.raises(ValueError):
                p(-3)
    def test_faster(self):
        for bits in [0,1,2,4,8,16,10,33,80,257]:
            assert parity_sparse_is_faster(bits)
        for bits in [15, 206, 250, 1000]:
            assert not parity_sparse_is_faster(bits)           

class Test_BerlekampMassey(object):
    def test1(self):
        for tailbits, initstate in [
            ([0,0],0x1d),
            ([0,1],0x19),
            ([1,0],0x11),
            ([1,1],0x15)
        ]:
            bits = [1,1,1,0,1,0,0,0]+tailbits
            p, n = berlekamp_massey(bits)
            e = GF2QuotientRing(p).wrap(initstate)  
            bits2 = [(e<<k).coeffs>>(n-1) for k in xrange(len(bits))] 
            assert bits == bits2
    def test2(self):
        for p, initstate in [
            (615,0x92),
            (456,0x37),
            (59857573,0x123),
            (0x10000000000b7,23)
        ]:
            field = GF2QuotientRing(p)
            e = field.wrap(initstate)
            bits = [(e<<k).coeffs>>(field.degree-1) for k in xrange(2*field.degree)]
            p, n = berlekamp_massey(bits)
            assert p == field.coeffs          

class Test_CosetRepresentative(object):
    def test1(self):
        for b in [0b1011101001,
                  0b1101110100,
                  0b0110111010,
                  0b0011011101,
                  0b1001101110,
                  0b0100110111,
                  0b1010011011,
                  0b1101001101,
                  0b1110100110,
                  0b0111010011, 
                  ]:
            assert coset_representative(b,10) == 0b0011011101   