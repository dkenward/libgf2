'''

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

'''
import pytest

from libgf2.gf2 import (GF2Element, GF2Polynomial, GF2QuotientRing)
from libgf2.lfsr import (PolyRingOverField, cantor_zassenhaus, _czsub)                 

@pytest.fixture
def prof(request):
    return PolyRingOverField(GF2QuotientRing(0x211))

class Test_PolyRingOverField(object):
    def test_add(self, prof):
        assert prof.add([1,1],[3,7]) == [2,6]
        assert prof.add([1,1],[3,7,0,1]) == [2,6,0,1]
    def test_mul(self, prof):
        assert prof.mul([5,1],[5,1,1]) == [17,0,4,1]
    def test_divmod(self, prof):
        assert prof.divmod([16,5,4,1],[5,1,1]) == ([5,1],[1,5])
    def test_sqrt(self, prof):
        assert prof.sqrt([17,0,1]) == [5,1]
        with pytest.raises(AssertionError):
            prof.sqrt([17,1,1])
    def test_normalize(self, prof):
        assert prof.normalize([3,3,0,1]) == [3,3,0,1]    
        assert prof.normalize([3,3,0,1,0,0,0]) == [3,3,0,1]  
    def test_mulmod(self, prof):
        assert prof.mulmod([5,1],[5,1,1],[7,2,1]) == [3,11]
    def test_powmod(self, prof):
        for k,r in enumerate([[1],
                              [5,1],
                              [17,0,1],
                              [84,17,5],
                              [257,1],
                              [295, 260, 1]]): 
            assert prof.powmod([5,1],[1,0,0,1],k) == r
    def test_gcd(self, prof):
        p1 = prof.mul([9,7,5,3,1],[38,44])
        p2 = prof.mul([9,7,5,3,1],[13,19])
        p1copy = p1[:]
        p2copy = p2[:]
        g = prof.gcd(p1,p2)
        assert g == [9,7,5,3,1]
        assert p1 == p1copy
        assert p2 == p2copy 
    def test_gcd2(self):
        f64 = GF2QuotientRing(0x43)
        prof = PolyRingOverField(f64)
        f = [10, 26, 55, 60,1]
        S = [13, 12, 32, 47]
        Scopy = S[:]
        g = prof.gcd(S,f)
        assert S == Scopy
    def test_divmod2(self):
        f64 = GF2QuotientRing(0x43)
        prof = PolyRingOverField(f64)
        f = [10, 26, 55, 60,1]
        S = [13, 12, 32, 47]
        Scopy = S[:]
        q,r = prof.divmod(S,f)
        assert S == Scopy
            
class Test_Cantor_Zassenhaus(object):
    def test1(self, prof):
        factors1 = [[2,1],[3,1],[7,1],[1,1],[19,1,1],[256,296,1]]
        p = prof.mul(*factors1)
        factors2,sq= cantor_zassenhaus(prof, p) 
        assert sorted(factors1) == sorted(factors2)
    def test2(self, prof):
        p = [1,3,7] + [0]*19 + [1]
        factors, sq = cantor_zassenhaus(prof, p)
        assert sq == [1]
        assert sorted(factors) == sorted([[230, 1], [367, 125, 1], [314, 290, 374, 1], [114, 155, 351, 429, 235, 433, 66, 121, 55, 457, 390, 228, 296, 282, 336, 493, 1]])
        p2 = prof.mul(*factors)
        assert p == p2
                                     