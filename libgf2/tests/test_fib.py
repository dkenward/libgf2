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

from libgf2.gf2 import _gf2mulxmod
from libgf2.fib import _fibmulxmod, _gf2_to_fib, _fibmulxnmod, _fib_to_gf2

def test_fib(request):
    fpoly = 0b100101
    gpoly = 0b101001
    mask  = 0b011111
    msb   = 0b010000
    n = 5
    fpolycomp = _fibmulxnmod(1,n-1,fpoly,mask)
    y = 1
    yg = 1
    for k, y_exp in enumerate([
        0b00001,
        0b00010,
        0b00101,
        0b01010,
        0b10101,
        0b01011,
        0b10111,
        0b01110,
        0b11101,
        0b11011,
        0b10110,
        0b01100,
        0b11000,
        0b10001,
        0b00011,
        0b00111,
        0b01111,
        0b11111,
        0b11110,
        0b11100,
        0b11001,
        0b10011,
        0b00110,
        0b01101,
        0b11010,
        0b10100,
        0b01001,
        0b10010,
        0b00100,
        0b01000,
        0b10000
        ]):
        assert y == y_exp, k
        y = _fibmulxmod(y, fpoly, mask)
        yg = _gf2mulxmod(yg, gpoly)
        print "{0:d} {1:05b} {2:05b}".format(k,yg,y)  
        assert y == _gf2_to_fib(yg, fpoly, mask)
        assert yg == _fib_to_gf2(y, fpolycomp, msb)
                           
                                      