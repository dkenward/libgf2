'''
Created on Oct 13, 2013

@author: jmsachs

Copyright 2013 Jason M Sachs

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

def _gf2mulmod(x,y,m):
    z = 0
    while x > 0:
        if (x & 1) != 0:
            z ^= y
        y <<= 1
        y2 = y ^ m
        if y2 < y:
            y = y2
        x >>= 1
    return z

def _gf2squaremod(a,N,squarecache):
    c = 0
    N2 = (N+1)//2
    ah = a >> N2
    c ^= a&1
    c ^= (ah&1)*squarecache[0]
    for i in xrange(1,N2):
        a >>= 1
        ah >>= 1
        c ^= (a&1) << (2*i)
        c ^= (ah&1) * squarecache[i]
    return c