``libgf2``: a Python module for computation in :math:`GF(2^n)`
==============================================================

This module is intended for exploration of linear
feedback shift registers (LFSRs) and other areas related
to binary Galois fields. 

At present there are two classes which are in a polished state:

``GF2QuotientRing`` -- this models the finite field :math:`GF(2^n)` as represented by the quotient ring :math:`GF(2)[x]/p(x)`, where the polynomial :math:`p(x)` is encoded as a bit vector of coefficients.

``GF2Element`` -- this models elements of within the quotient ring :math:`GF(2)[x]/p(x)`, where each element is a polynomial, encoded as a bit vector of coefficients.

The rest of the module is in a somewhat haphazard and undocumented state.
 
Licensing
---------

``libgf2`` is freely available under the Apache license.

Copyright 2013-2017 Jason M Sachs

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

