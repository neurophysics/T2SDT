# Type 2 SDT Analysis
Calculate the type 2 Signal Detection Theory (SDT) measure meta-d'
according to the method described in:

> Maniscalco, B., & Lau, H. (2012). A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings. Consciousness and Cognition, 21(1), 422-430. doi:10.1016/j.concog.2011.09.021

and

> Maniscalco, B., & Lau, H. (2014). Signal detection theory analysis of type 1 and type 2 data: meta-d', response-specific meta-d', and the unequal variance SDT mode. In S. M. Fleming & C. D. Frith (Eds.), The Cognitive Neuroscience of Metacognition (pp.25-66). Springer.

Only the equal variance approach and normally distributed inner decision
variables are currently supported. Additionally, currently only the overall
type 2 meta-d' is calculated (i.e., not response specific meta-d'
variables).

***************************************************************************
Disclosure:                                                       
-----------                                                       
This software comes as it is - there might be errors at runtime and results
might be wrong although the code was tested and did work as expected. Since
results might be wrong you must absolutely not use this software for a
medical purpuse - decisions concerning diagnosis, treatment or prophylaxis
***************************************************************************

Usage:
------
The class T2SDT implements the optimization of the type 2 SDT model.
As data, a confusion matrix (including confidence ratings) should be given.

The confusion matrix (including condidence ratings) can be calculated
from data using the function confusion_matrix:

```
conf_matrix = confusion_matrix(true_label, pred_label, rating)
```

After initialization, the fit() method of the class can be used to fit
the type 2 SDT model to the supplied data:

```
model = T2SDT(conf_matrix, adjust=True) # initialize the model
model.fit() # fit the model
# extract the parameters of the fitted model
d = model.d # the d' of the type 1 task
c = model.c # the response bias c of the type 1 task
meta_d = model.meta_d # the meta-d' of the type 2 task
```

The docstring included in the code provides more information about the
calculated parameters and the usage.

Notes:
------
The performance of this code was compared to the Matlab code available
at http://www.columbia.edu/~bsm2105/type2sdt/
Results were equivalent (this python implementation gave slightly larger
log-likelihoods of the results).

***************************************************************************

License:
--------
Copyright (c) 2017 Gunnar Waterstraat

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Author & Contact
----------------
Written by Gunnar Waterstraat

email: gunnar[dot]waterstraat[at]charite.de

