# EVE
This repository is official implementation of [**We Don't Need No Adam, All We Need Is EVE: On The Variance of Dual Learning Rate And Beyond**](https://arxiv.org/) using TensorFlow.

---

<font size = 4>EVE is equipped with two learning rates, where the 1st momentum is being obtained using short-term and long-term momenta, and 2nd momentum is a result of a residual learning paradigm. We have included all the interactive figures under the "figures" directory. The users and the community are encouraged to download these files and visualise the behaviour of the Adam optimiser vs. EVE in more granular detail.



---
## **How to use?!**
<font size = 4>Assuming you have built a TensorFlow model named "model", you will be able to compile the model using EVE as follows:
```python
from eve import EVE
...
model.compile(optimizer = EVE(learning_rate_1 = 0.001, learning_rate_2 = 0.0004), loss = ..., metrics = ...)
history = model.fit(...)
...
```
