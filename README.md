# MLRecap
Reimplementation of base ML concept for learning purpose

## Understandings.
Exponential based softmax cause heteroscedasticity.
ie:
```
[1, 2, 3, 40, 41]
[0.011494252873563218, 0.022988505747126436, 0.034482758620689655, 0.45977011494252873, 0.47126436781609193]
[3.1057958233902085e-18, 8.442428349625602e-18, 2.2948899570854758e-17, 0.26894142136999516, 0.7310585786300048]
```
thus acts a bit like rectification 

------
Could be useful to set d(sigm) as activation function.