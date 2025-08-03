# Log Probability

## Why would you use a negative [[log-probability]] in a loss function?

It's infinity at 0 and 0 at 1, which means at high confidence in something you get a low loss approaching 0, and at low confidence you get a high loss approaching infinity. It gives a strong [[gradient]] signal to the network to update its parameters.

## What is the equation for a sigmoid function?

$f(x) = \frac{1}{1 + e^{-x}}$
