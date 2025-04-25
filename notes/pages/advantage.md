---
alias: advantage function
---

- What shape is the advantage function? #card
    - $A(s,a) \in \mathbb{R}^{|A|}$, where $|A|$ is the number of actions.
    - It works in a fixed integer number of actions.
    - Same shape as the Q-value function.
- What is the intuition behind the advantage function? #card
    - The advantage function is a measure of how much better an action is compared to the average action in a given state.
    - Learning relative advantage is easier and has less variance than learning absolute values. Advantage is more relevant to decision making via argmax than absolute values.
- Define the advantage function in terms of the Q-value function and the value function. #card
  - $A(s,a) = Q(s,a) - V(s)$
- What are the ways you normalize an advantage function and why?
  - Max normalization: 
