Karan Kamath
kpk462

Question 1
Evaluation function crafted with respect to distance to nearest food, amount of food and distance to ghost.
The function is bounded by dividing these factors by some environment bound of each of them
(like size of the grid for distances), so as to ensure consistent behavior.

Question 2:
minval implemented with generic status as we choose to call min or max val based on the next agent and not
alternate as in textbk pseudocode.

Question 3:
Same as above for a-b pruning instead of minimax

Question 4:
minval -> expval
Probability calculated as 1/numOfAvailableMoves (Random choice).

Question 5:
Eval function crafted using factors that affect state value,
determining whether they are directly or inversely proportional,
and then optimizing powers to which they must be raised via trials

Factors Added In The Denominator (Inverse Relationship):
    1. Amount of food left (Inversely proportional to 10th power)
    2. Distance to nearest food
    3. Number of capsules left
    4. Distance to nearest capsule
Factors in Numerator (Direct Relationship):
    1. Distance to nearest ghost

Special Cases:
    1. Distance to ghost drops below 2: Flee (Negative Utility)
    2. Ghosts are scared: Chase (Eval function only depends on distance to ghost)
