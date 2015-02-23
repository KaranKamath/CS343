Karan Kamath
kpk462

Q1-Q4:

I've implemented a searchGeneric function that takes in a container
as an argument, and runs a generic seach using the graph search pseudo code.

The algorithm agnostic nature of that let me implement one line solutions
to the search problems by just changing the containers passed to search from
Stack to Queue and for Q3-4: PriorityQueueWithFunction.

Q5:

I've represented the state as a tuple of (currentPositon, currentlyUnvisitedCorners)
This is as the goal test needs to check if there are any unvisited corners.
Each move removes an unvisited corner if it happens to be visiting it.

The unvisited corners tuple is necessary as the search won't backtrack over
explored nodes unless it knows something about the state having changed
owing to a visit to some corner.

Q6:

The heuristic function designed solves the relaxed problem of reaching the
furthest unvisited corner (manhattan distance) considering no walls.
Hence, the heuristic is guaranteed to be admissable.

Q7:
The heuristic is designed to solve the relaxed problem of reaching the closest
food and then the furthest food from that position, considering no walls.

Hence, the heuristic is guaranteed admissability.

Q8:
The goal test just checks whether pacman reached any food.

The search is essentially a bfs.
