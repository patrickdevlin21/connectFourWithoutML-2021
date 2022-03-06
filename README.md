# connectFourWithoutML-2021

This is a project where I coded a five or so different algorithms to play Connect Four.  This is all using python without any machine learning (the idea being to use this as a benchmark and compare this to a machine learning algorithm).

There are several different AI levels, which have the following strategies.

Level 0: move randomly
Level 1: win in 1 if possible, else block opponent from winning in 1 if possible, else play randomly (favoring the center)
Level 2: think a few moves ahead with a key eye for tactics.  Else, play randomly (favoring the center)
Level 3: think a few moves ahead AND consider if one side can win by using a copy-cat strategy AND have a preference to "own" columns (i.e., maintain pending threats that prevent the opponent from playing in a certain column)
Level 4: same as level 3, but also consider if one side will win via zugswang

Level 5: alpha-beta pruning (experimental)
