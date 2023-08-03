# Automated Gaming

# TO VIEW IN ACTION, CLICK HERE

## Results

* I was able to make a snake that goes up to around __60__ on average after around _100_ rounds of training
* The snake reward system made it always go towards the food, too, which means it got a _higher score in a shorter time_ as well
* __HOW I AM UPDATING__: Implementing the _Hamilton Algorithm would be more effective and will give better results way more often

## Process

* I made a snake game using __PyGame__ that had a reward system for eating the food and punishing for dying
* I made a neural network that trains itself that can control the moves going either _left, right, or straight_ but never behind since that will kill it
* Implemented a reinforcement learning network that learned to get the food and what blocks around it are dangerous and let it deploy and record

## Project Inspiration

I always loved to see people make AI for retro games, and I wanted to take my __first step into reinforcement learning with games__. Since Snake is a rather simple game compared to the likes of Mario and Sonic (and due to copyright because Nintendo is ruthless), I chose this game to learn.

## What I would do better

* Like I said above, implement the Hamilton algorithm, which makes it go rows so its slower but can fill the entire board
* I spent time making the game instead of making an environment, which wasted a long time
* work out some quirks, like when it goes into circles forever or traps itself by punishing it for doing that
