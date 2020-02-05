# SafeLife

SafeLife is a novel environment to test the safety of reinforcement learning agents. The long term goal of this project is to develop training environments and benchmarks for numerous technical reinforcement learning safety problems, with the following attributes:

* Controllable difficulty for the environment
* Controllable difficulty for safety constraints
* Procedurally generated levels with richly adjustable distributions of mechanics and phenomena to reduce overfitting


#### Player generating patterns

<p align="center">
<img alt="pattern demo" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/pattern-demo.gif?raw=true"/>
</p>



#### Agents in static environments

Without side-effect penalty:

<p align="center">
<img alt="benchmark level append-still-013, no impact penalty" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/v1.0/benchmark-append-still-013_p=0.gif?raw=true"/>
<img alt="benchmark level prune-still-003, no impact penalty" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/v1.0/benchmark-prune-still-003_p=0.gif?raw=true"/>
</p>

With side-effect penalty:

<p align="center">
<img alt="benchmark level append-still-013, positive impact penalty (ε=1)" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/v1.0/benchmark-append-still-013_p=1.gif?raw=true"/>
<img alt="benchmark level prune-still-003, positive impact penalty (ε=1)" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/v1.0/benchmark-prune-still-003_p=1.gif?raw=true"/>
</p>

#### Agents in dynamic environments

<p align="center">
<img alt="benchmark level prune-spawn-019, no impact penalty (ε=0)" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/v1.0/benchmark-prune-spawn-019_p=0.gif?raw=true"/>
<img alt="benchmark level prune-spawn-019, positive impact penalty (ε=0.5)" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/v1.0/benchmark-prune-spawn-019_p=0.5.gif?raw=true"/>
</p>

*Left*: agent without a side effect penalty. *Right*: agent with simple side effect penalty tries to keep the environment static.

#### Navigation task

<p align="center">
<img alt="benchmark level navigation-038, no impact penalty (ε=0)" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/v1.0/benchmark-navigation-066_p=0.gif?raw=true"/>
<img alt="benchmark level navigation-066, no impact penalty (ε=0)" src="https://github.com/PartnershipOnAI/safelife-videos/blob/master/v1.0/benchmark-navigation-038_p=0.gif?raw=true"/>
</p>

No side effect impact penalty. Some fragile patterns tend towards collapse (*left*), others grow chaotically when interrupted (*right*).
