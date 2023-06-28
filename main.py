from fourrooms import Fourrooms


env = Fourrooms()
a = env.observation_space
print(a)
print("goal",env.goal)

b = env.reset()
print("b",b)

empty = env.empty_around(env.agent_location)

print(empty,"empty")

env.switch_goal()
print("goal change")

print("goal",env.goal)
b = env.reset()
print("b",b)

empty = env.empty_around(env.agent_location)

print(empty,"empty")