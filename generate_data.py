"""
Phrase Structured Grammar to produce SCAN commands
"""
import random

u = ["walk", "look", "run", "jump"]
new_u = ["hop", "skip", "jump" "juggle", "glance"]

x = {}

x["walk"] = " I_WALK"
x["look"] = " I_LOOK"
x["run"] = " I_RUN"
x["jump"] = " I_JUMP"

x["turn left"] = " I_TURN_LEFT"
x["turn right"] = " I_TURN_RIGHT"
x["turn opposite left"] = " I_TURN_LEFT" + " I_TURN_LEFT"
x["turn opposite right"] = " I_TURN_RIGHT" + " I_TURN_RIGHT"
x["turn around left"] = x["turn opposite left"] + x["turn opposite left"]
x["turn around right"] = x["turn opposite right"] + x["turn opposite right"]

#commands = x.keys()
#for key in x.keys():
for key in u:
	x[key + " left"] = x["turn left"] + x[key]
	x[key + " right"] = x["turn right"] + x[key]
	x[key + " opposite left"] = x["turn opposite left"] + x[key]
	x[key + " opposite right"] = x["turn opposite right"] + x[key]
	x[key + " around left"] = x["turn left"] + x[key] + x["turn left"] + x[key] + x["turn left"] + x[key] + x["turn left"] + x[key]
	x[key + " around right"] = x["turn right"] + x[key] + x["turn right"] + x[key] + x["turn right"] + x[key] + x["turn right"] + x[key]

#import pdb; pdb.set_trace()
cur = list(x.keys())
for key in cur:
	x[key + " twice"] = x[key] + x[key]
	x[key + " thrice"] = x[key] + x[key] + x[key]

cur = list(x.keys())
for k1 in cur:
	for k2 in cur:
		x[k1 + " and " + k2] = x[k1] + x[k2]
		x[k1 + " after " + k2] = x[k2] + x[k1]
"""
poop = list(x.keys())
crack = []
for i in range(len(poop)):
	if "walk" in poop[i] and "jump" in poop[i]:
		crack.append(poop[i])
import pdb; pdb.set_trace()
"""



keys = list(x.keys())
random.shuffle(keys)
random_x = [(key, x[key]) for key in keys]

with open("data_pel.txt", "w") as file:
	for item in random_x:
		file.write("IN: {0} OUT:{1} \n".format(item[0], item[1]))



