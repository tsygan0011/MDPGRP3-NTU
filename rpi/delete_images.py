import os


dirs = [r"/home/pi/MDP2/src/images/task1/captured",
       r"/home/pi/MDP2/src/images/task1/preprocess",
       r"/home/pi/MDP2/src/images/task1/results",
       r"/home/pi/MDP2/src/images/task1/runs",
       r"/home/pi/MDP2/src/images/task2/captured",
       r"/home/pi/MDP2/src/images/task2/preprocess",
       r"/home/pi/MDP2/src/images/task2/results",
       r"/home/pi/MDP2/src/images/task2/runs"]

for dir in dirs:
    for files in os.listdir(dir):
        os.remove(os.path.join(dir, files))
