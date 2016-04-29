import os
import subprocess

zero = "../../../"
first = "./baxter.sh"
second = "source devel/setup.bash"
third = "roslaunch lab3 baxter_right_hand_track.launch &"


# print os.getcwd()
# os.chdir(zero)
# print os.getcwd()
# os.system(first)

# print os.getcwd()
# os.system(second)
subprocess.call(third)
print 'hi'