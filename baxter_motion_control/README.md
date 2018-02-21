#baxter_motion_control
Control the motion of the baxter's left arm and move it to the desired position.

1) Run the baxter shell script with sim specified
	./baxter.sh sim

2) launch the gazebo simulator
	roslaunch baxter_gazebo baxter_world.launch

3) Run the python code
	rosrun baxter_motion_control joint_position_control.py

