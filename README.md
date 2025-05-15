# Voice-Command Interface for Context-Aware UV Disinfection
Hands-free control of a mobile-manipulator robot using Whisper (STT), a Vision-Language Model (VLM), and an LLM prompt pipeline.

<!-- > **Reference paper**  
> A. G. Sanchez, M. R. Miller, W. D. Smart, *A Hands-Free Interface for Disinfection During Tabletop Tasks*, RO-MAN 2024.  
> PDF included in this repository (`A_Hands-Free_Interface_for_Disinfection_During_Tabletop_Tasks.pdf`). -->

---

## Motivation & Scenario

Our long-term goal is a system that **works alongside a human performing multi-step tabletop tasks involving several objects**.  
The robot:

* **Monitors** which objects the human touches,  
* **Recognizes** when the task is finished, then  
* **Disinfects only those “contaminated” items**, minimizing disruption to the human coworker.

While the ultimate target is medical or laboratory workflows , we validate the approach in a **bartending task**: 
<!-- (extending our prior Ebola and COVID-19 disinfection research [1, 2]) -->

* Multiple drink ingredients sit on a bar.  
* A participant prepares any drink they choose, in any order.  
* The robot must flexibly understand the chosen recipe, track used items, and disinfect them afterward—asking for a demonstration if it lacks a tool path for a new object.
 <!-- [3]. -->

---

## System Flow

1. **Object Mapping & Labeling**  
   The robot scans the table, clusters the point cloud, and labels each object via the VLM.

2. **Initiation**  
   When a bartender approaches, the robot greets them and asks what drink they plan to make.

3. **Preparation**  
   From the spoken drink name, the LLM infers which labeled objects are required.

4. **Tracking**  
   As objects are manipulated, the robot marks them “contaminated” and monitors task progress.

5. **Disinfection**  
   On confirmation that the drink is complete, the robot disinfects the contaminated items.  
   * If a disinfection trajectory is missing, it requests a Programming-by-Demonstration (PbD) demo from the human operator.

![](repo_media/system_flow.png)
<!-- 
All explicit communication is **verbal**; no tablets, buttons, or markers are required.  
Only the set of objects that *can* be disinfected is pre-defined (each with a learned end-effector trajectory). -->




---

## Quick Start (to be completed)

```bash
# Clone & build
# TODO: add dependency and setup instructions
```

<!-- ```
sudo apt-get update
sudo apt-get install ros-melodic-fetch*
```

You also need to install the rviz_visual_tools for the cone marker. Further information [here](https://github.com/PickNikRobotics/rviz_visual_tools/blob/melodic-devel).
```
sudo apt-get install ros-melodic-rviz-visual-tools
```

The octomap dependencies need to be installed.
```
sudo apt-get install ros-melodic-octomap
sudo apt-get install ros-melodic-octomap-server
sudo apt-get install ros-melodic-octomap-mapping
```

You also need to pip and pip3 install:
* rospkg
* scipy
* sympy
* planar
* pyvista
* PyQt5


### Build
Add the package to your src file in your workspace.

```
git clone https://github.com/osuprg/fetch_disinfectant_project.git
cd ~/catkin_ws/
catkin_make
source devel/setup.bash
```

## Planar Disinfection
Run the below launch files and python node into separate terminals to get things started.

```
roslaunch fetch_disinfectant_project_moveit_config fetch_world.launch
```
```
roslaunch fetch_disinfectant_project_moveit_config disinfectant_project.launch
```
```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```
Click on the publish point feature and then click on one of the cubes in the octomap. This should populate an interactive marker at the location of the cube.

![](images/fetch.gif)

Once you have at least three (four preferred) markers up,  you will see a plane marked by these points and a lawnmower path defined by this plane at a height offset.

### Using the GUI
![](images/gui.png)
* Select "Plan Path" when you're ready with the lawnmower path and there are no collision error messages
* Select "Execute Path" if the planned path succeeds without any errors
* Select "Initial Pose" to take the arm to the initial position.
* Select "Tuck Arm" to take the arm back to its home position.


## Non-planar Disinfection
Run the below launch files and python node into separate terminals to get things started.
```
roslaunch fetch_disinfectant_project_moveit_config short_table_gazebo.launch
```
```
roslaunch fetch_disinfectant_project_moveit_config nonplanar_disinfection.launch
```
```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py

```

Select *Initial Pose* on the menu. This will allow the fetch to have an arm configuration that is more ideal before planning and executing a planned path.

In three separate terminals, run the following python nodes:
```
rosrun fetch_disinfectant_project_moveit_config head_movement.py
```
```
rosrun fetch_disinfectant_oject_moveit_config pcl_filter.py
```
```
rosrun fetch_disinfectant_project_move_config nonplanar_waypoint_generator.py
```

Similar to the planar disinfection section, click on the publish point feature on the top toolbar of RViz. Then click on one of the cubes in the octomap. This should populate an interactive marker at the location of the cube.


Once you have at least three (four preferred) markers up,  you will be able to see an arrow marker of the tool path.


![](images/nonplanar1.gif)

Then select the *Plan Path* button. If the planned trajectory behaves as desired, then select the *Execute Path* button. Once the path is complete, return the arm configuration by clicking on the *Initial Pose* button.
![](images/nonplanar2.gif) -->
