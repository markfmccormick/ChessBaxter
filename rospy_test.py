import roslaunch

package = 'kinect2_bridge'
executable = 'kinect2_bridge.launch'
node = roslaunch.core.Node(package, executable)

launch = roslaunch.scriptapi.ROSLaunch()
launch.start()

process = launch.launch(node)
print process.is_alive()
process.stop()
