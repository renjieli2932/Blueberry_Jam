# This message contains object shape and its position.
# (0, 0) is at top-left corner of image

Header header
int32[] circle_center      # center position of circle
int32[] square_center      # center position of square
int32[] triangle_center    # center position of triangle
int32   circle_radius      # positions of vertices, size is [3, 2]
geometry_msgs/Point32[] square_vertices    # positions of vertices, size is [4, 2]
geometry_msgs/Point32[] triangle_vertices  # radius of circle
