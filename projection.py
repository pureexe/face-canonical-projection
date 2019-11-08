import numpy as np
import matplotlib.pyplot as plt

position = np.load('data/canonical_vertices_righthand.npy')
position[:,2] = -position[:,2]
#Intrinsic
focal = 1/128
height = 256
intrinsic = np.array([
    [focal*height / 2,                 0,   0],
    [               0,  focal*height / 2,   0],
    [               0,                 0,        1]
]) 

focal_x = 64
focal_y = 43
intrinsic = np.array([
    [focal_x, 0, 128],
    [ 0, focal_y, 128],
    [ 0, 0,   1]
]) 


#create color
z_max = np.max(position)
z_min = np.min(np.min(position))
z_len = z_max - z_min
color = position[:,2]
color = (color - z_min)/ z_len

#projection
projected = np.matmul(intrinsic,position.T)
projected = (projected / projected[2,:]).T
projected = projected.astype(np.int32)
image = np.zeros((256,256,3))
for i in range(len(projected)):
    try:
        u,v,_ = projected[i]
        image[v,u,:] = color[i]
    except:
        pass
plt.imshow(image)
plt.imsave("saved_image/focal_{}_{}.png".format(focal_x,focal_y),image)
plt.show()
