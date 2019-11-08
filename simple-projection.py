import numpy as np
import matplotlib.pyplot as plt

position = np.load('data/canonical_vertices_righthand.npy')

projected = (position+1)*128
projected = projected.astype(np.int32)
image = np.zeros((256,256,3))
for i in range(len(projected)):
    try:
        u,v,c = projected[i]
        image[v,u,:] = c / 255.0
    except:
        pass
plt.imshow(image)
plt.show()