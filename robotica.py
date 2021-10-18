import numpy as np

def rotation_matrix(angle,axis):	
	k_hat=cross_matrix(axis)
	return np.identity(3)+np.sin(angle)*k_hat+(1-np.cos(angle))*k_hat.dot(k_hat)

def rot_x(angle):	
	return np.array(((1, 0, 0),(0, np.cos(angle), -np.sin(angle)), (0, np.sin(angle), np.cos(angle))))

def rot_y(angle):	
	return np.array(((np.cos(angle), 0, np.sin(angle)),(0, 1, 0), (-np.sin(angle),0, np.cos(angle))))

def rot_z(angle):	
	return np.array((( np.cos(angle), -np.sin(angle),0), (np.sin(angle), np.cos(angle),0),(0, 0, 1)))
				


def cross_matrix(v):	
	return np.array(((0, -v[2], v[1]),(v[2], 0, -v[0]), (-v[1], v[0], 0)))

def flow(x,u,dT):
		x+=u*dT
	return x



