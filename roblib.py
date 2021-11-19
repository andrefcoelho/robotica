import numpy as np

def rotation_matrix(angle,axis):	
	h_hat=skew(axis)
	return np.identity(3)+np.sin(angle)*h_hat+(1-np.cos(angle))*h_hat.dot(h_hat)

def rot_x(angle):	
	return np.array(((1, 0, 0),(0, np.cos(angle), -np.sin(angle)), (0, np.sin(angle), np.cos(angle))))

def rot_y(angle):	
	return np.array(((np.cos(angle), 0, np.sin(angle)),(0, 1, 0), (-np.sin(angle),0, np.cos(angle))))

def rot_z(angle):	
	return np.array((( np.cos(angle), -np.sin(angle),0), (np.sin(angle), np.cos(angle),0),(0, 0, 1)))
				
def skew(v):	
	return np.array(((0, -v[2], v[1]),(v[2], 0, -v[0]), (-v[1], v[0], 0)))

def integrator(u,y0,dT):
  y=np.zeros_like(u)
  y[0]=y0
  for i in range(len(y)-1):
    y[i+1]=y[i]+u[i]*dT
  return y

def flow(u,y0,dT):
	return y0+u*dT
	
def rectangularShape(center,width,height,angle):
  """
    Determina os quatro vértices de um retângulo dada a posição de seu centro (center), sua largura (width), sua altura (height) e sua orientação (angle). 
  """
  leftBottom=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).dot(np.array([-width/2,-height/2]))+center
  leftTop=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).dot(np.array([-width/2,height/2]))+center
  rightBottom=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).dot(np.array([width/2,-height/2]))+center
  rightTop=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]).dot(np.array([width/2,height/2]))+center
  points = [leftBottom,rightBottom,rightTop,leftTop]
  return points

def exp_so3(phi):
  normPhi=np.linalg.norm(phi)
  if normPhi<1e-15:
    return np.identity(3)
  else:
    phi_hat=skew(phi)
    return np.identity(3)+np.sin(normPhi)*phi_hat/normPhi+(1-np.cos(normPhi))*phi_hat.dot(phi_hat)/(normPhi*normPhi)

def log_so3(R):
  Tr=np.trace(R)-1
  if abs(Tr)>2:
    Tr=2*Tr/abs(Tr)
  if abs(Tr-3)<1e-10:
    return np.zeros_like(R)
  else:
    phi=np.arccos(0.5*Tr);
    if np.any(np.isnan(phi)) or np.linalg.norm(phi)<1e-10:
      return np.zeros_like(R)
    else:
      L_hat=phi/(2*np.sin(phi))*(R-R.T)
      return 0.5*(L_hat-L_hat.T)


def A(phi):
  normPhi=np.linalg.norm(phi)
  if normPhi<1e-15:
    return np.identity(3)
  else:
    phi_hat=skew(phi)
    return np.identity(3)+(1-np.cos(normPhi))/normPhi*phi_hat/normPhi + (1-np.cos(normPhi)/normPhi)*phi_hat.dot(phi_hat)/normPhi/normPhi

def exp_se3(phi,q):
  normPhi=np.linalg.norm(phi)
  if normPhi<1e-15:
    R=np.identity(3)
    p=[0,0,0]
  else:
    R=exp_so3(phi)
    p=A(phi).dot(q)
  return SE3(R,p)


class SE3:
  """Classe de matrizes em SE(3)"""
  def __init__(self, R, p):   #initiliza H dados R e p
    self.R=R               
    self.p=p
    self.value=np.concatenate((np.concatenate((R, [[p[0]],[p[1]],[p[2]]]), axis=1), [[0,0,0,1]]), axis=0)  #forma matricial de H 
    
  def dot(self,v):             #multiplicacao de um vetor v em R^3 por H
   return self.R.dot(v)+p  
