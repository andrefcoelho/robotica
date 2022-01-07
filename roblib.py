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
  if normPhi<1e-10:
    return np.identity(3)
  else:
    phi_hat=skew(phi)
    return np.identity(3)+np.sin(normPhi)*phi_hat/normPhi+(1-np.cos(normPhi))*phi_hat.dot(phi_hat)/(normPhi*normPhi)

def log_so3(R):
  Tr=np.trace(R)
  if abs(Tr-3)<1e-10:
    return np.zeros(3)
  else:
    psi=np.arccos(0.5*(Tr-1))
    if np.linalg.norm(psi)<1e-10:
      return np.zeros(3)
    else:
      L_hat=psi/(2*np.sin(psi))*(R-R.T)
      return vee(0.5*(L_hat-L_hat.T))
      
def log_se3(H):
  phi=log_so3(H.R)
  q=A_inv(phi).dot(H.p)
  return phi,q

def vee(V):
	return [V[2,1],V[0,2],V[1,0]]

def A(phi):
	normPhi=np.linalg.norm(phi)
	if normPhi<1e-15:
		return np.identity(3)
	else:
		phi_hat=skew(phi)
		return np.identity(3)+(1-np.cos(normPhi))/normPhi*phi_hat/normPhi + (1-np.sin(normPhi)/normPhi)*phi_hat.dot(phi_hat)/normPhi/normPhi
def A_inv(phi):
  normPhi=np.linalg.norm(phi)
  if normPhi<1e-15:
    return np.identity(3)
  else:
    phi_hat=skew(phi)
    alpha=0.5*normPhi/np.tan(0.5*normPhi)
    return np.identity(3)-0.5*phi_hat + (1-alpha)*phi_hat.dot(phi_hat)/normPhi/normPhi


def exp_se3(phi,q):
  R=exp_so3(phi)
  p=A(phi).dot(q)
  return SE3(R,p)
  
def se3_integrator_body(Vb,H0,dT):
  H=[]
  H.append(H0)
  for i in range(np.size(Vb,1)-1):
    H_=H[i].value.dot(exp_se3(Vb[3:6,i]*dT,Vb[0:3,i]*dT).value)
    H.append(SE3(H_[0:3,0:3],H_[0:3,3]))
  return H
def se3_integrator_spatial(Vs,H0,dT):
  H=[]
  H.append(H0)
  for i in range(np.size(Vs,1)-1):
    H_=exp_se3(Vs[3:6,i]*dT,Vs[0:3,i]*dT).value.dot(H[i].value)
    H.append(SE3(H_[0:3,0:3],H_[0:3,3]))
  return H



class SE3:
  """Classe de matrizes em SE(3)"""
  def __init__(self, R, p):   #initiliza H dados R e p
    self.R=R               
    self.p=p
    self.value=np.concatenate((np.concatenate((R, [[p[0]],[p[1]],[p[2]]]), axis=1), [[0,0,0,1]]), axis=0)  #forma matricial de H 
    
  def dot(self,v):             #multiplicacao de um vetor v em R^3 por H
   return self.R.dot(v)+p  


def Adjoint(H):
  return np.block([[H.R, skew(H.p).dot(H.R)],[np.zeros([3,3]), H.R]])
def Adjoint_inv(H):
  return np.block([[H.R.T, -skew(H.R.T.dot(H.p)).dot(H.R.T)],[np.zeros([3,3]), H.R.T]])  
