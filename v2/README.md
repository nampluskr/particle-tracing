Version 2

```
## Variables: u = [t, pos, vel, ang, omg]
# [2D]: u = [t, (x,y), (vx,vy), omega, theta]
# [3D]: u = [t, (x,y,z), (vx,vy,vz), (e1,e2,e3,e0), (w1,w2,w3)]
```

```python
def time_(u):
    return u[0] if u.ndim == 1 else u[:,0]

def pos_(u):
    if u.ndim == 1:
        return u[1:3] if u.size == 7 else u[1:4]
    else:
        return u[:,1:3] if u.shape[-1] == 7 else u[:,1:4]

def vel_(u):
    if u.ndim == 1:
        return u[3:5] if u.size == 7 else u[4:7]
    else:
        return u[:,3:5] if u.shape[-1] == 7 else u[:,4:7]

def ang_(u):
    if u.ndim == 1:
        return u[5] if u.size == 7 else u[7:11]
    else:
        return u[:,5] if u.shape[-1] == 7 else u[:,7:11]

def omg_(u):
    if u.ndim == 1:
        return u[6] if u.size == 7 else u[11:]
    else:
        return u[:,6] if u.shape[-1] == 7 else u[:,11:]
```
