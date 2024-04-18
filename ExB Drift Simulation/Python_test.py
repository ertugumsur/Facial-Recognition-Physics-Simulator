import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit


#Pre-Defined Do Not Change
x = 0; y = 1; z = 2

#Particle Constants
q_over_m_electron = 175882001076 # in Coulombs over kilograms
q_over_m_xenon = 733945 # in Coulombs over kilograms
q_over_m_argon = 2411817 # in Coulombs over kilograms
q_over_m_krypton = 1149838 # in Coulombs over kilograms
q_over_m = -1
eV = 30
speed = 100000
#Distance
L = 0.0005 # in meters

#Position Vector Start
P_x = 0; P_y = 0; P_z = 0 # in meters

#Velocity Vector Start
V_x = 3*(10**6); V_y = 0; V_z = 0 # in meters per seconds

# Magnetic Field Start
B_x = 0; B_y = 0; B_z = 0.03 # in Tesla

# Electric Field Start
E_x = 0; E_y = 300/L; E_z = 0 # in Volts per meters

#Acceleration Vector
A_x = 0; A_y = 0; A_z = 0 # in meters per squared seconds

#Angular Frequency
w = abs(q_over_m_electron)*B_z
T = 2*3.14159/w

#Time Constants
t0 = 0 #start time in seconds
t = 3*T #end time in seconds
dt = 0.0000000000001 #time interval between steps in seconds

p = np.array([P_x, P_y, P_z], dtype=float)
v = np.array([V_x, V_y,V_z], dtype=float)
a = np.array([A_x, A_y, A_z], dtype=float)
B = np.array([B_x, B_y, B_z], dtype=float)
E = np.array([E_x, E_y, E_z], dtype=float)

#Containers for graphing
px = np.array([0, p[x]])
py = np.array([0, p[y]])
vx = np.array([0, v[x]])
vy = np.array([0, v[y]])
ta = np.array([0, t0])
vx_contain = np.array([0])
i = 0
j = 0
k = 0
while t0 <= t:
    a[x] = q_over_m_electron*(E[x] + v[y]*B[z] - v[z]*B[y])
    a[y] = q_over_m_electron*(E[y] + v[z]*B[x] - v[x]*B[z])
    a[z] = q_over_m_electron*(E[z] + v[x]*B[y] - v[y]*B[x])
    v[x] = v[x] + a[x]*dt
    v[y] = v[y] + a[y]*dt
    v[z] = v[z] + a[z]*dt
    p[x] = p[x] + v[x]*dt
    p[y] = p[y] + v[y]*dt
    p[z] = p[z] + v[z]*dt
    t0 = t0 + dt
    px = np.append(px, p[x])
    py = np.append(py, p[y])
    vx = np.append(vx, v[x])
    vy = np.append(vy, v[y])
    ta = np.append(ta, t0)
    
    if (vx[-2] > vx[-3] and vx[-2] > v[-1]) and j ==0:
        #print(v[-2])
        vx_contain = np.append(vx_contain, v[x])
        j +=1

    if (vx[-2] < vx[-3] and vx[-2] < v[-1]) and k <= 1:
        #print(v[-2])
        vx_contain = np.append(vx_contain, v[x])
        k +=1

    if (py[-2] > 0 and p[y] < 0 ) or (py[-2] < 0 and p[y] > 0):
        i +=1
        if i == 2:
            #print(t0)
            s = 1
    if (py[-2] > py[-3] and py[-2] > p[y]) and j ==0:
        #print(p[y])
        j +=1
    if (py[-2] < py[-3] and py[-2] < p[y]) and k ==0:
        #print(p[y])
        k +=1
v_avg = px[-1] / t
print(v_avg)

#print(T)
fig1 = plt.figure("Motion Graph")
plt.plot(px,py, "--")
fig2 = plt.figure("Velocity Graph")
plt.plot(ta,vx)
#plt.plot(ta, vy)
fig3 = plt.figure("Y-Position Graph")
plt.plot(ta,py)
fig4 = plt.figure("X-Position Graph")
plt.plot(ta,px)
plt.show()

'''range = np.linspace(0.1,0.5,100)
rn = np.array([])
for x in range:
    rn = np.append(rn, simulate(x))
    
fig1 = plt.figure("Drift Velocity vs Magnetic Field")
plt.plot(range,rn)
plt.xlabel("Magnetic Field (T)")
plt.ylabel("Drift Velocity (m/s)")
plt.show()'''

'''def f(x,a,b):
    return a*(x**b)

par,cov = curve_fit(f, range, rn)

print(par[0])
print(par[1])

fig1 = plt.figure("Drift Velocity vs Electric Field")
plt.plot(range,rn,"ko", markersize=2)
plt.plot(range, f(range,par[0],par[1]))
plt.xlabel("Electric Field (V/m)")
plt.ylabel("Drift Velocity (m/s)")
plt.show()'''
