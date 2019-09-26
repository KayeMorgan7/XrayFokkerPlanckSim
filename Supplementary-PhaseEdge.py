#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 1 13:11:30 2019

@author: morgank

Built on 'slider_demo.py' from https://matplotlib.org/3.1.1/gallery/widgets/slider_demo.html
"""
import	numpy as	np
import	matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.15, bottom=0.5)
x = np.arange(-15.0, 15.0, 0.001)
a0 = 0.25
p0 = 1.5
b0 = 0.5
L0 = 0.3
R0 = 2
z0=0.5
k0=1.

#Intensity without sample
s = a0*np.sin(x/p0)+b0
l, = plt.plot(x, s, lw=1, color='grey')

#Intensity with sample
ms=(z0*(2*np.exp(-x**2)*x*(R0-L0))/(np.sqrt(np.pi)*k0)+1)*(a0*np.sin((x-np.arctan((np.exp(-x**2)*(R0-L0))/(np.sqrt(np.pi)*(k0/z0+(2*np.exp(-x**2)*x*(R0-L0))/(np.sqrt(np.pi))))))/p0)+b0)
m, = plt.plot(x, ms, lw=2, color='black')
plt.xlabel('Position')
plt.ylabel('Intensity')
plt.legend((l,m), ('Without sample', 'With sample'), fontsize='small')
plt.title('Phase edge')
  
plt.axis([-15.0, 15.0, 0, 1])

axcolor = 'lightgoldenrodyellow'
axp = plt.axes([0.35, 0.1, 0.55, 0.03], facecolor=axcolor)
axa = plt.axes([0.35, 0.15, 0.55, 0.03], facecolor=axcolor)
axb = plt.axes([0.35,0.2,0.55,0.03], facecolor=axcolor)
axL = plt.axes([0.35,0.25,0.55,0.03], facecolor=axcolor)
axR = plt.axes([0.35,0.3,0.55,0.03], facecolor=axcolor)
axz = plt.axes([0.35,0.35,0.55,0.03], facecolor=axcolor)


sp = Slider(axp, 'Period of illumination p', 0.1, 5.0, valinit=p0)
sa = Slider(axa, 'Amplitude of illumination a', 0.1, 0.5, valinit=a0)
sb = Slider(axb, 'Average intensity b', 0.1, 1.0, valinit=b0)
sL = Slider(axL, 'Phase shift on left', 0.01, 4, valinit=L0)
sR = Slider(axR, 'Phase shift on right', 0.01, 4, valinit=R0)
sz = Slider(axz, 'Propagation distance z', 0.01, 1.0, valinit=z0)

def update(val):
    a = sa.val
    p = sp.val
    b = sb.val
    L = sL.val
    R = sR.val
    z = sz.val
    l.set_ydata(a*np.sin(x/p)+b)
    ms=(z*(2*np.exp(-x**2)*x*(R-L))/(np.sqrt(np.pi)*k0)+1)*(a*np.sin((x-np.arctan((np.exp(-x**2)*(R-L))/(np.sqrt(np.pi)*(k0/z+(2*np.exp(-x**2)*x*(R-L))/(np.sqrt(np.pi))))))/p)+b)
    m.set_ydata(ms)
    
    fig.canvas.draw_idle()
sp.on_changed(update)
sa.on_changed(update)
sb.on_changed(update)
sL.on_changed(update)
sR.on_changed(update)
sz.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sp.reset()
    sa.reset()
    sb.reset()
    sL.reset()
    sR.reset()
    sz.reset()
button.on_clicked(reset)

plt.show()

