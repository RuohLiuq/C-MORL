import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

r0 = torch.load('rewardhopper0.pt')
r1 = torch.load('rewardhopper1.pt')
r2 = torch.load('rewardhopper2.pt')
r3 = torch.load('rewardhopper3.pt')
r4 = torch.load('rewardhopper4.pt')
r5 = torch.load('rewardhopper5.pt')

r0_ = torch.load('rewardant0.pt')
r1_ = torch.load('rewardant1.pt')
r2_ = torch.load('rewardant2.pt')
r3_ = torch.load('rewardant3.pt')
r4_ = torch.load('rewardant4.pt')
r5_ = torch.load('rewardant5.pt')

r = np.concatenate((r0,r1,r2,r3,r4,r5,r0_,r1_,r2_,r3_,r4_,r5_),axis=0)
initial_obj = np.concatenate([arr[4000:4001] for arr in [r0,r1,r2,r3,r4,r5]])

fig = plt.figure(figsize=[7, 7])
plt.plot(r0[:4000,0],r0[:4000,1],'rs', markersize=2)
plt.plot(r1[:4000,0],r1[:4000,1],'gs', markersize=2)
plt.plot(r2[:4000,0],r2[:4000,1],'bs', markersize=2)
plt.plot(r3[:4000,0],r3[:4000,1],'cs', markersize=2)
plt.plot(r4[:4000,0],r4[:4000,1],'ms', markersize=2)
plt.plot(r5[:4000,0],r5[:4000,1],'ys', markersize=2)
plt.plot(r0[4000:,0],r0[4000:,1],'ks', markersize=2)
plt.plot(r1[4000:,0],r1[4000:,1],'ks', markersize=2)
plt.plot(r2[4000:,0],r2[4000:,1],'ks', markersize=2)
plt.plot(r3[4000:,0],r3[4000:,1],'ks', markersize=2)
plt.plot(r4[4000:,0],r4[4000:,1],'ks', markersize=2)
plt.plot(r5[4000:,0],r5[4000:,1],'ks', markersize=2)

plt.plot(r0_[4000:,0],r0_[4000:,1],'ks', markersize=2)
plt.plot(r1_[4000:,0],r1_[4000:,1],'ks', markersize=2)
plt.plot(r2_[4000:,0],r2_[4000:,1],'ks', markersize=2)
plt.plot(r3_[4000:,0],r3_[4000:,1],'ks', markersize=2)
plt.plot(r4_[4000:,0],r4_[4000:,1],'ks', markersize=2)
plt.plot(r5_[4000:,0],r5_[4000:,1],'ks', markersize=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.ylim(0, 3500)
plt.xlim(0, 3500)
plt.title('Ant')
plt.savefig('antcpo.png')

from pymoo.factory import get_performance_indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
perf_ind = get_performance_indicator("hv", ref_point = np.zeros((r.shape[1])))
hv = perf_ind.do(-r)
print(hv)

non_dom = NonDominatedSorting().do(-r, only_non_dominated_front=True)        
non_dom_plot = r[non_dom]
fig = plt.figure(figsize=[7, 7])
plt.plot(non_dom_plot[:,0],non_dom_plot[:,1],'rs', markersize=2)
plt.plot(initial_obj[:,0],initial_obj[:,1],'bs', markersize=4)
plt.xlabel("X")
plt.ylabel("Y")
plt.ylim(0, 3500)
plt.xlim(0, 3500)
plt.title('Ant')
plt.savefig('antcpo_dom.png')



