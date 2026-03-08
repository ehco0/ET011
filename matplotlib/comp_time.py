import numpy as np
import matplotlib.pyplot as plt

n_eval = np.arange(1, 201)

cfd_time = 3 * 3600      
vsp_time = 3             
ai_time = 0.40625

cfd_cum = n_eval * cfd_time
vsp_cum = n_eval * vsp_time
ai_cum = n_eval * ai_time

plt.figure(figsize=(6,4))
plt.plot(n_eval, cfd_cum / 3600, label='CFD (768 cores)', linewidth=2)
plt.plot(n_eval, vsp_cum / 3600, label='OpenVSP (4 cores)', linewidth=2)
plt.plot(n_eval, ai_cum / 3600, label='Neural Network', linewidth=2)


plt.xlabel('Number of AOA Evaluations')
plt.ylabel('Cumulative Time (h)')
plt.yscale('log') 
plt.title('Cumulative Computational Cost')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comp time.png", dpi=1200)
plt.show()
