import numpy as np
import matplotlib.pyplot as plt

# === PARAMETRY OGÓLNE ===
fig_width = 6.7   # cale ~17 cm (pełna szerokość A4)
fig_height = 3.2  # proporcje wykresu
font_size = 9

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": font_size,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "dejavusans",  # pasuje do Arial
})

# === DANE ===
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x)

# === WYKRES ===
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.plot(x, y, color='black', linewidth=1.0, label=r'$y = \sin(x)$')

# === USTAWIENIA OSI ===
# Osie w dolnym i lewym miejscu
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# === STRZAŁKI NA OSIACH ===
arrow_style = dict(arrowstyle='->', color='black', lw=1.5)
ax.annotate('', xy=(2*np.pi + 0.3, 0), xytext=(-0.1, 0), arrowprops=arrow_style, clip_on=False)
ax.annotate('', xy=(0, 1.1), xytext=(0, -1.1), arrowprops=arrow_style, clip_on=False)

# === ZAKRES I TICKI ===
ax.set_xlim(0, 2*np.pi + 0.3)
ax.set_ylim(-1.1, 1.1)
ax.set_xticks([np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels([r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
ax.set_yticks([-1,-0.5,0,0.5, 1])

# === OPISY ===
ax.text(2*np.pi + 0.35, -0.05, r'$x$', fontsize=font_size, va='center')
ax.text(0.05, 1.12, r'$y$', fontsize=font_size, va='bottom')

# === OPIS FUNKCJI ===
ax.legend(loc='upper right', frameon=False, fontsize=font_size)

# === WYGLĄD I ZAPIS ===
ax.grid(False)
plt.tight_layout(pad=0.2)
plt.savefig("sinus.eps", format='eps', bbox_inches='tight')
plt.show()
