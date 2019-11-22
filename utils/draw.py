"""Real-time Parameter Monitor
"""
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt


class Monitor(object):
    def __init__(self, points=300, callback=None):
        self.points = points
        self.callback = callback

        fig, ax = plt.subplots()
        ax.set_ylim([-10, 5])
        ax.set_xlim([0, self.points])
        ax.set_autoscale_on(False)

        ax.set_xticks([])
        ax.set_yticks(range(-10, 6, 1))
        ax.grid(True)

        self.au25 = [None] * self.points
        self.au45 = [None] * self.points

        self.l_au25, = ax.plot(range(self.points), self.au25, label='AU25')
        self.l_au45, = ax.plot(range(self.points), self.au45, label='AU45')

        ax.legend(loc='upper center',
                  ncol=2, prop=font_manager.FontProperties(size=10))

        timer = fig.canvas.new_timer(interval=100)
        timer.add_callback(self.OnTimer, ax)
        timer.start()
        fig.show()

    def OnTimer(self, ax):
        tmp = self.callback()

        self.au25 = self.au25[1:] + [tmp[0][0]]
        self.au45 = self.au45[1:] + [tmp[0][1]]

        self.l_au25.set_ydata(self.au25)
        self.l_au45.set_ydata(self.au45)

        ax.draw_artist(self.l_au25)
        ax.draw_artist(self.l_au45)

        ax.figure.canvas.draw()
