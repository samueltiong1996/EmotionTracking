import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sqlite3
with sqlite3.connect('database.db') as db:
    c=db.cursor()


fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)


def animate(i):
    

    c.execute('SELECT rowid, happy, sad FROM graph')
    ids =[]
    emotion1 = []
    emotion2 = []

    for row in c.fetchall():
        ids.append(row[0])
        emotion1.append(row[1])
        emotion2.append(row[2])

    ax1.clear()            
    ax1.plot(ids,emotion1,'-')
    ax1.plot(ids,emotion2,color='orange')    

ani = animation.FuncAnimation(fig,animate,interval=1000)
plt.show()   